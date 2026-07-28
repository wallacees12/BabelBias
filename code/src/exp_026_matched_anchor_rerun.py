"""exp_026 — length-matched anchor re-run for the event-ladder corpus.

The Russia–Ukraine 2022 cell carries both the lowest EN diagonal pull in
the ladder and by far its longest anchors, so event recency is
confounded with anchor size (thesis §threats). Remediation, as
pre-specified there: truncate every anchor lead to the token length of
the *shortest anchor in its conflict family*, re-embed the truncated
leads (no new LLM calls), and recompute the row-centred EN diagonal
against the already-embedded responses.

Outputs (data/exp_026_matched_anchors/):
  - anchor_cache.json           truncated-anchor embeddings (resumable)
  - matched_gradient.csv        per-event original vs matched EN pull
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import tiktoken

from babelbias.embedding import embed_short
from babelbias.wiki import extract_lead
from build_conflict_gradient import MODELS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
OUT_DIR = DATA / "exp_026_matched_anchors"
OUT_DIR.mkdir(exist_ok=True)
CACHE_PATH = OUT_DIR / "anchor_cache.json"
GRADIENT = DATA / "exp_007_timeline" / "gradient_all_conflicts.csv"

ENC = tiktoken.get_encoding("cl100k_base")


def load_bank(event: str) -> dict:
    return json.loads((PROJECT_ROOT / "Code" / "prompts" / f"{event}.json").read_text())


def anchor_lead(event: str, slug: str, lang: str) -> str | None:
    raw = DATA / event / "raw" / f"{slug}_{lang}_raw.json"
    if not raw.exists():
        return None
    content = json.loads(raw.read_text()).get("content", "")
    return extract_lead(content) or None


def main() -> None:
    rows = list(csv.DictReader(open(GRADIENT)))
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}

    # ---- pass 1: collect every anchor lead and its token length ----------
    leads: dict[tuple[str, str, str], str] = {}
    fam_lengths: dict[str, list[int]] = defaultdict(list)
    event_meta: dict[str, dict] = {}
    for r in rows:
        event, fam = r["event"], r["conflict"]
        bank = load_bank(event)
        langs = bank["languages"]
        qid_slug = {p["id"]: p["wiki_anchor_slug"] for p in bank["prompts"]}
        event_meta[event] = {"fam": fam, "langs": langs, "qid_slug": qid_slug}
        for slug in sorted(set(qid_slug.values())):
            for lang in langs:
                text = anchor_lead(event, slug, lang)
                if text is None:
                    logger.warning("missing raw lead: %s %s %s", event, slug, lang)
                    continue
                leads[(event, slug, lang)] = text
                fam_lengths[fam].append(len(ENC.encode(text)))

    fam_min = {f: min(v) for f, v in fam_lengths.items()}
    for f in sorted(fam_min):
        logger.info("family %-16s min anchor length = %4d tokens (n=%d)",
                    f, fam_min[f], len(fam_lengths[f]))

    # ---- pass 2: truncate + embed (cached) -------------------------------
    anchors: dict[tuple[str, str, str], np.ndarray] = {}
    n_new = 0
    for (event, slug, lang), text in leads.items():
        fam = event_meta[event]["fam"]
        toks = ENC.encode(text)[: fam_min[fam]]
        key = f"{event}|{slug}|{lang}|{fam_min[fam]}"
        if key not in cache:
            vec = embed_short(ENC.decode(toks))
            if vec is None:
                logger.warning("embed failed: %s", key)
                continue
            cache[key] = vec
            n_new += 1
            if n_new % 25 == 0:
                CACHE_PATH.write_text(json.dumps(cache))
        a = np.asarray(cache[key], dtype=np.float32)
        anchors[(event, slug, lang)] = a / np.linalg.norm(a)
    CACHE_PATH.write_text(json.dumps(cache))
    logger.info("anchors embedded: %d (%d new API calls)", len(anchors), n_new)

    # ---- pass 3: recompute row-centred diagonals per event ---------------
    out_rows = []
    for r in rows:
        event, fam = r["event"], r["conflict"]
        meta = event_meta[event]
        langs, qid_slug = meta["langs"], meta["qid_slug"]
        # per provider: matrix[resp_lang][anchor_lang] -> list of cosines
        per_model_diag_en = []
        per_model_diag = defaultdict(list)  # lang -> per-model diagonal
        for model in MODELS:
            emb_dir = DATA / event / "llm_embeddings" / model / event
            if not emb_dir.is_dir():
                continue
            sums = defaultdict(float)
            counts = defaultdict(int)
            for f in emb_dir.glob("*.json"):
                stem = f.stem                      # qid..._lang_rNN
                parts = stem.rsplit("_", 2)
                qid, rlang = parts[0], parts[1]
                if qid not in qid_slug or rlang not in langs:
                    continue
                d = json.loads(f.read_text())
                v = d.get("embedding")
                if not v:
                    continue
                v = np.asarray(v, dtype=np.float32)
                v /= np.linalg.norm(v)
                slug = qid_slug[qid]
                for alang in langs:
                    a = anchors.get((event, slug, alang))
                    if a is None:
                        continue
                    sums[(rlang, alang)] += float(v @ a)
                    counts[(rlang, alang)] += 1
            if not counts:
                continue
            mat = {k: sums[k] / counts[k] for k in counts}
            # row-centre
            ok = True
            for rlang in langs:
                rowvals = [mat.get((rlang, al)) for al in langs]
                if any(x is None for x in rowvals):
                    ok = False
                    break
            if not ok:
                continue
            for rlang in langs:
                rowmean = np.mean([mat[(rlang, al)] for al in langs])
                per_model_diag[rlang].append(mat[(rlang, rlang)] - rowmean)
            per_model_diag_en.append(per_model_diag["en"][-1])
        if not per_model_diag_en:
            logger.warning("no providers computed for %s", event)
            continue
        out = {
            "conflict": fam,
            "event": event,
            "year": r["year"],
            "n_models": len(per_model_diag_en),
            "en_pull_orig": float(r["en_pull"]),
            "en_pull_matched": float(np.mean(per_model_diag_en)),
            "fam_min_tokens": fam_min[fam],
        }
        for lang in langs:
            if lang != "en" and per_model_diag[lang]:
                out[f"{lang}_pull_matched"] = float(np.mean(per_model_diag[lang]))
        out_rows.append(out)
        logger.info("%-18s %-28s orig %+0.3f  matched %+0.3f",
                    fam, event, out["en_pull_orig"], out["en_pull_matched"])

    fields = sorted({k for row in out_rows for k in row}, key=str)
    with open(OUT_DIR / "matched_gradient.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    logger.info("wrote %s", OUT_DIR / "matched_gradient.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
