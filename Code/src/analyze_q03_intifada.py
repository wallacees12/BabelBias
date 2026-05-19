"""Deep dive on q03_intifada: per-provider cosine matrices + phrase-level
ablation to isolate what's actually moving the embedding.

Outputs to stdout. Also writes:
  data/israel_palestine/analysis/q03_intifada_cosine_matrix.csv
  data/israel_palestine/analysis/q03_intifada_ablation.csv

The ablation:
  for the provider with the biggest HE↔AR ingroup pull,
  (1) translate HE and AR responses to EN via DeepL,
  (2) re-embed the translated text,
  (3) compare cosines against (EN/HE/AR) anchors.

If translated-to-EN versions converge (closer to each other and to EN
anchor), then the source-language embedding shift was largely lexical /
script-driven, not content. If they stay split, the content choices
themselves are pulling toward each language's anchor — the real
ingroup-bias finding.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code" / "src"))

EVENT = "israel_palestine"
QID = "q03_intifada"
LANGS = ("en", "he", "ar")
ANCHOR_SLUG = "Second_Intifada"

ANCHOR_DIR = ROOT / "data" / EVENT / "processed_leads"
EMB_ROOT   = ROOT / "data" / EVENT / "llm_embeddings"
OUT_DIR    = ROOT / "data" / EVENT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Embedding utility ────────────────────────────────────────────────────
def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_anchor(lang: str) -> np.ndarray:
    f = ANCHOR_DIR / f"{ANCHOR_SLUG}_{lang}.json"
    return np.asarray(json.load(open(f))["embedding"])


def load_responses(model: str) -> dict[str, list[np.ndarray]]:
    """Returns {lang: [embedding, ...]} across all repeats of q03."""
    out: dict[str, list[np.ndarray]] = {l: [] for l in LANGS}
    rdir = EMB_ROOT / model / EVENT
    for f in sorted(rdir.glob(f"{QID}_*.json")):
        try:
            rec = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not rec.get("embedding") or rec.get("refusal"):
            continue
        lang = rec.get("language")
        if lang in out:
            out[lang].append(np.asarray(rec["embedding"]))
    return out


# ── Phase 1: per-provider 3×3 cosine matrix ──────────────────────────────
def cosine_matrix(model: str, anchors: dict[str, np.ndarray]) -> np.ndarray | None:
    """Rows = response_lang, cols = anchor_lang. NaN where no answers."""
    resp = load_responses(model)
    M = np.full((3, 3), np.nan)
    for i, rl in enumerate(LANGS):
        if not resp[rl]:
            continue
        # Mean across repeats first → one representative response embedding.
        # (Equivalent to mean of cosines for L2-normalised vectors; close
        # enough for our purposes here.)
        for j, al in enumerate(LANGS):
            M[i, j] = float(np.mean([cos(r, anchors[al]) for r in resp[rl]]))
    return M


def row_centre(M: np.ndarray) -> np.ndarray:
    row_means = np.nanmean(M, axis=1, keepdims=True)
    return M - row_means


# ── Phase 3: phrase ablation via DeepL translate-to-EN + re-embed ────────
def translate_to_en(text: str, source_lang: str) -> str:
    from babelbias.translate import translate
    return translate(text, target_lang="en", source_lang=source_lang)


def embed_text(text: str) -> np.ndarray:
    """Use the same text-embedding-3-small the pipeline uses everywhere."""
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.asarray(resp.data[0].embedding)


def load_response_text(model: str, lang: str, repeat: int = 0) -> tuple[str, str]:
    """Return (prompt_text, response_text) for a single record."""
    f = (ROOT / "data" / EVENT / "llm_responses" / model / EVENT
         / f"{QID}_{lang}_r{repeat:02d}.json")
    rec = json.loads(f.read_text())
    return rec.get("prompt_text", ""), rec.get("response_text", "")


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    anchors = {l: load_anchor(l) for l in LANGS}

    models = sorted({p.name for p in EMB_ROOT.iterdir() if p.is_dir()})

    print("=" * 86)
    print(f"q03_intifada · 3×3 row-centred cosine matrices per provider")
    print("rows = response_lang  ·  cols = anchor_lang  ·  diagonal = ingroup pull")
    print("=" * 86)

    rows = []
    summary = []
    for m in models:
        M = cosine_matrix(m, anchors)
        if M is None or np.all(np.isnan(M)):
            continue
        C = row_centre(M)
        print(f"\n{m}")
        print(f"  raw cosine            EN-anch     HE-anch     AR-anch")
        for i, rl in enumerate(LANGS):
            cells = [f"{M[i,j]:+.3f}" if not np.isnan(M[i,j]) else "  n/a "
                     for j in range(3)]
            print(f"    {rl}-resp           {cells[0]:>8s}  {cells[1]:>8s}  {cells[2]:>8s}")
        print(f"  row-centred           EN-anch     HE-anch     AR-anch")
        for i, rl in enumerate(LANGS):
            cells = [f"{C[i,j]:+.3f}" if not np.isnan(C[i,j]) else "  n/a "
                     for j in range(3)]
            print(f"    {rl}-resp           {cells[0]:>8s}  {cells[1]:>8s}  {cells[2]:>8s}")
        # Save to CSV
        for i, rl in enumerate(LANGS):
            for j, al in enumerate(LANGS):
                rows.append({"model": m, "response_lang": rl, "anchor_lang": al,
                              "raw_cosine": M[i, j], "row_centred": C[i, j]})
        # Summary: HE vs AR ingroup-pull
        he_pull = C[1, 1] if not np.isnan(C[1, 1]) else np.nan
        ar_pull = C[2, 2] if not np.isnan(C[2, 2]) else np.nan
        en_pull = C[0, 0] if not np.isnan(C[0, 0]) else np.nan
        # AR-pull-toward-HE = C[2,1] (AR response, HE anchor)
        # HE-pull-toward-AR = C[1,2]
        divergence = ((C[1,1] - C[1,2]) + (C[2,2] - C[2,1]))   # symmetric HE↔AR gap
        summary.append({"model": m, "en_pull": en_pull, "he_pull": he_pull,
                        "ar_pull": ar_pull, "he_ar_gap": divergence})

    csv_path = OUT_DIR / "q03_intifada_cosine_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
    print(f"\n→ wrote {csv_path}")

    # ── Summary ranking ──
    print("\n" + "=" * 86)
    print("Per-provider ingroup pull + HE↔AR divergence (row-centred)")
    print("=" * 86)
    print(f"  {'provider':<35s} {'EN':>7s} {'HE':>7s} {'AR':>7s} {'HE↔AR_gap':>10s}")
    summary.sort(key=lambda r: -r["he_ar_gap"] if not np.isnan(r["he_ar_gap"]) else 0)
    for r in summary:
        def fmt(v): return f"{v:+.3f}" if not np.isnan(v) else "  n/a"
        print(f"  {r['model']:<35s} {fmt(r['en_pull']):>7s} {fmt(r['he_pull']):>7s} "
              f"{fmt(r['ar_pull']):>7s} {fmt(r['he_ar_gap']):>10s}")

    # ── Phase 3: ablation on highest-divergence provider ──
    top = next((r for r in summary if not np.isnan(r["he_ar_gap"])), None)
    if top is None:
        print("\nNo eligible provider for ablation.")
        return

    target = top["model"]
    print("\n" + "=" * 86)
    print(f"Phase 3 ablation · {target}  ·  highest HE↔AR gap = {top['he_ar_gap']:+.3f}")
    print(f"  → Translate HE and AR responses to EN, re-embed, compare cosines.")
    print(f"  → If translated-EN versions converge → embedding shift was language-form.")
    print(f"  → If they stay split → content choices are pulling toward each anchor.")
    print("=" * 86)

    he_prompt, he_resp = load_response_text(target, "he", 0)
    ar_prompt, ar_resp = load_response_text(target, "ar", 0)
    en_prompt, en_resp = load_response_text(target, "en", 0)

    print(f"\n  Translating HE→EN via DeepL…")
    he_en = translate_to_en(he_resp, "he")
    print(f"  Translating AR→EN via DeepL…")
    ar_en = translate_to_en(ar_resp, "ar")
    print(f"  Re-embedding 3 EN-form responses + 2 originals with text-embedding-3-small…")

    embeds = {
        "EN-response (native)":      embed_text(en_resp),
        "HE-response (native)":      embed_text(he_resp),       # Hebrew text
        "AR-response (native)":      embed_text(ar_resp),       # Arabic text
        "HE-response (→EN)":         embed_text(he_en),
        "AR-response (→EN)":         embed_text(ar_en),
    }

    print(f"\n  {'response variant':<28s} {'EN-anch':>9s} {'HE-anch':>9s} {'AR-anch':>9s}")
    abl_rows = []
    for tag, v in embeds.items():
        row = {"variant": tag}
        for al in LANGS:
            c = cos(v, anchors[al])
            row[f"cos_{al}"] = c
        # Row-centre across the 3 anchors
        vals = [row[f"cos_{l}"] for l in LANGS]
        rc = [v - np.mean(vals) for v in vals]
        for i, al in enumerate(LANGS):
            row[f"rc_{al}"] = rc[i]
        abl_rows.append(row)
        print(f"  {tag:<28s} {row['cos_en']:+.4f}  {row['cos_he']:+.4f}  {row['cos_ar']:+.4f}")

    print(f"\n  Row-centred (the bias signal):")
    print(f"  {'response variant':<28s} {'EN-anch':>9s} {'HE-anch':>9s} {'AR-anch':>9s}")
    for r in abl_rows:
        print(f"  {r['variant']:<28s} {r['rc_en']:+.4f}  {r['rc_he']:+.4f}  {r['rc_ar']:+.4f}")

    # Headline numbers
    he_native_he_pull = next(r["rc_he"] for r in abl_rows if "HE-response (native)" in r["variant"])
    he_en_he_pull     = next(r["rc_he"] for r in abl_rows if "HE-response (→EN)"   in r["variant"])
    ar_native_ar_pull = next(r["rc_ar"] for r in abl_rows if "AR-response (native)" in r["variant"])
    ar_en_ar_pull     = next(r["rc_ar"] for r in abl_rows if "AR-response (→EN)"   in r["variant"])

    print("\n  Ingroup-pull preserved through translation:")
    print(f"    HE: native ingroup pull {he_native_he_pull:+.3f}  →  "
          f"after-EN-translation {he_en_he_pull:+.3f}  "
          f"(retained {he_en_he_pull/he_native_he_pull*100:+.0f}%)")
    print(f"    AR: native ingroup pull {ar_native_ar_pull:+.3f}  →  "
          f"after-EN-translation {ar_en_ar_pull:+.3f}  "
          f"(retained {ar_en_ar_pull/ar_native_ar_pull*100:+.0f}%)")

    # Save ablation CSV
    abl_path = OUT_DIR / "q03_intifada_ablation.csv"
    with open(abl_path, "w", newline="") as f:
        if abl_rows:
            w = csv.DictWriter(f, fieldnames=abl_rows[0].keys())
            w.writeheader(); w.writerows(abl_rows)
    print(f"\n→ wrote {abl_path}")


if __name__ == "__main__":
    main()
