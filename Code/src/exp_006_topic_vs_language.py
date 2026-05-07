"""
exp_006 sub-analysis — topic-vs-language disentanglement.

For every (event, response_lang, model) cell, compute two cosines:

  on_topic_cos   = mean cosine(response, EVENT-anchor-in-same-lang)
  off_topic_cos  = mean cosine(response, sample of N=200 universal-control
                                          articles in same lang)

The DIFFERENCE — `topic_lift = on_topic - off_topic` — is the
topic-specific pull *above the same-language similarity floor*. If
topic_lift ≫ 0, the response cluster is being dragged toward its
event's content; if topic_lift ≈ 0, the apparent ingroup pull is
purely lexical/language-axis.

Output:
  data/<event>/analysis/exp_006_topic_vs_language.csv
  Aggregate summary printed to stdout.

Cost: $0 (operates entirely on cached embeddings).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from babelbias.event_bank import load_bank
from babelbias.paths import (
    DATA_ROOT,
    llm_embeddings_dir,
    processed_leads_dir,
)


N_CONTROL_SAMPLE = 200
RNG_SEED = 42

EVENTS = ("ru_uk_core", "israel_palestine", "india_pakistan",
          "taiwan_strait", "falklands")
MODELS = ("claude-haiku-4-5", "gpt-4o-mini", "gemini-2.5-flash",
          "grok-3-mini", "mercury-2", "deepseek-chat", "qwen-plus",
          "glm-4.5", "baidu/ernie-4.5-300b-a47b",
          "c4ai-aya-expanse-32b", "command-r7b-arabic-02-2025",
          "ollama:allam-7b", "ollama:taide-llama3-8b",
          "jamba-mini-2-2026-01")


def load_response_embeddings(event: str, model: str) -> dict:
    """{(qid, lang): [vec, ...]}"""
    d = llm_embeddings_dir(event) / model / event
    if not d.is_dir():
        return {}
    out: dict[tuple[str, str], list[np.ndarray]] = {}
    for f in d.glob("*.json"):
        try:
            rec = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("refusal") or not rec.get("embedding"):
            continue
        key = (rec["qid"], rec["language"])
        out.setdefault(key, []).append(np.asarray(rec["embedding"]))
    return out


def load_anchors(event: str, anchor_slugs: dict[str, str], langs: tuple[str, ...]) -> dict:
    """{(qid, lang): vec} for the event's Wikipedia anchors."""
    d = processed_leads_dir(event)
    out: dict[tuple[str, str], np.ndarray] = {}
    for qid, slug in anchor_slugs.items():
        for lang in langs:
            f = d / f"{slug}_{lang}.json"
            if not f.exists():
                continue
            rec = json.loads(f.read_text())
            out[(qid, lang)] = np.asarray(rec["embedding"])
    return out


def load_universal_controls(processed_dir: Path, lang: str,
                            n_sample: int = N_CONTROL_SAMPLE,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample N control article embeddings for the given language."""
    rng = rng or np.random.default_rng(RNG_SEED)
    paths = list(processed_dir.glob(f"CONTROL_*_{lang}.json"))
    if not paths:
        return np.empty((0,))
    if len(paths) > n_sample:
        paths = list(rng.choice(paths, size=n_sample, replace=False))
    vecs = []
    for f in paths:
        try:
            rec = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("type") != "control":
            continue
        vecs.append(rec["embedding"])
    return np.asarray(vecs, dtype=np.float64) if vecs else np.empty((0,))


def cosine_matrix(R: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Pairwise cosines between row-vectors of R and A."""
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True).clip(1e-12)
    An = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    return Rn @ An.T


def analyze_event(event: str) -> list[dict]:
    bank = load_bank(event)
    langs = bank.languages
    anchor_slugs = bank.anchor_slugs
    proc_dir = processed_leads_dir(event)

    # Load universal controls per language (sampled once per event).
    rng = np.random.default_rng(RNG_SEED)
    controls_by_lang: dict[str, np.ndarray] = {}
    for lang in langs:
        controls_by_lang[lang] = load_universal_controls(proc_dir, lang, rng=rng)

    rows: list[dict] = []
    for model in MODELS:
        responses = load_response_embeddings(event, model)
        if not responses:
            continue
        anchors = load_anchors(event, anchor_slugs, langs)

        # Aggregate per (response_lang) — average across qids that have
        # an anchor in that language.
        for lang in langs:
            on_topic_cosines: list[float] = []
            off_topic_cosines: list[float] = []
            n_resp = 0
            for qid, slug in anchor_slugs.items():
                resp_vecs = responses.get((qid, lang))
                if not resp_vecs:
                    continue
                R = np.asarray(resp_vecs)
                anchor = anchors.get((qid, lang))
                if anchor is not None:
                    on_topic_cosines.extend(
                        cosine_matrix(R, anchor[None, :]).flatten().tolist()
                    )
                ctrl = controls_by_lang.get(lang)
                if ctrl is not None and ctrl.size:
                    off_topic_cosines.extend(
                        cosine_matrix(R, ctrl).flatten().tolist()
                    )
                n_resp += len(R)
            if not on_topic_cosines or not off_topic_cosines:
                continue
            on_mean = float(np.mean(on_topic_cosines))
            off_mean = float(np.mean(off_topic_cosines))
            rows.append({
                "event":        event,
                "model":        model,
                "lang":         lang,
                "n_responses":  n_resp,
                "on_topic_cos": round(on_mean, 4),
                "off_topic_cos": round(off_mean, 4),
                "topic_lift":   round(on_mean - off_mean, 4),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", nargs="+", default=list(EVENTS))
    args = ap.parse_args()

    all_rows = []
    for event in args.events:
        try:
            all_rows.extend(analyze_event(event))
        except FileNotFoundError as e:
            print(f"  ⚠ {event}: {e}")

    if not all_rows:
        print("No data."); return

    df = pd.DataFrame(all_rows)

    # Per-event aggregate (mean across models): on_topic, off_topic, topic_lift
    print("\n=== Aggregate (mean across models) ===")
    print(f"{'event':<20} {'lang':<5} {'on_topic':>10} {'off_topic':>11} {'topic_lift':>12} {'n_models':>9}")
    print("-" * 70)
    agg = df.groupby(["event", "lang"]).agg(
        on=("on_topic_cos", "mean"),
        off=("off_topic_cos", "mean"),
        lift=("topic_lift", "mean"),
        n=("model", "count"),
    ).reset_index()
    for _, r in agg.iterrows():
        print(f"{r['event']:<20} {r['lang']:<5} {r['on']:>+10.4f} "
              f"{r['off']:>+11.4f} {r['lift']:>+12.4f} {int(r['n']):>9}")

    out_dir = DATA_ROOT / "Russia-Ukraine" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp_006_topic_vs_language.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
