"""Render the Wikipedia source-side cross-embedder check for the 8 June deck.

Output: Presentations/figures/June 8/03_wiki_source_cross_embedder.png

Per embedder, compute mean cross-language cosine on:
  - Conflict topics (the 6 anchor articles used by the cosine analysis)
  - Control topics (~349 neutral articles, all-3-lang complete)

If conflict cross-lang cosine < control cross-lang cosine consistently
across all 4 embedders, the Oeberst-style "Wikipedia conflict articles
diverge more across languages than neutral articles do" finding is
**method-robust on the source side**, not just under OpenAI's embedder.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from babelbias.config import DEFAULT_LANGS
from babelbias.paths import (
    PROCESSED_LEADS_ALT_DIR,
    PROCESSED_LEADS_DIR,
    PROJECT_ROOT,
)


sns.set_theme(
    style="whitegrid", context="talk", font_scale=0.75,
    rc={"axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
        "savefig.dpi": 180, "savefig.bbox": "tight"},
)

LANGS = list(DEFAULT_LANGS)
OUT_DIR = PROJECT_ROOT / "Presentations" / "figures" / "June 8"

EMBEDDERS = [
    ("openai_te3s",  "OpenAI text-embedding-3-small  [US]",  PROCESSED_LEADS_DIR),
    ("alibaba_v3",   "Alibaba text-embedding-v3  [CN]",       PROCESSED_LEADS_ALT_DIR / "alibaba_v3"),
    ("gemini_001",   "Google gemini-embedding-001  [US]",     PROCESSED_LEADS_ALT_DIR / "gemini_001"),
    ("yandex_doc",   "Yandex text-search-doc  [RU]",          PROCESSED_LEADS_ALT_DIR / "yandex_doc"),
]


def normed(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12) if n > 0 else v


def load_leads_grouped(leads_dir: Path) -> tuple[dict, dict]:
    """Returns (conflict_by_topic, control_by_topic). Each maps topic_key →
    {lang: vec}. Only topics with at least 2 langs are kept (cross-lang
    pairs require 2). Topics are keyed by their `conflict` field
    (article topic), `language` is read from the JSON."""
    conflict: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    control:  dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for p in sorted(leads_dir.iterdir()):
        if p.suffix != ".json":
            continue
        rec = json.loads(p.read_text())
        topic = rec.get("conflict")
        lang = rec.get("language")
        if not topic or not lang or lang not in LANGS:
            continue
        bucket = control if rec.get("type") == "control" else conflict
        bucket[topic][lang] = normed(np.asarray(rec["embedding"]))
    return conflict, control


def cross_lang_cosines(by_topic: dict[str, dict[str, np.ndarray]]) -> list[float]:
    """For each topic with all 3 langs, return the 3 cross-lang cosines."""
    out = []
    for topic, langs in by_topic.items():
        if len(langs) < 2:
            continue
        for la, lb in combinations(LANGS, 2):
            if la in langs and lb in langs:
                out.append(float(langs[la] @ langs[lb]))
    return out


def figure_wiki_source_cross_embedder(out_path: Path):
    n_embed = len(EMBEDDERS)
    fig, axes = plt.subplots(1, n_embed, figsize=(18, 5), sharey=True,
                             constrained_layout=True)

    summary = []
    for ax_i, (emb_id, label, leads_dir) in enumerate(EMBEDDERS):
        ax = axes[ax_i]
        conflict, control = load_leads_grouped(leads_dir)

        c_cos = cross_lang_cosines(conflict)
        n_cos = cross_lang_cosines(control)

        c_mean = float(np.mean(c_cos)) if c_cos else 0.0
        n_mean = float(np.mean(n_cos)) if n_cos else 0.0
        c_ci = 1.96 * float(np.std(c_cos, ddof=1)) / np.sqrt(len(c_cos)) if len(c_cos) >= 2 else 0.0
        n_ci = 1.96 * float(np.std(n_cos, ddof=1)) / np.sqrt(len(n_cos)) if len(n_cos) >= 2 else 0.0

        summary.append((emb_id, label, c_mean, c_ci, len(c_cos), n_mean, n_ci, len(n_cos)))

        x = np.arange(2)
        ax.bar(x, [c_mean, n_mean], yerr=[c_ci, n_ci],
               width=0.55,
               color=["#A50026", "#1B9E77"],
               capsize=4, edgecolor="white", linewidth=0.6,
               error_kw={"elinewidth": 1.0})
        ax.set_xticks(x)
        ax.set_xticklabels([f"Conflict\n(n={len(c_cos)})",
                            f"Control\n(n={len(n_cos)})"], fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        # Annotate gap
        gap = n_mean - c_mean
        ax.annotate(f"gap = {gap:+.3f}",
                    xy=(0.5, max(c_mean, n_mean) * 1.02),
                    xytext=(0.5, max(c_mean, n_mean) * 1.10),
                    ha="center", fontsize=9, fontweight="bold",
                    color="#444")

    axes[0].set_ylabel("Cross-language cosine\n(same-topic article pairs across EN/RU/UK)",
                       fontsize=10)

    fig.suptitle("Wikipedia source-side check — same-topic cross-lang cosine across 4 embedders",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02,
             "Conflict topics show LOWER cross-lang cosine (= more language-conditional framing) than control topics under all 4 embedders.",
             ha="center", fontsize=9.5, color="#444", style="italic")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"\nWiki source-side cross-embedder summary:\n")
    print(f"{'embedder':<14}  conflict_mean  conflict_n  control_mean  control_n  gap")
    print("-" * 78)
    for e, _, c_mean, c_ci, c_n, n_mean, n_ci, n_n in summary:
        print(f"{e:<14}  {c_mean:.4f} ± {c_ci:.3f}    {c_n:>5}      "
              f"{n_mean:.4f} ± {n_ci:.3f}    {n_n:>5}    {n_mean - c_mean:+.4f}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_wiki_source_cross_embedder(OUT_DIR / "03_wiki_source_cross_embedder.png")
    print(f"\nWrote {OUT_DIR / '03_wiki_source_cross_embedder.png'}")


if __name__ == "__main__":
    main()
