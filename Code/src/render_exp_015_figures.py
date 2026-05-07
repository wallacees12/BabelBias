"""Render the exp_015 cross-embedder robustness figures for the 8 June deck.

Headline figure: 01_robustness_4_embedder.png
    4-panel side-by-side bar chart. Each panel = one embedder (OpenAI baseline
    + 3 alts: Alibaba 🇨🇳, Gemini 🇺🇸, Yandex 🇷🇺). x-axis = response language;
    14 bars per group, one per provider, ecosystem-coloured (palette card
    carries the legend). Shared y-scale across panels for direct magnitude
    comparison.

The visual question for the slide is: "does the same shape appear in every
panel?" — and the punchline is **the sign of the diagonal pull is robust
across all 4 embedders, but the relative magnitudes differ**.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from babelbias.config import DEFAULT_LANGS
from babelbias.palette import MODEL_COLORS, ORDERED_MODELS
from babelbias.paths import (
    ANALYSIS_ALT_DIR,
    ANALYSIS_DIR,
    PROJECT_ROOT,
)


sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=0.75,
    rc={
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#333333",
        "axes.linewidth":    0.8,
        "grid.color":        "#e6e6e6",
        "grid.linewidth":    0.6,
        "savefig.dpi":       180,
        "savefig.bbox":      "tight",
    },
)

LANGS = list(DEFAULT_LANGS)  # ("en", "ru", "uk")
OUT_DIR = PROJECT_ROOT / "Presentations" / "figures" / "June 8"

# (embedder_id, panel_label, root_dir)
EMBEDDERS = [
    ("openai_te3s",  "OpenAI text-embedding-3-small  [US]",  ANALYSIS_DIR),
    ("alibaba_v3",   "Alibaba text-embedding-v3  [CN]",       ANALYSIS_ALT_DIR / "alibaba_v3"),
    ("gemini_001",   "Google gemini-embedding-001  [US]",     ANALYSIS_ALT_DIR / "gemini_001"),
    ("yandex_doc",   "Yandex text-search-doc  [RU]",          ANALYSIS_ALT_DIR / "yandex_doc"),
]


def load_diag_with_ci(root: Path, model: str, event: str):
    """Return per-language (diag_value, ci) for the row-centered cosine
    matrix. Diagonal cells are the "ingroup pull" — response in lang L
    pulled toward the L-language Wikipedia anchor."""
    rc_path = root / model / event / "anchor_heatmap_rowcentered.csv"
    ci_path = root / model / event / "anchor_heatmap_ci95.csv"
    if not rc_path.exists() or not ci_path.exists():
        return None
    rc = pd.read_csv(rc_path, index_col=0).to_numpy()
    ci = pd.read_csv(ci_path, index_col=0).to_numpy()
    return [(float(rc[i, i]), float(ci[i, i])) for i in range(3)]


def figure_4_embedder_robustness(out_path: Path, event: str = "ru_uk_core",
                                  title: str = None, footer: str = None):
    n_models = len(ORDERED_MODELS)
    bar_w = 0.06

    fig, axes = plt.subplots(1, len(EMBEDDERS), figsize=(20, 5.4), sharey=True,
                             constrained_layout=True)

    # Determine shared y-range from all data so panels are directly comparable.
    all_vals = []
    cached: dict[tuple[str, str], list[tuple[float, float]] | None] = {}
    for emb_id, _, root in EMBEDDERS:
        for m in ORDERED_MODELS:
            d = load_diag_with_ci(root, m, event)
            cached[(emb_id, m)] = d
            if d is not None:
                for v, c in d:
                    all_vals.append(v + c)
                    all_vals.append(v - c)
    y_max = max(all_vals) * 1.10
    y_min = min(all_vals) * 1.10

    for ax_i, (emb_id, label, _) in enumerate(EMBEDDERS):
        ax = axes[ax_i]
        x = np.arange(3)
        for i, model in enumerate(ORDERED_MODELS):
            diag = cached[(emb_id, model)]
            if diag is None:
                continue
            offset = (i - (n_models - 1) / 2) * bar_w
            vals = [d[0] for d in diag]
            cis  = [d[1] for d in diag]
            ax.bar(x + offset, vals, bar_w, yerr=cis,
                   color=MODEL_COLORS[model], capsize=1.5,
                   edgecolor="white", linewidth=0.4,
                   error_kw={"elinewidth": 0.6})

        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"resp = {l.upper()}" for l in LANGS])
        ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("Diagonal pull = own-language Wikipedia\nrow-centered cosine, 95% CI",
                       fontsize=10)

    fig.suptitle(title or "Embedding-space robustness — diagonal ingroup pull across 4 embedders × 14 providers",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02,
             footer or "Sign is method-robust (every (model, lang) cell positive); the EN >> RU > UK ordering is OpenAI-specific.",
             ha="center", fontsize=9.5, color="#444", style="italic")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_4_embedder_robustness(OUT_DIR / "01_robustness_4_embedder.png",
                                  event="ru_uk_core")
    print(f"Wrote {OUT_DIR / '01_robustness_4_embedder.png'}")
    figure_4_embedder_robustness(
        OUT_DIR / "02_robustness_4_embedder_debiased.png",
        event="ru_uk_core_debiased",
        title="Embedding-space robustness — DEBIASED diagonal pull (language subspace projected out)",
        footer="If the pattern is lexical, debiasing collapses it to zero. EN survival = framing/content; UK collapse = lexical overlap. Cross-embedder comparison.",
    )
    print(f"Wrote {OUT_DIR / '02_robustness_4_embedder_debiased.png'}")


if __name__ == "__main__":
    main()
