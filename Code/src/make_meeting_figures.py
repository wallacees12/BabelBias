"""
Build cross-model comparison figures for the next advisor meeting.

Reads the per-model analysis CSVs under data/Russia-Ukraine/analysis/<model>/
and writes presentation-ready PNGs to Presentations/figures/.

Two figures:
    1. Ingroup-pull bar chart — diagonal of the row-centered matrix, with 95%
       CI error bars, grouped by response language across the three models.
       Headline result in one glance.
    2. Side-by-side row-centered heatmaps — full 3x3 pattern for each model,
       shared color scale.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from babelbias.config import DEFAULT_LANGS
from babelbias.paths import ANALYSIS_DIR, PROJECT_ROOT

MODELS = ["gpt-4o-mini", "claude-haiku-4-5", "gemini-2.5-flash"]
EVENT = "ru_uk_core"
LANGS = list(DEFAULT_LANGS)
OUT_DIR = PROJECT_ROOT / "Presentations" / "figures"

MODEL_COLORS = {
    "gpt-4o-mini":      "#10a37f",  # OpenAI green
    "claude-haiku-4-5": "#d97757",  # Anthropic terracotta
    "gemini-2.5-flash": "#4285f4",  # Google blue
}


def load_model_matrices(model: str):
    d = ANALYSIS_DIR / model / EVENT
    rc = pd.read_csv(d / "anchor_heatmap_rowcentered.csv", index_col=0).to_numpy()
    ci = pd.read_csv(d / "anchor_heatmap_ci95.csv",        index_col=0).to_numpy()
    return rc, ci


def figure_ingroup_bars(out_path: Path):
    """Grouped bar chart: diagonal pull per (response language, model) with 95% CIs."""
    fig, ax = plt.subplots(figsize=(8, 4.8))

    x = np.arange(len(LANGS))
    width = 0.26

    for i, model in enumerate(MODELS):
        rc, ci = load_model_matrices(model)
        diag_vals = np.diag(rc)
        diag_cis  = np.diag(ci)
        offset = (i - 1) * width
        ax.bar(x + offset, diag_vals, width, yerr=diag_cis,
               label=model, color=MODEL_COLORS[model],
               capsize=4, edgecolor="white", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"response = {l.upper()}" for l in LANGS])
    ax.set_ylabel("Row-centered cosine\n(+ = pulled toward own-language Wikipedia)")
    ax.set_title("Ingroup bias: each language's response pulls toward its own-language Wikipedia\n"
                 "Row-centered diagonal, 95% CI across 9 questions × 10 samples", fontsize=11)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def figure_heatmap_grid(out_path: Path):
    """Three row-centered heatmaps side by side, shared color scale."""
    matrices = {m: load_model_matrices(m) for m in MODELS}
    abs_max = max(float(np.max(np.abs(rc))) for rc, _ in matrices.values())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    im = None
    for ax, model in zip(axes, MODELS):
        rc, ci = matrices[model]
        im = ax.imshow(rc, cmap="coolwarm", vmin=-abs_max, vmax=abs_max, aspect="equal")
        ax.set_xticks(range(3)); ax.set_xticklabels([l.upper() for l in LANGS])
        ax.set_yticks(range(3)); ax.set_yticklabels([l.upper() for l in LANGS])
        ax.set_xlabel("Wikipedia anchor")
        ax.set_title(model, fontsize=11)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{rc[i,j]:+.3f}\n±{ci[i,j]:.3f}",
                        ha="center", va="center", fontsize=9)
    axes[0].set_ylabel("Response language")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, label="Row-centered cosine")
    fig.suptitle("Response vs Wikipedia anchor (row-centered): consistent ingroup pull across providers",
                 fontsize=12)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_ingroup_bars(OUT_DIR / "01_ingroup_bars.png")
    figure_heatmap_grid(OUT_DIR / "02_heatmap_grid.png")
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
