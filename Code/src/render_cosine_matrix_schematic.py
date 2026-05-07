"""Render the 3x3 cosine-matrix schematic for the thesis Methods
chapter (figure 3.5, fig:cosine_matrix_schematic).

Output: Presentations/figures/methodology/cosine_matrix_schematic.png

Worked example uses the actual claude-haiku-4-5 q02 cosine numbers
from data/Russia-Ukraine/analysis/claude-haiku-4-5/ru_uk_core/
anchor_heatmap_mean.csv. Diagonal cells are filled to highlight the
ingroup-pull cells; off-diagonal cells are pale to mark
outgroup-distance cells. A small annotation strip on the right
explains what each region means.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from babelbias.paths import PROJECT_ROOT


sns.set_theme(
    style="white", context="talk", font_scale=0.78,
    rc={
        "savefig.dpi":  200,
        "savefig.bbox": "tight",
    },
)

OUT = (PROJECT_ROOT / "Presentations" / "figures" / "methodology"
       / "cosine_matrix_schematic.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

LANGS = ("en", "ru", "uk")
LANG_LABEL = {"en": "EN", "ru": "RU", "uk": "UK"}


def load_example_matrix() -> np.ndarray:
    """Use the q02 row of claude-haiku-4-5 row-centered cosine — a
    representative ingroup-pull pattern from the RU-UK headline."""
    p = (PROJECT_ROOT / "data" / "Russia-Ukraine" / "analysis"
         / "claude-haiku-4-5" / "ru_uk_core"
         / "anchor_heatmap_mean.csv")
    df = pd.read_csv(p, index_col=0)
    rows = [f"resp_{l}" for l in LANGS]
    cols = [f"wiki_{l}" for l in LANGS]
    return df.loc[rows, cols].to_numpy()


def main() -> None:
    M = load_example_matrix()

    fig = plt.figure(figsize=(11.5, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.set_axis_off()

    n = 3
    for i in range(n):
        for j in range(n):
            on_diag = (i == j)
            face = "#FCE7F3" if on_diag else "#F8FAFC"
            edge = "#BE185D" if on_diag else "#94A3B8"
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1,
                                        facecolor=face, edgecolor=edge,
                                        linewidth=1.4 if on_diag else 0.8))
            ax.text(j + 0.5, n - 1 - i + 0.5, f"{M[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=18, weight="700" if on_diag else "500",
                    color="#BE185D" if on_diag else "#475569")

    # Outer frame
    ax.add_patch(plt.Rectangle((0, 0), n, n, facecolor="none",
                                edgecolor="#0F172A", linewidth=1.6))

    # Tick labels
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels([LANG_LABEL[l] for l in LANGS], fontsize=14, weight="700")
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels([LANG_LABEL[l] for l in reversed(LANGS)],
                        fontsize=14, weight="700")
    ax.tick_params(axis="both", which="both", length=0, pad=8)

    # Axis labels
    ax.set_xlabel("Wikipedia anchor language", fontsize=12,
                   color="#0F172A", labelpad=10)
    ax.set_ylabel("LLM response language", fontsize=12,
                   color="#0F172A", labelpad=10)

    ax.set_xlim(-0.05, n + 0.05)
    ax.set_ylim(-0.05, n + 0.05)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(n) + 0.5)

    # Title above the grid
    ax.set_title("Mean cosine similarity per (response_lang × anchor_lang)\n"
                 "claude-haiku-4-5 · q02 'what happened in Crimea'",
                 fontsize=12, weight="600", pad=14, color="#0F172A")

    # ── Legend / explanation panel ──────────────────────────────────
    ax_legend.set_xlim(0, 10); ax_legend.set_ylim(0, 10)
    # Diagonal swatch
    ax_legend.add_patch(plt.Rectangle((0.4, 7.5), 1.0, 1.0,
                                       facecolor="#FCE7F3",
                                       edgecolor="#BE185D", linewidth=1.4))
    ax_legend.text(1.7, 8.0, "Diagonal — ingroup pull",
                    fontsize=12.5, weight="700", color="#BE185D",
                    va="center")
    ax_legend.text(1.7, 7.0,
                    "Response in language $\\ell$ vs.\n"
                    "Wikipedia anchor in same $\\ell$.",
                    fontsize=10.5, color="#334155", va="top")
    # Off-diagonal swatch
    ax_legend.add_patch(plt.Rectangle((0.4, 4.5), 1.0, 1.0,
                                       facecolor="#F8FAFC",
                                       edgecolor="#94A3B8", linewidth=0.8))
    ax_legend.text(1.7, 5.0, "Off-diagonal — outgroup",
                    fontsize=12.5, weight="700", color="#475569",
                    va="center")
    ax_legend.text(1.7, 4.0,
                    "Response in $\\ell$ vs. anchor\n"
                    "in a different language $\\ell'$.",
                    fontsize=10.5, color="#334155", va="top")
    # Row-centring formula
    ax_legend.add_patch(plt.Rectangle((0.4, 0.6), 9.2, 2.4,
                                       facecolor="#FEF3C7",
                                       edgecolor="#D97706", linewidth=1.0))
    ax_legend.text(5.0, 2.4,
                    "Row-centred ingroup pull",
                    ha="center", fontsize=12, weight="700", color="#D97706")
    ax_legend.text(5.0, 1.4,
                    r"$\Delta_{\mathrm{in}}(q,\ell) = c_{\ell\ell} -"
                    r" \overline{c}_{\ell\cdot}$",
                    ha="center", fontsize=14, color="#0F172A")

    fig.suptitle("3$\\times$3 cosine matrix layout",
                  fontsize=14, weight="bold", y=1.04, color="#0F172A")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
