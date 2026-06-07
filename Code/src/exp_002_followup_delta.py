"""
exp_002_followup — Δ-cosine between 400-tok and 800-tok sweeps.

Reads `analyze_bias.py` outputs from two parallel analysis trees:

    400-tok:  data/Russia-Ukraine/analysis_400tok/<model>/ru_uk_core/
    800-tok:  data/Russia-Ukraine/analysis/<model>/ru_uk_core/

For each of the 5 models (exp_001 set), computes per-(response_lang,
anchor_lang) cell deltas on:
  - raw mean cosine (anchor_heatmap_mean)
  - row-centered ingroup-pull (anchor_heatmap_rowcentered)

Writes:
  data/Russia-Ukraine/analysis/exp_002_delta_400_to_800.csv
  Presentations/figures/exp_002_followup/delta_ingroup_bar.png
  Presentations/figures/exp_002_followup/delta_heatmap_<model>.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from babelbias.paths import ANALYSIS_DIR, PROJECT_ROOT

MODELS = [
    "claude-haiku-4-5",
    "gpt-4o-mini",
    "gemini-2.5-flash",
    "grok-3-mini",
    "deepseek-chat",
]
EVENT = "ru_uk_core"
LANGS = ("en", "ru", "uk")

A800 = ANALYSIS_DIR
A400 = PROJECT_ROOT / "data" / "Russia-Ukraine" / "analysis_400tok"
FIG_DIR = PROJECT_ROOT / "Presentations" / "figures" / "exp_002_followup"


def load_matrix(root: Path, model: str, name: str) -> pd.DataFrame:
    p = root / model / EVENT / f"{name}.csv"
    df = pd.read_csv(p, index_col=0)
    return df


def cell_delta_rows() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        m400_raw = load_matrix(A400, m, "anchor_heatmap_mean")
        m800_raw = load_matrix(A800, m, "anchor_heatmap_mean")
        m400_rc = load_matrix(A400, m, "anchor_heatmap_rowcentered")
        m800_rc = load_matrix(A800, m, "anchor_heatmap_rowcentered")
        for rl in LANGS:
            for al in LANGS:
                rows.append({
                    "model": m,
                    "response_lang": rl,
                    "anchor_lang": al,
                    "mean_400": float(m400_raw.loc[f"resp_{rl}", f"wiki_{al}"]),
                    "mean_800": float(m800_raw.loc[f"resp_{rl}", f"wiki_{al}"]),
                    "delta_mean": float(m800_raw.loc[f"resp_{rl}", f"wiki_{al}"]) -
                                  float(m400_raw.loc[f"resp_{rl}", f"wiki_{al}"]),
                    "rowcentered_400": float(m400_rc.loc[f"resp_{rl}", f"wiki_{al}"]),
                    "rowcentered_800": float(m800_rc.loc[f"resp_{rl}", f"wiki_{al}"]),
                    "delta_rowcentered": float(m800_rc.loc[f"resp_{rl}", f"wiki_{al}"]) -
                                         float(m400_rc.loc[f"resp_{rl}", f"wiki_{al}"]),
                })
    return pd.DataFrame(rows)


def render_delta_heatmaps(df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        sub = df[df.model == m]
        mat = sub.pivot(index="response_lang", columns="anchor_lang",
                        values="delta_rowcentered").reindex(index=LANGS, columns=LANGS)
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        abs_max = max(0.001, float(np.max(np.abs(mat.values))))
        im = ax.imshow(mat.values, cmap="coolwarm", vmin=-abs_max, vmax=abs_max,
                       aspect="equal")
        ax.set_xticks(range(3)); ax.set_xticklabels([l.upper() for l in LANGS])
        ax.set_yticks(range(3)); ax.set_yticklabels([l.upper() for l in LANGS])
        ax.set_xlabel("Wikipedia anchor language")
        ax.set_ylabel("Response language")
        for i, rl in enumerate(LANGS):
            for j, al in enumerate(LANGS):
                v = mat.values[i, j]
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=10,
                        color="black")
        ax.set_title(m, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        out = FIG_DIR / f"delta_heatmap_{m}.png"
        plt.savefig(out, dpi=150)
        plt.close()


def render_delta_bar(df: pd.DataFrame):
    """Grouped bar chart: per-model EN/RU/UK Δ row-centered diagonal cosine
    between 400-tok (exp_001) and 800-tok (exp_002) sweeps. The visual
    point is that every bar sits near zero — the ingroup-pull signal is
    method-robust to a 2× token-budget change. Title-less per project
    no-in-figure-titles rule; description lives in the slide caption."""
    diag = df[df.response_lang == df.anchor_lang].copy()
    pivot = diag.pivot(index="model", columns="response_lang",
                       values="delta_rowcentered").reindex(index=MODELS,
                                                          columns=list(LANGS))

    lang_colors = {"en": "#2c7fb8", "ru": "#d7301f", "uk": "#fdae61"}
    n_models = len(MODELS)
    n_langs = len(LANGS)
    bar_w = 0.26
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, lang in enumerate(LANGS):
        vals = pivot[lang].values
        offset = (i - (n_langs - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w,
                      color=lang_colors[lang], label=lang.upper(),
                      edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.0006 if val >= 0 else -0.0006),
                    f"{val:+.3f}", ha="center", va=va, fontsize=7.5,
                    color="#333333")

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10)
    ax.set_ylabel("Δ ingroup row-centered cosine\n(800-tok minus 400-tok)",
                  fontsize=10)
    # Fix y-axis so the near-zero magnitude reads visually: ±0.015 brackets
    # the worst-case −0.011 (claude UK) and leaves headroom for value labels.
    ax.set_ylim(-0.015, 0.015)
    ax.legend(title="Response language", loc="upper right", frameon=False,
              fontsize=9, title_fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "delta_ingroup_bar.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


def main():
    df = cell_delta_rows()
    out_csv = ANALYSIS_DIR / "exp_002_delta_400_to_800.csv"
    df.to_csv(out_csv, index=False)
    print(f"Δ table: {out_csv}")

    diag = df[df.response_lang == df.anchor_lang].copy()
    summary = diag.groupby("response_lang")[["mean_400", "mean_800",
                                             "delta_mean",
                                             "rowcentered_400",
                                             "rowcentered_800",
                                             "delta_rowcentered"]].mean()
    print("\nDiagonal ingroup-pull summary (averaged over 5 models):")
    print(summary.round(4).to_string())

    render_delta_heatmaps(df)
    bar_path = render_delta_bar(df)
    print(f"\nFigures: {FIG_DIR}")
    print(f"  bar:     {bar_path}")
    print(f"  heatmaps: delta_heatmap_<model>.png × {len(MODELS)}")


if __name__ == "__main__":
    main()
