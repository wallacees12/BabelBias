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
        ax.set_title(f"{m} | Δ row-centered cosine\n(800-tok minus 400-tok)", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        out = FIG_DIR / f"delta_heatmap_{m}.png"
        plt.savefig(out, dpi=150)
        plt.close()


def render_delta_bar(df: pd.DataFrame):
    """One bar per (model, response_lang) showing change in diagonal
    (ingroup) row-centered cosine — i.e. resp_X aligned with wiki_X."""
    diag = df[df.response_lang == df.anchor_lang].copy()
    diag["label"] = diag["model"] + "\n" + diag["response_lang"].str.upper()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    colors = ["#2c7fb8" if l == "en" else "#41ab5d" if l == "ru" else "#fdae61"
              for l in diag.response_lang]
    bars = ax.bar(diag.label, diag.delta_rowcentered, color=colors)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Δ ingroup row-centered cosine\n(800-tok minus 400-tok)")
    ax.set_title("exp_002_followup · Diagonal ingroup-pull change per (model, language) "
                 "between 400-tok (exp_001) and 800-tok (exp_002) sweeps")
    for bar, val in zip(bars, diag.delta_rowcentered):
        offset = 0.002 if val >= 0 else -0.004
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:+.3f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=8)
    plt.xticks(rotation=0, fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "delta_ingroup_bar.png"
    plt.savefig(out, dpi=150)
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
