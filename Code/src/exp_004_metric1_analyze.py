"""
exp_004 Metric 1 — analyse the per-response Sonnet stance codes.

Reads `data/Russia-Ukraine/analysis/imaginary_metric1.csv` and emits:

  - imaginary_metric1_by_lang.csv       overall + per-language summary
  - imaginary_metric1_by_qid_lang.csv   (qid × lang) directional means
  - imaginary_metric1_by_model_lang.csv (model × lang) directional means
  - Presentations/figures/exp_004_metric1/by_lang_distribution.png
  - Presentations/figures/exp_004_metric1/qid_x_lang_heatmap.png
  - Presentations/figures/exp_004_metric1/model_x_lang_heatmap.png

Numeric "stance score" rule: +2/+1/0/-1/-2 → as-is; R → excluded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from babelbias.paths import ANALYSIS_DIR, PROJECT_ROOT

CSV_IN = ANALYSIS_DIR / "imaginary_metric1.csv"
FIG_DIR = PROJECT_ROOT / "Presentations" / "figures" / "exp_004_metric1"

LANGS = ("en", "ru", "uk")
NUMERIC = {"+2": 2, "+1": 1, "0": 0, "-1": -1, "-2": -2}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_IN, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.strip()
    df["score"] = df["code"].map(NUMERIC)  # R / NaN → NaN
    return df


def by_lang(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang in LANGS + ("ALL",):
        sub = df if lang == "ALL" else df[df.language == lang]
        n = len(sub)
        n_scored = sub.score.notna().sum()
        n_refusal = (sub.code == "R").sum()
        n_unparsed = sub.code.isna().sum() + (sub.code == "").sum() + (sub.code == "nan").sum()
        rows.append({
            "language": lang,
            "n_total": n,
            "n_scored": n_scored,
            "n_refusal": n_refusal,
            "n_unparsed": n_unparsed,
            "mean_score": float(sub.score.mean()) if n_scored else float("nan"),
            "share_pos": float((sub.score > 0).sum() / n_scored) if n_scored else float("nan"),
            "share_neg": float((sub.score < 0).sum() / n_scored) if n_scored else float("nan"),
            "share_zero": float((sub.score == 0).sum() / n_scored) if n_scored else float("nan"),
            "n_plus2": int((sub.score == 2).sum()),
            "n_plus1": int((sub.score == 1).sum()),
            "n_zero": int((sub.score == 0).sum()),
            "n_minus1": int((sub.score == -1).sum()),
            "n_minus2": int((sub.score == -2).sum()),
        })
    return pd.DataFrame(rows)


def by_qid_lang(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["qid", "language"]).agg(
        n=("score", "size"),
        n_scored=("score", "count"),
        mean_score=("score", "mean"),
        share_pos=("score", lambda s: float((s > 0).sum() / s.count()) if s.count() else float("nan")),
        share_neg=("score", lambda s: float((s < 0).sum() / s.count()) if s.count() else float("nan")),
    ).reset_index()
    return g


def by_model_lang(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "language"]).agg(
        n=("score", "size"),
        n_scored=("score", "count"),
        mean_score=("score", "mean"),
    ).reset_index()
    return g


def plot_by_lang(df: pd.DataFrame, out: Path):
    counts = (
        df[df.code.isin(["+2", "+1", "0", "-1", "-2", "R"])]
        .groupby(["language", "code"]).size().unstack("code", fill_value=0)
    )
    order = ["+2", "+1", "0", "-1", "-2", "R"]
    counts = counts.reindex(columns=order, fill_value=0).reindex(index=list(LANGS))

    palette = {
        "+2": "#67000d", "+1": "#fb6a4a", "0": "#bdbdbd",
        "-1": "#6baed6", "-2": "#08306b", "R": "#cccccc",
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bottom = np.zeros(len(LANGS))
    for code in order:
        vals = counts[code].to_numpy()
        ax.bar([l.upper() for l in LANGS], vals, bottom=bottom,
               color=palette[code], label=code,
               edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_ylabel("Number of responses")
    ax.legend(title="code", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_qid_x_lang(df: pd.DataFrame, out: Path):
    piv = df.pivot_table(index="qid", columns="language", values="score", aggfunc="mean")
    piv = piv.reindex(columns=list(LANGS))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    abs_max = max(0.05, float(np.max(np.abs(piv.values))))
    im = ax.imshow(piv.values, cmap="coolwarm", aspect="auto",
                   vmin=-abs_max, vmax=abs_max)
    ax.set_xticks(range(len(LANGS))); ax.set_xticklabels([l.upper() for l in LANGS])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    for i, qid in enumerate(piv.index):
        for j, lang in enumerate(LANGS):
            v = piv.values[i, j]
            txt = f"{v:+.2f}" if not np.isnan(v) else "·"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black")
    plt.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_model_x_lang(df: pd.DataFrame, out: Path):
    piv = df.pivot_table(index="model", columns="language", values="score", aggfunc="mean")
    piv = piv.reindex(columns=list(LANGS))
    piv = piv.sort_values("ru", na_position="last")
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    abs_max = max(0.05, float(np.max(np.abs(piv.fillna(0).values))))
    im = ax.imshow(piv.values, cmap="coolwarm", aspect="auto",
                   vmin=-abs_max, vmax=abs_max)
    ax.set_xticks(range(len(LANGS))); ax.set_xticklabels([l.upper() for l in LANGS])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    for i, model in enumerate(piv.index):
        for j, lang in enumerate(LANGS):
            v = piv.values[i, j]
            txt = f"{v:+.2f}" if not np.isnan(v) else "·"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main():
    df = load()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    by_l = by_lang(df)
    by_l.to_csv(ANALYSIS_DIR / "imaginary_metric1_by_lang.csv", index=False)
    print("=== By language ===")
    print(by_l.round(3).to_string(index=False))

    by_ql = by_qid_lang(df)
    by_ql.to_csv(ANALYSIS_DIR / "imaginary_metric1_by_qid_lang.csv", index=False)
    print("\n=== By (qid × language) ===")
    print(by_ql.round(3).to_string(index=False))

    by_ml = by_model_lang(df)
    by_ml.to_csv(ANALYSIS_DIR / "imaginary_metric1_by_model_lang.csv", index=False)
    print("\n=== By (model × language) ===")
    print(by_ml.round(3).to_string(index=False))

    plot_by_lang(df, FIG_DIR / "by_lang_distribution.png")
    plot_qid_x_lang(df, FIG_DIR / "qid_x_lang_heatmap.png")
    plot_model_x_lang(df, FIG_DIR / "model_x_lang_heatmap.png")
    print(f"\nFigures: {FIG_DIR}")


if __name__ == "__main__":
    main()
