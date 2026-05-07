"""Render the exp_017 tokenizer-robustness figure for the 8 June deck.

Two-panel composition:

  LEFT (Phase A motivation):
    Horizontal bars of RU-UK corpus_overlap (Qi et al. 2023 RankC metric)
    for the three inspectable embedder tokenizers. Visual question:
    "do these tokenizers carve RU and UK differently enough to matter?"
    Annotated with BLOOM's 0.76 reference.

  RIGHT (Phase B headline):
    Per-question UK→RU-anchor co-clustering share, grouped bars
    OpenAI te3s (bmcs=200, 12 clusters) vs Alibaba v3 (bmcs=150,
    11 clusters) — matched cluster resolution. Visual question:
    "is cluster-fusion an OpenAI artefact or a real phenomenon?"
    Punchline: total fusion comparable, but redistributed across
    questions — only q05/q09 are robust across embedders.

Output:
    Presentations/figures/June 8/05_exp_017_tokenizer_robustness.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from babelbias.paths import PROJECT_ROOT


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

EXP_DIR = PROJECT_ROOT / "Presentations" / "figures" / "exp_017"
OUT_DIR = PROJECT_ROOT / "Presentations" / "figures" / "June 8"

# Phase A — RU-UK corpus_overlap from exp_017_tokenizer_overlap.csv
PHASE_A_CORPUS_OVERLAP_PATH = (
    PROJECT_ROOT / "data" / "Russia-Ukraine" / "analysis"
    / "exp_017_tokenizer_overlap.csv"
)

# Question short-labels for the right panel (qid → short title)
QID_LABEL = {
    "q01_little_green_men":  "q01 little\ngreen men",
    "q02_crimea_2014":       "q02 Crimea\n2014",
    "q03_maidan_revolution": "q03 Maidan",
    "q04_referendum":        "q04 referendum",
    "q05_mh17":              "q05 MH17",
    "q06_crimea_belongs":    "q06 Crimea\nbelongs",
    "q07_pov_russia":        "q07 POV\nRussia",
    "q08_pov_ukraine":       "q08 POV\nUkraine",
    "q09_bandera":           "q09 Bandera",
}
QID_ORDER = list(QID_LABEL)

# Colours
COL_OPENAI   = "#2E75B6"   # US blue, matches palette
COL_ALIBABA  = "#D73027"   # CN red, matches palette
COL_ROBUST   = "#1B9E77"   # green for q05/q09 stable-fusion highlight
ROBUST_QIDS  = {"q05_mh17", "q09_bandera"}


def load_phase_a() -> pd.DataFrame:
    df = pd.read_csv(PHASE_A_CORPUS_OVERLAP_PATH)
    return df[df["lang_pair"] == "ru-uk"].set_index("tokenizer")


def load_phase_b() -> pd.DataFrame:
    """Return DataFrame with index=qid and columns=[openai, alibaba]."""
    oai = pd.read_csv(EXP_DIR / "openai_te3s" / "bmcs200" / "co_clustering_fractions.csv")
    ali = pd.read_csv(EXP_DIR / "alibaba_v3"  / "bmcs150" / "co_clustering_fractions.csv")
    oai_uk = oai[oai.response_lang == "uk"].set_index("qid")["share_with_ru_anchor"]
    ali_uk = ali[ali.response_lang == "uk"].set_index("qid")["share_with_ru_anchor"]
    return pd.DataFrame({"openai": oai_uk, "alibaba": ali_uk}).reindex(QID_ORDER)


def panel_phase_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    rows = [
        ("cl100k_base",   "OpenAI cl100k_base"),
        ("qwen2_proxy",   "Qwen2 (Alibaba proxy)"),
        ("cohere_ml_v3",  "Cohere ML-v3"),
    ]
    labels = [lbl for _, lbl in rows]
    values = [df.loc[k, "corpus_overlap"] for k, _ in rows]
    colors = ["#666666", COL_ALIBABA, "#1B9E77"]

    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.6)
    for yi, v in zip(y, values):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=10, color="#222")

    # BLOOM reference line at 0.76
    ax.axvline(0.76, color="#888", linestyle=":", linewidth=1.2)
    ax.text(0.76, len(rows) - 0.35, "BLOOM\n0.76", fontsize=8.5,
            color="#666", ha="center", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("RU-UK corpus_overlap  (Qi et al. 2023)", fontsize=10)
    ax.set_title("Phase A — tokenizer overlap on EN/RU/UK Wikipedia\n"
                 "(778 triplets, ~33M tokens)",
                 fontsize=11, fontweight="bold", pad=8, loc="left")
    ax.grid(axis="x", alpha=0.25)


def panel_phase_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    x = np.arange(len(df))
    bar_w = 0.36
    oai_vals = df["openai"].to_numpy()
    ali_vals = df["alibaba"].to_numpy()

    # Default colours; override q05/q09 with robust-fusion highlight
    edge_robust = [COL_ROBUST if q in ROBUST_QIDS else "white" for q in df.index]
    lw_robust   = [2.2        if q in ROBUST_QIDS else 0.6     for q in df.index]

    ax.bar(x - bar_w/2, oai_vals, bar_w, color=COL_OPENAI,
           edgecolor=edge_robust, linewidth=lw_robust,
           label="OpenAI te3s (bmcs=200, 12 clusters)")
    ax.bar(x + bar_w/2, ali_vals, bar_w, color=COL_ALIBABA,
           edgecolor=edge_robust, linewidth=lw_robust,
           label="Alibaba v3 (bmcs=150, 11 clusters)")

    ax.axhline(0.5, color="#999", linewidth=0.8, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels([QID_LABEL[q] for q in df.index], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0, 1.01, 0.25)])
    ax.set_ylabel("UK responses sharing a cluster\nwith the Russian Wikipedia anchor",
                  fontsize=10)
    ax.set_title("Phase B — does the cluster fusion survive a tokenizer change?\n"
                 "(matched cluster resolution; lead-only Wikipedia anchors)",
                 fontsize=11, fontweight="bold", pad=8, loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=9)


def render(out_path: Path) -> None:
    df_a = load_phase_a()
    df_b = load_phase_b()

    fig = plt.figure(figsize=(18, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.05, 2.4])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 2])

    panel_phase_a(ax_a, df_a)
    panel_phase_b(ax_b, df_b)

    fig.suptitle("exp_017 — Tokenizer leakage is not the explanation for cluster fusion",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.5, -0.04,
        "Phase A: 3 inspectable tokenizers all below BLOOM's 0.76 RU-UK overlap; Cohere lowest, "
        "Alibaba (Qwen-proxy) middle, cl100k highest.   "
        "Phase B: at matched cluster count, total UK→RU fusion is comparable (4 of 9 questions ≥87%) "
        "but redistributes across questions.   "
        "Only q05 MH17 and q09 Bandera (green outlines) fuse robustly across both embedders — "
        "the rest is embedder-conditional.",
        ha="center", fontsize=9.5, color="#444", style="italic", wrap=True,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "05_exp_017_tokenizer_robustness.png"
    render(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
