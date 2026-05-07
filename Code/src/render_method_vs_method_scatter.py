"""Method-vs-method convergent-validity scatter for the thesis
Discussion chapter.

For every (event, language) cell we have two independent metrics:

  * debiasing collapse % — how much of the diagonal ingroup pull
    survives the language-axis projection.
  * off-topic same-language baseline — how strongly responses
    cluster with random universal-control articles in the same
    language (no topic overlap).

A cell that is content-driven should have low collapse % AND a
small off-topic baseline. A cell that is language-similarity-driven
should have high collapse % AND a high off-topic baseline.

Plotting one against the other lets us see whether the two methods
agree on which cells are which. The Hebrew cell is highlighted in
red — both methods place it as the lone language-similarity outlier.

Output:
  Presentations/figures/methodology/method_vs_method_scatter.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from babelbias.paths import DATA_ROOT, PROJECT_ROOT


sns.set_theme(style="whitegrid", context="talk", font_scale=0.78,
              rc={"savefig.dpi": 200, "savefig.bbox": "tight",
                  "axes.spines.top": False, "axes.spines.right": False})

OUT = (PROJECT_ROOT / "Presentations" / "figures" / "methodology"
       / "method_vs_method_scatter.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

EVENTS = {
    "ru_uk_core":       ["en", "ru", "uk"],
    "israel_palestine": ["en", "he", "ar"],
    "india_pakistan":   ["en", "hi", "ur"],
    "taiwan_strait":    ["en", "zh"],
    "falklands":        ["en", "es"],
}
EVENT_SHORT = {"ru_uk_core": "RU-UK", "israel_palestine": "IL-PS",
               "india_pakistan": "IP", "taiwan_strait": "TW",
               "falklands": "FALK"}
LANG_LABEL = {"en": "EN", "ru": "RU", "uk": "UK", "he": "HE", "ar": "AR",
              "zh": "ZH", "es": "ES", "hi": "HI", "ur": "UR"}
LANG_COLOR = {
    "en": "#2E75B6", "ru": "#A50026", "uk": "#FDAE61",
    "he": "#1B9E77", "ar": "#66C2A5",
    "zh": "#984EA3", "es": "#F2B701",
    "hi": "#A6761D", "ur": "#E7298A",
}
MODELS = ["claude-haiku-4-5", "gpt-4o-mini", "gemini-2.5-flash",
          "grok-3-mini", "mercury-2", "deepseek-chat", "qwen-plus",
          "glm-4.5", "baidu/ernie-4.5-300b-a47b",
          "c4ai-aya-expanse-32b", "command-r7b-arabic-02-2025",
          "ollama:allam-7b", "ollama:taide-llama3-8b",
          "jamba-mini-2-2026-01"]


def collect() -> pd.DataFrame:
    # Method A — debiasing collapse %
    rows = []
    for event, langs in EVENTS.items():
        for lang in langs:
            raws, debs = [], []
            for m in MODELS:
                for tag, store in [(event, raws),
                                    (f"{event}_debiased", debs)]:
                    f = (DATA_ROOT / event / "analysis" / m / tag
                         / "anchor_heatmap_rowcentered.csv")
                    if not f.exists():
                        continue
                    df = pd.read_csv(f, index_col=0)
                    col, row = f"wiki_{lang}", f"resp_{lang}"
                    if col in df.columns and row in df.index:
                        v = df.at[row, col]
                        if pd.notna(v):
                            store.append(v)
            if not raws or not debs:
                continue
            rmean = float(np.mean(raws))
            dmean = float(np.mean(debs))
            collapse = (1 - dmean / rmean) * 100 if abs(rmean) > 0.02 else None
            if collapse is None:
                continue
            rows.append({"event": event, "lang": lang,
                          "raw": rmean, "deb": dmean,
                          "collapse_pct": collapse})

    coll_df = pd.DataFrame(rows)

    # Method B — off-topic same-language baseline
    tl = pd.read_csv(DATA_ROOT / "Russia-Ukraine" / "analysis"
                       / "exp_006_topic_vs_language.csv")
    base = (tl.groupby(["event", "lang"])["off_topic_cos"].mean()
            .reset_index().rename(columns={"off_topic_cos": "off_topic"}))

    return coll_df.merge(base, on=["event", "lang"], how="inner")


def main() -> None:
    df = collect()
    if df.empty:
        raise SystemExit("No cells to plot")

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    # Quadrant separators
    median_collapse = float(df["collapse_pct"].median())
    median_off      = float(df["off_topic"].median())
    ax.axvline(median_collapse, color="#94A3B8", lw=0.8, linestyle="--",
                alpha=0.7, zorder=1)
    ax.axhline(median_off, color="#94A3B8", lw=0.8, linestyle="--",
                alpha=0.7, zorder=1)

    # Quadrant labels (corner annotations)
    xmax = df["collapse_pct"].max() * 1.12
    ymax = df["off_topic"].max() * 1.18
    ax.set_xlim(-5, xmax)
    ax.set_ylim(0.04, ymax)
    ax.text(xmax * 0.97, 0.07, "content-driven\n(low collapse,\nlow language floor)",
             fontsize=10, color="#1B9E77", weight="700", ha="right",
             alpha=0.85)
    ax.text(xmax * 0.03, ymax * 0.95, "high baseline\n(language sticks\nto itself)",
             fontsize=10, color="#A50026", weight="700", ha="left",
             alpha=0.85)
    ax.text(xmax * 0.97, ymax * 0.95,
             "language-axis-only\n(both signals agree)",
             fontsize=10, color="#A50026", weight="800", ha="right",
             alpha=0.95)

    # Plot points
    for _, r in df.iterrows():
        is_he = (r["event"] == "israel_palestine" and r["lang"] == "he")
        color = LANG_COLOR.get(r["lang"], "#475569")
        ax.scatter(r["collapse_pct"], r["off_topic"],
                    s=320, color=color, edgecolors="#A50026" if is_he else "#0F172A",
                    linewidths=2.5 if is_he else 1.2,
                    zorder=4 if is_he else 3, alpha=0.92)

        label = f"{EVENT_SHORT[r['event']]}/{LANG_LABEL[r['lang']]}"
        if is_he:
            ax.annotate(label,
                         xy=(r["collapse_pct"], r["off_topic"]),
                         xytext=(r["collapse_pct"] - 12,
                                  r["off_topic"] + 0.04),
                         fontsize=12, weight="800", color="#A50026",
                         arrowprops=dict(arrowstyle="->",
                                          color="#A50026", lw=1.6),
                         bbox=dict(boxstyle="round,pad=0.4",
                                    facecolor="#fff3f0",
                                    edgecolor="#A50026", linewidth=1.4),
                         zorder=5)
        else:
            ax.text(r["collapse_pct"] + 1.5, r["off_topic"] + 0.005,
                     label, fontsize=9.5, color="#222", weight="600",
                     zorder=4)

    ax.set_xlabel("Method A — debiasing collapse % (raw → debiased)",
                   fontsize=11)
    ax.set_ylabel("Method B — off-topic same-language baseline",
                   fontsize=11)
    ax.grid(alpha=0.3)

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
