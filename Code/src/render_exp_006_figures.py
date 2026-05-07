"""Render the 22 June exp_006 deck figures.

Four publication-quality figures, all matplotlib/seaborn talk-context
on a coordinated palette anchored on babelbias.palette:

  01_cross_conflict_5panel.png   — HEADLINE
      5-panel grouped bar chart, one per event. Bars = per-language
      diagonal ingroup pull, averaged across the 14 providers, error
      bars = 95% CI across providers. Falklands explicitly marked as
      the Oeberst-confirmed null falsification anchor.

  02_topic_vs_language_scatter.png   — METHODOLOGY
      Scatter of (event × language) cells. X = off-topic same-language
      baseline (cosine to random universal-control articles in same
      lang). Y = on-topic cosine (cosine to event Wikipedia anchor).
      Points coloured by language family; Hebrew flagged as outlier.
      Diagonal y=x line = "no topic signal." Vertical distance above
      diagonal = topic-driven pull.

  03_raw_vs_debiased.png   — DEBIASING COLLAPSE HIERARCHY
      Per-(event, language) cell, sorted by debiasing collapse %.
      Two bars per cell: raw and debiased ingroup pull. Annotated with
      collapse %. EN cells (low collapse) at left, HE (96% collapse)
      at right.

  04_hebrew_convergent.png   — CONVERGENT-VALIDITY PAIR
      Two side-by-side panels: (left) % collapse under debiasing per
      language; (right) off-topic same-language baseline per language.
      Both peak at HE — two independent methods point to the same
      Hebrew edge case.

Output: Presentations/figures/22 June/*.png + symlinks into
Presentations/2026-06-22/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from babelbias.paths import DATA_ROOT, PROJECT_ROOT


sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=0.78,
    rc={
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#333333",
        "axes.linewidth":    0.9,
        "grid.color":        "#e6e6e6",
        "grid.linewidth":    0.6,
        "savefig.dpi":       180,
        "savefig.bbox":      "tight",
    },
)

OUT_DIR = PROJECT_ROOT / "Presentations" / "figures" / "22 June"

EVENTS = [
    ("ru_uk_core",       "Russia–Ukraine",     ["en", "ru", "uk"]),
    ("israel_palestine", "Israel–Palestine",   ["en", "he", "ar"]),
    ("india_pakistan",   "India–Pakistan",     ["en", "hi", "ur"]),
    ("taiwan_strait",    "Taiwan strait",      ["en", "zh"]),
    ("falklands",        "Falklands / Malvinas", ["en", "es"]),
]
EVENT_TITLES = {ev: title for ev, title, _ in EVENTS}

MODELS = [
    "claude-haiku-4-5", "gpt-4o-mini", "gemini-2.5-flash", "grok-3-mini",
    "mercury-2", "deepseek-chat", "qwen-plus", "glm-4.5",
    "baidu/ernie-4.5-300b-a47b", "c4ai-aya-expanse-32b",
    "command-r7b-arabic-02-2025", "ollama:allam-7b",
    "ollama:taide-llama3-8b", "jamba-mini-2-2026-01",
]

# Language-family colours (script-based grouping; aligned with the
# Slavic / Semitic / CJK / Romance / Indic logic in the thesis figure
# idea Thesis_Document_Ideas.md § "Language-axis PCA").
LANG_COLOR = {
    "en": "#2E75B6",   # English — US blue
    "ru": "#A50026",   # Slavic — China-red family overlap with RU-UK
    "uk": "#FDAE61",
    "he": "#1B9E77",   # Semitic — teal
    "ar": "#66C2A5",
    "zh": "#984EA3",   # CJK — purple
    "es": "#F2B701",   # Romance — amber
    "hi": "#A6761D",   # Indic — brown
    "ur": "#E7298A",
}
LANG_LABEL = {
    "en": "EN", "ru": "RU", "uk": "UK", "he": "HE", "ar": "AR",
    "zh": "ZH", "es": "ES", "hi": "HI", "ur": "UR",
}


# ── Data loaders ──────────────────────────────────────────────────────────


def load_event_diagonals(event: str, debiased: bool = False) -> pd.DataFrame:
    """Per-(model, lang) diagonal ingroup pull. Returns rows: model, lang, val."""
    suffix = f"{event}_debiased" if debiased else event
    rows = []
    for m in MODELS:
        f = (DATA_ROOT / event / "analysis" / m / suffix
             / "anchor_heatmap_rowcentered.csv")
        if not f.exists():
            continue
        df = pd.read_csv(f, index_col=0)
        for lang in df.index.str.replace("resp_", ""):
            row, col = f"resp_{lang}", f"wiki_{lang}"
            if col in df.columns and row in df.index:
                v = df.at[row, col]
                if pd.notna(v):
                    rows.append({"model": m, "lang": lang, "val": float(v)})
    return pd.DataFrame(rows)


def load_topic_lift_csv() -> pd.DataFrame:
    p = (DATA_ROOT / "Russia-Ukraine" / "analysis"
         / "exp_006_topic_vs_language.csv")
    return pd.read_csv(p)


def aggregate_by_lang(diag_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± 95% CI across models per language."""
    if diag_df.empty:
        return pd.DataFrame(columns=["lang", "mean", "ci95", "n"])
    out = []
    for lang, g in diag_df.groupby("lang"):
        v = g["val"].to_numpy()
        n = len(v)
        m = float(v.mean())
        ci = float(1.96 * v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out.append({"lang": lang, "mean": m, "ci95": ci, "n": n})
    return pd.DataFrame(out)


# ── Figure 1 — cross-conflict 5-panel headline ────────────────────────────


def figure_01_cross_conflict(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 5.4), sharey=True,
                              constrained_layout=True)

    panel_data = {}
    for event, title, langs in EVENTS:
        agg = aggregate_by_lang(load_event_diagonals(event))
        panel_data[event] = (title, langs, agg)

    y_min = -0.04
    y_max = max(d[2]["mean"].max() + d[2]["ci95"].max()
                for d in panel_data.values()
                if not d[2].empty) * 1.08

    for ax, (event, _, _) in zip(axes, EVENTS):
        title, langs, agg = panel_data[event]
        x = np.arange(len(langs))
        means, cis, colors = [], [], []
        for lang in langs:
            row = agg[agg.lang == lang]
            if row.empty:
                means.append(np.nan); cis.append(0); colors.append("#cccccc")
            else:
                means.append(float(row["mean"].iloc[0]))
                cis.append(float(row["ci95"].iloc[0]))
                colors.append(LANG_COLOR.get(lang, "#888888"))

        bars = ax.bar(x, means, yerr=cis, capsize=4,
                      color=colors, edgecolor="white", linewidth=0.8,
                      error_kw={"elinewidth": 1.2, "ecolor": "#333"})
        # Inline value annotations
        for xi, m_, c_ in zip(x, means, cis):
            if np.isnan(m_):
                continue
            ax.text(xi, m_ + c_ + 0.008, f"{m_:+.2f}",
                    ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold", color="#222")

        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABEL.get(l, l.upper()) for l in langs],
                            fontweight="bold")
        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", alpha=0.3)

    # Falklands annotation — the deliberate Oeberst-null falsification anchor.
    falk_ax = axes[-1]
    falk_ax.annotate(
        "Oeberst-predicted\nnull conflict — confirmed",
        xy=(0.5, 0.06), xycoords="axes fraction",
        fontsize=10, fontweight="bold", ha="center", color="#A50026",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3f0",
                  edgecolor="#A50026", linewidth=1.2),
    )

    # Star the strongest cell (IP UR by raw cosine).
    ip_ax = axes[2]
    ip_ax.annotate(
        "strongest non-EN\ningroup pull observed",
        xy=(2, 0.21), xytext=(1.6, 0.34),
        fontsize=9.5, ha="left", color="#1B9E77", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1B9E77", lw=1.5),
    )

    axes[0].set_ylabel(
        "Diagonal ingroup pull\n(row-centered cosine, mean ± 95% CI across providers)",
        fontsize=10.5,
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2 — topic-vs-language scatter ──────────────────────────────────


def figure_02_topic_vs_language(out_path: Path) -> None:
    df = load_topic_lift_csv()
    # Aggregate across models per (event, lang).
    agg = (df.groupby(["event", "lang"])
              .agg(on=("on_topic_cos", "mean"),
                   off=("off_topic_cos", "mean"),
                   n=("model", "count"))
              .reset_index())
    agg["lift"] = agg["on"] - agg["off"]

    fig, ax = plt.subplots(figsize=(11.5, 7.5), constrained_layout=True)

    # y = x reference (no topic signal)
    ax.plot([0, 0.85], [0, 0.85], color="#bbb", linewidth=1.2,
            linestyle="--", zorder=1, label="y = x  (no topic signal)")

    for _, row in agg.iterrows():
        color = LANG_COLOR.get(row["lang"], "#888888")
        ax.scatter(row["off"], row["on"], s=320, color=color,
                   edgecolor="#222", linewidth=1.3, zorder=3,
                   alpha=0.92)
        # Label = event short + lang
        evt_short = {"ru_uk_core": "RU-UK", "israel_palestine": "IL-PS",
                     "india_pakistan": "IP", "taiwan_strait": "TW",
                     "falklands": "FALK"}[row["event"]]
        offset_x, offset_y = 0.012, 0.0
        # Hebrew gets the special treatment.
        if row["event"] == "israel_palestine" and row["lang"] == "he":
            offset_x, offset_y = -0.04, 0.018
            ax.annotate(
                f"{evt_short}/{LANG_LABEL[row['lang']]}\n"
                f"lift +{row['lift']:.2f}",
                xy=(row["off"], row["on"]),
                xytext=(row["off"] - 0.15, row["on"] + 0.06),
                fontsize=11, fontweight="bold", color="#A50026",
                arrowprops=dict(arrowstyle="->", color="#A50026", lw=1.6),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3f0",
                          edgecolor="#A50026", linewidth=1.4),
            )
        else:
            ax.text(row["off"] + offset_x, row["on"] + offset_y,
                    f"{evt_short}/{LANG_LABEL[row['lang']]}",
                    fontsize=9.5, va="center", color="#222",
                    fontweight="medium")

    ax.set_xlabel("Off-topic same-language baseline\n"
                  "mean cosine to random universal-control articles in same language",
                  fontsize=11)
    ax.set_ylabel("On-topic cosine\n"
                  "mean cosine to event Wikipedia anchor in same language",
                  fontsize=11)
    ax.set_xlim(0, 0.50)
    ax.set_ylim(0.30, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ── Figure 3 — raw vs debiased collapse hierarchy ─────────────────────────


def figure_03_raw_vs_debiased(out_path: Path) -> None:
    rows = []
    for event, title, langs in EVENTS:
        raw = aggregate_by_lang(load_event_diagonals(event, debiased=False))
        deb = aggregate_by_lang(load_event_diagonals(event, debiased=True))
        for lang in langs:
            r = raw[raw.lang == lang]
            d = deb[deb.lang == lang]
            if r.empty or d.empty:
                continue
            rmean = float(r["mean"].iloc[0])
            dmean = float(d["mean"].iloc[0])
            collapse = (1 - dmean / rmean) * 100 if abs(rmean) > 1e-6 else 0
            rows.append({
                "event":    event,
                "event_short": {"ru_uk_core": "RU-UK", "israel_palestine": "IL-PS",
                                "india_pakistan": "IP", "taiwan_strait": "TW",
                                "falklands": "FALK"}[event],
                "lang":     lang,
                "raw":      rmean,
                "deb":      dmean,
                "collapse": collapse,
                "label":    f"{event[:5]}/{LANG_LABEL.get(lang, lang.upper())}",
            })
    df = pd.DataFrame(rows).sort_values("collapse")
    df["xlabel"] = df["event_short"] + "  " + df["lang"].map(LANG_LABEL)

    fig, ax = plt.subplots(figsize=(13, 6.8), constrained_layout=True)
    x = np.arange(len(df))
    bar_w = 0.36

    raw_bars = ax.bar(x - bar_w/2, df["raw"], bar_w,
                      color="#5B9BD5", edgecolor="white", linewidth=0.8,
                      label="Raw cosine ingroup pull")
    deb_bars = ax.bar(x + bar_w/2, df["deb"], bar_w,
                      color="#A50026", edgecolor="white", linewidth=0.8,
                      label="After language-axis debiasing")

    # Collapse % annotation above each pair
    for xi, (raw, deb, col) in enumerate(zip(df["raw"], df["deb"], df["collapse"])):
        top = max(raw, deb)
        ax.text(xi, top + 0.012, f"{col:.0f}%↓" if col > 0 else "—",
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold",
                color="#A50026" if col > 50 else "#555")

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df["xlabel"], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Diagonal ingroup pull\n(row-centered cosine, mean across providers)",
                  fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    # Highlight the Hebrew cell with a callout
    he_idx = df[(df.event == "israel_palestine") & (df.lang == "he")].index
    if len(he_idx):
        he_pos = list(df.index).index(he_idx[0])
        ax.axvspan(he_pos - 0.5, he_pos + 0.5, color="#fff3f0", alpha=0.7, zorder=0)
        ax.annotate(
            "HE: 96% collapse →\nlexical artefact, not framing",
            xy=(he_pos, df.loc[he_idx[0], "raw"]),
            xytext=(he_pos - 3.5, df.loc[he_idx[0], "raw"] + 0.07),
            fontsize=10, color="#A50026", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#A50026", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3f0",
                      edgecolor="#A50026", linewidth=1.4),
        )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ── Figure 4 — Hebrew convergent validity (paired methods) ────────────────


def figure_04_hebrew_convergent(out_path: Path) -> None:
    # Method A — debiasing collapse % per language (averaged across events
    # the lang appears in).
    # Method A — debiasing collapse % per language. Skip cells where the
    # raw signal is too small to support a meaningful collapse-% metric
    # (|raw| < 0.02). That drops Taiwan-ZH, where the raw pull is near zero
    # so any debiasing change is a percentage of noise, not signal.
    coll_rows = []
    for event, _, langs in EVENTS:
        raw = aggregate_by_lang(load_event_diagonals(event, debiased=False))
        deb = aggregate_by_lang(load_event_diagonals(event, debiased=True))
        for lang in langs:
            r = raw[raw.lang == lang]
            d = deb[deb.lang == lang]
            if r.empty or d.empty:
                continue
            rmean, dmean = float(r["mean"].iloc[0]), float(d["mean"].iloc[0])
            if abs(rmean) < 0.02:  # raw signal too small for meaningful %
                continue
            coll_rows.append({"lang": lang, "collapse_pct": (1 - dmean/rmean) * 100})
    coll_df = (pd.DataFrame(coll_rows).groupby("lang", as_index=False)["collapse_pct"]
                .mean().sort_values("collapse_pct", ascending=False))

    # Method B — off-topic same-language baseline per language.
    tl = load_topic_lift_csv()
    base_df = (tl.groupby("lang", as_index=False)["off_topic_cos"]
                 .mean().sort_values("off_topic_cos", ascending=False))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), constrained_layout=True)

    # Panel A
    ax = axes[0]
    colors = ["#A50026" if l == "he" else LANG_COLOR.get(l, "#888")
              for l in coll_df["lang"]]
    ax.barh(coll_df["lang"].map(LANG_LABEL), coll_df["collapse_pct"],
            color=colors, edgecolor="white", linewidth=0.8)
    for i, v in enumerate(coll_df["collapse_pct"]):
        ax.text(v + 1, i, f"{v:+.0f}%", va="center", fontsize=10,
                fontweight="bold" if coll_df.iloc[i]["lang"] == "he" else "normal",
                color="#A50026" if coll_df.iloc[i]["lang"] == "he" else "#222")
    ax.invert_yaxis()
    ax.set_xlabel("Debiasing collapse % (raw → debiased)", fontsize=11)
    ax.set_title("Method A — language-axis debiasing",
                 fontsize=12.5, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.3)

    # Panel B
    ax = axes[1]
    colors = ["#A50026" if l == "he" else LANG_COLOR.get(l, "#888")
              for l in base_df["lang"]]
    ax.barh(base_df["lang"].map(LANG_LABEL), base_df["off_topic_cos"],
            color=colors, edgecolor="white", linewidth=0.8)
    for i, v in enumerate(base_df["off_topic_cos"]):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=10,
                fontweight="bold" if base_df.iloc[i]["lang"] == "he" else "normal",
                color="#A50026" if base_df.iloc[i]["lang"] == "he" else "#222")
    ax.invert_yaxis()
    ax.set_xlabel("Off-topic same-language baseline cosine", fontsize=11)
    ax.set_title("Method B — off-topic same-language baseline",
                 fontsize=12.5, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.3)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ("01_cross_conflict_5panel.png",        figure_01_cross_conflict),
        ("02_topic_vs_language_scatter.png",    figure_02_topic_vs_language),
        ("03_raw_vs_debiased.png",              figure_03_raw_vs_debiased),
        ("04_hebrew_convergent.png",            figure_04_hebrew_convergent),
    ]
    for filename, render in targets:
        out = OUT_DIR / filename
        print(f"Rendering {filename} …")
        render(out)
        print(f"  → {out}")
    print("\nAll figures rendered.")


if __name__ == "__main__":
    main()
