"""Re-render the cross-conflict and event-ladder thesis figures with the
canonical palettes (babelbias.palette.LANG_COLOR / FAMILY_COLOR).

All figures are title-less per the project figure rule; per-panel conflict
identifiers are drawn as in-axes data labels, never as axes titles.
Outputs go straight to Report/tex/figures/ under the thesis filenames.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from babelbias.palette import FAMILY_COLOR, FAMILY_LABEL, LANG_COLOR, LANG_LABEL

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "Report" / "tex" / "figures"

sns.set_theme(
    style="whitegrid", context="paper", font_scale=1.15,
    rc={
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
        "savefig.dpi": 220, "savefig.bbox": "tight",
    },
)

GREY = "#8A8F98"

# (slug, event-analysis dir, langs, negative pole, positive pole)
STANCE_CONFLICTS = [
    ("ru_uk", "Russia-Ukraine", "ru_uk_core", ["en", "ru", "uk"],
     "Ukrainian framing", "Russian framing"),
    ("in_pk", "india_pakistan", "india_pakistan", ["en", "hi", "ur"],
     "Pakistani framing", "Indian framing"),
    ("il_ps", "israel_palestine", "israel_palestine", ["en", "he", "ar"],
     "Palestinian framing", "Israeli framing"),
    ("taiwan", "taiwan_strait", "taiwan_strait", ["en", "zh"],
     "Taiwanese framing", "PRC framing"),
    ("falklands", "falklands", "falklands", ["en", "es"],
     "Argentine framing", "British framing"),
]

LADDER_ORDER = ["ru_uk", "il_ps", "in_pk", "cn_jp", "gr_tr", "pl_ru",
                "cs_sk", "no_sv"]
DISPLAY_TO_SLUG = {
    "Russia–Ukraine": "ru_uk", "Israel–Palestine": "il_ps",
    "India–Pakistan": "in_pk", "China–Japan": "cn_jp",
    "Greece–Turkey": "gr_tr", "Poland–Russia": "pl_ru",
    "Velvet Divorce": "cs_sk", "Norway–Sweden": "no_sv",
    "Velvet Divorce (settled)": "cs_sk", "Norway–Sweden (settled)": "no_sv",
}
STANCE_TO_SLUG = {
    "ru_uk_core": "ru_uk", "israel_palestine": "il_ps",
    "india_pakistan": "in_pk", "china_japan": "cn_jp",
    "greece_turkey": "gr_tr", "poland_russia": "pl_ru",
    "czech_slovak": "cs_sk", "norway_sweden": "no_sv",
}


def _panel_tag(ax, text: str) -> None:
    """Conflict identifier as an in-axes data label (not an axes title)."""
    ax.text(0.02, 0.985, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=10.5, fontweight="bold", color="#333333")


# ── Figure 1: stance 5-panel ────────────────────────────────────────────────

def _stance_cells(event_dir: str, langs: list[str]) -> pd.DataFrame:
    """Canonical estimator: per-model per-language mean stance from the
    per-response projections; keep models with >= 2 language cells
    (Yandex included where it answered)."""
    df = pd.read_csv(DATA / event_dir / "analysis" / "exp_021_stance_axis.csv")
    cell = df.pivot_table(index="model", columns="lang", values="stance")
    cell = cell[[l for l in langs if l in cell.columns]]
    return cell[cell.notna().sum(axis=1) >= 2]


def render_stance_5panel() -> Path:
    fig = plt.figure(figsize=(12.5, 7.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], hspace=0.42,
                          wspace=0.55)
    axes = {
        "ru_uk": fig.add_subplot(gs[:, 0]),
        "in_pk": fig.add_subplot(gs[0, 1]),
        "il_ps": fig.add_subplot(gs[0, 2]),
        "taiwan": fig.add_subplot(gs[1, 1]),
        "falklands": fig.add_subplot(gs[1, 2]),
    }
    for fam, event_dir, event, langs, neg, pos in STANCE_CONFLICTS:
        ax = axes[fam]
        cell = _stance_cells(event_dir, langs)
        gap = cell.max(axis=1) - cell.min(axis=1)
        cell = cell.loc[gap.sort_values().index]
        small = fam != "ru_uk"
        for i, (model, row) in enumerate(cell.iterrows()):
            ax.plot([row.min(), row.max()], [i, i], color="#c9ced6",
                    lw=1.3, zorder=1)
            for lang in langs:
                if pd.notna(row.get(lang)):
                    ax.scatter(row[lang], i, s=26 if small else 46,
                               color=LANG_COLOR[lang], zorder=3)
        ax.set_yticks(range(len(cell)))
        ax.set_yticklabels(
            [m.replace("baidu/", "").replace("ollama:", "") + (" (local)" if "ollama:" in m else "")
             for m in cell.index],
            fontsize=6.6 if small else 8.6)
        ax.axvline(0.0, color="#333333", lw=0.8, ls=":")
        ax.set_xlabel(f"$\\leftarrow$ {neg}   $\\cdot$   {pos} $\\rightarrow$",
                      fontsize=7.4 if small else 9.5)
        ax.tick_params(axis="x", labelsize=6.8 if small else 8.5)
        _panel_tag(ax, FAMILY_LABEL[fam])
        handles = [plt.Line2D([0], [0], marker="o", ls="", ms=5.5,
                              color=LANG_COLOR[l], label=LANG_LABEL[l])
                   for l in langs]
        ax.set_ylim(-3.4 if small else -2.2, len(cell) + (1.4 if small else 0.8))
        ax.legend(handles=handles, loc="lower right",
                  fontsize=6.2 if small else 8.2, frameon=False,
                  borderpad=0.2, handletextpad=0.25, labelspacing=0.25)
    out = OUT_DIR / "exp_021_cross_conflict_5panel.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


# ── Figure 2: ladder overlay ────────────────────────────────────────────────

def render_overlay() -> Path:
    df = pd.read_csv(DATA / "exp_007_timeline" / "gradient_all_conflicts.csv")
    df = df.sort_values("year").reset_index(drop=True)
    xpos = {ev: i for i, ev in enumerate(df["event"])}

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    settled = {"cs_sk", "no_sv"}
    s_vals = df[df["conflict"].isin(settled)]["en_pull"]
    ax.axhspan(s_vals.min(), s_vals.max(), color="#000000", alpha=0.06,
               zorder=0)
    ax.text(0.3, s_vals.max() + 0.004, "settled-separation band",
            fontsize=8.5, color="#666666")
    live = df[df["year"] >= 2022]
    if len(live):
        ax.axvspan(min(xpos[e] for e in live["event"]) - 0.5,
                   len(df) - 0.5, color="#A50026", alpha=0.05, zorder=0)
    for fam in [f for f in LADDER_ORDER if f not in settled]:
        sub = df[df["conflict"] == fam]
        if sub.empty:
            continue
        xs = [xpos[e] for e in sub["event"]]
        ax.plot(xs, sub["en_pull"], "-o", color=FAMILY_COLOR[fam],
                lw=1.8, ms=6, label=FAMILY_LABEL[fam], zorder=3)
    for fam in settled:
        sub = df[df["conflict"] == fam]
        for _, r in sub.iterrows():
            ax.scatter(xpos[r["event"]], r["en_pull"], marker="X", s=70,
                       color=FAMILY_COLOR[fam], zorder=4)
            ax.annotate(FAMILY_LABEL[fam].replace(" (settled)", ""),
                        (xpos[r["event"]], r["en_pull"]),
                        xytext=(5, -11), textcoords="offset points",
                        fontsize=8, color="#555555")
    ax.axhline(0.0, color="#333333", lw=0.7, ls=":")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["year"], rotation=55, fontsize=7.5)
    ax.set_xlabel("event year (one slot per ladder event)")
    ax.set_ylabel("EN diagonal ingroup pull (row-centred cosine)")
    ax.set_ylim(-0.015, 0.315)
    ax.legend(loc="lower center", fontsize=8.5, frameon=True, ncol=3,
              bbox_to_anchor=(0.5, 0.02))
    out = OUT_DIR / "exp_007_conflict_overlay_enpull.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


# ── Figure 3: refusal by conflict ──────────────────────────────────────────

def render_refusal() -> Path:
    df = pd.read_csv(DATA / "exp_007_timeline" / "refusal_by_event.csv")
    df["slug"] = df["conflict"].map(DISPLAY_TO_SLUG)
    order = [f for f in LADDER_ORDER if f in set(df["slug"])]
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for i, fam in enumerate(order):
        sub = df[df["slug"] == fam]
        xs = i + rng.uniform(-0.18, 0.18, len(sub))
        ax.scatter(xs, sub["refusal_pct"], s=26, color=FAMILY_COLOR[fam],
                   alpha=0.85, zorder=3)
        ax.hlines(sub["refusal_pct"].mean(), i - 0.3, i + 0.3,
                  color="#333333", lw=1.6, zorder=4)
        outliers = sub[sub["refusal_pct"] > 5].sort_values(
            "refusal_pct", ascending=False)
        for k, (_, r) in enumerate(outliers.iterrows()):
            if r["event"] == "ru_uk_core":
                ev = "2014 case study"
            else:
                ev = r["event"].split("_", 1)[1].replace("_", " ")
            ax.annotate(f"{ev} · {r['lang']}", (i, r["refusal_pct"]),
                        xytext=(14, 4 - 10 * k),
                        textcoords="offset points",
                        fontsize=7.4, color="#444444", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.5,
                                        color="#aaaaaa",
                                        shrinkA=0, shrinkB=2))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([FAMILY_LABEL[f].replace(" (settled)", "\n(settled)")
                        for f in order], fontsize=8.3)
    ax.set_ylabel("refusal rate per (event, language) cell (%)")
    ax.set_ylim(-0.3, 8.8)
    out = OUT_DIR / "exp_007_refusal_by_conflict.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


# ── Figure 4: convergent validity ──────────────────────────────────────────

def render_convergent() -> Path:
    # five-conflict sweep gap: canonical estimator (see _stance_cells)
    sweep_gap = {}
    for fam, event_dir, event, langs, _, _ in STANCE_CONFLICTS:
        cell = _stance_cells(event_dir, langs)
        sweep_gap[fam] = float((cell.max(axis=1) - cell.min(axis=1)).mean())
    # ladder gap: spread of pooled per-language means
    s = pd.read_csv(DATA / "exp_007_timeline" / "stance_all.csv")
    s["slug"] = s["conflict"].map(STANCE_TO_SLUG)
    ladder_gap = (s.groupby(["slug", "lang"])["mean_stance"].mean()
                    .groupby("slug").agg(lambda v: v.max() - v.min()))

    shared = [f for f in sweep_gap if f in ladder_gap.index]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lim = 0.16
    ax.plot([0, lim], [0, lim], color="#bbbbbb", lw=1.0, ls="--", zorder=1)
    ax.text(lim * 0.86, lim * 0.905, "identity", fontsize=8.5,
            color="#888888", rotation=42)
    for fam in shared:
        x, y = sweep_gap[fam], float(ladder_gap[fam])
        ax.scatter(x, y, s=130, color=FAMILY_COLOR[fam], zorder=3)
        ax.annotate(FAMILY_LABEL[fam], (x, y), xytext=(9, -3),
                    textcoords="offset points", fontsize=9.5)
    ax.set_xlim(0, lim), ax.set_ylim(0, lim)
    ax.set_xlabel("cross-language stance gap — five-conflict sweep")
    ax.set_ylabel("cross-language stance gap — event-ladder corpus")
    out = OUT_DIR / "exp_007_convergent_validity.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


# ── Figure 5: judge stance by conflict ─────────────────────────────────────

def render_judge() -> Path:
    df = pd.read_csv(DATA / "exp_007_timeline" / "judge_stance.csv")
    df["slug"] = df["conflict"].map(
        {"Russia-Ukraine": "ru_uk", "Israel-Palestine": "il_ps",
         "India-Pakistan": "in_pk", "China-Japan": "cn_jp",
         "Greece-Turkey": "gr_tr", "Poland-Russia": "pl_ru",
         "Czech-Slovak (settled)": "cs_sk",
         "Norway-Sweden (settled)": "no_sv"})
    cell = df.groupby(["slug", "lang"])["stance"].mean().reset_index()
    fam_mean = cell.groupby("slug")["stance"].mean().sort_values()
    order = list(fam_mean.index)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for i, fam in enumerate(order):
        if fam in ("cs_sk", "no_sv"):
            ax.axhspan(i - 0.42, i + 0.42, color="#000000", alpha=0.05,
                       zorder=0)
        sub = cell[cell["slug"] == fam]
        ax.plot([sub["stance"].min(), sub["stance"].max()], [i, i],
                color="#c9ced6", lw=1.3, zorder=1)
        sub = sub.sort_values("stance").reset_index(drop=True)
        prev_x = None
        for k, r in sub.iterrows():
            ax.scatter(r["stance"], i, s=64, color=FAMILY_COLOR[fam],
                       zorder=3)
            below = prev_x is not None and abs(r["stance"] - prev_x) < 0.05
            ax.annotate(r["lang"], (r["stance"], i),
                        xytext=(0, -15 if below else 8),
                        textcoords="offset points", fontsize=7.6,
                        ha="center", color="#444444")
            prev_x = r["stance"]
    ax.axvline(0.0, color="#333333", lw=0.9)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FAMILY_LABEL[f] for f in order], fontsize=9)
    ax.set_xlabel("mean judge stance per (family, language) cell "
                  "($-1$ = first-named party's framing, $+1$ = second)")
    out = OUT_DIR / "exp_007_judge_stance_by_conflict.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


# ── Figure 6: method-vs-method disentanglement ─────────────────────────────

EVENTS_DEBIAS = [
    ("ru_uk", DATA / "Russia-Ukraine" / "analysis", "ru_uk_core",
     ["en", "ru", "uk"]),
    ("il_ps", DATA / "israel_palestine" / "analysis", "israel_palestine",
     ["en", "he", "ar"]),
    ("in_pk", DATA / "india_pakistan" / "analysis", "india_pakistan",
     ["en", "hi", "ur"]),
    ("taiwan", DATA / "taiwan_strait" / "analysis", "taiwan_strait",
     ["en", "zh"]),
    ("falklands", DATA / "falklands" / "analysis", "falklands",
     ["en", "es"]),
]


def _diag_means(analysis_dir: Path, event: str, langs, debiased: bool):
    suffix = f"{event}_debiased" if debiased else event
    per_lang = defaultdict(list)
    pats = [f"*/{suffix}/anchor_heatmap_rowcentered.csv",
            f"baidu/*/{suffix}/anchor_heatmap_rowcentered.csv"]
    for pat in pats:
        for p in analysis_dir.glob(pat):
            if "yandex" in str(p):
                continue
            m = pd.read_csv(p, index_col=0)
            for lang in langs:
                r, c = f"resp_{lang}", f"wiki_{lang}"
                if r in m.index and c in m.columns:
                    per_lang[lang].append(float(m.loc[r, c]))
    return {l: float(np.nanmean(v)) for l, v in per_lang.items() if v}


def render_method_vs_method() -> Path:
    topic = pd.read_csv(DATA / "Russia-Ukraine" / "analysis"
                        / "exp_006_topic_vs_language.csv")
    topic = topic[~topic["model"].str.contains("yandex")]
    off = topic.groupby(["event", "lang"])["off_topic_cos"].mean()

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    pts = []
    for fam, adir, event, langs in EVENTS_DEBIAS:
        raw = _diag_means(adir, event, langs, False)
        deb = _diag_means(adir, event, langs, True)
        for lang in langs:
            if lang not in raw or raw[lang] <= 0.005:
                continue
            collapse = (1 - deb[lang] / raw[lang]) * 100
            key_event = event if (event, lang) in off.index else None
            if key_event is None:
                continue
            y = float(off[(key_event, lang)])
            pts.append((fam, lang, collapse, y))
    for fam, lang, x, y in pts:
        ax.scatter(x, y, s=120, color=FAMILY_COLOR[fam],
                   edgecolor="#333333", lw=0.7, zorder=3)
        ax.annotate(f"{FAMILY_LABEL[fam].split('–')[0][:4]}·{lang.upper()}",
                    (x, y), xytext=(7, 5), textcoords="offset points",
                    fontsize=8.2, color="#333333")
    xs = [p[2] for p in pts]
    ys = [p[3] for p in pts]
    ax.axvline(np.median(xs), color="#aab2bd", lw=0.8, ls="--", zorder=1)
    ax.axhline(np.median(ys), color="#aab2bd", lw=0.8, ls="--", zorder=1)
    ax.text(0.99, 0.02, "content-driven:\nlow collapse, low language floor",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color="#1B9E77")
    ax.text(0.99, 0.90, "language-axis-dominated",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color="#A50026")
    ax.set_xlabel("debiasing collapse of the diagonal pull (%)")
    ax.set_ylabel("off-topic same-language baseline (mean cosine)")
    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=8,
                          color=FAMILY_COLOR[f], label=FAMILY_LABEL[f])
               for f, *_ in EVENTS_DEBIAS]
    ax.legend(handles=handles, loc="center right", fontsize=8.2,
              frameon=True)
    out = OUT_DIR / "exp_006_method_vs_method.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


# ── Figure 7: colour key ───────────────────────────────────────────────────

def render_color_key() -> Path:
    from babelbias.palette import MODEL_COLORS, PALETTE_GROUPS

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 0.85, 1.1]})
    for ax in axes:
        ax.set_axis_off()

    def swatch_col(ax, entries, header):
        ax.text(0.0, 1.0, header, fontsize=10, fontweight="bold",
                va="top", color="#333333")
        y = 0.90
        for label, color, indent in entries:
            if color is None:
                ax.text(0.0, y, label, fontsize=8.6, fontweight="bold",
                        va="center", color="#555555")
            else:
                ax.add_patch(plt.Rectangle((0.02 + indent, y - 0.022),
                                           0.07, 0.044, facecolor=color,
                                           edgecolor="#333333", lw=0.5,
                                           transform=ax.transAxes,
                                           clip_on=False))
            if color is not None:
                ax.text(0.12 + indent, y, label, fontsize=8.4, va="center",
                        color="#222222")
            y -= 0.062
        return y

    prov = []
    for group, models in PALETTE_GROUPS:
        prov.append((group, None, 0.0))
        for m in models:
            prov.append((m.replace("baidu/", "").replace("ollama:", ""),
                         MODEL_COLORS[m], 0.03))
    swatch_col(axes[0], prov, "Provider (training ecosystem)")
    swatch_col(axes[1],
               [(LANG_LABEL[l], LANG_COLOR[l], 0.0) for l in LANG_COLOR],
               "Prompt / response language")
    swatch_col(axes[2],
               [(FAMILY_LABEL[f], FAMILY_COLOR[f], 0.0) for f in FAMILY_COLOR],
               "Conflict family")
    out = OUT_DIR / "palette_legend.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def render_topic_vs_language() -> Path:
    """Empirical on-topic vs off-topic scatter per (event, language)."""
    t = pd.read_csv(DATA / "Russia-Ukraine" / "analysis"
                    / "exp_006_topic_vs_language.csv")
    t = t[~t["model"].str.contains("yandex")]
    agg = t.groupby(["event", "lang"])[["on_topic_cos",
                                        "off_topic_cos"]].mean()
    ev_fam = {"ru_uk_core": "ru_uk", "israel_palestine": "il_ps",
              "india_pakistan": "in_pk", "taiwan_strait": "taiwan",
              "falklands": "falklands"}
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    lim = float(agg.max().max()) + 0.08
    ax.plot([0, lim], [0, lim], color="#bbbbbb", lw=1.0, ls="--", zorder=1)
    ax.text(0.84 * lim, 0.885 * lim, "$y = x$ (no topic lift)",
            fontsize=8.5, color="#888888", rotation=42)
    # The five EN cells crowd top-left; stagger their labels so none
    # collide (offsets in points, tuned per family).
    en_offsets = {"in_pk": (-46, 10), "falklands": (-52, -1),
                  "ru_uk": (-50, -8), "taiwan": (-50, -18),
                  "il_ps": (9, -12)}
    for (event, lang), row in agg.iterrows():
        fam = ev_fam.get(event)
        if fam is None:
            continue
        ax.scatter(row["off_topic_cos"], row["on_topic_cos"], s=110,
                   color=LANG_COLOR.get(lang, "#666666"),
                   edgecolor="#333333", lw=0.7, zorder=3)
        off = en_offsets.get(fam, (7, 4)) if lang == "en" else (7, 4)
        ax.annotate(f"{FAMILY_LABEL[fam].split(chr(8211))[0][:4]}"
                    f"·{lang.upper()}",
                    (row["off_topic_cos"], row["on_topic_cos"]),
                    xytext=off, textcoords="offset points",
                    fontsize=8.2, color="#333333")
    ax.set_xlim(0, lim), ax.set_ylim(0, lim)
    ax.set_xlabel("off-topic same-language baseline (mean cosine)")
    ax.set_ylabel("on-topic anchor cosine (mean)")
    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=7,
                          color=LANG_COLOR[l], label=LANG_LABEL[l])
               for l in ["en", "ru", "uk", "he", "ar", "hi", "ur",
                         "zh", "es"]]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              frameon=True, ncol=2)
    out = OUT_DIR / "topic_vs_language_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def render_anchor_length_control() -> Path:
    """EN pull vs mean anchor lead length (tokens) per ladder event."""
    import json as _json
    import sys

    import tiktoken
    sys.path.insert(0, str(PROJECT_ROOT / "Code" / "src"))
    from babelbias.wiki import extract_lead

    enc = tiktoken.get_encoding("cl100k_base")
    df = pd.read_csv(DATA / "exp_007_timeline"
                     / "gradient_all_conflicts.csv")
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for _, r in df.iterrows():
        event, fam = r["event"], r["conflict"]
        bank = _json.loads((PROJECT_ROOT / "Code" / "prompts"
                            / f"{event}.json").read_text())
        lens = []
        for prompt in bank["prompts"]:
            slug = prompt["wiki_anchor_slug"]
            for lang in bank["languages"]:
                raw = DATA / event / "raw" / f"{slug}_{lang}_raw.json"
                if raw.exists():
                    lead = extract_lead(
                        _json.loads(raw.read_text()).get("content", ""))
                    if lead:
                        lens.append(len(enc.encode(lead)))
        if not lens:
            logger.warning("no leads for %s", event)
            continue
        x = float(np.mean(lens))
        marker = "X" if fam in ("cs_sk", "no_sv") else "o"
        ax.scatter(x, r["en_pull"], s=64, marker=marker,
                   color=FAMILY_COLOR[fam], zorder=3)
        if event == "ruuk_2022_invasion":
            ax.annotate("2022 invasion", (x, r["en_pull"]),
                        xytext=(8, -12), textcoords="offset points",
                        fontsize=8.5, color="#333333")
    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=7,
                          color=FAMILY_COLOR[f], label=FAMILY_LABEL[f])
               for f in LADDER_ORDER]
    ax.legend(handles=handles, loc="upper right", fontsize=7.6,
              frameon=True, ncol=2)
    ax.set_xlabel("mean anchor lead length across the event's languages "
                  "(cl100k tokens)")
    ax.set_ylabel("EN diagonal ingroup pull (row-centred cosine)")
    out = OUT_DIR / "exp_007_fig6_anchor_length_control.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def print_stance_table() -> None:
    for fam, event_dir, event, langs, _, _ in STANCE_CONFLICTS:
        cell = _stance_cells(event_dir, langs)
        gap = cell.max(axis=1) - cell.min(axis=1)
        print(f"TABLE {FAMILY_LABEL[fam]:20s} mean {gap.mean():+.3f} "
              f"max {gap.max():+.3f} n {len(cell)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    render_stance_5panel()
    render_overlay()
    render_refusal()
    render_convergent()
    render_judge()
    render_method_vs_method()
    render_color_key()
    render_topic_vs_language()
    render_anchor_length_control()
    print_stance_table()
