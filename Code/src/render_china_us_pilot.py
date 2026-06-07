"""Presentation graphics for the China-US (COVID-origins) provider-country
pilot. PILOT data on un-reviewed DeepL prompts — filenames carry a `_pilot`
suffix and the result is preliminary (see exp_023 report).

Two figures, each rendered as a static PNG, a sequence of click-build frames
(for §5b PowerPoint Appear/Fade On-Click reveals), and an assembled GIF:

  1. Convergence strip — the "dog that didn't bark". Every provider plotted on
     the US-framing ↔ China-framing stance axis, coloured by HQ ecosystem
     (US blues, China reds). The punchline is visual: the US and China group
     means nearly coincide and the clouds interleave — provider country does
     not separate the framing. All providers sit left of 0 (China-framing lean).

  2. Three-gaps bar — provider-country gap vs the two cross-language gaps
     (China-US, Russia-Ukraine). The axis we flipped to is the shortest bar.

Title-less per project figure rule; colours from babelbias.palette.
Outputs under Presentations/figures/exp_021/ (+ a builds/ subdir).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from babelbias.palette import MODEL_COLORS, PALETTE_GROUPS

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "data" / "china_us" / "analysis" / "exp_021_stance_axis_summary.csv"
RANKING = ROOT / "Presentations" / "figures" / "exp_021" / "cross_conflict_ranking.csv"
OUT_DIR = ROOT / "Presentations" / "figures" / "exp_021"
BUILD_DIR = OUT_DIR / "china_us_pilot_builds"

LANGS = ("en", "zh-cn", "zh-tw")
LANG_MARKER = {"en": "o", "zh-cn": "s", "zh-tw": "^"}
US_BLUE = "#2E75B6"
CN_RED = "#D73027"
INK = "#0F172A"

PERRESP = ROOT / "data" / "china_us" / "analysis" / "exp_021_stance_axis.csv"
H1_QIDS = {"q01_origin", "q03_wiv", "q04_responsibility", "q06_information_control"}
THEME = {
    "q01_origin": "origin attribution",
    "q02_lab_leak": "lab-leak credible?",
    "q03_wiv": "WIV role",
    "q04_responsibility": "responsibility",
    "q05_who_investigation": "WHO mission",
    "q06_information_control": "information control",
    "q07_pov_lab_leak": "POV: lab-leak",
    "q08_pov_zoonotic": "POV: zoonotic",
    "q09_li_wenliang": "Li Wenliang",
}


# ── data ──────────────────────────────────────────────────────────────────
def load_summary() -> dict[str, dict[str, float]]:
    prov: dict[str, dict[str, float]] = defaultdict(dict)
    with SUMMARY.open() as fh:
        for r in csv.DictReader(fh):
            prov[r["model"]][r["lang"]] = float(r["mean"])
    return prov


def group_of(model: str) -> str:
    for name, members in PALETTE_GROUPS:
        if model in members:
            return name
    return "other"


def provider_mean(d: dict[str, float]) -> float:
    vals = [d[l] for l in LANGS if l in d]
    return sum(vals) / len(vals)


def short(model: str) -> str:
    return model.split("/")[-1].replace("ollama:", "")


def cross_language_gap(prov: dict[str, dict[str, float]]) -> float:
    """Mean over providers of (max-lang mean − min-lang mean)."""
    spreads = []
    for d in prov.values():
        vals = [d[l] for l in LANGS if l in d]
        if len(vals) >= 2:
            spreads.append(max(vals) - min(vals))
    return sum(spreads) / len(spreads)


def ru_uk_language_gap() -> float:
    if not RANKING.exists():
        return 0.033  # committed 8-June headline value
    with RANKING.open() as fh:
        for r in csv.DictReader(fh):
            if r["event"] == "ru_uk_core":
                return float(r["mean_provider_spread"])
    return 0.033


# ── figure 1: convergence strip ─────────────────────────────────────────────
def draw_convergence(ax, prov, order, groups_shown: set[str], show_lines: bool,
                     show_callout: bool, gap: float) -> None:
    sns.set_theme(style="white", context="talk", font_scale=0.7)
    us_mean = group_mean(prov, "US")
    cn_mean = group_mean(prov, "China (CAC-regulated)")

    ax.axvline(0.0, color="#94A3B8", lw=1.0, zorder=1)
    # pole anchors at the extremes (top of plot)
    ax.scatter([0.171], [-2.0], marker="|", s=400, color=US_BLUE, zorder=2)
    ax.scatter([-0.305], [-2.0], marker="|", s=400, color=CN_RED, zorder=2)
    ax.text(0.171, -2.45, "US-framing pole", ha="center", va="bottom",
            fontsize=8, color=US_BLUE, weight="600")
    ax.text(-0.305, -2.45, "China-framing pole", ha="center", va="bottom",
            fontsize=8, color=CN_RED, weight="600")

    for y, model in enumerate(order):
        grp = group_of(model)
        if grp not in groups_shown:
            continue
        color = MODEL_COLORS.get(model, "#888")
        d = prov[model]
        for l in LANGS:
            if l in d:
                ax.scatter(d[l], y, marker=LANG_MARKER[l], s=70,
                           color=color, edgecolor="white", linewidth=0.6,
                           zorder=4)
        # connecting whisker across languages
        vals = [d[l] for l in LANGS if l in d]
        if len(vals) >= 2:
            ax.plot([min(vals), max(vals)], [y, y], color=color, lw=2,
                    alpha=0.45, zorder=3)

    if show_lines:
        ax.axvline(us_mean, color=US_BLUE, lw=2, ls="--", zorder=5)
        ax.axvline(cn_mean, color=CN_RED, lw=2, ls="--", zorder=5)
        ax.text(us_mean - 0.004, -0.5, "US mean", ha="right", va="bottom",
                fontsize=8.5, color=US_BLUE, weight="700")
        ax.text(cn_mean + 0.004, -0.5, "China mean", ha="left", va="bottom",
                fontsize=8.5, color=CN_RED, weight="700")
    if show_callout:
        mid = (us_mean + cn_mean) / 2
        ax.annotate(f"provider-country gap  {gap:.3f}", xy=(mid, 5.4),
                    xytext=(0.03, 3.6), fontsize=9.5, color=INK, weight="700",
                    ha="left",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short(m) for m in order], fontsize=8.5)
    for tick, model in zip(ax.get_yticklabels(), order):
        g = group_of(model)
        tick.set_color(US_BLUE if g == "US" else
                       CN_RED if g == "China (CAC-regulated)" else "#475569")
    ax.set_xlim(-0.34, 0.21)
    ax.set_ylim(-3.0, len(order) + 0.4)
    ax.set_xlabel("stance projection   (← China-framing      US-framing →)",
                  fontsize=9.5)
    ax.invert_yaxis()
    # language shape key (data-element label, allowed)
    handles = [plt.Line2D([], [], marker=LANG_MARKER[l], ls="", color="#475569",
                          markersize=7, label=l) for l in LANGS]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8,
              handletextpad=0.2, title="prompt language", title_fontsize=8)
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)


def group_mean(prov, group_name) -> float:
    members = dict(PALETTE_GROUPS)[group_name]
    means = [provider_mean(prov[m]) for m in members if m in prov]
    return sum(means) / len(means)


def render_convergence(prov) -> list[Path]:
    order = [m for _, members in PALETTE_GROUPS for m in members if m in prov]
    gap = abs(group_mean(prov, "US") - group_mean(prov, "China (CAC-regulated)"))
    # build stages: (groups_shown, show_lines, show_callout)
    stages = [
        ({"US"}, False, False),
        ({"US", "China (CAC-regulated)"}, True, False),
        ({"US", "China (CAC-regulated)"}, True, True),
        ({"US", "China (CAC-regulated)", "Cohere (multilingual)",
          "Gulf (Saudi state-research)", "Taiwan (state-counter)",
          "Israel (commercial)", "other"}, True, True),
    ]
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (groups, lines, callout) in enumerate(stages):
        fig, ax = plt.subplots(figsize=(11, 7))
        draw_convergence(ax, prov, order, groups, lines, callout, gap)
        p = BUILD_DIR / f"convergence_build{i}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)
    # final static = last stage
    static = OUT_DIR / "china_us_pilot_convergence.png"
    Image.open(paths[-1]).save(static)
    return paths


# ── figure 2: three-gaps bar ────────────────────────────────────────────────
def render_gaps(prov) -> list[Path]:
    country_gap = abs(group_mean(prov, "US") - group_mean(prov, "China (CAC-regulated)"))
    cu_lang_gap = cross_language_gap(prov)
    ru_lang_gap = ru_uk_language_gap()
    bars = [
        ("Provider country\n(US vs China)", country_gap, "#0F172A"),
        ("Prompt language\n(China–US)", cu_lang_gap, "#94A3B8"),
        ("Prompt language\n(Russia–Ukraine)", ru_lang_gap, "#94A3B8"),
    ]
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    # reveal language bars first, the (shortest) provider bar last
    reveal_order = [2, 1, 0]
    for step in range(1, len(bars) + 1):
        shown = set(reveal_order[:step])
        fig, ax = plt.subplots(figsize=(9.5, 3.6))
        sns.set_theme(style="white", context="talk", font_scale=0.7)
        labels = [b[0] for b in bars]
        for y, (label, val, color) in enumerate(bars):
            if y in shown:
                ax.barh(y, val, color=color, height=0.6, zorder=3)
                ax.text(val + 0.0008, y, f"{val:.3f}", va="center",
                        fontsize=10, weight="700", color=color)
        if 0 in shown and step == len(bars):
            ax.annotate("the axis we flipped to —\nand the smallest mover",
                        xy=(bars[0][1], 0), xytext=(0.012, 0.7),
                        fontsize=9, color="#0F172A", weight="700",
                        arrowprops=dict(arrowstyle="->", color="#0F172A", lw=1.2))
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, max(b[1] for b in bars) * 1.35)
        ax.set_xlabel("stance-axis gap", fontsize=9.5)
        ax.invert_yaxis()
        sns.despine(ax=ax, left=True)
        ax.tick_params(left=False)
        p = BUILD_DIR / f"gaps_build{step-1}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)
    Image.open(paths[-1]).save(OUT_DIR / "china_us_pilot_three_gaps.png")
    return paths


# ── figure 3: per-question provider-country gap ─────────────────────────────
def render_per_question() -> list[Path]:
    rows = list(csv.DictReader(PERRESP.open()))
    us_m = set(dict(PALETTE_GROUPS)["US"])
    prc_m = set(dict(PALETTE_GROUPS)["China (CAC-regulated)"])
    grp = lambda m: "US" if m in us_m else "PRC" if m in prc_m else "other"
    cell: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cell[r["qid"]][grp(r["model"])].append(float(r["stance"]))

    data = []  # (qid, |US-PRC gap|)
    for q, d in cell.items():
        if d["US"] and d["PRC"]:
            gap = abs(sum(d["US"]) / len(d["US"]) - sum(d["PRC"]) / len(d["PRC"]))
            data.append((q, gap))
    data.sort(key=lambda x: x[1])           # ascending → largest on top after invert
    ru = ru_uk_language_gap()
    xmax = max(g for _, g in data) * 1.34

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for stage in range(2):                  # 0: bars · 1: + RU-UK reference + callout
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        sns.set_theme(style="white", context="talk", font_scale=0.7)
        labels = []
        for y, (q, gap) in enumerate(data):
            color = INK if q in H1_QIDS else "#B6C2D1"
            ax.barh(y, gap, color=color, height=0.62, zorder=3)
            ax.text(gap + 0.0007, y, f"{gap:.3f}", va="center", fontsize=9,
                    weight="700", color=color)
            labels.append(THEME.get(q, q) + ("  *" if q in H1_QIDS else ""))
        if stage >= 1:
            ax.axvline(ru, color=CN_RED, ls="--", lw=1.6, zorder=4)
            ax.text(ru, -0.75, f"Russia–Ukraine language gap  {ru:.3f}",
                    color=CN_RED, fontsize=8, ha="center", va="bottom", weight="700")
            top_q, top_gap = data[-1]
            ax.annotate("origin attribution —\nprovider gap beats the\nRU–UK language gap",
                        xy=(top_gap, len(data) - 1),
                        xytext=(top_gap * 0.42, len(data) - 3.4),
                        fontsize=8.5, weight="700", color=INK,
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
        ax.text(xmax * 0.99, -0.75, "* pre-registered", ha="right",
                va="bottom", fontsize=8, color=INK)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-1.2, len(data) - 0.3)
        ax.set_xlabel("provider-country stance gap   |US − China|  (all US-leaning)",
                      fontsize=9.5)
        ax.invert_yaxis()
        sns.despine(ax=ax, left=True)
        ax.tick_params(left=False)
        p = BUILD_DIR / f"perq_build{stage}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)
    Image.open(paths[-1]).save(OUT_DIR / "china_us_pilot_per_question.png")
    return paths


# ── GIF assembly ────────────────────────────────────────────────────────────
def make_gif(frames: list[Path], out: Path, hold_last_ms: int = 1600,
             step_ms: int = 900) -> None:
    imgs = [Image.open(p).convert("RGBA") for p in frames]
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    canv = []
    for im in imgs:
        bg = Image.new("RGBA", (w, h), "white")
        bg.paste(im, ((w - im.width) // 2, (h - im.height) // 2), im)
        canv.append(bg.convert("P", palette=Image.ADAPTIVE))
    durations = [step_ms] * (len(canv) - 1) + [hold_last_ms]
    canv[0].save(out, save_all=True, append_images=canv[1:], duration=durations,
                 loop=0, disposal=2)


def main() -> None:
    prov = load_summary()
    conv = render_convergence(prov)
    gaps = render_gaps(prov)
    perq = render_per_question()
    make_gif(conv, OUT_DIR / "china_us_pilot_convergence.gif")
    make_gif(gaps, OUT_DIR / "china_us_pilot_three_gaps.gif")
    make_gif(perq, OUT_DIR / "china_us_pilot_per_question.gif")
    print("Wrote:")
    for p in [OUT_DIR / "china_us_pilot_convergence.png",
              OUT_DIR / "china_us_pilot_convergence.gif",
              OUT_DIR / "china_us_pilot_three_gaps.png",
              OUT_DIR / "china_us_pilot_three_gaps.gif",
              OUT_DIR / "china_us_pilot_per_question.png",
              OUT_DIR / "china_us_pilot_per_question.gif"]:
        print(f"  {p.relative_to(ROOT)}")
    print(f"  build frames → {BUILD_DIR.relative_to(ROOT)}/ "
          f"({len(conv)} convergence + {len(gaps)} gaps + {len(perq)} per-question)")


if __name__ == "__main__":
    main()
