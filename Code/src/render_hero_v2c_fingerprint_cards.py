"""Hero figure v2-C — 15-Card Bias Fingerprint grid.

A trading-card layout of all 15 providers grouped by training-data
ecosystem.  Each card carries:

- Ecosystem-colour flag stripe at the top
- Model logo
- Display name + parameter-size badge
- EN-ingroup-pull sparkline across the 5 conflicts (one bar per conflict)
- Refusal-mode indicator (Yandex only — graded refusal staircase)

Rows correspond to ecosystems; horizontal scan within a row reveals
the bias fingerprint shared by providers in the same ecosystem.

Per BabelBias figure rule: no in-image titles or captions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Rectangle

from babelbias.hero_assets import (
    CONFLICTS,
    ECOSYSTEM_COLOR,
    PROVIDERS,
    YANDEX_REFUSAL,
    load_logo,
)

ROOT = Path(__file__).resolve().parents[2]
EXP006_CSV = ROOT / "data" / "Russia-Ukraine" / "analysis" / "exp_006_topic_vs_language.csv"
OUT_DIR = ROOT / "Presentations" / "figures" / "hero"

CARD_W = 2.4
CARD_H = 3.5
COL_GAP = 0.30
ROW_GAP = 0.45

PULL_AXIS_MAX = 0.40  # cap for the sparkline x-scale


def load_pulls() -> dict[tuple[str, str], float]:
    df = pd.read_csv(EXP006_CSV)
    out: dict[tuple[str, str], float] = {}
    for ev_key, _, native in CONFLICTS:
        sub = df[df["event"] == ev_key]
        for prov in PROVIDERS:
            cell = sub[sub["model"] == prov.key]
            en = cell[cell["lang"] == "en"]["topic_lift"]
            nat = cell[cell["lang"].isin(native)]["topic_lift"]
            if en.empty or nat.empty:
                continue
            out[(prov.key, ev_key)] = float(en.mean() - nat.mean())
    return out


def ecosystem_order() -> list[str]:
    seen = []
    for p in PROVIDERS:
        if p.ecosystem not in seen:
            seen.append(p.ecosystem)
    return seen


def providers_by_eco() -> dict[str, list]:
    out: dict[str, list] = {}
    for p in PROVIDERS:
        out.setdefault(p.ecosystem, []).append(p)
    return out


def draw_card(ax: plt.Axes, prov, pulls: dict[tuple[str, str], float],
              x: float, y: float) -> None:
    eco_color = ECOSYSTEM_COLOR[prov.ecosystem]

    # Card shadow
    ax.add_patch(Rectangle(
        (x + 0.03, y - 0.04),
        CARD_W, CARD_H,
        facecolor="#dddddd", edgecolor="none",
        zorder=1,
    ))
    # Card body
    ax.add_patch(FancyBboxPatch(
        (x, y), CARD_W, CARD_H,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor="white",
        edgecolor=eco_color,
        linewidth=1.2,
        zorder=2,
    ))
    # Flag stripe at top
    stripe_h = 0.30
    ax.add_patch(Rectangle(
        (x, y + CARD_H - stripe_h), CARD_W, stripe_h,
        facecolor=eco_color, edgecolor="none",
        zorder=3,
    ))
    ax.text(
        x + CARD_W / 2, y + CARD_H - stripe_h / 2,
        prov.ecosystem.upper(),
        ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="white",
        zorder=4,
    )

    # Logo region
    logo_cx = x + CARD_W / 2
    logo_cy = y + CARD_H - 0.95
    try:
        img = load_logo(prov.logo, size_px=200)
        oi = OffsetImage(np.asarray(img), zoom=0.32)
        ab = AnnotationBbox(oi, (logo_cx, logo_cy),
                            frameon=False, pad=0, zorder=5)
        ax.add_artist(ab)
    except Exception:
        ax.text(logo_cx, logo_cy, prov.display.split()[0],
                ha="center", va="center", fontsize=10, color="#444",
                zorder=5)

    # Display name
    ax.text(
        x + CARD_W / 2, y + CARD_H - 1.65,
        prov.display,
        ha="center", va="center",
        fontsize=9.6, fontweight="bold", color="#222",
        zorder=5,
    )

    # Param badge
    ax.text(
        x + CARD_W / 2, y + CARD_H - 1.92,
        prov.param_label,
        ha="center", va="center",
        fontsize=7.6, color="#666", style="italic",
        zorder=5,
    )

    # Sparkline: per-conflict EN ingroup pull (or graded Yandex refusal).
    spark_x0 = x + 0.18
    spark_x1 = x + CARD_W - 0.18
    spark_y0 = y + 0.32
    spark_y1 = y + 1.10
    spark_w = spark_x1 - spark_x0
    spark_h = spark_y1 - spark_y0
    ax.plot(
        [spark_x0, spark_x1], [spark_y0, spark_y0],
        color="#bbb", linewidth=0.7, zorder=4,
    )

    n = len(CONFLICTS)
    bar_w = spark_w / (n * 1.5)
    is_yandex = (prov.key == "yandexgpt")

    if is_yandex:
        # Render refusal staircase (red bars scaled to 100% = full height)
        for i, (ev_key, lbl, _) in enumerate(CONFLICTS):
            cx = spark_x0 + (i + 0.5) * spark_w / n
            pct = YANDEX_REFUSAL[ev_key]
            h = (pct / 100.0) * spark_h
            ax.add_patch(Rectangle(
                (cx - bar_w / 2, spark_y0), bar_w, h,
                facecolor="#A50026", edgecolor="white", linewidth=0.4,
                zorder=5,
            ))
            ax.text(
                cx, spark_y0 - 0.02, lbl,
                ha="center", va="top", fontsize=7.0, color="#666",
                zorder=5,
            )
            ax.text(
                cx, spark_y0 + h + 0.005, f"{pct}%",
                ha="center", va="bottom", fontsize=6.8, color="#A50026",
                zorder=5,
            )
        ax.text(
            x + CARD_W / 2, y + 1.20,
            "refusal rate", ha="center", va="bottom",
            fontsize=8.0, fontweight="bold", color="#A50026", zorder=5,
        )
    else:
        for i, (ev_key, lbl, _) in enumerate(CONFLICTS):
            v = pulls.get((prov.key, ev_key))
            if v is None or not np.isfinite(v):
                cx = spark_x0 + (i + 0.5) * spark_w / n
                ax.text(
                    cx, spark_y0 + spark_h / 2, "—",
                    ha="center", va="center", fontsize=8, color="#ccc",
                    zorder=5,
                )
                ax.text(
                    cx, spark_y0 - 0.02, lbl,
                    ha="center", va="top", fontsize=7.0, color="#bbb",
                    zorder=5,
                )
                continue
            v = max(0.0, v)
            cx = spark_x0 + (i + 0.5) * spark_w / n
            h = (v / PULL_AXIS_MAX) * spark_h
            ax.add_patch(Rectangle(
                (cx - bar_w / 2, spark_y0), bar_w, h,
                facecolor=eco_color, edgecolor="white", linewidth=0.4,
                zorder=5,
            ))
            ax.text(
                cx, spark_y0 - 0.02, lbl,
                ha="center", va="top", fontsize=7.0, color="#666",
                zorder=5,
            )
            ax.text(
                cx, spark_y0 + h + 0.005, f"{v:+.2f}",
                ha="center", va="bottom", fontsize=6.8, color="#222",
                zorder=5,
            )
        ax.text(
            x + CARD_W / 2, y + 1.20,
            "EN ingroup pull per conflict",
            ha="center", va="bottom",
            fontsize=8.0, color="#444", zorder=5,
        )


def render(out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pulls = load_pulls()

    eco_groups = providers_by_eco()
    eco_order = ecosystem_order()
    max_cols = max(len(g) for g in eco_groups.values())
    n_rows = len(eco_order)

    left_margin = 2.6  # for ecosystem labels
    fig_w_units = max_cols * (CARD_W + COL_GAP) + left_margin + 0.3
    fig_h_units = n_rows * (CARD_H + ROW_GAP) + 0.5
    # Render at 1:1 between axis units and inches for predictable sizing.
    fig, ax = plt.subplots(figsize=(fig_w_units, fig_h_units), dpi=160)
    ax.set_xlim(0, fig_w_units)
    ax.set_ylim(0, fig_h_units)
    ax.set_aspect("equal")
    ax.axis("off")

    for ri, eco in enumerate(eco_order):
        row_y = fig_h_units - 0.3 - (ri + 1) * (CARD_H + ROW_GAP) + ROW_GAP
        # Ecosystem colour bar
        ax.add_patch(Rectangle(
            (0.4, row_y), 0.12, CARD_H,
            facecolor=ECOSYSTEM_COLOR[eco], edgecolor="none",
        ))
        # Row label on the left
        ax.text(
            0.7, row_y + CARD_H / 2,
            eco,
            ha="left", va="center",
            fontsize=14, fontweight="bold",
            color=ECOSYSTEM_COLOR[eco],
        )

        for ci, prov in enumerate(eco_groups[eco]):
            col_x = left_margin + ci * (CARD_W + COL_GAP)
            draw_card(ax, prov, pulls, col_x, row_y)

    base = out_dir / "fig_v2c_fingerprint_cards"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base.with_suffix(".png")


if __name__ == "__main__":
    print(f"wrote {render()}")
