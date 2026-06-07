"""Hero figure v2-A — Provider Constellation Map (with regional insets).

A world basemap with one HQ pin per city, plus two zoom-inset panels
(SF Bay Area, Beijing/Hangzhou corridor) that carry the dense clusters
of US and Chinese providers as full-size compass glyphs.  Each glyph
is a disk-with-logo at its centre and a 5-spoke radial compass; spoke
length = EN-ingroup-pull magnitude for that conflict.

YandexGPT — which refused 100% of RU-UK and was graded across the other
conflicts in exp_022 — is rendered as a refusal-X glyph at Moscow rather
than a compass.

Per BabelBias figure rule: no in-image titles or footers.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, ConnectionPatch, Rectangle
from matplotlib.lines import Line2D

from babelbias.hero_assets import (
    CONFLICTS,
    ECOSYSTEM_COLOR,
    PROVIDERS,
    YANDEX_REFUSAL,
    load_logo,
)

ROOT = Path(__file__).resolve().parents[2]
EXP006_CSV = ROOT / "data" / "Russia-Ukraine" / "analysis" / "exp_006_topic_vs_language.csv"
GEOJSON = ROOT / "assets" / "geo" / "countries.geojson"
OUT_DIR = ROOT / "Presentations" / "figures" / "hero"

# Spoke angles for the 5 conflicts: top of clock (90°) → clockwise.
SPOKE_ANGLES_DEG = [90, 18, -54, -126, 162]


def load_ingroup_pull() -> dict[tuple[str, str], float]:
    df = pd.read_csv(EXP006_CSV)
    out: dict[tuple[str, str], float] = {}
    for ev_key, _, native in CONFLICTS:
        sub = df[df["event"] == ev_key]
        for prov in PROVIDERS:
            if prov.key == "yandexgpt":
                continue
            cell = sub[sub["model"] == prov.key]
            en = cell[cell["lang"] == "en"]["topic_lift"]
            nat = cell[cell["lang"].isin(native)]["topic_lift"]
            if en.empty or nat.empty:
                continue
            out[(prov.key, ev_key)] = float(en.mean() - nat.mean())
    return out


def draw_compass_glyph(ax: plt.Axes, cx: float, cy: float,
                       prov, pulls: dict[tuple[str, str], float],
                       disk_r: float, spoke_max: float,
                       logo_zoom: float,
                       is_yandex: bool = False) -> None:
    """Draw a single compass-glyph at (cx, cy) in axis coords."""
    eco_color = ECOSYSTEM_COLOR[prov.ecosystem]

    if not is_yandex:
        for ci, (event_key, _, _) in enumerate(CONFLICTS):
            v = pulls.get((prov.key, event_key))
            if v is None or not np.isfinite(v):
                continue
            v = max(0.0, v)
            length = disk_r + 0.15 * spoke_max + (v / 0.40) * spoke_max
            theta = np.deg2rad(SPOKE_ANGLES_DEG[ci])
            x0 = cx + (disk_r + 0.15 * spoke_max) * np.cos(theta)
            y0 = cy + (disk_r + 0.15 * spoke_max) * np.sin(theta)
            x1 = cx + length * np.cos(theta)
            y1 = cy + length * np.sin(theta)
            ax.plot([x0, x1], [y0, y1],
                    color=eco_color, linewidth=2.0,
                    solid_capstyle="round", alpha=0.92, zorder=5)
            ax.scatter(x1, y1, s=9, color=eco_color,
                       edgecolor="white", linewidth=0.4, zorder=6)

    # Disk under the logo
    ax.add_patch(Circle((cx, cy), disk_r,
                        facecolor="white",
                        edgecolor=eco_color, linewidth=1.6,
                        zorder=7))

    if is_yandex:
        s = disk_r * 0.75
        ax.plot([cx - s, cx + s], [cy - s, cy + s],
                color="#A50026", linewidth=2.6, zorder=9)
        ax.plot([cx - s, cx + s], [cy + s, cy - s],
                color="#A50026", linewidth=2.6, zorder=9)
    else:
        try:
            img = load_logo(prov.logo, size_px=140)
            oi = OffsetImage(np.asarray(img), zoom=logo_zoom)
            ab = AnnotationBbox(oi, (cx, cy),
                                frameon=False, pad=0, zorder=8)
            ax.add_artist(ab)
        except Exception:
            ax.text(cx, cy, prov.display.split()[0],
                    ha="center", va="center", fontsize=6.5, zorder=8)

    # Display name above the compass
    label_y = cy + disk_r + 1.10 * spoke_max
    ax.text(cx, label_y, prov.display,
            ha="center", va="bottom",
            fontsize=8.4, fontweight="bold", color=eco_color, zorder=10)


def draw_world(ax: plt.Axes, pulls: dict[tuple[str, str], float]) -> None:
    """World basemap with country tinting and single-provider HQ glyphs.

    Dense clusters (SF Bay, Beijing/Hangzhou) are NOT drawn here —
    just a single ecosystem-coloured pin per such city.  Insets handle
    the detail.
    """
    world = gpd.read_file(GEOJSON)
    world = world[world["ADMIN"] != "Antarctica"]
    world.plot(ax=ax, color="#f1ede4",
               edgecolor="#c8c2b1", linewidth=0.4, zorder=1)

    iso_to_eco: dict[str, str] = {}
    for p in PROVIDERS:
        iso_to_eco.setdefault(p.country_iso_a3, p.ecosystem)
    iso_col = "ADM0_A3"
    for iso, eco in iso_to_eco.items():
        m = world[world[iso_col] == iso]
        if not m.empty:
            m.plot(ax=ax, color=ECOSYSTEM_COLOR[eco], alpha=0.13,
                   edgecolor=ECOSYSTEM_COLOR[eco], linewidth=0.5, zorder=2)

    # Sets of providers grouped by HQ city.
    grouped: dict[tuple[float, float], list] = {}
    for p in PROVIDERS:
        grouped.setdefault((p.hq_lon, p.hq_lat), []).append(p)

    # On the world map we only render single-provider HQs (and Yandex);
    # multi-provider HQs get a single ecosystem-coloured pin with leader
    # lines drawn later to the insets.
    # Combine all China providers into one Beijing-area pin since
    # Beijing and Hangzhou are co-located at world-map scale.
    china_provs = [p for p in PROVIDERS if p.ecosystem == "China"]
    if china_provs:
        # Use Beijing coords as the single pin position.
        ax.add_patch(Circle((116.4, 39.9), 5.2,
                            facecolor=ECOSYSTEM_COLOR["China"],
                            edgecolor="white", linewidth=1.0, zorder=4))
        ax.text(116.4, 39.9, str(len(china_provs)),
                ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=5)
        ax.text(116.4, 33, "Beijing + Hangzhou\n→ see inset",
                ha="center", va="top",
                fontsize=8.6, color=ECOSYSTEM_COLOR["China"],
                fontweight="bold", style="italic", zorder=5)

    for (lon, lat), provs in grouped.items():
        if any(p.ecosystem == "China" for p in provs):
            continue  # already handled above
        if len(provs) > 1:
            eco = provs[0].ecosystem
            n_provs = len(provs)
            ax.add_patch(Circle((lon, lat), 5.2,
                                facecolor=ECOSYSTEM_COLOR[eco],
                                edgecolor="white", linewidth=1.0, zorder=4))
            ax.text(lon, lat, str(n_provs),
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white", zorder=5)
            ax.text(lon, lat - 7,
                    f"{provs[0].hq_city}\n→ see inset",
                    ha="center", va="top",
                    fontsize=8.6, color=ECOSYSTEM_COLOR[eco],
                    fontweight="bold", style="italic", zorder=5)
            continue
        # Single provider — full compass glyph
        p = provs[0]
        draw_compass_glyph(ax, lon, lat, p, pulls,
                           disk_r=3.2, spoke_max=8.0, logo_zoom=0.16,
                           is_yandex=(p.key == "yandexgpt"))
        if p.key == "yandexgpt":
            text = " / ".join(
                f"{lbl} {YANDEX_REFUSAL[k]}%"
                for (k, lbl, _) in CONFLICTS
            )
            ax.text(lon + 14, lat + 4, f"refused: {text}",
                    fontsize=7.6, color="#A50026",
                    ha="left", va="center", zorder=10)

    ax.set_xlim(-178, 200)
    ax.set_ylim(-58, 84)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_inset(inset_ax: plt.Axes, providers, pulls,
               title: str, title_color: str) -> None:
    """Lay out a cluster of providers in a compact regional inset."""
    inset_ax.set_xlim(0, 10)
    inset_ax.set_ylim(0, 10)
    inset_ax.set_aspect("equal")

    # Background card
    inset_ax.add_patch(Rectangle(
        (0.2, 0.2), 9.6, 9.6,
        facecolor="#fafaf6", edgecolor=title_color, linewidth=1.4,
        zorder=1,
    ))
    inset_ax.text(0.5, 9.4, title,
                  ha="left", va="top", fontsize=11, fontweight="bold",
                  color=title_color, zorder=2)

    # Arrange providers in a row (or grid).  Pick column count by
    # roster size to fill the inset gracefully.
    n = len(providers)
    if n <= 2:
        cols = n
    elif n <= 4:
        cols = 2
    else:
        cols = 3
    rows = (n + cols - 1) // cols
    cell_w = 9.0 / cols
    cell_h = 8.4 / rows

    disk_r = 0.55 if rows == 1 else 0.48
    spoke_max = 1.4 if rows == 1 else 1.1
    logo_zoom = 0.18 if rows == 1 else 0.15

    for i, p in enumerate(providers):
        r = i // cols
        c = i % cols
        cx = 0.5 + cell_w * (c + 0.5)
        cy = 8.6 - cell_h * (r + 0.5)
        draw_compass_glyph(inset_ax, cx, cy, p, pulls,
                           disk_r=disk_r, spoke_max=spoke_max,
                           logo_zoom=logo_zoom)

    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for s in inset_ax.spines.values():
        s.set_visible(False)


def render(out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pulls = load_ingroup_pull()

    # Identify dense clusters (>=2 providers at same HQ city).
    by_hq: dict[tuple[float, float], list] = {}
    for p in PROVIDERS:
        by_hq.setdefault((p.hq_lon, p.hq_lat), []).append(p)

    # Build dense-cluster insets ordered by ecosystem.  We bundle co-located
    # ecosystems together (SF + Mountain View + Palo Alto for US; Beijing +
    # Hangzhou for China).
    us_inset = [p for p in PROVIDERS if p.ecosystem == "US"]
    cn_inset = [p for p in PROVIDERS if p.ecosystem == "China"]
    ca_inset = [p for p in PROVIDERS if p.ecosystem == "Cohere"]

    # Figure layout: world map fills bottom 60%; insets sit above as 3
    # panels (US, China, Cohere) plus a legend strip.
    fig = plt.figure(figsize=(18, 13), dpi=160)
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.05, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.05, wspace=0.08,
        left=0.02, right=0.98, top=0.98, bottom=0.02,
    )

    ax_us = fig.add_subplot(gs[0, 0])
    ax_cn = fig.add_subplot(gs[0, 1])
    ax_ca = fig.add_subplot(gs[0, 2])
    ax_world = fig.add_subplot(gs[1, :])

    draw_inset(ax_us, us_inset, pulls,
               "United States — San Francisco Bay Area",
               ECOSYSTEM_COLOR["US"])
    draw_inset(ax_cn, cn_inset, pulls,
               "China — Beijing & Hangzhou (CAC-regulated)",
               ECOSYSTEM_COLOR["China"])
    draw_inset(ax_ca, ca_inset, pulls,
               "Canada — Cohere (multilingual)",
               ECOSYSTEM_COLOR["Cohere"])

    draw_world(ax_world, pulls)

    # Compass-key legend in the lower-left of the world panel.
    key_cx, key_cy = -150, -38
    key_r_world = 6.5
    for ci, (_, conflict_label, _) in enumerate(CONFLICTS):
        theta = np.deg2rad(SPOKE_ANGLES_DEG[ci])
        x0 = key_cx + key_r_world * 0.4 * np.cos(theta)
        y0 = key_cy + key_r_world * 0.4 * np.sin(theta)
        x1 = key_cx + key_r_world * 1.5 * np.cos(theta)
        y1 = key_cy + key_r_world * 1.5 * np.sin(theta)
        ax_world.plot([x0, x1], [y0, y1],
                      color="#444", linewidth=1.6, zorder=11)
        ax_world.text(
            key_cx + key_r_world * 2.0 * np.cos(theta),
            key_cy + key_r_world * 2.0 * np.sin(theta),
            conflict_label,
            ha="center", va="center", fontsize=8, color="#222", zorder=11,
        )
    ax_world.add_patch(Circle((key_cx, key_cy), key_r_world * 0.4,
                              facecolor="white", edgecolor="#444",
                              linewidth=1.2, zorder=11))
    ax_world.text(key_cx, key_cy - key_r_world * 2.6,
                  "Each spoke = one conflict; spoke length scales with EN ingroup pull",
                  ha="center", va="top", fontsize=7.6, color="#444", zorder=11)

    # Ecosystem legend top-right of world panel.
    leg_x = 195
    leg_y0 = 70
    ax_world.text(leg_x, leg_y0 + 6, "Training-data ecosystem",
                  ha="right", va="bottom", fontsize=9, fontweight="bold",
                  color="#222", zorder=11)
    for i, (eco, hexcol) in enumerate(ECOSYSTEM_COLOR.items()):
        y = leg_y0 - i * 7
        ax_world.add_patch(Circle((leg_x - 4, y), 1.6,
                                   facecolor=hexcol,
                                   edgecolor="white", linewidth=0.4, zorder=11))
        ax_world.text(leg_x - 8, y, eco, ha="right", va="center",
                      fontsize=8.5, color="#222", zorder=11)

    base = out_dir / "fig_v2a_constellation_map"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base.with_suffix(".png")


if __name__ == "__main__":
    print(f"wrote {render()}")
