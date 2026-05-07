"""Render the BabelBias methodology pipeline diagram for the thesis
Methods chapter (figure 3.1, fig:pipeline_diagram).

Output:
  Presentations/figures/methodology/pipeline_diagram.png

Design: horizontal flow, NN-architecture aesthetic.

  - Inputs as simple labeled rectangles (data icons).
  - Pre-processing as small grey boxes.
  - Embedding step shown as four parallel transformer-style stacks
    (the OpenAI primary embedder and three alt-embedders), with the
    multi-head sub-block visible inside each stack.
  - Cosine pivot as a literal 3x3 grid with the diagonal coloured
    so the ingroup-pull cells are immediately legible.
  - Analysis branches as compact symbolic tags (delta, debiased
    delta, cluster glyph, scatter glyph, dotted box for the
    falsification check) — text kept short enough to fit.

All labels short. Long captions live in the LaTeX caption, not in
the figure itself.
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


LOGOS = Path(__file__).resolve().parents[2] / "assets" / "model_logos"


def load_logo(name: str, target_area: int = 6000) -> np.ndarray:
    """Load a logo from assets/model_logos/<name> and rescale so its
    pixel-area is roughly `target_area`. Wordmarks and squarish marks
    end up similar visual weight."""
    path = LOGOS / name
    if path.suffix.lower() == ".svg":
        png_bytes = cairosvg.svg2png(url=str(path), output_width=240)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    else:
        img = Image.open(path).convert("RGBA")
    w, h = img.size
    aspect = w / h
    new_h = max(10, int(round((target_area / aspect) ** 0.5)))
    new_w = max(10, int(round(new_h * aspect)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(img)


def place_logo(ax, name: str, xy: tuple[float, float], zoom: float = 0.40) -> None:
    arr = load_logo(name)
    oi = OffsetImage(arr, zoom=zoom)
    ab = AnnotationBbox(oi, xy, frameon=False, pad=0, zorder=5)
    ax.add_artist(ab)

OUT = (Path(__file__).resolve().parents[2]
       / "Presentations" / "figures" / "methodology" / "pipeline_diagram.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


# ── Palette ──────────────────────────────────────────────────────────────
COL_INPUT       = "#E0F2FE"; COL_INPUT_E       = "#0284C7"
COL_PROCESS     = "#F1F5F9"; COL_PROCESS_E     = "#475569"
COL_EMBED       = "#FEF3C7"; COL_EMBED_E       = "#D97706"
COL_EMBED_INNER = "#FDE68A"
COL_PIVOT       = "#FCE7F3"; COL_PIVOT_E       = "#BE185D"
COL_PIVOT_DIAG  = "#BE185D"
COL_OUT_RAW     = "#1F4E79"
COL_OUT_DEBIAS  = "#A50026"
COL_OUT_EVOC    = "#1B9E77"
COL_OUT_TOPIC   = "#984EA3"
COL_OUT_TOK     = "#475569"


def box(ax, x, y, w, h, text, fc, ec, fs=10.0, weight="600"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs, color="#0F172A",
            weight=weight, zorder=4, multialignment="center")
    return x + w / 2, y + h / 2


def arrow(ax, src, dst, color="#475569", lw=1.4, style="-|>",
          connectionstyle="arc3,rad=0", linestyle="-", alpha=1.0):
    a = FancyArrowPatch(
        src, dst,
        arrowstyle=style, mutation_scale=14,
        color=color, linewidth=lw, zorder=2,
        connectionstyle=connectionstyle,
        linestyle=linestyle, alpha=alpha,
        shrinkA=4, shrinkB=4,
    )
    ax.add_patch(a)


def transformer_stack(ax, x, y, w, h, label, n_layers=3,
                      face=COL_EMBED, edge=COL_EMBED_E,
                      inner=COL_EMBED_INNER):
    """Draw a transformer-style stack: outer rounded box + n_layers
    inner sub-blocks each split into Attn / FFN halves, with the
    embedder label below the stack."""
    # Outer container
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=face, edgecolor=edge, linewidth=1.4, zorder=3,
    )
    ax.add_patch(p)
    # Layer sub-blocks (stacked vertically inside the container)
    inner_pad_x = 0.5
    inner_pad_y = 0.5
    inner_w = w - 2 * inner_pad_x
    layer_h = (h - 2 * inner_pad_y) / n_layers - 0.25
    for i in range(n_layers):
        ly = y + inner_pad_y + i * (layer_h + 0.25)
        # Attention half
        ax.add_patch(Rectangle(
            (x + inner_pad_x, ly), inner_w * 0.55, layer_h,
            facecolor=inner, edgecolor=edge, linewidth=0.6, zorder=4,
        ))
        # FFN half
        ax.add_patch(Rectangle(
            (x + inner_pad_x + inner_w * 0.55, ly),
            inner_w * 0.45, layer_h,
            facecolor="white", edgecolor=edge, linewidth=0.6, zorder=4,
        ))
    # Label inside top of container
    ax.text(x + w / 2, y + h + 0.7, label,
            ha="center", va="bottom", fontsize=9, color=edge,
            weight="700", zorder=5)


def cosine_grid(ax, x, y, side, langs=("en", "ru", "uk")):
    """3×3 cosine-matrix glyph. Diagonal cells are filled, off-diagonal
    are pale. Returns the centre point."""
    cell = side / 3
    for i in range(3):
        for j in range(3):
            on_diag = (i == j)
            ax.add_patch(Rectangle(
                (x + j * cell, y + (2 - i) * cell), cell, cell,
                facecolor=COL_PIVOT_DIAG if on_diag else "white",
                edgecolor=COL_PIVOT_E, linewidth=0.9, zorder=4,
            ))
    # Outer rounded frame
    ax.add_patch(FancyBboxPatch(
        (x - 0.2, y - 0.2), side + 0.4, side + 0.4,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="none", edgecolor=COL_PIVOT_E, linewidth=1.4, zorder=3,
    ))
    return x + side / 2, y + side / 2


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_aspect("auto")
    ax.set_axis_off()

    # ── Inputs ─────────────────────────────────────────────────────────
    # Compact input boxes; small logo on top, label below.
    inp_w, inp_h = 14, 8
    inp_x = 2.5

    def logo_box(yc: float, label: str, logo_name: str | None,
                 zoom: float = 0.20) -> tuple[float, float]:
        y = yc - inp_h / 2
        ax.add_patch(FancyBboxPatch(
            (inp_x, y), inp_w, inp_h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=COL_INPUT, edgecolor=COL_INPUT_E, linewidth=1.4,
            zorder=3,
        ))
        if logo_name is not None:
            place_logo(ax, logo_name,
                       (inp_x + inp_w / 2, yc + 1.4), zoom=zoom)
        ax.text(inp_x + inp_w / 2, yc - 2.2, label,
                ha="center", va="center", fontsize=10, color="#0F172A",
                weight="600", zorder=4, multialignment="center")
        return inp_x + inp_w, yc

    in1 = logo_box(46.5, "Wikipedia\nanchors",  "wikipedia.png", zoom=0.30)
    in2 = logo_box(35.5, "Universal\ncontrols", "wikipedia.png", zoom=0.22)
    # 9-question bank: text-only with a "?" emblem in the input colour
    in3_centre = 24.5
    ax.add_patch(FancyBboxPatch(
        (inp_x, in3_centre - inp_h / 2), inp_w, inp_h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=COL_INPUT, edgecolor=COL_INPUT_E, linewidth=1.4, zorder=3,
    ))
    ax.text(inp_x + inp_w / 2, in3_centre + 1.4, "?",
            ha="center", va="center", fontsize=22, color=COL_INPUT_E,
            weight="800", zorder=4)
    ax.text(inp_x + inp_w / 2, in3_centre - 2.2, "9-question\nbank",
            ha="center", va="center", fontsize=10, color="#0F172A",
            weight="600", zorder=4, multialignment="center")
    in3 = (inp_x + inp_w, in3_centre)

    # 14-LLM box: 2×2 cluster of representative provider logos
    in4_centre = 13.5
    ax.add_patch(FancyBboxPatch(
        (inp_x, in4_centre - inp_h / 2), inp_w, inp_h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=COL_INPUT, edgecolor=COL_INPUT_E, linewidth=1.4, zorder=3,
    ))
    # Three representative logos in a horizontal row (one per ecosystem
    # the deck cares about: US/Anthropic, China/Qwen, Russia/Yandex).
    cluster_logos = [("anthropic.svg", 0.18),
                     ("qwen.png",      0.10),
                     ("yandex.svg",    0.18)]
    cluster_xs = [inp_x + inp_w * f for f in (0.22, 0.50, 0.78)]
    for (logo, zoom), x in zip(cluster_logos, cluster_xs):
        place_logo(ax, logo, (x, in4_centre + 1.4), zoom=zoom)
    ax.text(inp_x + inp_w / 2, in4_centre - 2.2, "14 LLMs",
            ha="center", va="center", fontsize=10, color="#0F172A",
            weight="600", zorder=4)
    in4 = (inp_x + inp_w, in4_centre)

    # ── Pre-process ────────────────────────────────────────────────────
    proc_w, proc_h = 11, 4
    proc_x = 21
    p1     = box(ax, proc_x, 46.5 - proc_h / 2, proc_w, proc_h, "Leads",
                 COL_PROCESS, COL_PROCESS_E, fs=10.5)
    p_ctrl = box(ax, proc_x, 35.5 - proc_h / 2, proc_w, proc_h, "Filter",
                 COL_PROCESS, COL_PROCESS_E, fs=10.5)
    # LLM sweep sits between in3 (24.5) and in4 (13.5), centred ~19
    p_sweep= box(ax, proc_x, 19.0 - proc_h / 2, proc_w, proc_h, "LLM sweep",
                 COL_PROCESS, COL_PROCESS_E, fs=10.5)

    # ── Embedding stack: 4 transformer-style stacks side by side ───────
    emb_x0 = 38; stack_w = 4.6; stack_gap = 1.0; stack_h = 22
    stack_y = 22
    stacks = [
        ("OpenAI\nte3s", "#FEF3C7", "#D97706", "#FDE68A"),
        ("Alibaba\nv3",  "#FEE2E2", "#B91C1C", "#FECACA"),
        ("Gemini\n001",  "#DBEAFE", "#1D4ED8", "#BFDBFE"),
        ("Yandex\nsearch","#EDE9FE","#6D28D9", "#DDD6FE"),
    ]
    for i, (label, face, edge, inner) in enumerate(stacks):
        sx = emb_x0 + i * (stack_w + stack_gap)
        transformer_stack(ax, sx, stack_y, stack_w, stack_h, label,
                          n_layers=4, face=face, edge=edge, inner=inner)
    emb_band_centre_x = emb_x0 + 2 * (stack_w + stack_gap) - stack_gap / 2
    emb_band_left  = emb_x0 - 1
    emb_band_right = emb_x0 + 4 * (stack_w + stack_gap) - stack_gap

    # ── Cosine pivot: literal 3×3 grid ─────────────────────────────────
    cos_side = 8
    cos_x = 70
    cos_y = stack_y + (stack_h - cos_side) / 2
    cos_centre = cosine_grid(ax, cos_x, cos_y, cos_side)
    # Axis labels around the grid
    ax.text(cos_x + cos_side / 2, cos_y - 1.6, "anchor lang",
            ha="center", va="top", fontsize=9, color=COL_PIVOT_E,
            weight="600")
    ax.text(cos_x - 1.6, cos_y + cos_side / 2, "response lang",
            ha="center", va="center", fontsize=9, color=COL_PIVOT_E,
            weight="600", rotation=90)

    # ── Analysis outputs (compact symbolic tags) ───────────────────────
    out_w, out_h = 12.5, 4.8
    out_x = 84.5
    o_raw   = box(ax, out_x, 49,    out_w, out_h, r"$\Delta_{\mathrm{in}}$  raw",
                  "white", COL_OUT_RAW,    fs=12)
    o_deb   = box(ax, out_x, 41.5,  out_w, out_h, r"$\Delta_{\mathrm{in}}$  debiased",
                  "white", COL_OUT_DEBIAS, fs=12)
    o_evoc  = box(ax, out_x, 34,    out_w, out_h, "EVoC clusters",
                  "white", COL_OUT_EVOC,   fs=11)
    o_topic = box(ax, out_x, 26.5,  out_w, out_h, "Topic lift",
                  "white", COL_OUT_TOPIC,  fs=11)
    o_tok   = box(ax, out_x, 19,    out_w, out_h, "Tokenizer\ncontrol",
                  "white", COL_OUT_TOK,    fs=10.5)

    # ── Arrows: inputs → pre-process ───────────────────────────────────
    arrow(ax, in1, p1)
    arrow(ax, in2, p_ctrl)
    arrow(ax, in3, p_sweep, connectionstyle="arc3,rad=-0.05")
    arrow(ax, in4, p_sweep, connectionstyle="arc3,rad=0.05")

    # ── pre-process → embedding band ──────────────────────────────────
    band_top    = (emb_band_left, stack_y + stack_h * 0.85)
    band_mid    = (emb_band_left, stack_y + stack_h * 0.50)
    band_bot    = (emb_band_left, stack_y + stack_h * 0.15)
    arrow(ax, p1,     band_top,                      connectionstyle="arc3,rad=0")
    arrow(ax, p_ctrl, band_mid,                      connectionstyle="arc3,rad=0")
    arrow(ax, p_sweep,band_bot,                      connectionstyle="arc3,rad=0.04")

    # ── embedding band → cosine pivot ──────────────────────────────────
    arrow(ax, (emb_band_right, stack_y + stack_h / 2),
          (cos_x - 0.4, cos_centre[1]),
          color=COL_EMBED_E, lw=1.8)

    # ── cosine pivot → analyses (fan out) ──────────────────────────────
    cos_right = (cos_x + cos_side + 0.4, cos_centre[1])
    arrow(ax, cos_right, o_raw,   color=COL_OUT_RAW,    lw=1.8,
          connectionstyle="arc3,rad=-0.18")
    arrow(ax, cos_right, o_deb,   color=COL_OUT_DEBIAS, lw=1.8,
          connectionstyle="arc3,rad=-0.06")
    arrow(ax, cos_right, o_evoc,  color=COL_OUT_EVOC,   lw=1.8,
          connectionstyle="arc3,rad=0.0")
    arrow(ax, cos_right, o_topic, color=COL_OUT_TOPIC,  lw=1.8,
          connectionstyle="arc3,rad=0.10")
    arrow(ax, cos_right, o_tok,   color=COL_OUT_TOK,    lw=1.8,
          connectionstyle="arc3,rad=0.18")

    # ── skip connections: controls → debiased + topic-lift ────────────
    # Routed as an "underbus" beneath the main flow so the line never
    # crosses the embedding stacks, the cosine grid, or any text.
    bus_y = 2.6
    bus_x_left  = proc_x + proc_w / 2
    bus_x_right = out_x - 2.0

    # Drop down from filter-box bottom (no arrowhead — it's a feeder).
    ax.plot([bus_x_left, bus_x_left], [33.5, bus_y],
            color="#94A3B8", lw=1.0, linestyle=(0, (5, 3)),
            alpha=0.85, zorder=1)
    # Horizontal underbus.
    ax.plot([bus_x_left, bus_x_right], [bus_y, bus_y],
            color="#94A3B8", lw=1.0, linestyle=(0, (5, 3)),
            alpha=0.85, zorder=1)
    # Two L-shaped risers with arrowheads. Each arrow uses the
    # angle-style connection so the corner is sharp and the head sits
    # right at the destination box edge.
    arrow(ax, (bus_x_right, bus_y), (out_x, 44.0),
          color=COL_OUT_DEBIAS, lw=1.0, linestyle=(0, (5, 3)),
          alpha=0.85,
          connectionstyle="angle,angleA=90,angleB=180,rad=0")
    arrow(ax, (bus_x_right, bus_y), (out_x, 28.9),
          color=COL_OUT_TOPIC, lw=1.0, linestyle=(0, (5, 3)),
          alpha=0.85,
          connectionstyle="angle,angleA=90,angleB=180,rad=0")

    # ── Band headers ──────────────────────────────────────────────────
    band_y = 58
    ax.text(10, band_y, "INPUTS",      fontsize=10.5, color="#475569",
            weight="700", ha="center")
    ax.text(28, band_y, "PRE-PROCESS", fontsize=10.5, color="#475569",
            weight="700", ha="center")
    ax.text((emb_band_left + emb_band_right) / 2, band_y, "EMBEDDING",
            fontsize=10.5, color="#475569", weight="700", ha="center")
    ax.text(cos_x + cos_side / 2, band_y, "PIVOT",
            fontsize=10.5, color="#475569", weight="700", ha="center")
    ax.text(out_x + out_w / 2, band_y, "ANALYSIS",
            fontsize=10.5, color="#475569", weight="700", ha="center")

    # Band separators
    for x in (19, 35, 67.5, 82.5):
        ax.axvline(x, ymin=0.05, ymax=0.92,
                   color="#E2E8F0", linewidth=0.8, zorder=0)

    # ── Legend ────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color="#475569", linewidth=1.6,
               label="primary data flow"),
        Line2D([0], [0], color=COL_OUT_DEBIAS, linewidth=1.0,
               linestyle=(0, (5, 3)),
               label="control-corpus skip → debiasing / topic lift"),
    ]
    leg = ax.legend(handles=legend_handles, loc="lower center",
                    bbox_to_anchor=(0.5, -0.02),
                    frameon=False, ncol=2, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color("#334155")

    plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
