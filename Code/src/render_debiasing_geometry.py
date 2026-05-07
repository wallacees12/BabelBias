"""Render the language-axis debiasing geometric illustration for the
thesis Methods chapter (figure 3.6, fig:debiasing_geometry).

Output: Presentations/figures/methodology/debiasing_geometry.png

A 2-D schematic. Three clusters of control points (one per language)
define per-language centroids; the language axis is the line through
the centroids; one example response point is shown together with its
projection onto the orthogonal complement (the debiased coordinate).

The figure is illustrative — the data points are synthetic 2-D toys,
chosen so the geometric relationships read clearly. The actual
1{,}536-D pipeline does the same projection in a higher-dimensional
space.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from babelbias.paths import PROJECT_ROOT


sns.set_theme(
    style="white", context="talk", font_scale=0.78,
    rc={
        "savefig.dpi":  200,
        "savefig.bbox": "tight",
    },
)

OUT = (PROJECT_ROOT / "Presentations" / "figures" / "methodology"
       / "debiasing_geometry.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Language palette consistent with the rest of the deck
COL = {"en": "#2E75B6", "ru": "#A50026", "uk": "#FDAE61"}
COL_AXIS    = "#0F172A"
COL_RESP    = "#1B9E77"
COL_PROJ    = "#984EA3"
COL_BG_AXIS = "#FCE7F3"


def main() -> None:
    rng = np.random.default_rng(7)

    # Per-language centroids: roughly collinear so the language axis is
    # obvious in 2-D. Slight off-line jitter shows the centroids span a
    # 2-D subspace in this toy (in practice the language subspace is
    # high-dim).
    centroids = {
        "en": np.array([1.5,  0.6]),
        "ru": np.array([3.5,  3.4]),
        "uk": np.array([5.5,  6.1]),
    }

    # Sample control points around each centroid
    points = {}
    for lang, c in centroids.items():
        points[lang] = c + rng.normal(scale=[0.55, 0.55], size=(40, 2))

    # Language axis = line through the centroids (here approximately
    # the diagonal x≈y). In code we'd PCA the centred centroids; for
    # the schematic, plot the line through the means.
    cs = np.stack(list(centroids.values()))
    cmean = cs.mean(axis=0)
    centred = cs - cmean
    U, S, Vt = np.linalg.svd(centred, full_matrices=False)
    axis_dir = Vt[0]                         # principal language axis
    perp_dir = np.array([-axis_dir[1], axis_dir[0]])  # orthogonal complement

    # Example response point — sits off the axis, clearly visible
    resp = np.array([4.3, 1.6])

    # Project response onto the language axis and the orthogonal
    # complement (anchored at cmean).
    rel = resp - cmean
    along = rel @ axis_dir
    across = rel @ perp_dir
    proj_axis = cmean + along * axis_dir
    proj_perp = cmean + across * perp_dir
    debiased = resp - along * axis_dir   # the residual after removing the lang axis

    # ── Figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6.4), constrained_layout=True)

    # Soft band along the language axis to emphasise "this is the dim
    # that gets projected out"
    t = np.linspace(-6, 6, 200)
    band_pts = cmean[None, :] + t[:, None] * axis_dir[None, :]
    n_pts    = cmean[None, :] + t[:, None] * perp_dir[None, :]
    # Construct a thin rectangular band centred on the axis line:
    band_w = 0.55
    band_top    = band_pts + band_w * perp_dir
    band_bottom = band_pts - band_w * perp_dir
    band_poly = np.concatenate([band_top, band_bottom[::-1]], axis=0)
    ax.fill(band_poly[:, 0], band_poly[:, 1],
            color=COL_BG_AXIS, alpha=0.55, zorder=1, edgecolor="none")

    # Control points per language
    for lang, P in points.items():
        ax.scatter(P[:, 0], P[:, 1], s=42, color=COL[lang], alpha=0.55,
                    edgecolors="white", linewidths=0.6, zorder=3,
                    label=f"controls ({lang.upper()})")

    # Centroids (large markers)
    for lang, c in centroids.items():
        ax.scatter(*c, s=320, color=COL[lang], edgecolors="#0F172A",
                    linewidths=1.6, zorder=5, marker="X")
        ax.annotate(f"$\\bar{{c}}_{{{lang.upper()}}}$",
                     xy=c, xytext=(c[0] + 0.25, c[1] + 0.55),
                     fontsize=13, weight="700", color=COL[lang], zorder=6)

    # Language axis line (long)
    L = 4.5
    ax.plot([cmean[0] - L*axis_dir[0], cmean[0] + L*axis_dir[0]],
             [cmean[1] - L*axis_dir[1], cmean[1] + L*axis_dir[1]],
             color=COL_AXIS, linewidth=2.0, zorder=4)
    # Axis label
    label_pt = cmean + (L - 0.4) * axis_dir
    ax.text(label_pt[0] + 0.2, label_pt[1] - 0.4,
             r"language axis $\hat{\boldsymbol{\ell}}$",
             fontsize=12.5, weight="700", color=COL_AXIS, zorder=6)

    # Response point
    ax.scatter(*resp, s=260, color=COL_RESP, edgecolors="#0F172A",
                linewidths=1.6, zorder=6, marker="o")
    ax.annotate("response  $\\mathbf{r}$",
                 xy=resp, xytext=(resp[0] + 0.35, resp[1] - 0.30),
                 fontsize=12.5, weight="700", color=COL_RESP, zorder=6)

    # Project response onto axis (dashed)
    ax.plot([resp[0], proj_axis[0]], [resp[1], proj_axis[1]],
             color=COL_PROJ, linewidth=1.6, linestyle=(0, (4, 3)), zorder=5)
    ax.scatter(*proj_axis, s=120, color=COL_PROJ,
                edgecolors="#0F172A", linewidths=1.0, zorder=6, marker="s")
    ax.annotate("language component\n"
                 "$(\\mathbf{r} \\cdot \\hat{\\boldsymbol{\\ell}})\\,"
                 "\\hat{\\boldsymbol{\\ell}}$",
                 xy=proj_axis,
                 xytext=(proj_axis[0] - 4.5, proj_axis[1] - 1.2),
                 fontsize=11, color=COL_PROJ, weight="600",
                 multialignment="center",
                 arrowprops=dict(arrowstyle="->", color=COL_PROJ,
                                  lw=1.2, alpha=0.8), zorder=6)

    # Debiased / orthogonal residual
    ax.annotate("",
                 xy=debiased, xytext=resp,
                 arrowprops=dict(arrowstyle="->", color=COL_RESP,
                                  lw=2.0), zorder=6)
    ax.scatter(*debiased, s=200, color="white", edgecolors=COL_RESP,
                linewidths=2.0, zorder=6, marker="o")
    ax.annotate("debiased response\n"
                 "$\\mathbf{r}_{\\perp}$",
                 xy=debiased,
                 xytext=(debiased[0] - 4.0, debiased[1] + 1.0),
                 fontsize=11.5, color=COL_RESP, weight="700",
                 multialignment="center",
                 arrowprops=dict(arrowstyle="->", color=COL_RESP,
                                  lw=1.2), zorder=6)

    ax.set_xlim(-2, 9.5)
    ax.set_ylim(-1, 8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Header / title
    ax.set_title("Language-axis projection — geometry of debiasing",
                  fontsize=14, weight="bold", color="#0F172A", pad=12)

    # Legend (custom, only the control swatches and the axis)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="white",
                    markerfacecolor=COL["en"], markeredgecolor="white",
                    markersize=10, label="EN controls"),
        plt.Line2D([0], [0], marker="o", color="white",
                    markerfacecolor=COL["ru"], markeredgecolor="white",
                    markersize=10, label="RU controls"),
        plt.Line2D([0], [0], marker="o", color="white",
                    markerfacecolor=COL["uk"], markeredgecolor="white",
                    markersize=10, label="UK controls"),
        plt.Line2D([0], [0], marker="X", color="none",
                    markerfacecolor="#475569", markeredgecolor="#0F172A",
                    markersize=12, label="per-language centroid"),
        plt.Line2D([0], [0], color=COL_AXIS, linewidth=2.0,
                    label="language axis $\\hat{\\boldsymbol{\\ell}}$"),
        plt.Line2D([0], [0], color=COL_RESP, linewidth=2.0,
                    label="$\\mathbf{r} \\to \\mathbf{r}_{\\perp}$  "
                          "(projected residual)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
                frameon=True, fontsize=9.5, framealpha=0.95,
                edgecolor="#CBD5E1")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
