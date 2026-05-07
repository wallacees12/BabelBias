"""Render the language-axis debiasing geometric illustration in 3-D.

Output: Presentations/figures/methodology/debiasing_geometry.png

Minimal 3-D schematic:
  * a tilted plane = the language subspace
  * a vertical line perpendicular to the plane = the debiased
    direction (orthogonal complement)
  * a response point above the plane, decomposed via a perpendicular
    drop into a language component (on the plane) and a debiased
    residual (the height above the plane)

Language-specific labels are intentionally absent — the caption
identifies the subspace as the span of per-language centroids in the
RU--UK case and generalises to L languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from babelbias.paths import PROJECT_ROOT


OUT = (PROJECT_ROOT / "Presentations" / "figures" / "methodology"
       / "debiasing_geometry.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

COL_PLANE = "#94A3B8"
COL_PLANE_EDGE = "#475569"
COL_RESP  = "#1B9E77"
COL_PERP  = "#0F172A"


def main() -> None:
    # Plane spanned by two orthonormal in-plane vectors u, v.
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    n = np.cross(u, v)                 # plane normal
    n = n / np.linalg.norm(n)

    # Plane mesh (square).
    s = np.linspace(-2, 2, 12)
    t = np.linspace(-2, 2, 12)
    S, T = np.meshgrid(s, t)
    X = S * u[0] + T * v[0]
    Y = S * u[1] + T * v[1]
    Z = S * u[2] + T * v[2]

    # Response point and decomposition.
    proj_on_plane = 0.6 * u + 0.5 * v          # language component
    resp = proj_on_plane + 1.7 * n             # response = language + residual

    fig = plt.figure(figsize=(8.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.04, 0.04, 0.92, 0.86])
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    # Translucent plane + outline.
    ax.plot_surface(X, Y, Z, color=COL_PLANE, alpha=0.22,
                    edgecolor="none", zorder=1)
    edge = np.array([
        [-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0], [-2, -2, 0]
    ])
    ax.plot(edge[:, 0], edge[:, 1], edge[:, 2],
            color=COL_PLANE_EDGE, linewidth=1.0, zorder=2)

    # Perpendicular axis through the origin (the debiased direction).
    ax.plot([0, 0], [0, 0], [-0.6, 2.6],
            color=COL_PERP, linewidth=2.0, linestyle=(0, (4, 3)), zorder=3)
    ax.text(0.05, 0.05, 2.7, "orthogonal complement\n(debiased direction)",
            fontsize=11, weight="700", color=COL_PERP, zorder=6)

    # Plane label.
    ax.text(1.65, -1.95, 0, "language subspace",
            fontsize=12, weight="700", color=COL_PLANE_EDGE, zorder=6)

    # Response point + perpendicular drop.
    ax.scatter(*resp, s=240, color=COL_RESP, edgecolors="#0F172A",
               linewidths=1.6, depthshade=False, zorder=6)
    ax.text(resp[0] + 0.10, resp[1] + 0.10, resp[2] + 0.20,
            "response", fontsize=12, weight="700",
            color=COL_RESP, zorder=7)

    # Foot of the perpendicular (language component).
    ax.scatter(*proj_on_plane, s=160, facecolors="white",
               edgecolors=COL_RESP, linewidths=2.0, depthshade=False,
               zorder=6)
    ax.text(proj_on_plane[0] + 0.10, proj_on_plane[1] + 0.10,
            proj_on_plane[2] - 0.30,
            "language\ncomponent",
            fontsize=10.5, weight="600", color=COL_RESP, zorder=7,
            multialignment="left")

    # Dashed perpendicular drop.
    ax.plot([resp[0], proj_on_plane[0]],
            [resp[1], proj_on_plane[1]],
            [resp[2], proj_on_plane[2]],
            color=COL_RESP, linestyle="--", linewidth=1.6, zorder=5)

    # Bracket-style label for the residual height.
    midpoint = (resp + proj_on_plane) / 2
    ax.text(midpoint[0] + 0.20, midpoint[1] - 0.20, midpoint[2],
            "debiased\nresidual",
            fontsize=11, weight="700", color=COL_RESP, zorder=7)

    # Camera & cosmetics.
    ax.view_init(elev=18, azim=-58)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_zlim(-0.8, 3.0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor("white")
        pane.set_alpha(0.0)

    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
