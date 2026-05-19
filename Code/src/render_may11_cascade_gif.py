"""Animated decomposition cascade for the May 11 deck.

Visualises the headline +0.18 EN ingroup-pull number being built up
step by step:

    1. raw cosine                    EN -> EN  ~ 0.74
    2. row-centred (subtract mean)               +0.18  <- ingroup pull
    3. after language-axis debiasing             +0.16
    4. ratio EN-vs-UK after debiasing           ~16x

Each frame holds for ~1 sec, then the next bar fades in.

Output: Presentations/figures/May 11/12_cascade_decomposition.gif
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch

from babelbias.paths import PROJECT_ROOT


OUT = (PROJECT_ROOT / "Presentations" / "figures" / "May 11"
       / "12_cascade_decomposition.gif")
OUT.parent.mkdir(parents=True, exist_ok=True)


# Stages: (label, value, color, annotation)
STAGES = [
    ("Raw cosine\nEN response × EN anchor", 0.74, "#5B9BD5",
     "the on-diagonal cosine"),
    ("- row mean", 0.74 - 0.56, "#1F4E79",
     "subtract the row mean → row-centred"),
    ("Row-centred\n(EN ingroup pull)", 0.18, "#1F4E79",
     "+0.18 — the headline number"),
    ("After debiasing", 0.16, "#A50026",
     "language axis projected out"),
    ("Residual", 0.16, "#A50026",
     "content-driven signal survives"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)

    n = len(STAGES)
    x = np.arange(n)
    bar_w = 0.6
    y_max = 0.85

    # Pre-draw faint placeholder bars so the canvas doesn't reflow.
    placeholder_color = "#E2E8F0"
    bar_artists = []
    label_artists = []
    annot_artists = []

    for i, (label, val, color, annot) in enumerate(STAGES):
        bar = ax.bar(x[i], 0, bar_w, color=color,
                      edgecolor="white", linewidth=0.8, zorder=3)
        bar_artists.append(bar[0])
        # Stage label below each bar
        ax.text(x[i], -0.08, label, ha="center", va="top",
                fontsize=10, color="#0F172A", weight="600")
        # Top-of-bar annotation, hidden initially
        ann = ax.text(x[i], 0, "", ha="center", va="bottom",
                       fontsize=11, color="#0F172A", weight="700")
        label_artists.append(ann)
        # Side annotation (e.g. "subtract the row mean")
        side = ax.text(x[i], -0.20, "", ha="center", va="top",
                        fontsize=9, color="#475569", style="italic",
                        wrap=True)
        annot_artists.append(side)

    # Cosmetic frame
    ax.axhline(0, color="#0F172A", linewidth=0.8, zorder=2)
    ax.set_ylim(-0.30, y_max)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks([])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([f"{t:.1f}" for t in [0, 0.2, 0.4, 0.6, 0.8]])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Cosine similarity", fontsize=10.5, color="#0F172A")
    ax.grid(axis="y", alpha=0.25)

    # Connecting arrows between stages (drawn lazily as frames advance)
    arrow_artists: list = []

    FPS = 30
    HOLD_FRAMES = 36                  # 1.2 s hold per stage
    GROW_FRAMES = 24                  # 0.8 s smooth grow-in
    total_frames = (HOLD_FRAMES + GROW_FRAMES) * n + HOLD_FRAMES

    def ease(t: float) -> float:
        if t < 0.5:
            return 4 * t ** 3
        return 1 - ((-2 * t + 2) ** 3) / 2

    def update(frame):
        # Determine which stage we're on and how much it's grown
        active = 0
        local = frame
        for i in range(n):
            block = HOLD_FRAMES + GROW_FRAMES if i > 0 else GROW_FRAMES
            if local < block:
                active = i
                progress = local / max(block - HOLD_FRAMES, 1) if i > 0 else local / GROW_FRAMES
                break
            local -= block
        else:
            active = n - 1
            progress = 1.0

        progress = ease(float(np.clip(progress, 0, 1)))

        for i, (label, val, color, annot) in enumerate(STAGES):
            if i < active:
                bar_artists[i].set_height(val)
                label_artists[i].set_position((x[i], val + 0.015))
                label_artists[i].set_text(f"{val:+.2f}" if i > 0 else f"{val:.2f}")
                annot_artists[i].set_text(annot)
            elif i == active:
                bar_artists[i].set_height(val * progress)
                label_artists[i].set_position((x[i], val * progress + 0.015))
                label_artists[i].set_text(
                    (f"{val * progress:+.2f}" if i > 0 else f"{val * progress:.2f}")
                    if progress > 0.05 else ""
                )
                annot_artists[i].set_text(annot if progress > 0.4 else "")
            else:
                bar_artists[i].set_height(0)
                label_artists[i].set_text("")
                annot_artists[i].set_text("")

        return bar_artists + label_artists + annot_artists

    ani = FuncAnimation(fig, update, frames=total_frames,
                         interval=1000 / FPS, blit=False)

    print(f"Saving GIF to {OUT}…")
    ani.save(OUT, writer=PillowWriter(fps=FPS))
    plt.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
