"""Hero figure 3 — Wikipedia-anchor ternary.

Equilateral triangle with the EN / RU / UK Wikipedia anchors at the three
corners.  Each plotted point is a (provider, qid, response_language) cell
whose barycentric coordinates come from the mean cosines to the three
anchors (after softmax normalisation so the triple sums to 1).

Source: `data/Russia-Ukraine/analysis/<provider>/ru_uk_core/anchor_per_question.csv`
columns `<resp_lang>-><wiki_lang>` give the per-question mean cosines.

The visual punchline of the figure is the UK-response pile-up on the
RU-UK edge tilted toward the RU corner — the structural fusion finding
from exp_016 made geometric.

Per BabelBias figure rule: no in-image titles or captions.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pubfig

from babelbias.palette import ORDERED_MODELS

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "data" / "Russia-Ukraine" / "analysis"
OUT_DIR = ROOT / "Presentations" / "figures" / "hero"

LANGS = ["en", "ru", "uk"]
LANG_COLOR = {
    "en": "#1F4E79",  # US blue (matches palette)
    "ru": "#A50026",  # Russia/CAC red
    "uk": "#F2B701",  # Saffron — visually distinct, no project conflict
}
LANG_LABEL = {"en": "EN responses", "ru": "RU responses", "uk": "UK responses"}

# Equilateral triangle corner coordinates (anchor language → vertex).
CORNERS = {
    "en": np.array([0.0, 0.0]),
    "ru": np.array([1.0, 0.0]),
    "uk": np.array([0.5, np.sqrt(3) / 2]),
}


def softmax_weights(c_en: float, c_ru: float, c_uk: float, tau: float = 12.0) -> np.ndarray:
    """Convert three cosine similarities into normalised barycentric weights.

    Cosines on text-embedding-3-small live in [≈0.2, ≈0.85] so a softmax
    with temperature 1.0 produces nearly-uniform weights.  Using tau = 12
    sharpens the projection without saturating to corners.
    """
    z = np.array([c_en, c_ru, c_uk], dtype=float) * tau
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def project(weights: np.ndarray) -> np.ndarray:
    """Barycentric → cartesian using the equilateral corner layout."""
    return (
        weights[0] * CORNERS["en"]
        + weights[1] * CORNERS["ru"]
        + weights[2] * CORNERS["uk"]
    )


def load_points() -> pd.DataFrame:
    """Return a DataFrame with columns model, qid, lang, x, y."""
    rows = []
    for model in ORDERED_MODELS:
        csv = ANALYSIS_DIR / model / "ru_uk_core" / "anchor_per_question.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            for lang in LANGS:
                col_en = f"{lang}->en"
                col_ru = f"{lang}->ru"
                col_uk = f"{lang}->uk"
                if col_en not in df.columns:
                    continue
                w = softmax_weights(r[col_en], r[col_ru], r[col_uk])
                x, y = project(w)
                rows.append({
                    "model": model, "qid": r["qid"], "lang": lang,
                    "x": x, "y": y,
                })
    return pd.DataFrame(rows)


def render(out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = load_points()

    # Use pubfig's Nature theme so font/line weights match the other two
    # hero figures, then drop a custom triangle on top.
    theme = pubfig.get_theme("default")
    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=150)
    ax.set_aspect("equal")
    ax.axis("off")

    # Triangle frame
    tri_x = [CORNERS["en"][0], CORNERS["ru"][0], CORNERS["uk"][0], CORNERS["en"][0]]
    tri_y = [CORNERS["en"][1], CORNERS["ru"][1], CORNERS["uk"][1], CORNERS["en"][1]]
    ax.plot(tri_x, tri_y, color="#555", linewidth=1.0, zorder=1)

    # Internal gridlines: thirds and centroid for visual anchoring.
    centroid = (CORNERS["en"] + CORNERS["ru"] + CORNERS["uk"]) / 3.0
    for corner in CORNERS.values():
        mid_opp = (
            sum(c for k, c in CORNERS.items() if not np.allclose(c, corner))
            / 2.0
        )
        ax.plot(
            [corner[0], mid_opp[0]],
            [corner[1], mid_opp[1]],
            color="#cccccc", linewidth=0.5, linestyle=":", zorder=1,
        )
    ax.scatter(*centroid, s=10, color="#999", marker="x", zorder=2)

    # Scatter per response language
    for lang in LANGS:
        sub = pts[pts["lang"] == lang]
        ax.scatter(
            sub["x"], sub["y"],
            s=24,
            color=LANG_COLOR[lang],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.4,
            label=LANG_LABEL[lang],
            zorder=3,
        )
        # Class centroid as a larger ringed marker for at-a-glance grouping.
        cx, cy = sub["x"].mean(), sub["y"].mean()
        ax.scatter(
            cx, cy,
            s=200,
            facecolor="none",
            edgecolor=LANG_COLOR[lang],
            linewidth=2.0,
            zorder=4,
        )
        ax.scatter(
            cx, cy,
            s=30,
            color=LANG_COLOR[lang],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )

    # Corner labels (the Wikipedia anchors)
    label_offset = {
        "en": (-0.06, -0.06),
        "ru": (+0.06, -0.06),
        "uk": (0.00, +0.10),
    }
    label_ha = {"en": "right", "ru": "left", "uk": "center"}
    label_va = {"en": "top", "ru": "top", "uk": "bottom"}
    for lang, corner in CORNERS.items():
        dx, dy = label_offset[lang]
        ax.text(
            corner[0] + dx, corner[1] + dy,
            f"{lang.upper()} Wikipedia\nanchor",
            ha=label_ha[lang], va=label_va[lang],
            fontsize=10.5, fontweight="bold", color="#333",
        )

    # Legend below the triangle — keeps the apex label clear.
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
        fontsize=9.5,
        handletextpad=0.5,
        labelspacing=0.3,
        ncol=3,
        columnspacing=1.4,
    )
    for handle in leg.legend_handles:
        handle.set_alpha(0.9)

    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(-0.26, 1.12)

    pubfig.batch_export(
        fig,
        out_dir / "fig3_anchor_ternary",
        formats=("png", "pdf"),
        spec="nature",
        width="double",
        dpi=300,
        trim=True,
    )
    plt.close(fig)
    return out_dir / "fig3_anchor_ternary.png"


if __name__ == "__main__":
    print(f"wrote {render()}")
