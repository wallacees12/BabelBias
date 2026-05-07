"""Render the Mercury q01 Crimea-coverage stacked bar for the 25 May deck.

Reads `data/Russia-Ukraine/analysis/mercury_q01_crimea_coverage.csv`
(per-response Sonnet codes from `score_exp_003_mercury_crimea.py`)
and writes one figure to:

    Presentations/figures/May 25/09_mercury_q01_crimea_bar.png

Slide 9 of the 25 May Urman deck. Single visual: 3 stacked bars (EN /
RU / UK), each n=10, segments coloured by 4-level Crimea-coverage
code (2 Primary / 1 Secondary / 0 Absent / R Refusal).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from babelbias.paths import ANALYSIS_DIR, PROJECT_ROOT

CSV_IN = ANALYSIS_DIR / "mercury_q01_crimea_coverage.csv"
OUT = PROJECT_ROOT / "Presentations" / "figures" / "May 25" / "09_mercury_q01_crimea_bar.png"

LANGS = ("en", "ru", "uk")
ORDER = ("2", "1", "0", "R")
LABELS = {
    "2": "Primary (leads with Crimea)",
    "1": "Secondary (mentions Crimea)",
    "0": "Absent (no Crimea)",
    "R": "Refusal / empty",
}
COLORS = {
    "2": "#08306b",
    "1": "#6baed6",
    "0": "#fdae61",
    "R": "#bdbdbd",
}


def main() -> None:
    df = pd.read_csv(CSV_IN, dtype={"code": str})
    counts = (
        df.groupby(["language", "code"]).size().unstack("code", fill_value=0)
        .reindex(index=list(LANGS), columns=list(ORDER), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    bottom = np.zeros(len(LANGS))
    for code in ORDER:
        vals = counts[code].to_numpy()
        ax.barh(
            [l.upper() for l in LANGS], vals, left=bottom,
            color=COLORS[code], label=LABELS[code], edgecolor="white", linewidth=0.6,
        )
        for y, v in enumerate(vals):
            if v <= 0:
                continue
            ax.text(bottom[y] + v / 2, y, str(int(v)), ha="center", va="center",
                    fontsize=10, color="white" if code in ("2",) else "black")
        bottom += vals

    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("count of n=10 responses")
    ax.set_title(
        "Mercury-2 · q01 'little green men' · same prompt, three languages\n"
        "(coding by Sonnet 4.6 against pre-registered 4-level rubric)",
        fontsize=10, loc="left",
    )
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=160)
    plt.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
