"""5x9 small-multiples cosine grid for the May 11 deck.

A 5-row x 9-column grid of mini 3x3 cosine matrices — one per
(provider, qid). Lets the eye spot per-question, per-provider patterns
at a glance.

Output: Presentations/figures/May 11/11_provider_question_small_multiples.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from babelbias.palette import MODEL_COLORS
from babelbias.paths import (
    LLM_EMBEDDINGS_DIR, PROCESSED_LEADS_DIR, PROJECT_ROOT,
)


sns.set_theme(style="white", context="talk", font_scale=0.65,
              rc={"savefig.dpi": 200, "savefig.bbox": "tight"})

OUT = (PROJECT_ROOT / "Presentations" / "figures" / "May 11"
       / "11_provider_question_small_multiples.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

EVENT = "ru_uk_core"
LANGS = ("en", "ru", "uk")
LANG_LABEL = {"en": "EN", "ru": "RU", "uk": "UK"}
MODELS = [
    "claude-haiku-4-5", "gpt-4o-mini", "gemini-2.5-flash",
    "grok-3-mini", "deepseek-chat",
]
MODEL_LABEL = {
    "claude-haiku-4-5":  "Claude\nHaiku 4.5",
    "gpt-4o-mini":        "GPT-4o\nmini",
    "gemini-2.5-flash":   "Gemini\n2.5 Flash",
    "grok-3-mini":        "Grok-3\nmini",
    "deepseek-chat":      "DeepSeek\nchat",
}
QID_LABEL = {
    "q01_little_green_men": "q01", "q02_crimea_2014": "q02",
    "q03_maidan_revolution": "q03", "q04_referendum": "q04",
    "q05_mh17": "q05", "q06_crimea_belongs": "q06",
    "q07_pov_russia": "q07", "q08_pov_ukraine": "q08",
    "q09_bandera": "q09",
}
ANCHOR_SLUGS = {
    "q01_little_green_men":  "Little_green_men",
    "q02_crimea_2014":       "2014_Russian_annexation_of_Crimea",
    "q03_maidan_revolution": "Revolution_of_Dignity",
    "q04_referendum":        "2014_Crimean_status_referendum",
    "q05_mh17":              "Malaysia_Airlines_Flight_17",
    "q06_crimea_belongs":    "2014_Russian_annexation_of_Crimea",
    "q07_pov_russia":        "2014_Russian_annexation_of_Crimea",
    "q08_pov_ukraine":       "2014_Russian_annexation_of_Crimea",
    "q09_bandera":           "Stepan_Bandera",
}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_matrix(model: str, qid: str,
                   anchors: dict[tuple[str, str], np.ndarray],
                   responses: dict[tuple[str, str], list[np.ndarray]]
                   ) -> np.ndarray:
    M = np.full((3, 3), np.nan)
    for i, r_lang in enumerate(LANGS):
        R = responses.get((qid, r_lang), [])
        if not R:
            continue
        for j, a_lang in enumerate(LANGS):
            anchor = anchors[(qid, a_lang)]
            M[i, j] = float(np.mean([_cos(r, anchor) for r in R]))
    # Row-centre.
    row_means = np.nanmean(M, axis=1, keepdims=True)
    return M - row_means


def main() -> None:
    # Anchors are model-independent
    anchors: dict[tuple[str, str], np.ndarray] = {}
    for qid, slug in ANCHOR_SLUGS.items():
        for lang in LANGS:
            with open(PROCESSED_LEADS_DIR / f"{slug}_{lang}.json") as f:
                anchors[(qid, lang)] = np.asarray(json.load(f)["embedding"])

    # Load all 5 models' response embeddings
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for model in MODELS:
        rdir = LLM_EMBEDDINGS_DIR / model / EVENT
        responses: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        for f in rdir.glob("*.json"):
            try:
                rec = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not rec.get("embedding") or rec.get("refusal"):
                continue
            responses[(rec["qid"], rec["language"])].append(
                np.asarray(rec["embedding"])
            )
        for qid in ANCHOR_SLUGS:
            matrices[(model, qid)] = compute_matrix(model, qid, anchors, responses)

    # Shared color scale across all cells
    all_vals = np.concatenate([m.flatten() for m in matrices.values()])
    abs_max = float(np.nanmax(np.abs(all_vals)))

    fig, axes = plt.subplots(
        len(MODELS), len(ANCHOR_SLUGS),
        figsize=(15.5, 9.0), constrained_layout=True,
    )

    qids = list(ANCHOR_SLUGS.keys())
    for i, model in enumerate(MODELS):
        for j, qid in enumerate(qids):
            ax = axes[i, j]
            M = matrices[(model, qid)]
            im = ax.imshow(M, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
                            aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for k in range(3):
                for l in range(3):
                    v = M[k, l]
                    if np.isnan(v): continue
                    ax.text(l, k, f"{v:+.2f}".replace("+0.", "+.").replace("-0.", "-."),
                            ha="center", va="center",
                            fontsize=6.5,
                            color="white" if abs(v) > abs_max * 0.5 else "#0F172A",
                            fontweight="600")
            # Top row: question labels
            if i == 0:
                ax.set_title(QID_LABEL[qid], fontsize=10, weight="700",
                             pad=4, color="#0F172A")
            # Left column: model labels
            if j == 0:
                ax.set_ylabel(MODEL_LABEL[model], fontsize=9, weight="700",
                              color=MODEL_COLORS.get(model, "#0F172A"),
                              rotation=0, ha="right", va="center", labelpad=12)

    # Shared colorbar
    cbar_ax = fig.add_axes([1.005, 0.20, 0.012, 0.6])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Row-centred cosine\n(diagonal = ingroup pull)",
                 fontsize=9, color="#0F172A")
    cb.ax.tick_params(labelsize=8)

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
