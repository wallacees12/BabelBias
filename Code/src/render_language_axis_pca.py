"""Language-axis PCA over the universal-control corpus.

Loads every embedded universal-control article (data/universal_controls/
processed_leads/CONTROL_*_<lang>.json) for the 10 target languages,
computes the per-language centroid, and projects the centroids into the
first two principal components of the embedding space.

This is the methodology-chapter / discussion-chapter visualisation the
language-axis debiasing operates on: each language carves out its own
direction in the embedding space, and the orthogonal complement of the
span of these centroids is what survives debiasing.

Hypothesised pattern: language families cluster.
  * Slavic — RU, UK
  * Semitic — HE, AR
  * Indic — HI, UR
  * Romance — ES, PT
  * CJK — ZH (alone)
  * Germanic — EN (alone)

Output:
  Presentations/figures/methodology/language_axis_pca.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from babelbias.paths import DATA_ROOT, PROJECT_ROOT


sns.set_theme(style="whitegrid", context="talk", font_scale=0.78,
              rc={"savefig.dpi": 200, "savefig.bbox": "tight",
                  "axes.spines.top": False, "axes.spines.right": False})

OUT = (PROJECT_ROOT / "Presentations" / "figures" / "methodology"
       / "language_axis_pca.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

LANGS = ("en", "ru", "uk", "he", "ar", "zh", "es", "pt", "hi", "ur")
LANG_LABEL = {"en": "EN", "ru": "RU", "uk": "UK", "he": "HE", "ar": "AR",
              "zh": "ZH", "es": "ES", "pt": "PT", "hi": "HI", "ur": "UR"}
LANG_FAMILY = {
    "en": "Germanic",
    "ru": "Slavic",  "uk": "Slavic",
    "he": "Semitic", "ar": "Semitic",
    "zh": "CJK",
    "es": "Romance", "pt": "Romance",
    "hi": "Indic",   "ur": "Indic",
}
FAMILY_COLOR = {
    "Germanic": "#2E75B6",
    "Slavic":   "#A50026",
    "Semitic":  "#1B9E77",
    "CJK":      "#984EA3",
    "Romance":  "#F2B701",
    "Indic":    "#E7298A",
}


def load_lang_embeddings(lang: str) -> np.ndarray:
    src = DATA_ROOT / "universal_controls" / "processed_leads"
    vecs = []
    for f in src.glob(f"CONTROL_*_{lang}.json"):
        try:
            rec = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        emb = rec.get("embedding")
        if emb:
            vecs.append(emb)
    if not vecs:
        return np.empty((0, 0))
    return np.asarray(vecs, dtype=np.float64)


def main() -> None:
    # Load per-language sets and flatten with language labels
    Xs, langs = [], []
    n_per_lang = {}
    for lang in LANGS:
        E = load_lang_embeddings(lang)
        if E.size == 0:
            continue
        Xs.append(E)
        langs.extend([lang] * len(E))
        n_per_lang[lang] = len(E)
    if not Xs:
        raise SystemExit("No universal-control embeddings found.")
    X = np.vstack(Xs)
    langs_arr = np.asarray(langs)

    # Global PCA via SVD on centred data
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    PC = Vt[:2]                     # first two principal directions
    Y  = Xc @ PC.T                  # (n, 2) projected coordinates

    # Per-language centroids (in PC space)
    centroids = {}
    for lang in LANGS:
        if lang not in n_per_lang:
            continue
        centroids[lang] = Y[langs_arr == lang].mean(axis=0)

    # ── Figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    # Sample of points (alpha low) to show within-language scatter
    rng = np.random.default_rng(7)
    for lang in LANGS:
        if lang not in n_per_lang:
            continue
        mask = langs_arr == lang
        idx = np.flatnonzero(mask)
        sample = rng.choice(idx, size=min(120, len(idx)), replace=False)
        family = LANG_FAMILY[lang]
        color = FAMILY_COLOR[family]
        ax.scatter(Y[sample, 0], Y[sample, 1], s=18, color=color,
                    alpha=0.18, edgecolors="none", zorder=2)

    # Centroid markers + labels
    for lang, c in centroids.items():
        family = LANG_FAMILY[lang]
        color = FAMILY_COLOR[family]
        ax.scatter(*c, s=420, color=color, edgecolors="#0F172A",
                    linewidths=1.8, zorder=5, marker="X")
        ax.annotate(LANG_LABEL[lang],
                     xy=c, xytext=(c[0] + 0.08 * abs(c[0] + 1),
                                    c[1] + 0.02 * abs(c[1] + 1) + 0.08),
                     fontsize=14, weight="800", color=color, zorder=6,
                     bbox=dict(boxstyle="round,pad=0.18",
                                facecolor="white", edgecolor=color,
                                linewidth=1.2))

    # Variance explained
    var_explained = (S[:2] ** 2) / (S ** 2).sum() * 100
    ax.set_xlabel(f"PC 1   ({var_explained[0]:.1f}% of variance)",
                   fontsize=11)
    ax.set_ylabel(f"PC 2   ({var_explained[1]:.1f}% of variance)",
                   fontsize=11)

    # Family legend
    families_seen = sorted({LANG_FAMILY[l] for l in centroids})
    legend_handles = [
        plt.Line2D([0], [0], marker="X", color="none",
                    markerfacecolor=FAMILY_COLOR[f],
                    markeredgecolor="#0F172A", markersize=14, label=f)
        for f in families_seen
    ]
    leg = ax.legend(handles=legend_handles, loc="upper right",
                     frameon=True, fontsize=10.5,
                     title="Language family", title_fontsize=11,
                     labelcolor="#0F172A", framealpha=0.96,
                     edgecolor="#CBD5E1")
    leg.get_title().set_color("#0F172A"); leg.get_title().set_weight("700")
    ax.grid(alpha=0.3)

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
