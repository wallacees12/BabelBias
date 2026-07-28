"""Render the thesis results figures that live in Report/tex/figures/.

Two figures, both title-less per the project figure rule (captions are
carried by the LaTeX ``\\caption{}``):

exp_014_forest.png
    Forest plot of the row-centred diagonal ingroup pull per provider and
    language on the ru_uk_core sweep, with 95% CIs. Visual claim: the
    EN > RU > UK ordering is unanimous across the matrix, and the TAIDE
    contested-language cells sit at or below zero.
    Source: data/Russia-Ukraine/analysis/exp_015_diagonal_summary.csv
    (OpenAI panel).

exp_004_two_shapes.png
    Scatter of the imaginary-conflict question bank: cross-lingual content
    divergence (as a multiple of the same-language baseline) against the
    largest per-language |stance mean|. Visual claim: free-form prose
    diverges without direction; attribution commits without divergence.
    Sources: imaginary_metric2.csv, imaginary_metric1_by_qid_lang.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "data" / "Russia-Ukraine" / "analysis"
OUT_DIR = PROJECT_ROOT / "Report" / "tex" / "figures"

LANG_COLOR = {"en": "#2E75B6", "ru": "#A50026", "uk": "#1A9850"}
LANG_LABEL = {"en": "English", "ru": "Russian", "uk": "Ukrainian"}

PROVIDER_LABEL = {
    "baidu/ernie-4.5-300b-a47b": "ernie-4.5-300b-a47b",
    "ollama:allam-7b": "ALLaM-7B (local)",
    "ollama:taide-llama3-8b": "TAIDE-Llama3-8B (local)",
}

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.15,
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    },
)


def render_forest() -> Path:
    """Forest plot of per-provider diagonal pull with 95% CIs."""
    df = pd.read_csv(ANALYSIS_DIR / "exp_014_qclustered_cis.csv")
    df = df.rename(columns={"provider": "model"})
    df = df.sort_values("en", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    offsets = {"en": 0.22, "ru": 0.0, "uk": -0.22}
    for lang in ("en", "ru", "uk"):
        vals = df[lang]
        cis = df[f"ci_{lang}"]
        ys = np.arange(len(df)) + offsets[lang]
        ax.errorbar(
            vals, ys, xerr=cis, fmt="o", ms=5.5, lw=0, elinewidth=1.1,
            capsize=2.2, color=LANG_COLOR[lang], label=LANG_LABEL[lang],
        )
        ax.axvline(vals.mean(), color=LANG_COLOR[lang], ls=":", lw=1.0, alpha=0.55)

    ax.axvline(0.0, color="#333333", lw=0.9)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels([PROVIDER_LABEL.get(m, m) for m in df["model"]], fontsize=9)
    ax.set_xlabel("Row-centred diagonal ingroup pull (cosine)")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.set_ylim(-0.7, len(df) - 0.3)

    out = OUT_DIR / "exp_014_forest.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def render_two_shapes() -> Path:
    """Divergence-vs-direction scatter for the imaginary question bank."""
    m2 = pd.read_csv(ANALYSIS_DIR / "imaginary_metric2.csv")
    m2["same"] = m2[["same_en", "same_ru", "same_uk"]].mean(axis=1)
    m2["cross"] = m2[["x_en_ru", "x_en_uk", "x_ru_uk"]].mean(axis=1)
    ratio = (m2.groupby("qid")["cross"].mean() / m2.groupby("qid")["same"].mean())

    m1 = pd.read_csv(ANALYSIS_DIR / "imaginary_metric1_by_qid_lang.csv")
    stance = m1.groupby("qid")["mean_score"].apply(lambda s: s.abs().max())

    qid_label = {
        "q01_responsibility": "responsibility",
        "q02_terminology": "terminology",
        "q03_war_crime": "war crime",
        "q04_belongs": "belongs to",
        "q05_news_report": "news report",
    }

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for qid in ratio.index:
        x, y = ratio[qid], stance[qid]
        ax.scatter(x, y, s=110, color="#1F4E79", zorder=3)
        dx, dy = (0.03, 0.012)
        ha = "left"
        if qid == "q05_news_report":  # rightmost point — label to the left
            dx, ha = -0.03, "right"
        ax.annotate(qid_label[qid], (x, y), xytext=(x + dx, y + dy),
                    fontsize=10.5, ha=ha, color="#333333")

    ax.set_xlabel("Cross-lingual content divergence "
                  "($\\times$ same-language baseline)")
    ax.set_ylabel("Largest per-language $|$stance mean$|$")
    ax.set_xlim(1.55, 3.05)
    ax.set_ylim(-0.03, 0.62)

    out = OUT_DIR / "exp_004_two_shapes.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


EVENTS = [
    ("Russia–Ukraine", PROJECT_ROOT / "data" / "Russia-Ukraine" / "analysis",
     "ru_uk_core", ["en", "ru", "uk"]),
    ("Israel–Palestine", PROJECT_ROOT / "data" / "israel_palestine" / "analysis",
     "israel_palestine", ["en", "he", "ar"]),
    ("India–Pakistan", PROJECT_ROOT / "data" / "india_pakistan" / "analysis",
     "india_pakistan", ["en", "hi", "ur"]),
    ("Taiwan strait", PROJECT_ROOT / "data" / "taiwan_strait" / "analysis",
     "taiwan_strait", ["en", "zh"]),
    ("Falklands", PROJECT_ROOT / "data" / "falklands" / "analysis",
     "falklands", ["en", "es"]),
]


def _diag_means(analysis_dir: Path, event: str, langs: list[str],
                debiased: bool) -> dict[str, float]:
    """Mean row-centred diagonal per language across providers on disk."""
    suffix = f"{event}_debiased" if debiased else event
    per_lang: dict[str, list[float]] = {l: [] for l in langs}
    for csv_path in analysis_dir.glob(f"*/{suffix}/anchor_heatmap_rowcentered.csv"):
        if "yandex" in str(csv_path):
            continue
        df = pd.read_csv(csv_path, index_col=0)
        for lang in langs:
            row, col = f"resp_{lang}", f"wiki_{lang}"
            if row in df.index and col in df.columns:
                per_lang[lang].append(float(df.loc[row, col]))
    # Baidu's analysis sits one level deeper (model id contains a slash).
    for csv_path in analysis_dir.glob(f"baidu/*/{suffix}/anchor_heatmap_rowcentered.csv"):
        df = pd.read_csv(csv_path, index_col=0)
        for lang in langs:
            row, col = f"resp_{lang}", f"wiki_{lang}"
            if row in df.index and col in df.columns:
                per_lang[lang].append(float(df.loc[row, col]))
    # nanmean: isolated NaN cells (e.g. aya's Urdu row) must not void a language.
    return {l: float(np.nanmean(v)) for l, v in per_lang.items() if v}


def render_debias_dumbbell() -> Path:
    """Raw vs language-axis-debiased diagonal pull per (family, language)."""
    rows = []
    for family, analysis_dir, event, langs in EVENTS:
        raw = _diag_means(analysis_dir, event, langs, debiased=False)
        deb = _diag_means(analysis_dir, event, langs, debiased=True)
        for lang in langs:
            if lang in raw and lang in deb:
                rows.append((family, lang, raw[lang], deb[lang]))

    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    ys, labels = [], []
    y = 0.0
    prev_family = None
    for family, lang, r, d in rows:
        if prev_family is not None and family != prev_family:
            y -= 0.9  # gap between family blocks
        prev_family = family
        color = "#2E75B6" if lang == "en" else "#A50026"
        ax.plot([r, d], [y, y], color=color, lw=1.6, zorder=2)
        ax.scatter([r], [y], s=62, color=color, zorder=3)
        ax.scatter([d], [y], s=62, facecolor="white", edgecolor=color,
                   lw=1.6, zorder=3)
        if r > 0.005:
            note = f"−{(1 - d / r) * 100:.0f}%"
        else:
            note = "raw < 0"
        ax.annotate(note, (max(r, d), y), xytext=(8, -3.5),
                    textcoords="offset points", fontsize=8.5, color="#555555")
        ys.append(y)
        labels.append(f"{family} · {lang.upper()}")
        y -= 1.0

    ax.axvline(0.0, color="#333333", lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Row-centred diagonal ingroup pull (cosine)")
    handles = [
        plt.Line2D([0], [0], marker="o", color="#666666", lw=0, ms=8,
                   label="raw"),
        plt.Line2D([0], [0], marker="o", markerfacecolor="white",
                   markeredgecolor="#666666", color="none", ms=8,
                   label="debiased"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=9)

    out = OUT_DIR / "exp_006_debias_dumbbell.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


ANCHOR_SLUGS = {
    "Little_green_men": "Little green men",
    "2014_Russian_annexation_of_Crimea": "Annexation",
    "Revolution_of_Dignity": "Maidan",
    "2014_Crimean_status_referendum": "Referendum",
    "Malaysia_Airlines_Flight_17": "MH17",
    "Stepan_Bandera": "Bandera",
}


def render_evoc_anchor_map() -> Path:
    """UMAP of the ru_uk_core response corpus with Wikipedia anchors."""
    import json

    import umap

    emb_root = PROJECT_ROOT / "data" / "Russia-Ukraine" / "llm_embeddings"
    leads = PROJECT_ROOT / "data" / "Russia-Ukraine" / "processed_leads"

    vecs, langs = [], []
    for f in sorted(emb_root.glob("*/ru_uk_core/*.json")):
        if "yandex" in str(f):
            continue
        d = json.loads(f.read_text())
        if not d.get("embedding"):
            continue
        vecs.append(d["embedding"])
        langs.append(f.stem.rsplit("_", 2)[-2])
    n_resp = len(vecs)

    anchor_meta = []
    for slug, short in ANCHOR_SLUGS.items():
        for lang in ("en", "ru", "uk"):
            d = json.loads((leads / f"{slug}_{lang}.json").read_text())
            vecs.append(d["embedding"])
            anchor_meta.append((short, lang))

    X = np.asarray(vecs, dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    coords = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                       random_state=42).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    langs_arr = np.asarray(langs)
    for lang in ("en", "ru", "uk"):
        mask = langs_arr == lang
        ax.scatter(coords[:n_resp][mask, 0], coords[:n_resp][mask, 1],
                   s=7, color=LANG_COLOR[lang], alpha=0.35, lw=0,
                   label=LANG_LABEL[lang], rasterized=True)
    anchor_xy = coords[n_resp:]
    for (short, lang), (x, yc) in zip(anchor_meta, anchor_xy):
        ax.scatter(x, yc, s=340, marker="*", facecolor=LANG_COLOR[lang],
                   edgecolor="black", lw=1.1, zorder=5)
    # Merge labels for co-located anchors of the same article — the RU/UK
    # co-location IS the finding, and separate labels would overprint.
    from collections import defaultdict
    by_slug = defaultdict(list)
    for (short, lang), (x, yc) in zip(anchor_meta, anchor_xy):
        by_slug[short].append((lang, x, yc))
    groups: list[tuple[str, str, float, float]] = []
    for short, entries in by_slug.items():
        used = [False] * len(entries)
        for i, (lang_i, xi, yi) in enumerate(entries):
            if used[i]:
                continue
            group = [(lang_i, xi, yi)]
            used[i] = True
            for j in range(i + 1, len(entries)):
                lang_j, xj, yj = entries[j]
                if not used[j] and (xi - xj) ** 2 + (yi - yj) ** 2 < 1.2 ** 2:
                    group.append((lang_j, xj, yj))
                    used[j] = True
            gx = np.mean([g[1] for g in group])
            gy = np.mean([g[2] for g in group])
            tag = "+".join(g[0].upper() for g in group)
            label = short if len(group) == 3 else f"{short} ({tag})"
            groups.append((short, label, gx, gy))

    groups_out = []
    for short, label, gx, gy in groups:
        # If another group of the same article sits just left at similar
        # height, its label would collide rightward — flip this one left.
        near_left = any(s == short and abs(oy - gy) < 1.0 and 0 < gx - ox < 6
                        for s, _, ox, oy in groups)
        if near_left:
            groups_out.append((label, gx, gy, (9, -14), "left"))
        else:
            groups_out.append((label, gx, gy, (-9, 7), "right")
                              if any(s == short and abs(oy - gy) < 1.0
                                     and 0 < ox - gx < 6
                                     for s, _, ox, oy in groups)
                              else (label, gx, gy, (9, 7), "left"))
    for label, gx, gy, offset, ha in groups_out:
        ax.annotate(label, (gx, gy), xytext=offset,
                    textcoords="offset points", fontsize=8, ha=ha,
                    color="#111111", zorder=6)
    star = plt.Line2D([0], [0], marker="*", color="none", ms=15,
                      markerfacecolor="#cccccc", markeredgecolor="black",
                      label="Wikipedia anchor")
    handles, hlabels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [star], labels=hlabels + ["Wikipedia anchor"],
              loc="best", frameon=True, fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")

    out = OUT_DIR / "exp_016_anchor_map.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    render_forest()
    render_two_shapes()
    render_debias_dumbbell()
    render_evoc_anchor_map()
