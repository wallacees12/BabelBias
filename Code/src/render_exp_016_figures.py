"""
Render paper-grade figures for exp_016 (EVoC cluster exploration).

Three deliverables, each saved as both PDF (vector, for the thesis) and
PNG (300 dpi raster, for slides):

  fig01_umap_quad.{pdf,png}        4-panel UMAP — language / qid / EVoC cluster / outlier-vs-frontier
  fig02_purity_bars.{pdf,png}      Headline purity numbers, three sets compared
  fig03_model_fingerprint.{pdf,png} Per-model top-3-cluster concentration

Figures use a tight, journal-column-friendly style (~7" wide for two-column
layout). Categorical palettes chosen for colour-blind safety
(ColorBrewer Set2 / tab10), with the 3 outlier providers explicitly highlighted.

Run after the embeddings exist (post exp_001/002 and any newly-embedded
providers). Outputs to Presentations/figures/exp_016/.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from babelbias.paths import LLM_EMBEDDINGS_DIR, PROJECT_ROOT


OUT_DIR = PROJECT_ROOT / "Presentations" / "figures" / "exp_016"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Models we treat as outliers in the 15-model set — see exp_016 report
OUTLIERS = {"yandexgpt", "ollama:taide-llama3-8b", "ollama:allam-7b"}

# Stable categorical palettes (ColorBrewer-ish, colour-blind safe)
LANG_PALETTE = {
    "en": "#1f77b4",  # blue
    "ru": "#d62728",  # red
    "uk": "#2ca02c",  # green
}

# Style
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handlelength": 1.0,
})


def load_data():
    rows = []
    vecs = []
    for f in LLM_EMBEDDINGS_DIR.rglob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not d.get("embedding"):
            continue
        rows.append({
            "model": d.get("model"),
            "qid": d.get("qid"),
            "language": d.get("language"),
        })
        vecs.append(d["embedding"])
    X = np.array(vecs, dtype=np.float32)
    return X, rows


def normalise_l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def fit_evoc(X):
    from evoc import EVoC
    print("Fitting EVoC ...")
    m = EVoC(base_min_cluster_size=30, max_layers=10, random_state=42)
    m.fit_predict(X)
    return m


def fit_umap(X):
    import umap
    print("Fitting UMAP ...")
    return umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                     random_state=42).fit_transform(X)


def purity(cluster_labels, true_labels):
    by = defaultdict(list)
    for c, t in zip(cluster_labels, true_labels):
        if c == -1:
            continue
        by[c].append(t)
    if not by:
        return 0.0
    return float(np.mean([Counter(v).most_common(1)[0][1] / len(v)
                          for v in by.values()]))


def save_both(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  → {name}.pdf + {name}.png")


def fig01_umap_quad(coords, languages, qids, clusters, models):
    """4-panel UMAP coloured by language / qid / EVoC cluster / outlier-vs-frontier."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.5), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.05, hspace=0.18, left=0.06, right=0.99,
                        top=0.96, bottom=0.05)

    # (a) by language
    ax = axes[0, 0]
    for lang, color in LANG_PALETTE.items():
        m = np.array([l == lang for l in languages])
        ax.scatter(coords[m, 0], coords[m, 1], s=4, c=color,
                   alpha=0.6, label=lang.upper(), linewidths=0)
    ax.set_title("(a) coloured by response language", loc="left",
                 fontweight="bold")
    ax.legend(loc="upper right", markerscale=2)

    # (b) by qid (9 questions, tab10)
    ax = axes[0, 1]
    qid_unique = sorted(set(qids))
    cmap = plt.get_cmap("tab10")
    for i, q in enumerate(qid_unique):
        m = np.array([qq == q for qq in qids])
        # short-form label
        short = q.split("_", 1)[1] if "_" in q else q
        short = short.replace("_", " ")[:18]
        ax.scatter(coords[m, 0], coords[m, 1], s=4, c=[cmap(i)],
                   alpha=0.6, label=short, linewidths=0)
    ax.set_title("(b) coloured by question (qid)", loc="left",
                 fontweight="bold")
    ax.legend(loc="upper right", markerscale=2, fontsize=6,
              ncol=1, handletextpad=0.3, borderpad=0.3)

    # (c) by EVoC cluster
    ax = axes[1, 0]
    cl_unique = sorted(set(c for c in clusters if c != -1))
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(cl_unique):
        m = np.array([cc == c for cc in clusters])
        ax.scatter(coords[m, 0], coords[m, 1], s=4,
                   c=[cmap((i * 7) % 20)], alpha=0.6, linewidths=0)
    # noise points in light grey
    noise = np.array([cc == -1 for cc in clusters])
    if noise.any():
        ax.scatter(coords[noise, 0], coords[noise, 1], s=3, c="#cccccc",
                   alpha=0.4, linewidths=0)
    ax.set_title(f"(c) coloured by EVoC cluster "
                 f"(n={len(cl_unique)})", loc="left", fontweight="bold")

    # (d) outlier vs frontier
    ax = axes[1, 1]
    is_outlier = np.array([m in OUTLIERS for m in models])
    ax.scatter(coords[~is_outlier, 0], coords[~is_outlier, 1], s=4,
               c="#9aa6b2", alpha=0.4, label=f"frontier ({(~is_outlier).sum()})",
               linewidths=0)
    # plot outliers per-model with distinct colours
    outlier_palette = {"yandexgpt": "#d62728",
                       "ollama:taide-llama3-8b": "#ff7f0e",
                       "ollama:allam-7b": "#9467bd"}
    for name, color in outlier_palette.items():
        m = np.array([mm == name for mm in models])
        if m.any():
            label = name.replace("ollama:", "")
            ax.scatter(coords[m, 0], coords[m, 1], s=8, c=color,
                       alpha=0.85, label=f"{label} ({m.sum()})",
                       linewidths=0)
    ax.set_title("(d) outlier providers highlighted", loc="left",
                 fontweight="bold")
    ax.legend(loc="upper right", markerscale=1.5)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    fig.text(0.5, 0.005, "UMAP-1", ha="center", fontsize=8, color="#666")
    fig.text(0.005, 0.5, "UMAP-2", va="center", rotation="vertical",
             fontsize=8, color="#666")
    save_both(fig, "fig01_umap_quad")


def fig02_purity_bars(rows, layer0, all_models):
    """Grouped bar chart: 5-model vs all-15 vs frontier-only purity."""
    languages = [r["language"] for r in rows]
    qids = [r["qid"] for r in rows]
    models = [r["model"] for r in rows]

    # All 15
    p_all_lang = purity(layer0, languages)
    p_all_qid = purity(layer0, qids)
    p_all_mdl = purity(layer0, models)

    # Frontier only — recompute on the subset
    keep = [i for i, r in enumerate(rows) if r["model"] not in OUTLIERS]
    front_layer = [layer0[i] for i in keep]
    front_lang = [languages[i] for i in keep]
    front_qid = [qids[i] for i in keep]
    front_mdl = [models[i] for i in keep]
    p_front_lang = purity(front_layer, front_lang)
    p_front_qid = purity(front_layer, front_qid)
    p_front_mdl = purity(front_layer, front_mdl)

    # 5-model baseline (hard-coded from original exp_016 numbers)
    p_5_lang, p_5_qid, p_5_mdl = 0.932, 0.956, 0.283
    rand_5, rand_15, rand_12 = 1/5, 1/15, 1/12

    sets = ["5-model\noriginal", "all 15\nmodels", "frontier-only\n(12)"]
    lang_v = [p_5_lang, p_all_lang, p_front_lang]
    qid_v  = [p_5_qid,  p_all_qid,  p_front_qid]
    mdl_v  = [p_5_mdl,  p_all_mdl,  p_front_mdl]
    rand_v = [rand_5, rand_15, rand_12]

    x = np.arange(len(sets))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bars_l = ax.bar(x - w, lang_v, w, label="language", color="#1f77b4")
    bars_q = ax.bar(x,     qid_v,  w, label="question",  color="#2ca02c")
    bars_m = ax.bar(x + w, mdl_v,  w, label="model",     color="#d62728")

    # random baseline as a small grey dash above each model bar
    for xi, rv in zip(x + w, rand_v):
        ax.hlines(rv, xi - w/2.2, xi + w/2.2, color="#444", lw=1.2,
                  linestyle=(0, (3, 2)))
    # baseline legend entry — invisible bar for legend ordering
    ax.plot([], [], color="#444", lw=1.2, linestyle=(0, (3, 2)),
            label="random baseline (model)")

    for bars in (bars_l, bars_q, bars_m):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, v + 0.018, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(sets)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("mean per-cluster purity")
    ax.set_title("Purity by axis × cluster set (EVoC Layer 0)",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower left", ncol=2)
    save_both(fig, "fig02_purity_bars")


def fig03_model_fingerprint(rows, layer0):
    """Per-model: % of responses in the model's top-3 most-occupied clusters."""
    models = [r["model"] for r in rows]
    per_model = defaultdict(Counter)
    for c, m in zip(layer0, models):
        if c == -1:
            continue
        per_model[m][c] += 1

    items = []
    for m, counts in per_model.items():
        total = sum(counts.values())
        top3 = sum(n for _, n in counts.most_common(3))
        items.append((m, top3 / total, len(counts), total))
    items.sort(key=lambda x: x[1])  # ascending → highest at top of barh

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ys = np.arange(len(items))
    vals = [it[1] for it in items]
    colors = ["#d62728" if it[0] in OUTLIERS else "#4c72b0" for it in items]
    bars = ax.barh(ys, vals, color=colors, height=0.7)

    labels = []
    for it in items:
        name, share, n_clu, n = it
        # short display name
        disp = name.replace("ollama:", "ollama:").replace(
            "hf:", "hf:").replace("CohereLabs/", "")
        if len(disp) > 30:
            disp = disp[:28] + "…"
        labels.append(f"{disp}  (n_clusters={n_clu})")

    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of responses in the model's top-3 most-occupied clusters")
    ax.axvline(3 / 57, color="#888", lw=0.8, linestyle=(0, (3, 2)),
               label="random baseline (3/57)")
    for y, v in zip(ys, vals):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=7)

    ax.set_title("Per-model embedding-cluster fingerprint  "
                 "(higher = model's responses are more concentrated)",
                 loc="left", fontweight="bold", fontsize=9)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_both(fig, "fig03_model_fingerprint")


def main():
    X, rows = load_data()
    print(f"Loaded {len(rows)} embeddings · {len(set(r['model'] for r in rows))} models")
    Xn = normalise_l2(X)
    evoc_model = fit_evoc(Xn)
    layer0 = list(evoc_model.cluster_layers_[0])

    coords = fit_umap(Xn)

    languages = [r["language"] for r in rows]
    qids = [r["qid"] for r in rows]
    models = [r["model"] for r in rows]

    print("\nRendering paper figures ...")
    fig01_umap_quad(coords, languages, qids, layer0, models)
    fig02_purity_bars(rows, layer0, models)
    fig03_model_fingerprint(rows, layer0)
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
