"""
EVoC-based cluster exploration on the LLM-response embeddings (exp_016).

Loads every response embedding from
`data/Russia-Ukraine/llm_embeddings/<model>/ru_uk_core/*.json`, runs EVoC
to get a multi-resolution cluster hierarchy, projects to 2D with UMAP for
visualisation, and reports per-layer purity against three candidate
labellings: language, model, question (qid).

Outputs to `Presentations/figures/exp_016/`:
- `umap_by_<labelling>.png`     — 2D scatter coloured by language / model / qid
- `umap_by_evoc_layer<i>.png`   — 2D scatter coloured by EVoC cluster at layer i
- `purity_by_layer.csv`         — table of (layer, label, mean_purity, n_clusters)
- `evoc_summary.txt`            — quick-look numbers + interpretation hints
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from babelbias.paths import LLM_EMBEDDINGS_DIR, PROJECT_ROOT

EMBED_ROOT = LLM_EMBEDDINGS_DIR
BASE_OUT_DIR = PROJECT_ROOT / "Presentations" / "figures" / "exp_016"


def load_embeddings():
    rows = []
    vecs = []
    for f in EMBED_ROOT.rglob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        emb = d.get("embedding")
        if not emb:
            continue
        rows.append({
            "model": d.get("model"),
            "qid": d.get("qid"),
            "theme": d.get("theme"),
            "language": d.get("language"),
            "type": d.get("type", "response"),
        })
        vecs.append(emb)
    X = np.array(vecs, dtype=np.float32)
    print(f"Loaded {len(rows)} embeddings × {X.shape[1]} dims "
          f"from {len(set(r['model'] for r in rows))} models")
    return X, rows


def normalise_l2(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def purity(cluster_labels, true_labels):
    """Mean per-cluster majority-class fraction."""
    by_cluster = defaultdict(list)
    for c, t in zip(cluster_labels, true_labels):
        if c == -1:  # noise points (HDBSCAN convention)
            continue
        by_cluster[c].append(t)
    if not by_cluster:
        return 0.0, 0
    purities = []
    for members in by_cluster.values():
        if not members:
            continue
        most_common = Counter(members).most_common(1)[0][1]
        purities.append(most_common / len(members))
    return float(np.mean(purities)), len(by_cluster)


def render_scatter(coords2d, labels, title, out_path, palette=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    unique = sorted(set(labels), key=lambda x: (str(type(x)), str(x)))
    cmap = plt.get_cmap("tab20" if len(unique) <= 20 else "nipy_spectral")
    colors = palette or [cmap(i / max(1, len(unique) - 1)) for i in range(len(unique))]
    for i, u in enumerate(unique):
        mask = np.array([l == u for l in labels])
        ax.scatter(coords2d[mask, 0], coords2d[mask, 1],
                   s=6, alpha=0.6, label=str(u), color=colors[i])
    ax.set_title(title)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    leg = ax.legend(fontsize=7, markerscale=2, loc="best", frameon=True)
    if len(unique) > 25:
        leg.remove()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def layer2_cross_lingual_breakdown(layer, rows, out_path):
    """For each cluster on a given layer, print qid-dominant + (en/ru/uk) mix.

    This is the table that revealed q09-Bandera-fully-cross-lingual /
    q02–q05-EN-vs-RU+UK divergence in the 5-model exp_016 Layer 2.
    """
    languages = [r["language"] for r in rows]
    qids = [r["qid"] for r in rows]

    by_cluster = defaultdict(list)
    for c, lang, qid in zip(layer, languages, qids):
        if c == -1:
            continue
        by_cluster[c].append((lang, qid))

    rows_out = []
    rows_out.append("cluster,size,en,ru,uk,dominant_qid,dominant_qid_share")
    print(f"\nLayer cross-lingual breakdown ({len(by_cluster)} clusters):")
    print(f"{'cl':>4} {'n':>5} {'en':>4} {'ru':>4} {'uk':>4}  "
          f"{'top_qid':<12} {'qid_share':>9}")
    print("-" * 56)
    for cl in sorted(by_cluster.keys()):
        members = by_cluster[cl]
        n = len(members)
        c_lang = Counter(l for l, _ in members)
        c_qid = Counter(q for _, q in members)
        top_qid, top_qid_n = c_qid.most_common(1)[0]
        share = top_qid_n / n
        en, ru, uk = c_lang.get("en", 0), c_lang.get("ru", 0), c_lang.get("uk", 0)
        print(f"{cl:>4} {n:>5} {en:>4} {ru:>4} {uk:>4}  "
              f"{top_qid:<12} {share:>9.2f}")
        rows_out.append(f"{cl},{n},{en},{ru},{uk},{top_qid},{share:.4f}")

    out_path.write_text("\n".join(rows_out))
    print(f"  → wrote {out_path.name}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-min-cluster-size", type=int, default=30,
                   help="EVoC base_min_cluster_size (default 30). "
                        "Larger values force a deeper hierarchy.")
    p.add_argument("--out-suffix", type=str, default=None,
                   help="Suffix appended to the output dir, e.g. 'bmcs60' "
                        "writes to Presentations/figures/exp_016/bmcs60/. "
                        "Default = no suffix (writes to exp_016/ root).")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = BASE_OUT_DIR / args.out_suffix if args.out_suffix else BASE_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    X, rows = load_embeddings()
    if len(rows) == 0:
        print("No embeddings found; aborting.")
        return

    # L2-normalise for cosine-flavoured distance
    Xn = normalise_l2(X)

    # ---- EVoC fit ----
    # Larger base_min_cluster_size forces a deeper hierarchy with coarser top
    # layers — defaults gave us only 2 near-identical fine layers.
    print(f"\nFitting EVoC (base_min_cluster_size={args.base_min_cluster_size}, "
          f"random_state={args.random_state}) ...")
    from evoc import EVoC
    model = EVoC(base_min_cluster_size=args.base_min_cluster_size,
                 max_layers=10, random_state=args.random_state)
    cluster_labels = model.fit_predict(Xn)
    layers = getattr(model, "cluster_layers_", None)
    if layers is None:
        layers = [cluster_labels]
    print(f"EVoC produced {len(layers)} hierarchy layers.")
    for i, layer in enumerate(layers):
        n_clusters = len(set(layer)) - (1 if -1 in layer else 0)
        n_noise = int(np.sum(np.array(layer) == -1))
        print(f"  layer {i}: {n_clusters} clusters · {n_noise} noise pts")

    # ---- UMAP for visualisation only ----
    print("\nFitting UMAP for 2D visualisation ...")
    import umap
    coords = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                       random_state=42).fit_transform(Xn)

    languages = [r["language"] for r in rows]
    models = [r["model"] for r in rows]
    qids = [r["qid"] for r in rows]

    # ---- Render per-labelling scatters ----
    print(f"\nRendering scatters to {out_dir} ...")
    render_scatter(coords, languages, "UMAP — coloured by response language",
                   out_dir / "umap_by_language.png")
    render_scatter(coords, models, "UMAP — coloured by model",
                   out_dir / "umap_by_model.png")
    render_scatter(coords, qids, "UMAP — coloured by question (qid)",
                   out_dir / "umap_by_qid.png")

    for i, layer in enumerate(layers):
        render_scatter(coords, list(layer),
                       f"UMAP — coloured by EVoC layer {i} "
                       f"({len(set(layer))} clusters)",
                       out_dir / f"umap_by_evoc_layer{i}.png")

    # ---- Purity table ----
    print("\nPurity per (layer × labelling):")
    print(f"{'layer':>6} {'n_clusters':>11} "
          f"{'lang_purity':>12} {'model_purity':>13} {'qid_purity':>11}")
    print("-" * 60)
    purity_csv = ["layer,n_clusters,n_noise,lang_purity,model_purity,qid_purity"]
    for i, layer in enumerate(layers):
        labels_arr = np.array(layer)
        n_noise = int(np.sum(labels_arr == -1))
        p_lang, n_clu = purity(layer, languages)
        p_model, _ = purity(layer, models)
        p_qid, _ = purity(layer, qids)
        print(f"{i:>6} {n_clu:>11} {p_lang:>12.3f} {p_model:>13.3f} {p_qid:>11.3f}")
        purity_csv.append(f"{i},{n_clu},{n_noise},{p_lang:.4f},{p_model:.4f},{p_qid:.4f}")

    (out_dir / "purity_by_layer.csv").write_text("\n".join(purity_csv))

    # ---- Summary ----
    summary_lines = [
        f"exp_016 EVoC cluster exploration",
        f"=================================",
        f"",
        f"Input: {len(rows)} response embeddings · "
        f"{len(set(models))} models · {len(set(languages))} languages · "
        f"{len(set(qids))} questions",
        f"",
        f"EVoC layers: {len(layers)}",
    ]
    for i, layer in enumerate(layers):
        n_clu = len(set(layer)) - (1 if -1 in layer else 0)
        n_noise = int(np.sum(np.array(layer) == -1))
        p_lang, _ = purity(layer, languages)
        p_model, _ = purity(layer, models)
        p_qid, _ = purity(layer, qids)
        # Heuristic interpretation
        dominant = max([("language", p_lang), ("model", p_model), ("qid", p_qid)],
                       key=lambda x: x[1])
        summary_lines.append(
            f"  layer {i}: {n_clu} clusters, {n_noise} noise · "
            f"dominant axis = {dominant[0]} (purity {dominant[1]:.3f})"
        )
    summary_lines.append("")
    summary_lines.append(
        "Interpretation: layer N is dominated by axis A if A's purity is highest.\n"
        "Expected pattern (if cross-lingual divergence is real):\n"
        "  coarse layers → language is dominant\n"
        "  finer layers  → qid (question) becomes dominant within each language\n"
        "  finest layers → model becomes a sub-axis (per-model stylistic clusters)\n"
        "If model purity is dominant at any layer, that's a finding: model identity\n"
        "is a stronger structural signal in the embeddings than language or topic."
    )
    (out_dir / "evoc_summary.txt").write_text("\n".join(summary_lines))

    # ---- Cross-lingual breakdown for every layer ----
    # 5-model exp_016 found at the coarsest layer that q09 Bandera came out
    # fully cross-lingual while q02–q05 split EN off from RU+UK. We write
    # one breakdown per layer so the seed-stability aggregator has a stable
    # filename to read regardless of how many layers EVoC produced.
    for i, layer in enumerate(layers):
        layer2_cross_lingual_breakdown(
            list(layer), rows,
            out_dir / f"layer{i}_cross_lingual_breakdown.csv",
        )

    print(f"\nWrote summary + figures to {out_dir}")


if __name__ == "__main__":
    main()
