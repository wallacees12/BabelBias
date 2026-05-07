"""Cross-embedder EVoC clustering for the 8 June deck.

Re-runs the exp_016 unsupervised-clustering test on each of the 4
embedder spaces (OpenAI baseline + Alibaba + Gemini + Yandex) and
produces:

  - 04_evoc_4_embedder_lang.png  — 4-panel UMAP-by-language scatter
  - per-embedder purity table (printed + CSV)

If the (qid × language) cell structure that the OpenAI baseline
recovers (lang_purity 0.977, qid_purity 0.988, model_purity 0.255 in
exp_016) re-emerges in non-OpenAI embedding spaces, the methodology
defence gains a *second* independent piece of evidence — beyond the
cosine-magnitude finding from figure 01.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from babelbias.palette import ORDERED_MODELS
from babelbias.paths import (
    LLM_EMBEDDINGS_ALT_DIR,
    LLM_EMBEDDINGS_DIR,
    PROJECT_ROOT,
)
from cluster_explore import normalise_l2, purity

# Restrict every embedder to the same 14 cosine-eligible providers so
# OpenAI vs alt-embedder cluster purity numbers are apples-to-apples
# (the OpenAI tree happens to also contain yandexgpt response files —
# 270 refusal boilerplate embeddings that aren't in the alt-embedder set).
COSINE_ELIGIBLE_MODELS = set(ORDERED_MODELS)


sns.set_theme(
    style="whitegrid", context="talk", font_scale=0.75,
    rc={"axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333333", "savefig.dpi": 180,
        "savefig.bbox": "tight"},
)

OUT_DIR = PROJECT_ROOT / "Presentations" / "figures" / "June 8"

EMBEDDERS = [
    ("openai_te3s",  "OpenAI text-embedding-3-small  [US]",  LLM_EMBEDDINGS_DIR),
    ("alibaba_v3",   "Alibaba text-embedding-v3  [CN]",       LLM_EMBEDDINGS_ALT_DIR / "alibaba_v3"),
    ("gemini_001",   "Google gemini-embedding-001  [US]",     LLM_EMBEDDINGS_ALT_DIR / "gemini_001"),
    ("yandex_doc",   "Yandex text-search-doc  [RU]",          LLM_EMBEDDINGS_ALT_DIR / "yandex_doc"),
]

LANG_COLOR = {"en": "#11A579", "ru": "#E73F74", "uk": "#3969AC"}


def load_embeddings(embed_root: Path):
    """Load response embeddings for ru_uk_core under embed_root, restricted
    to the 14 cosine-eligible providers."""
    rows, vecs = [], []
    for f in embed_root.rglob("*.json"):
        if "/ru_uk_core/" not in str(f):
            continue
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        emb = d.get("embedding")
        if not emb:
            continue
        if d.get("model") not in COSINE_ELIGIBLE_MODELS:
            continue
        rows.append({
            "model":    d.get("model"),
            "qid":      d.get("qid"),
            "language": d.get("language"),
        })
        vecs.append(emb)
    if not vecs:
        return np.zeros((0, 0)), []
    X = np.array(vecs, dtype=np.float32)
    return X, rows


def run_for_embedder(emb_id: str, label: str, embed_root: Path):
    """Returns (coords_2d, languages, purity_record)."""
    print(f"\n=== {emb_id}: loading from {embed_root} ===")
    X, rows = load_embeddings(embed_root)
    print(f"  loaded {len(rows)} embeddings × {X.shape[1]} dims")
    if len(rows) == 0:
        return None, None, None

    Xn = normalise_l2(X)
    languages = [r["language"] for r in rows]
    models    = [r["model"]    for r in rows]
    qids      = [r["qid"]      for r in rows]

    # EVoC fit
    from evoc import EVoC
    model = EVoC(base_min_cluster_size=30, max_layers=10, random_state=42)
    cluster_labels = model.fit_predict(Xn)
    layers = getattr(model, "cluster_layers_", [cluster_labels])
    print(f"  EVoC: {len(layers)} layers")

    # Pick the layer closest to the 27 (qid × lang) cell count.
    target = 27
    chosen_layer_idx = min(
        range(len(layers)),
        key=lambda i: abs(len(set(layers[i])) - target),
    )
    chosen = list(layers[chosen_layer_idx])
    n_clu = len(set(chosen)) - (1 if -1 in chosen else 0)

    p_lang,  _ = purity(chosen, languages)
    p_qid,   _ = purity(chosen, qids)
    p_model, _ = purity(chosen, models)
    print(f"  layer {chosen_layer_idx} ({n_clu} clusters): "
          f"lang={p_lang:.3f} · qid={p_qid:.3f} · model={p_model:.3f}")

    # UMAP for visualisation
    import umap
    coords = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                       random_state=42).fit_transform(Xn)

    record = {
        "embedder": emb_id,
        "label": label,
        "n_embeddings": len(rows),
        "dims": X.shape[1],
        "n_layers": len(layers),
        "chosen_layer": chosen_layer_idx,
        "n_clusters_at_chosen": n_clu,
        "lang_purity": p_lang,
        "qid_purity": p_qid,
        "model_purity": p_model,
    }
    return coords, languages, record


def figure_4_panel_umap(panels: list[tuple[str, np.ndarray, list[str]]], out_path: Path):
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 5.4),
                             constrained_layout=True)
    for ax, (label, coords, langs) in zip(axes, panels):
        for lang, col in LANG_COLOR.items():
            mask = np.array([l == lang for l in langs])
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=4, alpha=0.55, color=col, label=lang.upper(),
                       linewidths=0)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    axes[0].legend(loc="lower left", fontsize=9, frameon=False, markerscale=3)
    fig.suptitle("Unsupervised clustering — UMAP coloured by response language across 4 embedders",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02,
             "If the language structure re-emerges in every embedder space, the (qid × language) cell partition is method-robust.",
             ha="center", fontsize=9.5, color="#444", style="italic")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = []
    records = []
    for emb_id, label, root in EMBEDDERS:
        coords, langs, rec = run_for_embedder(emb_id, label, root)
        if coords is None:
            print(f"  (skip {emb_id} — no embeddings)")
            continue
        panels.append((label, coords, langs))
        records.append(rec)

    figure_4_panel_umap(panels, OUT_DIR / "04_evoc_4_embedder_lang.png")
    print(f"\nWrote {OUT_DIR / '04_evoc_4_embedder_lang.png'}")

    # Print + save the purity table
    print("\nPer-embedder EVoC purity (at the layer closest to 27 clusters):")
    print(f"{'embedder':<14}  {'n_emb':>5} {'dims':>4} "
          f"{'layer':>5} {'n_clu':>5}  "
          f"{'lang':>6} {'qid':>6} {'model':>6}")
    print("-" * 76)
    for r in records:
        print(f"{r['embedder']:<14}  {r['n_embeddings']:>5} {r['dims']:>4} "
              f"{r['chosen_layer']:>5} {r['n_clusters_at_chosen']:>5}  "
              f"{r['lang_purity']:>6.3f} {r['qid_purity']:>6.3f} "
              f"{r['model_purity']:>6.3f}")

    csv_path = PROJECT_ROOT / "data" / "Russia-Ukraine" / "analysis" / "exp_015_evoc_purity_per_embedder.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(records[0].keys())
        for r in records:
            w.writerow(r.values())
    print(f"\nSaved purity table → {csv_path}")


if __name__ == "__main__":
    main()
