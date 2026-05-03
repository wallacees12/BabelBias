"""
EVoC fit with Wikipedia anchors injected (exp_016 follow-up).

Loads:
- All 4,049 response embeddings from llm_embeddings/.
- 18 Wikipedia anchor embeddings (6 unique articles × 3 languages, slugs
  copied from analyze_bias.ANCHOR_SLUGS — q02/q06/q07/q08 share the
  Crimea-annexation article so we use 6 not 9 articles).

Fits EVoC at the requested base_min_cluster_size and reports:
- Anchor cluster assignments (which cluster each Wikipedia article fell into)
- Per (qid, response_lang) cell: fraction of responses sharing a cluster
  with the same-language anchor, vs the en/ru/uk anchors of that qid.

The headline question this answers: do model responses cluster *with*
their own-language Wikipedia article, or do they form their own
provider regions of embedding space?

Output: Presentations/figures/exp_016/with_anchors/{...}
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from babelbias.paths import LLM_EMBEDDINGS_DIR, PROCESSED_DIR, PROJECT_ROOT


# Mirrors analyze_bias.ANCHOR_SLUGS — kept inline so this script doesn't
# import from analyze_bias (which has heavier deps).
ANCHOR_SLUGS = {
    "q01_little_green_men": "Little_green_men",
    "q02_crimea_2014":      "2014_Russian_annexation_of_Crimea",
    "q03_maidan_revolution":"Revolution_of_Dignity",
    "q04_referendum":       "2014_Crimean_status_referendum",
    "q05_mh17":             "Malaysia_Airlines_Flight_17",
    "q06_crimea_belongs":   "2014_Russian_annexation_of_Crimea",
    "q07_pov_russia":       "2014_Russian_annexation_of_Crimea",
    "q08_pov_ukraine":      "2014_Russian_annexation_of_Crimea",
    "q09_bandera":          "Stepan_Bandera",
}
LANGS = ("en", "ru", "uk")
BASE_OUT_DIR = PROJECT_ROOT / "Presentations" / "figures" / "exp_016"


def load_response_embeddings():
    rows, vecs = [], []
    for f in LLM_EMBEDDINGS_DIR.rglob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not d.get("embedding"):
            continue
        rows.append({
            "kind": "response",
            "model": d.get("model"),
            "qid": d.get("qid"),
            "language": d.get("language"),
        })
        vecs.append(d["embedding"])
    return rows, vecs


def load_anchor_embeddings():
    """One unique anchor per (slug × lang) — deduped across qids."""
    seen_slugs = set()
    rows, vecs = [], []
    for qid, slug in ANCHOR_SLUGS.items():
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        for lang in LANGS:
            path = PROCESSED_DIR / f"{slug}_{lang}.json"
            d = json.loads(path.read_text())
            rows.append({
                "kind": "anchor",
                "model": "wikipedia",
                "qid": qid,            # representative qid for this slug
                "slug": slug,
                "language": lang,
            })
            vecs.append(d["embedding"])
    return rows, vecs


def normalise_l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def co_clustering_fractions(labels, rows):
    """For each (qid, response_lang) cell, compute the fraction of
    responses that share a cluster with each anchor language for that
    qid's slug.

    We use the qid's slug to identify the anchor — q02/q06/q07/q08 all
    point at the Crimea-annexation slug, so they share the same 3
    anchors.
    """
    qid_to_slug = ANCHOR_SLUGS

    # Anchor cluster lookup keyed by (slug, lang)
    anchor_cluster = {}
    for c, r in zip(labels, rows):
        if r["kind"] == "anchor":
            anchor_cluster[(r["slug"], r["language"])] = int(c)

    # Group responses by (qid, response_lang)
    resp_clusters = defaultdict(list)
    for c, r in zip(labels, rows):
        if r["kind"] == "response":
            resp_clusters[(r["qid"], r["language"])].append(int(c))

    out_rows = ["qid,response_lang,n,anchor_cluster_en,anchor_cluster_ru,"
                "anchor_cluster_uk,share_with_en_anchor,share_with_ru_anchor,"
                "share_with_uk_anchor,share_with_own_lang_anchor"]
    print(f"\n{'qid':<22} {'r_lang':>6} {'n':>4} "
          f"{'aclEN':>6} {'aclRU':>6} {'aclUK':>6}  "
          f"{'%EN':>5} {'%RU':>5} {'%UK':>5}  {'%own':>5}")
    print("-" * 88)
    for qid in sorted({r["qid"] for r in rows if r["kind"] == "response"}):
        slug = qid_to_slug.get(qid)
        if slug is None:
            continue
        a_en = anchor_cluster.get((slug, "en"), -99)
        a_ru = anchor_cluster.get((slug, "ru"), -99)
        a_uk = anchor_cluster.get((slug, "uk"), -99)
        for r_lang in LANGS:
            members = resp_clusters.get((qid, r_lang), [])
            n = len(members)
            if n == 0:
                continue
            ctr = Counter(members)
            f_en = ctr.get(a_en, 0) / n
            f_ru = ctr.get(a_ru, 0) / n
            f_uk = ctr.get(a_uk, 0) / n
            f_own = {"en": f_en, "ru": f_ru, "uk": f_uk}[r_lang]
            print(f"{qid:<22} {r_lang:>6} {n:>4} "
                  f"{a_en:>6} {a_ru:>6} {a_uk:>6}  "
                  f"{f_en*100:>4.0f}% {f_ru*100:>4.0f}% {f_uk*100:>4.0f}%  "
                  f"{f_own*100:>4.0f}%")
            out_rows.append(
                f"{qid},{r_lang},{n},{a_en},{a_ru},{a_uk},"
                f"{f_en:.4f},{f_ru:.4f},{f_uk:.4f},{f_own:.4f}"
            )
    return out_rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-min-cluster-size", type=int, default=200)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-suffix", type=str, default="with_anchors")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = BASE_OUT_DIR / args.out_suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    print("Loading responses ...")
    r_rows, r_vecs = load_response_embeddings()
    print(f"  {len(r_rows)} response embeddings")

    print("Loading Wikipedia anchors ...")
    a_rows, a_vecs = load_anchor_embeddings()
    print(f"  {len(a_rows)} anchor embeddings "
          f"({len({r['slug'] for r in a_rows})} unique articles × "
          f"{len(LANGS)} langs)")

    rows = r_rows + a_rows
    X = np.array(r_vecs + a_vecs, dtype=np.float32)
    Xn = normalise_l2(X)
    print(f"\nCombined matrix: {Xn.shape}")

    print(f"\nFitting EVoC (bmcs={args.base_min_cluster_size}, "
          f"random_state={args.random_state}) ...")
    from evoc import EVoC
    model = EVoC(base_min_cluster_size=args.base_min_cluster_size,
                 max_layers=10, random_state=args.random_state)
    cluster_labels = model.fit_predict(Xn)
    layers = getattr(model, "cluster_layers_", [cluster_labels])
    deepest = list(layers[-1])
    print(f"EVoC produced {len(layers)} layers; using deepest "
          f"({len(set(c for c in deepest if c != -1))} clusters).")

    # Anchor-only summary
    print("\nAnchor cluster assignments:")
    print(f"{'slug':<40} {'lang':>4} {'cluster':>8}")
    print("-" * 56)
    anchor_csv = ["slug,language,cluster"]
    for c, r in zip(deepest, rows):
        if r["kind"] == "anchor":
            print(f"{r['slug']:<40} {r['language']:>4} {c:>8}")
            anchor_csv.append(f"{r['slug']},{r['language']},{c}")
    (out_dir / "anchor_cluster_assignments.csv").write_text("\n".join(anchor_csv))

    # Co-clustering fractions
    print("\nCo-clustering: per (qid, response_lang), fraction of "
          "responses sharing a cluster with each anchor language")
    co_csv = co_clustering_fractions(deepest, rows)
    (out_dir / "co_clustering_fractions.csv").write_text("\n".join(co_csv))

    # Cluster-mix breakdown including anchors (so anchor positions are visible)
    by_cluster = defaultdict(lambda: {"en": 0, "ru": 0, "uk": 0,
                                       "anchor_slugs_langs": []})
    for c, r in zip(deepest, rows):
        if c == -1:
            continue
        if r["kind"] == "response":
            by_cluster[c][r["language"]] += 1
        else:
            by_cluster[c]["anchor_slugs_langs"].append(
                f"{r['slug']}/{r['language']}")
    print("\nCluster composition (responses + anchors):")
    print(f"{'cl':>4} {'en':>4} {'ru':>4} {'uk':>4}  anchors")
    print("-" * 80)
    cmix_csv = ["cluster,en,ru,uk,anchor_count,anchors"]
    for c in sorted(by_cluster.keys()):
        d = by_cluster[c]
        anchors_str = "; ".join(d["anchor_slugs_langs"]) or "—"
        print(f"{c:>4} {d['en']:>4} {d['ru']:>4} {d['uk']:>4}  {anchors_str}")
        cmix_csv.append(f"{c},{d['en']},{d['ru']},{d['uk']},"
                        f"{len(d['anchor_slugs_langs'])},"
                        f"\"{anchors_str}\"")
    (out_dir / "cluster_composition_with_anchors.csv").write_text(
        "\n".join(cmix_csv))

    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
