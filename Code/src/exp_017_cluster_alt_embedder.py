"""
exp_017 Phase B — re-run the exp_016 EVoC + Wikipedia-anchor projection
under an alternative embedder, to test whether the
"UK responses cluster with the Russian anchor at 82-87%" finding
(exp_016) survives a tokenizer change.

Mirrors `cluster_with_wiki_anchors.py` (exp_016) but is parametrised
on an `--embedder` flag so we can swap the embedding source without
touching the original (reproducibility).

For openai_te3s, anchor source is `processed_leads/` (lead-only) — NOT
the full-page `processed/` exp_016 used. Both embedders are then
compared on the same lead-only anchor content, isolating the tokenizer
variable.

Usage:
    python -m exp_017_cluster_alt_embedder --embedder openai_te3s
    python -m exp_017_cluster_alt_embedder --embedder alibaba_v3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from babelbias.embedding import ALL_EMBEDDERS
from babelbias.paths import (
    LLM_EMBEDDINGS_ALT_DIR,
    LLM_EMBEDDINGS_DIR,
    PROCESSED_LEADS_ALT_DIR,
    PROCESSED_LEADS_DIR,
    PROJECT_ROOT,
)


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
LANGS = ("en", "ru", "uk")
BASE_OUT_DIR = PROJECT_ROOT / "Presentations" / "figures" / "exp_017"


def resolve_paths(embedder: str) -> tuple[Path, Path]:
    """Return (responses_root, anchors_root) for the given embedder."""
    if embedder == "openai_te3s":
        return LLM_EMBEDDINGS_DIR, PROCESSED_LEADS_DIR
    if embedder in ALL_EMBEDDERS:
        return (
            LLM_EMBEDDINGS_ALT_DIR / embedder,
            PROCESSED_LEADS_ALT_DIR / embedder,
        )
    raise ValueError(f"Unknown embedder: {embedder}. Expected one of {ALL_EMBEDDERS}")


def load_response_embeddings(root: Path) -> tuple[list[dict], list[list[float]]]:
    rows, vecs = [], []
    for f in root.rglob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not d.get("embedding"):
            continue
        rows.append({
            "kind":     "response",
            "model":    d.get("model"),
            "qid":      d.get("qid"),
            "language": d.get("language"),
        })
        vecs.append(d["embedding"])
    return rows, vecs


def load_anchor_embeddings(anchors_root: Path) -> tuple[list[dict], list[list[float]]]:
    """One unique anchor per (slug × lang) — deduped across qids."""
    seen_slugs: set[str] = set()
    rows, vecs = [], []
    for qid, slug in ANCHOR_SLUGS.items():
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        for lang in LANGS:
            path = anchors_root / f"{slug}_{lang}.json"
            if not path.exists():
                print(f"  WARNING: missing anchor {path.name}")
                continue
            d = json.loads(path.read_text())
            rows.append({
                "kind":     "anchor",
                "model":    "wikipedia",
                "qid":      qid,
                "slug":     slug,
                "language": lang,
            })
            vecs.append(d["embedding"])
    return rows, vecs


def normalise_l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def co_clustering_fractions(labels: list[int], rows: list[dict]) -> list[str]:
    anchor_cluster: dict[tuple[str, str], int] = {}
    for c, r in zip(labels, rows):
        if r["kind"] == "anchor":
            anchor_cluster[(r["slug"], r["language"])] = int(c)

    resp_clusters: dict[tuple[str, str], list[int]] = defaultdict(list)
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
        slug = ANCHOR_SLUGS.get(qid)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedder", required=True, choices=list(ALL_EMBEDDERS))
    p.add_argument("--base-min-cluster-size", type=int, default=200,
                   help="EVoC bmcs (matches exp_016 protocol)")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = BASE_OUT_DIR / args.embedder / f"bmcs{args.base_min_cluster_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_root, anchors_root = resolve_paths(args.embedder)
    print(f"Embedder:        {args.embedder}")
    print(f"Responses root:  {responses_root}")
    print(f"Anchors root:    {anchors_root}")
    print(f"Output dir:      {out_dir}")

    print("\nLoading responses ...")
    r_rows, r_vecs = load_response_embeddings(responses_root)
    print(f"  {len(r_rows)} response embeddings")
    if not r_rows:
        raise SystemExit(f"No responses under {responses_root}")

    print("Loading Wikipedia anchors (lead-only) ...")
    a_rows, a_vecs = load_anchor_embeddings(anchors_root)
    n_slugs = len({r["slug"] for r in a_rows})
    print(f"  {len(a_rows)} anchor embeddings ({n_slugs} unique slugs × {len(LANGS)} langs)")

    rows = r_rows + a_rows
    X = np.array(r_vecs + a_vecs, dtype=np.float32)
    Xn = normalise_l2(X)
    print(f"\nCombined matrix: {Xn.shape}")

    print(f"\nFitting EVoC (bmcs={args.base_min_cluster_size}, "
          f"random_state={args.random_state}) ...")
    from evoc import EVoC
    model = EVoC(base_min_cluster_size=args.base_min_cluster_size,
                 max_layers=10, random_state=args.random_state)
    model.fit_predict(Xn)
    layers = getattr(model, "cluster_layers_", None)
    if layers is None:
        raise SystemExit("EVoC did not expose cluster_layers_; check version.")
    deepest = list(layers[-1])
    n_clusters = len({c for c in deepest if c != -1})
    print(f"EVoC produced {len(layers)} layers; deepest has {n_clusters} clusters.")

    print("\nAnchor cluster assignments:")
    print(f"{'slug':<40} {'lang':>4} {'cluster':>8}")
    print("-" * 56)
    anchor_csv = ["slug,language,cluster"]
    for c, r in zip(deepest, rows):
        if r["kind"] == "anchor":
            print(f"{r['slug']:<40} {r['language']:>4} {c:>8}")
            anchor_csv.append(f"{r['slug']},{r['language']},{c}")
    (out_dir / "anchor_cluster_assignments.csv").write_text("\n".join(anchor_csv))

    print("\nCo-clustering: per (qid, response_lang), fraction of "
          "responses sharing a cluster with each anchor language")
    co_csv = co_clustering_fractions(deepest, rows)
    (out_dir / "co_clustering_fractions.csv").write_text("\n".join(co_csv))

    by_cluster: dict[int, dict] = defaultdict(
        lambda: {"en": 0, "ru": 0, "uk": 0, "anchor_slugs_langs": []})
    for c, r in zip(deepest, rows):
        if c == -1:
            continue
        if r["kind"] == "response":
            by_cluster[c][r["language"]] += 1
        else:
            by_cluster[c]["anchor_slugs_langs"].append(
                f"{r['slug']}/{r['language']}")
    cmix_csv = ["cluster,en,ru,uk,anchor_count,anchors"]
    print("\nCluster composition (responses + anchors):")
    print(f"{'cl':>4} {'en':>4} {'ru':>4} {'uk':>4}  anchors")
    print("-" * 80)
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
