"""
Fetch Wikipedia articles for a set of conflict topics in multiple languages,
embed them, and cache both raw text and embeddings.

Topic source is selectable on the CLI:

    embed_conflicts.py excel                    # default Excel sheet
    embed_conflicts.py excel --path path/to.xlsx
    embed_conflicts.py modern                   # built-in 2022-invasion conflict list
    embed_conflicts.py topics "Topic A" "Topic B" --langs en,ru,uk

Outputs:
    raw text       -> data/Russia-Ukraine/raw/<safe_name>_<lang>_raw.json
    embedding json -> data/Russia-Ukraine/processed/<safe_name>_<lang>.json
"""

import argparse
import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from babelbias.config import DEFAULT_LANGS, LANG_MAP
from babelbias.embedding import embed
from babelbias.paths import CONFLICT_EXCEL, PROCESSED_DIR, RAW_DIR
from babelbias.wiki import fetch_with_cache, resolve_langlinks, safe_name

# 2022-invasion-era topics; previously a separate script (embed_modern_conflicts.py).
MODERN_CONFLICTS = [
    "2022 Russian theft of Ukrainian grain",
    "Bucha massacre",
    "Deportation of Ukrainian children during the Russian invasion of Ukraine",
    "International Criminal Court investigation in Ukraine",
    "Izium mass graves",
    "Makiivka surrender incident",
    "Murder of Yevgeny Nuzhin",
    "Russian filtration camps for Ukrainians",
    "Russian strikes on hospitals during the Russian invasion of Ukraine",
    "Russian torture chambers in Ukraine",
    "Sexual violence in the Russian invasion of Ukraine",
    "Torture and castration of a Ukrainian POW in Pryvillia",
    "Torture of Russian soldiers in Mala Rohan",
    "Use of incendiary weapons in the Russo-Ukrainian War",
    "War crimes in the Russian invasion of Ukraine",
]


def topics_from_excel(path) -> dict[str, list[str]]:
    """Return {conflict_name: [lang_codes]} from the source Excel sheet.
    The sheet has a sparse 'conflict' column (one row per language version)."""
    df = pd.read_excel(path)
    df["conflict"] = df["conflict"].ffill()
    out: dict[str, list[str]] = {}
    for conflict in df["conflict"].dropna().unique():
        langs = (
            df.loc[df["conflict"] == conflict, "language version"]
            .str.lower()
            .map(LANG_MAP)
            .dropna()
            .unique()
            .tolist()
        )
        out[conflict] = langs
    return out


def topics_with_default_langs(topics: list[str]) -> dict[str, list[str]]:
    return {t: list(DEFAULT_LANGS) for t in topics}


def embed_conflicts(topics_with_langs: dict[str, list[str]]) -> list[dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for conflict_name, target_langs in topics_with_langs.items():
        print(f"\nTargeting: {conflict_name}")
        slug = safe_name(conflict_name)

        available = resolve_langlinks(conflict_name)
        if available is None:
            print(f"  EN page not found — skipping.")
            continue

        for lang in target_langs:
            if lang not in available:
                print(f"  no {lang} version — skipping.")
                continue

            remote_title = available[lang]
            processed_path = PROCESSED_DIR / f"{slug}_{lang}.json"
            raw_path = RAW_DIR / f"{slug}_{lang}_raw.json"

            if processed_path.exists():
                print(f"  {lang}: cached embedding")
                with open(processed_path) as f:
                    results.append(json.load(f))
                continue

            print(f"  {lang}: fetching '{remote_title}'")
            content = fetch_with_cache(lang, remote_title, raw_path)
            if not content:
                continue

            print(f"  {lang}: embedding")
            vec = embed(content)
            if vec is None:
                continue

            record = {
                "conflict": conflict_name,
                "language": lang,
                "title": remote_title,
                "embedding": vec,
            }
            with open(processed_path, "w") as f:
                json.dump(record, f)
            results.append(record)

    return results


def print_cross_lingual_similarities(results: list[dict]) -> None:
    by_conflict: dict[str, dict[str, list[float]]] = {}
    for r in results:
        by_conflict.setdefault(r["conflict"], {})[r["language"]] = r["embedding"]

    print("\nCross-lingual cosine similarities")
    print("-" * 40)
    for name, langs in by_conflict.items():
        if len(langs) < 2:
            continue
        print(f"\n{name}")
        codes = list(langs.keys())
        sims = cosine_similarity(np.array([langs[c] for c in codes]))
        for i, j in combinations(range(len(codes)), 2):
            print(f"  {codes[i]} vs {codes[j]}: {sims[i][j]:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)

    p_excel = sub.add_parser("excel", help="Read topics + per-topic langs from the source Excel.")
    p_excel.add_argument("--path", default=str(CONFLICT_EXCEL))

    sub.add_parser("modern", help=f"Embed the built-in {len(MODERN_CONFLICTS)} 2022-invasion topics in en/ru/uk.")

    p_topics = sub.add_parser("topics", help="Embed an explicit list of topic names.")
    p_topics.add_argument("topics", nargs="+", help="Wikipedia article titles (English).")
    p_topics.add_argument("--langs", default=",".join(DEFAULT_LANGS),
                          help="Comma-separated language codes (default: en,ru,uk).")

    args = ap.parse_args()

    if args.source == "excel":
        topics = topics_from_excel(args.path)
    elif args.source == "modern":
        topics = topics_with_default_langs(MODERN_CONFLICTS)
    else:
        langs = [l.strip() for l in args.langs.split(",")]
        topics = {t: langs for t in args.topics}

    results = embed_conflicts(topics)
    print_cross_lingual_similarities(results)


if __name__ == "__main__":
    main()
