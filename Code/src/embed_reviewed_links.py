"""
Embed the conflicts that survived manual review in reviewed_links.csv.
Only topics with EN+RU+UK Wikipedia versions are kept.
"""

import json

import pandas as pd

from babelbias.config import DEFAULT_LANGS
from babelbias.embedding import embed
from babelbias.paths import PROCESSED_DIR, RAW_DIR, REVIEWED_LINKS_CSV
from babelbias.wiki import fetch_with_cache, resolve_langlinks, safe_name


def process_reviewed_links():
    if not REVIEWED_LINKS_CSV.exists():
        print(f"Reviewed links CSV not found at {REVIEWED_LINKS_CSV}")
        return

    df = pd.read_csv(REVIEWED_LINKS_CSV)
    to_embed = df[(df["keep"] == True) & (df["reviewed"] == True)]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Embedding {len(to_embed)} reviewed conflicts...")

    for _, row in to_embed.iterrows():
        conflict_name = row["title"]
        print(f"\nTargeting: {conflict_name}")
        slug = safe_name(conflict_name, max_len=60)

        available = resolve_langlinks(conflict_name)
        if available is None:
            print(f"  EN page not found — skipping")
            continue

        if not all(c in available for c in DEFAULT_LANGS):
            print(f"  missing one of {DEFAULT_LANGS} — skipping")
            continue

        for lang in DEFAULT_LANGS:
            remote_title = available[lang]
            processed_path = PROCESSED_DIR / f"{slug}_{lang}.json"
            raw_path = RAW_DIR / f"{slug}_{lang}_raw.json"

            if processed_path.exists():
                print(f"  {lang}: cached embedding")
                continue

            print(f"  {lang}: fetching '{remote_title}'")
            content = fetch_with_cache(lang, remote_title, raw_path)
            if not content:
                continue

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


if __name__ == "__main__":
    process_reviewed_links()
