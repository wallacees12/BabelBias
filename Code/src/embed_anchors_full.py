"""
Embed the *full text* (not lead-only) of every Wikipedia anchor article
referenced by analyze_bias.py's ANCHOR_SLUGS.

Reads the cached raw article JSON from data/Russia-Ukraine/raw/ and writes
chunked-and-averaged embeddings to data/Russia-Ukraine/processed/. This is
the full-page counterpart to processed_leads/, used for the lead-vs-full
sensitivity check (responding to Dr Urman's 27 April note on lead-section
restriction).

Two of the six anchor slugs (the two Crimea-related ones) were already
embedded as full pages by embed_conflicts.py from the source Excel sheet.
This script fills in the other four:

    Little_green_men · Revolution_of_Dignity ·
    Malaysia_Airlines_Flight_17 · Stepan_Bandera
"""

import json

from babelbias.config import DEFAULT_LANGS
from babelbias.embedding import embed
from babelbias.paths import PROCESSED_DIR, RAW_DIR

ANCHOR_SLUGS = [
    "Little_green_men",
    "2014_Russian_annexation_of_Crimea",
    "Revolution_of_Dignity",
    "2014_Crimean_status_referendum",
    "Malaysia_Airlines_Flight_17",
    "Stepan_Bandera",
]


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    embedded = skipped = missing = 0
    for slug in ANCHOR_SLUGS:
        for lang in DEFAULT_LANGS:
            out_path = PROCESSED_DIR / f"{slug}_{lang}.json"
            if out_path.exists():
                skipped += 1
                continue
            raw_path = RAW_DIR / f"{slug}_{lang}_raw.json"
            if not raw_path.exists():
                print(f"  missing raw: {raw_path.name}")
                missing += 1
                continue
            data = json.loads(raw_path.read_text())
            content = data.get("content", "")
            if not content:
                print(f"  empty content: {raw_path.name}")
                missing += 1
                continue
            print(f"  embedding {slug} {lang} ({len(content)} chars)")
            vec = embed(content)
            if vec is None:
                print(f"  FAIL")
                continue
            record = {
                "conflict": slug.replace("_", " "),
                "language": lang,
                "title": data.get("title", slug),
                "embedding": vec,
            }
            out_path.write_text(json.dumps(record))
            embedded += 1

    print(f"\nDone. embedded={embedded}  skipped={skipped}  missing_raw={missing}")


if __name__ == "__main__":
    main()
