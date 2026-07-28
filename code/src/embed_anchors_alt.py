"""Embed Wikipedia lead-section anchors for exp_015 with an alternative embedder.

Re-extracts the lead from `data/.../raw/<slug>_<lang>_raw.json` and embeds
it with the chosen alt embedder. Only the 6 unique anchor slugs used by
the `ru_uk_core` 3×3 cosine analysis are processed (q01–q09 collapse to
6 unique Wikipedia topics — see ANCHOR_SLUGS in analyze_bias.py).

Output: `data/.../processed_leads_alt/<embedder>/<slug>_<lang>.json`

Used by `analyze_bias.py --wiki-root <alt>` to recompute the per-embedder
3×3 cosine matrix.
"""

import argparse
import json
import time
from pathlib import Path

from babelbias.embedding import (
    ALL_EMBEDDERS,
    embed_short_alt,
    embedder_label,
    embedder_out_dir,
)
from babelbias.paths import PROCESSED_LEADS_DIR, RAW_DIR
from babelbias.wiki import extract_lead


# 6 unique anchor slugs used by ru_uk_core (q06, q07, q08 share the Crimea slug).
ANCHOR_SLUGS = (
    "Little_green_men",
    "2014_Russian_annexation_of_Crimea",
    "Revolution_of_Dignity",
    "2014_Crimean_status_referendum",
    "Malaysia_Airlines_Flight_17",
    "Stepan_Bandera",
)
LANGS = ("en", "ru", "uk")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="openai_te3s",
                    choices=list(ALL_EMBEDDERS),
                    help="Embedding model to use.")
    ap.add_argument("--out-root", type=Path, default=PROCESSED_LEADS_DIR,
                    help="Base output root. Alt embedders route to "
                         "{out_root}_alt/{embedder}/ automatically.")
    args = ap.parse_args()

    routed = embedder_out_dir(args.out_root, args.embedder)
    routed.mkdir(parents=True, exist_ok=True)

    label = embedder_label(args.embedder)
    print(f"Embedding {len(ANCHOR_SLUGS)} slugs × {len(LANGS)} langs "
          f"= {len(ANCHOR_SLUGS) * len(LANGS)} anchors")
    print(f"  embedder: {args.embedder} ({label})")
    print(f"  output:   {routed}")

    made = skipped = missing = 0
    for slug in ANCHOR_SLUGS:
        for lang in LANGS:
            out_path = routed / f"{slug}_{lang}.json"
            if out_path.exists():
                skipped += 1
                continue

            raw_path = RAW_DIR / f"{slug}_{lang}_raw.json"
            if not raw_path.exists():
                print(f"  MISSING: {raw_path.name}")
                missing += 1
                continue

            with open(raw_path) as f:
                data = json.load(f)

            lead = extract_lead(data.get("content", ""))
            if not lead:
                print(f"  no lead content: {raw_path.name}")
                continue

            vec = embed_short_alt(lead, args.embedder)
            if vec is None:
                continue

            with open(out_path, "w") as f:
                json.dump({
                    "conflict": slug.replace("_", " "),
                    "language": lang,
                    "title": data.get("title", slug.replace("_", " ")),
                    "embedding_model": label,
                    "embedding": vec,
                    "type": "conflict",
                }, f)

            made += 1
            if args.embedder in ("openai_te3s", "cohere_ml_v3"):
                time.sleep(0.05)

    print(f"Done. {made} new · {skipped} existed · {missing} missing")


if __name__ == "__main__":
    main()
