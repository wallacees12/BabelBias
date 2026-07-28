"""Embed neutral-control Wikipedia leads with an alternative embedder
(exp_015 cross-embedder debiasing). Reads CONTROL_*_raw.json files from
data/.../raw/ and writes per-embedder output to
data/.../processed_leads_alt/<embedder>/CONTROL_*.json.

Used to estimate the language axis from controls under each alt
embedder, so analyze_bias.py --debias can produce a debiased 3×3
cosine matrix per embedder.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="openai_te3s",
                    choices=list(ALL_EMBEDDERS))
    ap.add_argument("--out-root", type=Path, default=PROCESSED_LEADS_DIR)
    args = ap.parse_args()

    routed = embedder_out_dir(args.out_root, args.embedder)
    routed.mkdir(parents=True, exist_ok=True)
    label = embedder_label(args.embedder)

    files = sorted(p for p in RAW_DIR.iterdir()
                   if p.name.startswith("CONTROL_") and p.name.endswith("_raw.json"))
    print(f"Embedding {len(files)} control leads")
    print(f"  embedder: {args.embedder} ({label})")
    print(f"  output:   {routed}")

    made = skipped = empty = errors = 0
    for src in files:
        out_name = src.name.replace("_raw.json", ".json")
        out_path = routed / out_name
        if out_path.exists():
            skipped += 1
            continue

        with open(src) as f:
            data = json.load(f)
        lead = extract_lead(data.get("content", ""))
        if not lead:
            empty += 1
            continue

        try:
            vec = embed_short_alt(lead, args.embedder)
        except Exception as e:
            print(f"  ERROR ({type(e).__name__}): {src.name} — {e}")
            errors += 1
            continue
        if vec is None:
            empty += 1
            continue

        # Parse "CONTROL_<topic>_<lang>_raw.json" → topic + lang
        stem = src.name.removeprefix("CONTROL_").removesuffix("_raw.json")
        parts = stem.split("_")
        lang = parts[-1]
        topic = " ".join(parts[:-1])

        with open(out_path, "w") as f:
            json.dump({
                "conflict": topic,
                "language": lang,
                "title": data.get("title", topic),
                "embedding_model": label,
                "embedding": vec,
                "type": "control",
            }, f)

        made += 1
        if args.embedder in ("openai_te3s", "cohere_ml_v3"):
            time.sleep(0.05)

    print(f"Done. {made} new · {skipped} existed · {empty} empty/no-lead · {errors} errors")


if __name__ == "__main__":
    main()
