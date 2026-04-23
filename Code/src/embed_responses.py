"""
Embed LLM response texts so they sit in the same vector space as the
Wikipedia lead embeddings in data/Russia-Ukraine/processed_leads/.

Input:  data/Russia-Ukraine/llm_responses/<model>/<event>/<qid>_<lang>.json
Output: data/Russia-Ukraine/llm_embeddings/<model>/<event>/<qid>_<lang>.json
"""

import argparse
import json
import time
from pathlib import Path

from babelbias.config import EMBEDDING_MODEL
from babelbias.embedding import embed_short
from babelbias.paths import LLM_EMBEDDINGS_DIR, LLM_RESPONSES_DIR


def run(model: str, event: str, in_root: Path, out_root: Path):
    in_dir = in_root / model / event
    out_dir = out_root / model / event
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        raise SystemExit(f"No responses at {in_dir}. Run prompt_llms.py first.")

    files = sorted(p for p in in_dir.iterdir() if p.suffix == ".json")
    print(f"Embedding {len(files)} responses from {in_dir}...")

    made = skipped = 0
    for src in files:
        out_path = out_dir / src.name
        if out_path.exists():
            skipped += 1
            continue

        with open(src) as f:
            rec = json.load(f)

        vec = embed_short(rec.get("response_text", ""))
        if vec is None:
            print(f"  skip (empty response): {src.name}")
            continue

        with open(out_path, "w") as f:
            json.dump({
                "event": rec["event"],
                "model": rec["model"],
                "qid": rec["qid"],
                "theme": rec.get("theme"),
                "language": rec["language"],
                "embedding_model": EMBEDDING_MODEL,
                "embedding": vec,
                "type": "llm_response",
            }, f)

        made += 1
        time.sleep(0.05)

    print(f"Done. {made} new embeddings, {skipped} already existed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--event", default="ru_uk_core")
    ap.add_argument("--in-root", type=Path, default=LLM_RESPONSES_DIR)
    ap.add_argument("--out-root", type=Path, default=LLM_EMBEDDINGS_DIR)
    args = ap.parse_args()
    run(args.model, args.event, args.in_root, args.out_root)


if __name__ == "__main__":
    main()
