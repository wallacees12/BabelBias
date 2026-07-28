"""
Embed LLM response texts so they sit in the same vector space as the
Wikipedia lead embeddings.

Input:  data/Russia-Ukraine/llm_responses/<model>/<event>/<qid>_<lang>.json
Output (default OpenAI):
        data/Russia-Ukraine/llm_embeddings/<model>/<event>/<qid>_<lang>.json
Output (alt embedders, e.g. --embedder bge_m3):
        data/Russia-Ukraine/llm_embeddings_alt/bge_m3/<model>/<event>/<qid>_<lang>.json
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
from babelbias.paths import llm_embeddings_dir, llm_responses_dir
from babelbias.refusal import is_refusal


def run(model: str, event: str, in_root: Path, out_root: Path, embedder: str):
    in_dir = in_root / model / event
    routed_root = embedder_out_dir(out_root, embedder)
    out_dir = routed_root / model / event
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        raise SystemExit(f"No responses at {in_dir}. Run prompt_llms.py first.")

    files = sorted(p for p in in_dir.iterdir() if p.suffix == ".json")
    label = embedder_label(embedder)
    print(f"Embedding {len(files)} responses from {in_dir}")
    print(f"  embedder: {embedder} ({label})")
    print(f"  output:   {out_dir}")

    made = skipped = refusals = 0
    for src in files:
        out_path = out_dir / src.name
        if out_path.exists():
            skipped += 1
            continue

        with open(src) as f:
            rec = json.load(f)

        refusal = is_refusal(rec)
        if refusal:
            refusals += 1

        try:
            vec = embed_short_alt(rec.get("response_text", ""), embedder)
        except Exception as e:
            print(f"  ERROR ({type(e).__name__}): {src.name} — {e}")
            continue
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
                "finish_reason": rec.get("finish_reason"),
                "refusal": refusal,
                "embedding_model": label,
                "embedding": vec,
                "type": "llm_response",
            }, f)

        made += 1
        # API embedders need a tiny pacing delay; local ones don't.
        if embedder in ("openai_te3s", "cohere_ml_v3"):
            time.sleep(0.05)

    print(f"Done. {made} new embeddings, {skipped} already existed, "
          f"{refusals} flagged as refusals/content-filter.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--event", default="ru_uk_core")
    ap.add_argument("--in-root", type=Path, default=None,
                    help="Override response root. Default = "
                         "`event_root(event) / llm_responses`.")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Override embedding root. Default = "
                         "`event_root(event) / llm_embeddings`. Alt "
                         "embedders route to {out_root}_alt/{embedder}/ "
                         "automatically.")
    ap.add_argument("--embedder", default="openai_te3s",
                    choices=list(ALL_EMBEDDERS),
                    help="Embedding model to use. Default = OpenAI baseline.")
    args = ap.parse_args()
    in_root  = args.in_root  or llm_responses_dir(args.event)
    out_root = args.out_root or llm_embeddings_dir(args.event)
    run(args.model, args.event, in_root, out_root, args.embedder)


if __name__ == "__main__":
    main()
