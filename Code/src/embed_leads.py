"""
Embed the lead section (everything before the first '==' header) of every
cached Wikipedia article under `data/<event>/raw/`.

Outputs to `data/<event>/processed_leads/` — same JSON shape as
embed_conflicts.py so analyze_bias.py / visualize scripts can mix them.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from babelbias.embedding import embed_short
from babelbias.paths import processed_leads_dir, raw_dir
from babelbias.wiki import extract_lead


def parse_filename(filename: str) -> tuple[str, str, bool]:
    """('Bucha_massacre_en_raw.json',) -> ('Bucha massacre', 'en', is_control=False).
    Control files are prefixed CONTROL_ and the prefix is stripped from the topic."""
    stem = filename.removesuffix("_raw.json")
    parts = stem.split("_")
    lang = parts[-1]
    is_control = filename.startswith("CONTROL_")
    topic_parts = parts[1:-1] if is_control else parts[:-1]
    return " ".join(topic_parts), lang, is_control


def _embed_one(raw_path, out_dir):
    """Worker for parallel embedding. Returns (raw_path.name, status_str)."""
    out_path = out_dir / raw_path.name.replace("_raw.json", ".json")
    if out_path.exists():
        return raw_path.name, "skip-exists"
    try:
        with open(raw_path) as f:
            data = json.load(f)
        lead = extract_lead(data.get("content", ""))
        if not lead:
            return raw_path.name, "no-lead"
        vec = embed_short(lead)
        if vec is None:
            return raw_path.name, "embed-none"
        topic, lang, is_control = parse_filename(raw_path.name)
        record = {
            "conflict":  topic,
            "language":  lang,
            "title":     data["title"],
            "embedding": vec,
            "type":      "control" if is_control else "conflict",
        }
        with open(out_path, "w") as f:
            json.dump(record, f)
        return raw_path.name, "ok"
    except Exception as e:
        return raw_path.name, f"error: {type(e).__name__}: {e}"


def embed_leads(event: str = "ru_uk_core", max_workers: int = 8) -> None:
    out_dir = processed_leads_dir(event)
    in_dir  = raw_dir(event)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.iterdir() if p.name.endswith("_raw.json"))
    print(f"Embedding {len(files)} lead sections from {in_dir} "
          f"({max_workers} threads)")

    counts = {"ok": 0, "skip-exists": 0, "no-lead": 0, "embed-none": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_embed_one, p, out_dir) for p in files]
        for i, fut in enumerate(as_completed(futures), 1):
            name, status = fut.result()
            key = "error" if status.startswith("error") else status
            counts[key] = counts.get(key, 0) + 1
            if status.startswith("error"):
                print(f"  ✗ {name}: {status}")
            elif i % 200 == 0:
                print(f"  [{i}/{len(files)}] ok={counts['ok']}  "
                      f"skip={counts['skip-exists']}  err={counts['error']}",
                      flush=True)
    print(f"Done. {counts}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", default="ru_uk_core")
    args = ap.parse_args()
    embed_leads(args.event)


if __name__ == "__main__":
    main()
