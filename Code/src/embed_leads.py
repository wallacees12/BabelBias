"""
Embed the lead section (everything before the first '==' header) of every
cached Wikipedia article in data/Russia-Ukraine/raw/.

Outputs to data/Russia-Ukraine/processed_leads/ — same JSON shape as
embed_conflicts.py so analyze_bias.py / visualize scripts can mix them.
"""

import json
import time

from babelbias.embedding import embed_short
from babelbias.paths import PROCESSED_LEADS_DIR, RAW_DIR
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


def embed_leads():
    PROCESSED_LEADS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in RAW_DIR.iterdir() if p.name.endswith("_raw.json"))
    print(f"Embedding {len(files)} lead sections...")

    for raw_path in files:
        out_path = PROCESSED_LEADS_DIR / raw_path.name.replace("_raw.json", ".json")
        if out_path.exists():
            continue

        print(f"  {raw_path.name}")
        with open(raw_path) as f:
            data = json.load(f)

        lead = extract_lead(data.get("content", ""))
        if not lead:
            print(f"    no lead content — skipping")
            continue

        vec = embed_short(lead)
        if vec is None:
            continue

        topic, lang, is_control = parse_filename(raw_path.name)
        record = {
            "conflict": topic,
            "language": lang,
            "title": data["title"],
            "embedding": vec,
            "type": "control" if is_control else "conflict",
        }
        with open(out_path, "w") as f:
            json.dump(record, f)
        time.sleep(0.1)


if __name__ == "__main__":
    embed_leads()
