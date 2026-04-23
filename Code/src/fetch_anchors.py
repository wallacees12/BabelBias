"""
Fetch the four Wikipedia leads missing from processed_leads/ that the
prompt_llms.py ru_uk_core bank needs anchors for.

Uses langlinks on the English page to resolve foreign titles, then writes raw
files in the same format embed_leads.py expects so it picks them up next run.
"""

import json

from babelbias.config import DEFAULT_LANGS
from babelbias.paths import RAW_DIR
from babelbias.wiki import fetch_with_cache, resolve_langlinks

# slug (used for filenames / processed_leads lookup) -> English Wikipedia title
ANCHORS = {
    "Stepan_Bandera":              "Stepan Bandera",
    "Malaysia_Airlines_Flight_17": "Malaysia Airlines Flight 17",
    "Little_green_men":            "Little green men (Russo-Ukrainian War)",
    "Revolution_of_Dignity":       "Revolution of Dignity",
}


def fetch():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for slug, en_title in ANCHORS.items():
        print(f"\n== {slug} ==")
        available = resolve_langlinks(en_title)
        if available is None:
            print(f"  EN page not found: {en_title}")
            continue

        for lang in DEFAULT_LANGS:
            if lang not in available:
                print(f"  no {lang} langlink — skipping")
                continue
            title = available[lang]
            raw_path = RAW_DIR / f"{slug}_{lang}_raw.json"

            if raw_path.exists():
                print(f"  {lang}: already on disk")
                continue

            print(f"  {lang}: '{title}' -> {raw_path.name}")
            content = fetch_with_cache(
                lang, title, raw_path,
                extra_meta={"type": "conflict", "topic": slug},
                sleep_s=0.3,
            )
            if content is None:
                print(f"  {lang}: page does not exist")


if __name__ == "__main__":
    fetch()
