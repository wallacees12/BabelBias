"""
Harvest neutral control articles (en/ru/uk triples) into data/Russia-Ukraine/raw/.

Two strategies:

    fetch_controls.py list                 # hardcoded list of high-level topics
    fetch_controls.py random --count 300   # random RU mainspace pages, keyword-filtered

Output filename format (shared):
    CONTROL_<safe_en_topic>_<lang>_raw.json
"""

import argparse
import json
import re
import time

import requests

from babelbias.config import DEFAULT_LANGS, WIKI_USER_AGENT
from babelbias.paths import RAW_DIR
from babelbias.wiki import get_wiki, resolve_langlinks, safe_name

# ---------------------------------------------------------------------------
# Hardcoded list strategy
# ---------------------------------------------------------------------------

CONTROL_TOPICS = [
    "Photosynthesis", "Solar System", "Quantum mechanics", "Ancient Egypt",
    "Renaissance", "Microscope", "Beethoven", "Albert Einstein",
    "Great Wall of China", "Amazon Rainforest", "Oxygen", "DNA",
    "Mathematics", "Architecture", "Philosophy", "Printing press",
    "Steam engine", "Industrial Revolution", "Library of Alexandria", "Viking",
    "Marie Curie", "Leonardo da Vinci", "Human body", "Volcano",
    "Global warming", "Internet", "Computer", "Algorithm",
    "Galaxy", "Black hole", "Evolution", "Periodic table",
    "Chemical element", "World War I",
    "French Revolution", "Magna Carta", "United Nations", "Olympic Games",
    "Chess", "Agriculture", "Philosophy of science", "Logic",
    "Astronomy", "Botany", "Zoology", "Geography", "Geology",
]

# ---------------------------------------------------------------------------
# Random-page strategy
# ---------------------------------------------------------------------------

API_URL_RANDOM = "https://ru.wikipedia.org/w/api.php"

# Reject any candidate whose title or EN lead matches these — we want zero
# geopolitical loading in the control set.
CONFLICT_KEYWORDS = re.compile(
    r"\b("
    r"russia|russian|soviet|ussr|kremlin|moscow|putin|"
    r"ukraine|ukrainian|kyiv|kiev|zelensky|"
    r"crimea|donbas|donetsk|luhansk|chechen|"
    r"nato|warsaw pact|"
    r"war|invasion|annexation|occupation|battle|military|army|"
    r"missile|tank|soldier|armed forces|conflict|"
    r"belarus|georgia|moldova|transnistria"
    r")\b",
    re.IGNORECASE,
)


def looks_conflict(*texts: str) -> bool:
    return any(CONFLICT_KEYWORDS.search(t) for t in texts if t)


# ---------------------------------------------------------------------------
# Shared save helper
# ---------------------------------------------------------------------------

def control_path(topic: str, lang: str):
    return RAW_DIR / f"CONTROL_{safe_name(topic)}_{lang}_raw.json"


def already_have(topic: str) -> bool:
    return any(control_path(topic, lang).exists() for lang in DEFAULT_LANGS)


def save_control_triple(topic: str, titles_by_lang: dict[str, str],
                        reject_if_lead_looks_conflict: bool = False) -> bool:
    """Fetch all three languages and save them as CONTROL_*.json. Returns True
    on full success. Skips writing anything if any language fails."""
    payloads = {}
    for lang, remote_title in titles_by_lang.items():
        page = get_wiki(lang).page(remote_title)
        if not page.exists() or not page.text.strip():
            return False
        if reject_if_lead_looks_conflict and lang == "en":
            lead = page.text.split("==")[0]
            if looks_conflict(lead):
                return False
        payloads[lang] = {
            "title": remote_title,
            "content": page.text,
            "type": "control",
            "topic": topic,
            "url": page.fullurl,
        }
        time.sleep(0.1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for lang, payload in payloads.items():
        with open(control_path(topic, lang), "w") as f:
            json.dump(payload, f, ensure_ascii=False)
    return True


# ---------------------------------------------------------------------------
# Strategy: hardcoded list
# ---------------------------------------------------------------------------

def fetch_from_list(topics: list[str]) -> None:
    for topic in topics:
        print(f"Checking control: {topic}")

        if already_have(topic):
            print("  already on disk")
            continue

        available = resolve_langlinks(topic)
        if available is None:
            print("  EN page not found")
            continue
        if not all(c in available for c in DEFAULT_LANGS):
            print("  missing ru or uk version")
            continue

        titles = {lang: available[lang] for lang in DEFAULT_LANGS}
        save_control_triple(topic, titles)


# ---------------------------------------------------------------------------
# Strategy: random sampling
# ---------------------------------------------------------------------------

def get_random_titles(session: requests.Session, n: int) -> list[str]:
    resp = session.get(
        API_URL_RANDOM,
        params={
            "action": "query", "format": "json",
            "list": "random", "rnnamespace": 0,
            "rnlimit": min(n, 20),
        },
        timeout=30,
    )
    resp.raise_for_status()
    return [r["title"] for r in resp.json()["query"]["random"]]


def get_langlinks_via_api(session: requests.Session, title: str) -> dict[str, str]:
    resp = session.get(
        API_URL_RANDOM,
        params={
            "action": "query", "format": "json",
            "titles": title, "prop": "langlinks", "lllimit": 500,
        },
        timeout=30,
    )
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return {ll["lang"]: ll["*"] for ll in page.get("langlinks", [])}


def fetch_random(target_count: int, batch_size: int, page_limit: int) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": WIKI_USER_AGENT})

    saved = inspected = 0
    rejected = {"no_langs": 0, "conflict": 0, "fetch": 0, "dup": 0}

    while saved < target_count and inspected < page_limit:
        try:
            ru_titles = get_random_titles(session, batch_size)
        except Exception as e:
            print(f"  random batch failed: {e}")
            time.sleep(2)
            continue

        for ru_title in ru_titles:
            inspected += 1

            try:
                langlinks = get_langlinks_via_api(session, ru_title)
            except Exception as e:
                print(f"  langlinks failed for {ru_title!r}: {e}")
                time.sleep(1)
                continue

            if "en" not in langlinks or "uk" not in langlinks:
                rejected["no_langs"] += 1
                continue

            en_title = langlinks["en"]
            if looks_conflict(en_title):
                rejected["conflict"] += 1
                continue
            if already_have(en_title):
                rejected["dup"] += 1
                continue

            titles = {"en": en_title, "ru": ru_title, "uk": langlinks["uk"]}
            print(f"  [{saved+1}/{target_count}] candidate: {en_title!r} "
                  f"(ru={ru_title!r}, uk={langlinks['uk']!r})")
            try:
                ok = save_control_triple(en_title, titles, reject_if_lead_looks_conflict=True)
            except Exception as e:
                print(f"    fetch failed: {e}")
                rejected["fetch"] += 1
                continue

            if ok:
                saved += 1
            else:
                rejected["fetch"] += 1

            if saved >= target_count:
                break
            time.sleep(0.2)

    print(f"\nDone. saved={saved}  inspected={inspected}  "
          + "  ".join(f"rejected_{k}={v}" for k, v in rejected.items()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    sub.add_parser("list", help=f"Fetch the built-in {len(CONTROL_TOPICS)} curated control topics.")

    p_random = sub.add_parser("random", help="Sample random RU pages, keyword-filtered.")
    p_random.add_argument("--count", type=int, default=300, help="How many new controls to add.")
    p_random.add_argument("--batch-size", type=int, default=20, help="Random titles per API call (max 20).")
    p_random.add_argument("--page-limit", type=int, default=5000, help="Hard cap on pages inspected.")

    args = ap.parse_args()

    if args.mode == "list":
        fetch_from_list(CONTROL_TOPICS)
    else:
        fetch_random(args.count, args.batch_size, args.page_limit)


if __name__ == "__main__":
    main()
