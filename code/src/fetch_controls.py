"""
Harvest neutral control articles into `data/<event>/raw/` for the
language-axis debiasing step.

Strategies:

    fetch_controls.py list   --event ru_uk_core         # curated list, langlink-resolved
    fetch_controls.py random --event ru_uk_core --count 300   # random sampling

Controls are saved as triples (or N-tuples for events with > 3 langs)
keyed off the EN topic — only complete sets across every language in
the event bank are written, so the debias estimator never sees gaps.

Output filename format (shared):
    CONTROL_<safe_en_topic>_<lang>_raw.json
"""

import argparse
import json
import re
import time

import requests

from babelbias.config import WIKI_USER_AGENT
from babelbias.event_bank import load_bank
from babelbias.paths import raw_dir
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

def control_path(event: str, topic: str, lang: str):
    return raw_dir(event) / f"CONTROL_{safe_name(topic)}_{lang}_raw.json"


def already_have(event: str, topic: str, languages: tuple[str, ...]) -> bool:
    return all(control_path(event, topic, lang).exists() for lang in languages)


def save_control_triple(event: str, topic: str, titles_by_lang: dict[str, str],
                        reject_if_lead_looks_conflict: bool = False) -> bool:
    """Fetch every requested language and save them as CONTROL_*.json.
    Returns True on full success. Skips writing anything if any language
    fails — the language-axis estimator must see complete tuples.
    """
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

    out_dir = raw_dir(event)
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang, payload in payloads.items():
        with open(control_path(event, topic, lang), "w") as f:
            json.dump(payload, f, ensure_ascii=False)
    return True


# ---------------------------------------------------------------------------
# Strategy: hardcoded list
# ---------------------------------------------------------------------------

def existing_en_control_titles(source_event: str) -> list[str]:
    """Return the EN titles of every CONTROL_*_en_raw.json triple under
    `data/<source_event>/raw/`. Used by `from-event` to port a vetted
    neutral set across events without re-running random sampling.
    """
    src_dir = raw_dir(source_event)
    titles = []
    for p in sorted(src_dir.glob("CONTROL_*_en_raw.json")):
        try:
            with p.open(encoding="utf-8") as fh:
                rec = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        title = rec.get("title")
        if title:
            titles.append(title)
    return titles


def fetch_from_list(event: str, topics: list[str], languages: tuple[str, ...]) -> None:
    saved = skipped = no_en = missing_lang = 0
    for topic in topics:
        print(f"Checking control: {topic}")

        if already_have(event, topic, languages):
            print("  already on disk")
            skipped += 1
            continue

        available = resolve_langlinks(topic)
        if available is None:
            print("  EN page not found")
            no_en += 1
            continue
        missing = [c for c in languages if c not in available]
        if missing:
            print(f"  missing langlink(s): {missing}")
            missing_lang += 1
            continue

        titles = {lang: available[lang] for lang in languages}
        if save_control_triple(event, topic, titles):
            saved += 1
    print(f"\nDone. {saved} new triples saved, {skipped} already on disk, "
          f"{no_en} EN-page-missing, {missing_lang} missing one or more langlinks.")


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
            if already_have("ru_uk_core", en_title, ("en", "ru", "uk")):
                rejected["dup"] += 1
                continue

            titles = {"en": en_title, "ru": ru_title, "uk": langlinks["uk"]}
            print(f"  [{saved+1}/{target_count}] candidate: {en_title!r} "
                  f"(ru={ru_title!r}, uk={langlinks['uk']!r})")
            try:
                ok = save_control_triple("ru_uk_core", en_title, titles, reject_if_lead_looks_conflict=True)
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

    p_list = sub.add_parser("list",
                            help=f"Fetch the built-in {len(CONTROL_TOPICS)} curated topics, "
                                 "langlink-resolved into the event's languages.")
    p_list.add_argument("--event", default="ru_uk_core")

    p_port = sub.add_parser("from-event",
                            help="Port the EN control titles from another "
                                 "event into this event's languages. Re-uses "
                                 "the already-vetted-neutral RU-UK control set "
                                 "without re-running random sampling.")
    p_port.add_argument("--event", required=True,
                        help="Target event (e.g. israel_palestine).")
    p_port.add_argument("--source-event", default="ru_uk_core",
                        help="Source event whose CONTROL_*_en_raw.json titles "
                             "are langlink-resolved into the target event's "
                             "languages. Default = ru_uk_core (1,049 vetted "
                             "neutral topics).")

    p_random = sub.add_parser("random",
                              help="Sample random RU pages, keyword-filtered. "
                                   "Currently RU-pivoted (not yet event-generic) — "
                                   "use `list` mode for non-RU events.")
    p_random.add_argument("--event", default="ru_uk_core")
    p_random.add_argument("--count", type=int, default=300, help="How many new controls to add.")
    p_random.add_argument("--batch-size", type=int, default=20, help="Random titles per API call (max 20).")
    p_random.add_argument("--page-limit", type=int, default=5000, help="Hard cap on pages inspected.")

    args = ap.parse_args()

    bank = load_bank(args.event)
    if args.mode == "list":
        fetch_from_list(args.event, CONTROL_TOPICS, bank.languages)
    elif args.mode == "from-event":
        titles = existing_en_control_titles(args.source_event)
        print(f"Porting {len(titles)} EN titles from {args.source_event!r} "
              f"into {args.event!r} languages {bank.languages}\n")
        fetch_from_list(args.event, titles, bank.languages)
    else:
        if set(bank.languages) != {"en", "ru", "uk"}:
            raise SystemExit(
                "random-mode is currently RU-pivoted (queries ru.wikipedia "
                "and assumes en/ru/uk langlinks). For non-RU events, use "
                "`fetch_controls.py list --event <slug>` or "
                "`fetch_controls.py from-event --event <slug>` instead."
            )
        fetch_random(args.count, args.batch_size, args.page_limit)


if __name__ == "__main__":
    main()
