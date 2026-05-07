"""
Harvest neutral Wikipedia articles available across the full BabelBias
roadmap language set.

Strategy: random-walk EN Wikipedia mainspace, query langlinks once per
candidate, keep articles that exist in *every* TARGET_LANG. Conflict-
keyword filter rejects geopolitically-loaded titles before fetching.

Stops when `--target-count` complete tuples are saved (default 1,000)
or `--page-limit` candidates have been inspected.

Storage:
    data/universal_controls/raw/CONTROL_<safe_topic>_<lang>_raw.json
        — same naming convention as fetch_controls.py so
          fetch_controls.py from-universal can port them into any
          event whose languages are a subset of TARGET_LANGS.

Long-running (~1.5–3 hr wall for 1,000 keeps at ~5–8% yield) but
purely free Wikipedia API. Run as a background job.

Usage:
    python -m harvest_universal_controls --target-count 1000
    python -m harvest_universal_controls --target-count 200 --page-limit 4000  # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from babelbias.config import WIKI_USER_AGENT
from babelbias.paths import DATA_ROOT
from babelbias.wiki import get_wiki, safe_name


# Locked language set covering the full exp_006 roadmap:
#   en (control)         · ru/uk (RU-UK)
#   he/ar (IL-PS)         · zh (Taiwan)
#   es/pt (Falklands +    · hi/ur (India-Pakistan)
#         Latin Am)
TARGET_LANGS: tuple[str, ...] = (
    "en", "ru", "uk", "he", "ar", "zh", "es", "pt", "hi", "ur",
)

UNIVERSAL_DIR = DATA_ROOT / "universal_controls"
RAW_DIR = UNIVERSAL_DIR / "raw"
MANIFEST_PATH = UNIVERSAL_DIR / "manifest.json"

# Random-walk pivot. Rationale: the smallest target Wikipedia is the
# tightest selection filter — any article that exists in ur.wikipedia
# (~200k articles) is by editorial selection translation-worthy, so the
# probability it ALSO exists in the larger wikis (en/ru/es/zh/...) is
# much higher than random EN walks (which are dominated by stubs).
# Empirically: en-pivot yield 0.6%, ur-pivot expected 30-50%.
PIVOT_LANG = "ur"


def api_url(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"

# Conflict-keyword filter — generalised across the roadmap. Rejects any
# candidate whose EN title or lead matches. Order matters loosely (most
# common first for early-exit-style speed). We deliberately stay narrow
# — strictly geopolitical/violence terms, not nationality words like
# "Indian" or "British" which appear in many neutral biographies and
# placenames.
CONFLICT_KEYWORDS = re.compile(
    r"("
    # general violence / conflict
    r"\bwar\b|\binvasion\b|annexation|occupation|\bbattle\b|military|"
    r"\barmy\b|\bnavy\b|missile|\btank\b|soldier|armed forces|"
    r"\bconflict\b|terror(?:ism|ist)?|genocide|massacre|atrocit|"
    r"insurgenc|guerrilla|militia|intifada|uprising|revolt|\bcoup\b|"
    r"rebellion|\briot\b|airstrike|bombing|"
    # specific incident families that slip through generic word filters
    r"\bflight\s*\d+|\bMH\s*1?7\b|\bMH\s*370\b|9/11|september\s*11|"
    r"plane crash|aviation incident|maritime disaster|"
    r"treaty of|peace of|operation\s+[A-Z]\w+|"
    # RU-UK
    r"russia|russian|soviet|\bussr\b|kremlin|moscow|putin|"
    r"ukrain|kyiv|\bkiev\b|zelensky|"
    r"crimea|donbas|donetsk|luhansk|chechen|"
    r"belarus|\bgeorgia\b|moldova|transnistria|"
    # Israel-Palestine
    r"israel|palestin|\bgaza\b|hamas|hezbollah|"
    r"jerusalem|west bank|nakba|zionis|\bidf\b|"
    # Taiwan / China
    r"\btaiwan\b|tibet|hong kong|tiananmen|xinjiang|uyghur|\bplc\b|"
    # India-Pakistan
    r"kashmir|partition of india|\bpunjab\b|"
    # Latin Am
    r"falklands|malvinas|"
    # Cyprus / Kurdish
    r"\bcyprus\b|kurdish|kurdistan|"
    # multilateral
    r"\bnato\b|warsaw pact|cold war"
    r")",
    re.IGNORECASE,
)


def looks_conflict(*texts: str) -> bool:
    return any(CONFLICT_KEYWORDS.search(t) for t in texts if t)


def control_path(topic: str, lang: str) -> Path:
    return RAW_DIR / f"CONTROL_{safe_name(topic)}_{lang}_raw.json"


def have_complete_tuple(topic: str) -> bool:
    return all(control_path(topic, l).exists() for l in TARGET_LANGS)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(encoding="utf-8") as fh:
            return json.load(fh)
    return {"target_langs": list(TARGET_LANGS), "topics": []}


def save_manifest(manifest: dict) -> None:
    UNIVERSAL_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def _get_with_backoff(session: requests.Session, url: str, params: dict,
                      max_attempts: int = 6, timeout: int = 30) -> requests.Response:
    """Wrap session.get with exponential backoff on HTTP 429
    (Wikipedia bot-traffic rate limit).
    """
    for attempt in range(max_attempts):
        resp = session.get(url, params=params, timeout=timeout)
        if resp.status_code == 429:
            wait = 2 ** attempt  # 1, 2, 4, 8, 16, 32 sec
            print(f"    429 rate-limited, sleeping {wait}s (attempt {attempt+1}/{max_attempts})",
                  flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def get_random_titles(session: requests.Session, lang: str, n: int) -> list[str]:
    resp = _get_with_backoff(
        session, api_url(lang),
        params={
            "action": "query", "format": "json",
            "list":   "random", "rnnamespace": 0,
            "rnlimit": min(n, 20),
        },
    )
    return [r["title"] for r in resp.json()["query"]["random"]]


def get_langlinks(session: requests.Session, title: str,
                  pivot_lang: str = "en") -> dict[str, str]:
    """Single-title langlinks query against the pivot Wikipedia.
    Includes 429-aware backoff."""
    resp = _get_with_backoff(
        session, api_url(pivot_lang),
        params={
            "action": "query", "format": "json",
            "titles": title, "prop": "langlinks", "lllimit": 500,
        },
    )
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return {ll["lang"]: ll["*"] for ll in page.get("langlinks", [])}


def get_langlinks_batch(session: requests.Session,
                        titles: list[str],
                        pivot_lang: str = "en",
                        max_workers: int = 3) -> dict[str, dict[str, str]]:
    """Threaded single-title langlinks lookup against the pivot Wikipedia.

    `lllimit` caps total langlinks across a multi-title API response,
    so popular pages starve out the rest of a batch (Mount Everest
    returned 1 langlink in a 4-title batch vs 211 alone). Threading
    single-title calls avoids that and is gentle enough at
    `max_workers=3` to stay under Wikipedia's bot-traffic ceiling.
    """
    out: dict[str, dict[str, str]] = {}
    if not titles:
        return out

    def _one(title: str) -> tuple[str, dict[str, str]]:
        try:
            return title, get_langlinks(session, title, pivot_lang)
        except Exception:
            return title, {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, t) for t in titles]
        for f in as_completed(futures):
            title, langs = f.result()
            out[title] = langs
    return out


def _fetch_one(topic: str, lang: str, remote_title: str) -> tuple[str, dict | None]:
    """Worker: fetch one (lang, title) page. Returns (lang, payload-or-None)."""
    try:
        page = get_wiki(lang).page(remote_title)
        if not page.exists() or not page.text.strip():
            return lang, None
        return lang, {
            "title":   remote_title,
            "content": page.text,
            "type":    "control",
            "topic":   topic,
            "url":     page.fullurl,
        }
    except Exception:
        return lang, None


def save_tuple(topic: str, titles_by_lang: dict[str, str],
               max_workers: int = 3) -> bool:
    """Fetch every TARGET_LANG version in parallel and save them.

    Returns True only if the entire tuple lands successfully — if any
    language fails, no files are written (the manifest must never see
    partial tuples).
    """
    payloads: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_fetch_one, topic, lang, remote_title)
            for lang, remote_title in titles_by_lang.items()
        ]
        for fut in as_completed(futures):
            lang, payload = fut.result()
            if payload is None:
                # One language failed → no point fetching the rest. Cancel
                # the still-pending futures and return early.
                for f in futures:
                    f.cancel()
                return False
            payloads[lang] = payload

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for lang, payload in payloads.items():
            p = control_path(topic, lang)
            with p.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            written.append(p)
        return True
    except Exception:
        for p in written:
            p.unlink(missing_ok=True)
        return False


def harvest(target_count: int, batch_size: int, page_limit: int,
            sleep_s: float, pivot_lang: str) -> None:
    UNIVERSAL_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": WIKI_USER_AGENT})
    manifest = load_manifest()
    saved_topics: set[str] = set(manifest["topics"])

    inspected = 0
    rejected = {"no_langs": 0, "conflict": 0, "fetch": 0, "dup": 0,
                "pivot_only_en": 0}

    print(f"Harvesting up to {target_count} universal tuples in "
          f"{len(TARGET_LANGS)} languages: {TARGET_LANGS}\n"
          f"  pivot:        random walk on {pivot_lang}.wikipedia.org "
          f"({'EN-target — long-tail bias' if pivot_lang == 'en' else 'small-edition pivot — high yield'})\n"
          f"  already have: {len(saved_topics)} on disk\n"
          f"  page_limit:   {page_limit}, batch_size: {batch_size}, sleep_s: {sleep_s}\n")

    need_non_pivot = [l for l in TARGET_LANGS if l != pivot_lang]
    while len(saved_topics) < target_count and inspected < page_limit:
        try:
            pivot_titles = get_random_titles(session, pivot_lang, batch_size)
        except Exception as e:
            print(f"  random batch failed: {e}", flush=True)
            time.sleep(2)
            continue

        # Resolve langlinks first; we need the EN title (for the canonical
        # `topic` filename + conflict-keyword filter) which only comes from
        # langlinks if we're pivoting on a non-EN language.
        try:
            batch_langlinks = get_langlinks_batch(session, pivot_titles,
                                                  pivot_lang=pivot_lang,
                                                  max_workers=3)
        except Exception as e:
            print(f"  langlinks batch failed: {e}", flush=True)
            time.sleep(2)
            continue

        for pivot_title in pivot_titles:
            inspected += 1
            langlinks = batch_langlinks.get(pivot_title, {})
            # The pivot page itself counts as one of the target langs,
            # so the surviving link map needs the OTHER 9 plus pivot.
            full_titles = {pivot_lang: pivot_title, **langlinks}
            missing = [l for l in TARGET_LANGS if l not in full_titles]
            if missing:
                rejected["no_langs"] += 1
                continue

            en_title = full_titles["en"]
            if looks_conflict(en_title):
                rejected["conflict"] += 1
                continue
            if have_complete_tuple(en_title):
                rejected["dup"] += 1
                saved_topics.add(en_title)
                continue

            titles_by_lang = {l: full_titles[l] for l in TARGET_LANGS}
            print(f"  [{len(saved_topics)+1}/{target_count}] candidate: {en_title!r}",
                  flush=True)
            try:
                ok = save_tuple(en_title, titles_by_lang, max_workers=3)
            except Exception as e:
                print(f"    fetch failed: {e}", flush=True)
                rejected["fetch"] += 1
                continue

            if ok:
                saved_topics.add(en_title)
                manifest["topics"] = sorted(saved_topics)
                save_manifest(manifest)
                if len(saved_topics) >= target_count:
                    break
            else:
                rejected["fetch"] += 1
            time.sleep(sleep_s)

        if inspected and inspected % 100 < batch_size:
            yield_pct = 100 * len(saved_topics) / max(inspected, 1)
            print(f"  [{inspected} inspected] saved={len(saved_topics)}  "
                  f"yield={yield_pct:.1f}%  rejects={rejected}", flush=True)
        # Pace the outer loop a touch to keep Wikipedia happy across
        # sustained runs — even with backoff baked in, gentler is safer.
        time.sleep(0.5)

    yield_pct = 100 * len(saved_topics) / max(inspected, 1)
    print(f"\nDone. saved={len(saved_topics)}  inspected={inspected}  "
          f"yield={yield_pct:.1f}%  "
          + "  ".join(f"rejected_{k}={v}" for k, v in rejected.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-count", type=int, default=1000,
                    help="Stop when this many complete tuples have been saved.")
    ap.add_argument("--batch-size", type=int, default=20,
                    help="Random titles per API call (max 20).")
    ap.add_argument("--page-limit", type=int, default=50_000,
                    help="Hard cap on candidate articles inspected.")
    ap.add_argument("--sleep-s", type=float, default=0.5,
                    help="Pacing between successful saves (seconds). Increase "
                         "if Wikipedia returns 429s.")
    ap.add_argument("--pivot-lang", default=PIVOT_LANG,
                    help="Wikipedia edition to random-walk. Smaller editions "
                         "(ur ~200k articles) bias toward universal-popularity "
                         "topics; en walks pull from the long tail of stubs.")
    args = ap.parse_args()
    harvest(args.target_count, args.batch_size, args.page_limit,
            args.sleep_s, args.pivot_lang)


if __name__ == "__main__":
    main()
