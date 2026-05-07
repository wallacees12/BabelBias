"""
Discover the universal-control language coverage of the existing RU-UK
control corpus.

For every CONTROL_*_en_raw.json under `data/ru_uk_core/raw/`, query
Wikipedia langlinks once and cache the per-title coverage map. Then
report how many titles survive when filtered to various candidate
universal language sets.

Output:
    data/universal_controls/coverage_map.json
    data/universal_controls/coverage_summary.txt

Once a target language set is chosen, downstream callers can reuse this
cache to fetch the actual articles per event without re-querying
Wikipedia.

Usage:
    python -m discover_universal_controls --refresh   # re-query langlinks
    python -m discover_universal_controls             # use cache, just report
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from babelbias.paths import DATA_ROOT, raw_dir
from babelbias.wiki import resolve_langlinks


CACHE_DIR = DATA_ROOT / "universal_controls"
COVERAGE_PATH = CACHE_DIR / "coverage_map.json"
SUMMARY_PATH = CACHE_DIR / "coverage_summary.txt"

# Candidate universal language sets to evaluate. The roadmap covers:
# - RU-UK core: ru, uk
# - Israel-Palestine: he, ar
# - Taiwan-strait: zh (handles both scripts on the canonical zh.wikipedia)
# - Falklands / Latin American: es, pt
# - India-Pakistan / Kashmir: hi, ur, bn
# - Russo-Georgian: ka
# - Cyprus: el, tr
# - Senkaku/Diaoyu: ja
# - Western Sahara / Kurdish: includes ar/tr already
# - Northern Ireland: ga
TIERS = {
    "T1 — RU-UK only":              {"en", "ru", "uk"},
    "T2 — RU-UK + IL-PS":           {"en", "ru", "uk", "he", "ar"},
    "T3 — T2 + Taiwan + Falklands": {"en", "ru", "uk", "he", "ar", "zh", "es"},
    "T4 — T3 + India/Pakistan":     {"en", "ru", "uk", "he", "ar", "zh", "es", "hi", "ur"},
    "T5 — T4 + Latin/Cyprus":       {"en", "ru", "uk", "he", "ar", "zh", "es", "hi", "ur", "pt", "el", "tr"},
}


def existing_en_titles(source_event: str = "ru_uk_core") -> list[str]:
    out = []
    for p in sorted(raw_dir(source_event).glob("CONTROL_*_en_raw.json")):
        try:
            with p.open(encoding="utf-8") as fh:
                rec = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        title = rec.get("title")
        if title:
            out.append(title)
    return out


def load_cache() -> dict[str, list[str]]:
    if COVERAGE_PATH.exists():
        with COVERAGE_PATH.open(encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_cache(coverage: dict[str, list[str]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with COVERAGE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(coverage, fh, ensure_ascii=False, indent=2)


def refresh_coverage(source_event: str = "ru_uk_core",
                     sleep_s: float = 0.3) -> dict[str, list[str]]:
    titles = existing_en_titles(source_event)
    coverage = load_cache()
    print(f"Refreshing langlinks for {len(titles)} EN control titles "
          f"({len(coverage)} already cached) …")
    new = 0
    failed = 0
    for i, title in enumerate(titles, 1):
        if title in coverage:
            continue
        try:
            available = resolve_langlinks(title)
        except Exception as e:
            print(f"  [{i}/{len(titles)}] {title!r}: {type(e).__name__}: {e}")
            failed += 1
            continue
        if available is None:
            coverage[title] = []
        else:
            coverage[title] = sorted(available.keys())
        new += 1
        if new % 25 == 0:
            print(f"  [{i}/{len(titles)}] resolved {new} new titles, "
                  f"{failed} failed")
            save_cache(coverage)  # checkpoint
        time.sleep(sleep_s)
    save_cache(coverage)
    print(f"\nDone. {new} new resolutions, {failed} failed, "
          f"total cached = {len(coverage)}.")
    return coverage


def report(coverage: dict[str, list[str]]) -> str:
    total = len(coverage)
    lines = [f"Universal-control language-coverage report",
             f"Total EN titles cached: {total}\n"]
    # Per-language coverage histogram
    lang_counts: Counter = Counter()
    for langs in coverage.values():
        lang_counts.update(langs)
    lines.append("Per-language coverage (top 25):")
    for lang, count in lang_counts.most_common(25):
        pct = 100 * count / total if total else 0
        lines.append(f"  {lang:<8} {count:>5}  ({pct:5.1f}%)")
    lines.append("")
    # Per-tier intersection
    lines.append("Intersection size for candidate language tiers:")
    for tier_name, lang_set in TIERS.items():
        kept = [t for t, ls in coverage.items() if lang_set.issubset(ls)]
        lines.append(f"  {tier_name:<35}  {len(kept):>5}  ({sorted(lang_set)})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="Query Wikipedia langlinks for any title not yet "
                         "in the cache. Slow (~30 min for the full RU-UK "
                         "1,049-title set; respects a 0.3s rate limit).")
    ap.add_argument("--source-event", default="ru_uk_core")
    args = ap.parse_args()

    coverage = (refresh_coverage(args.source_event)
                if args.refresh else load_cache())
    if not coverage:
        raise SystemExit(
            "No coverage cached. Run with --refresh first."
        )

    summary = report(coverage)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nWrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
