"""
Fetch every Wikipedia lead-anchor article declared by an event's prompt
bank. Writes raw JSON in the format `embed_leads.py` consumes.

Two modes:

  1. **Bank-driven (post-exp_006, recommended).** With `--event <slug>`,
     read the prompt bank, walk every prompt's `wiki_anchor_slug` +
     `wiki_titles` map, and fetch the per-language Wikipedia article
     directly. No langlink resolution needed when the bank already
     declares the per-language titles.

  2. **Langlink fallback (legacy).** With `--event <slug> --use-langlinks`
     OR `--anchors slug=EN_title,…`, resolve the foreign-language titles
     via the English page's langlinks API. This is the legacy path used
     when the original ru_uk_core bank had no `wiki_titles` field.

Output:  `data/<event>/raw/<slug>_<lang>_raw.json`
"""

from __future__ import annotations

import argparse

from babelbias.event_bank import load_bank
from babelbias.paths import raw_dir
from babelbias.wiki import fetch_with_cache, resolve_langlinks


def fetch_from_bank(event: str, use_langlinks: bool = False) -> None:
    bank = load_bank(event)
    out_dir = raw_dir(event)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group prompts by anchor slug — q06/q07/q08 etc. share an article.
    by_slug: dict[str, dict] = {}
    for p in bank.prompts:
        slug = p.wiki_anchor_slug
        if slug not in by_slug:
            by_slug[slug] = {
                "wiki_titles": dict(p.wiki_titles),
                "wiki_anchor_topic": p.wiki_anchor_topic,
            }
        else:
            # Merge per-language titles across prompts that share a slug
            # (defensive — they should already match).
            for lang, title in p.wiki_titles.items():
                by_slug[slug]["wiki_titles"].setdefault(lang, title)

    print(f"Fetching {len(by_slug)} unique anchor article(s) "
          f"× {len(bank.languages)} language(s) into {out_dir}")

    for slug, meta in by_slug.items():
        print(f"\n== {slug} ==")
        titles = meta["wiki_titles"]
        if use_langlinks or not titles:
            en_title = titles.get("en") or meta["wiki_anchor_topic"]
            resolved = resolve_langlinks(en_title)
            if resolved is None:
                print(f"  EN page not found: {en_title}")
                continue
            titles = {l: resolved[l] for l in bank.languages if l in resolved}

        for lang in bank.languages:
            title = titles.get(lang)
            if title is None:
                print(f"  no {lang} title in bank — skipping")
                continue
            raw_path = out_dir / f"{slug}_{lang}_raw.json"
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", default="ru_uk_core",
                    help="Event slug whose bank declares the anchors to fetch.")
    ap.add_argument("--use-langlinks", action="store_true",
                    help="Resolve per-language titles via the English page's "
                         "langlinks API instead of the bank's `wiki_titles` "
                         "field. Use when the bank has English titles only.")
    args = ap.parse_args()
    fetch_from_bank(args.event, args.use_langlinks)


if __name__ == "__main__":
    main()
