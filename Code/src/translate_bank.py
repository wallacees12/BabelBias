"""
Translate / cross-check a prompt bank with DeepL.

Three modes:

  --diff       : print every (prompt, target_lang) cell where the bank's
                 existing translation differs from the DeepL translation
                 of the EN source. Read-only.

  --update     : overwrite the bank's translations with DeepL's, for the
                 chosen target languages. EN source is preserved.

  --fill-only  : only translate cells that are missing or empty in the
                 bank — preserves existing human-verified translations.
                 Useful when wiring in a new language.

By default, runs `--diff`. EN is always treated as the source.

Usage:
    python -m translate_bank --event israel_palestine --diff
    python -m translate_bank --event israel_palestine --update --langs he,ar
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from babelbias.event_bank import load_bank
from babelbias.paths import PROMPTS_DIR
from babelbias.translate import translate_batch


def _bank_path(event: str) -> Path:
    return PROMPTS_DIR / f"{event}.json"


def _load_raw(event: str) -> dict:
    with _bank_path(event).open(encoding="utf-8") as fh:
        return json.load(fh)


def _save_raw(event: str, raw: dict) -> None:
    """Write JSON keeping non-ASCII characters readable (no \\uXXXX escapes)."""
    path = _bank_path(event)
    backup = path.with_suffix(".json.bak")
    shutil.copy2(path, backup)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(raw, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"  Wrote {path} (backup: {backup.name})")


def diff_mode(event: str, target_langs: list[str]) -> None:
    bank = load_bank(event)
    print(f"== DeepL diff vs bank — event={event} ==\n")

    for lang in target_langs:
        if lang == "en":
            continue
        en_texts = [p.text["en"] for p in bank.prompts]
        deepl_texts = translate_batch(en_texts, lang, source_lang="en")

        print(f"\n--- {lang.upper()} ---")
        for p, deepl_t in zip(bank.prompts, deepl_texts):
            bank_t = p.text.get(lang, "")
            if bank_t.strip() == deepl_t.strip():
                print(f"  ✓ {p.id}")
            else:
                print(f"  ✗ {p.id}")
                print(f"    bank:  {bank_t}")
                print(f"    deepl: {deepl_t}")


def update_mode(event: str, target_langs: list[str], fill_only: bool) -> None:
    raw = _load_raw(event)
    print(f"== DeepL {'fill-only' if fill_only else 'update'} — event={event} ==\n")

    for lang in target_langs:
        if lang == "en":
            continue
        # Pick prompts that need translation (all of them, or only empty cells
        # in fill-only mode)
        idx_to_translate = []
        en_texts = []
        for i, p in enumerate(raw["prompts"]):
            if fill_only and p["text"].get(lang, "").strip():
                continue
            idx_to_translate.append(i)
            en_texts.append(p["text"]["en"])

        if not en_texts:
            print(f"\n--- {lang.upper()} ---  nothing to translate")
            continue

        print(f"\n--- {lang.upper()} ---  translating {len(en_texts)} prompts")
        deepl_texts = translate_batch(en_texts, lang, source_lang="en")
        for i, deepl_t in zip(idx_to_translate, deepl_texts):
            qid = raw["prompts"][i]["id"]
            old = raw["prompts"][i]["text"].get(lang, "")
            raw["prompts"][i]["text"][lang] = deepl_t
            tag = "fill" if fill_only or not old else "overwrite"
            print(f"  {tag} {qid}: {deepl_t}")
            if old and not fill_only:
                print(f"    (was: {old})")

    _save_raw(event, raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", required=True)
    ap.add_argument("--langs", default=None,
                    help="Comma-separated target langs. Default = bank's "
                         "`languages` field minus 'en'.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--diff", action="store_true",
                   help="Print divergences between bank and DeepL (default).")
    g.add_argument("--update", action="store_true",
                   help="Overwrite bank translations with DeepL.")
    g.add_argument("--fill-only", action="store_true",
                   help="Translate only cells that are empty in the bank.")
    args = ap.parse_args()

    bank = load_bank(args.event)
    if args.langs:
        target_langs = [l.strip() for l in args.langs.split(",")]
    else:
        target_langs = [l for l in bank.languages if l != "en"]

    if args.update:
        update_mode(args.event, target_langs, fill_only=False)
    elif args.fill_only:
        update_mode(args.event, target_langs, fill_only=True)
    else:
        diff_mode(args.event, target_langs)


if __name__ == "__main__":
    main()
