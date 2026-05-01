"""Terminal review tool for saved LLM responses.

For each matching response, print the prompt, the response in its
detected language, and an English translation. Step through with
Enter; quit with q. Translations are cached on disk so re-running
the tool is free.

Examples:
    # Walk through every YandexGPT refusal across the EN/RU/UK sweep
    python review_responses.py --model yandexgpt --refusals-only

    # Just q01 across all providers
    python review_responses.py --qid q01 --model gpt-4o-mini

    # Resume from the 12th record (after quitting earlier)
    python review_responses.py --model yandexgpt --start 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from babelbias.paths import ENV_PATH, LLM_RESPONSES_DIR
from babelbias.refusal import is_refusal

load_dotenv(ENV_PATH)

TRANSLATIONS_DIR = LLM_RESPONSES_DIR.parent / "translations"
TRANSLATOR_MODEL = "gpt-4o-mini"


# ANSI ---------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()


def _c(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s


def bold(s: str) -> str:        return _c(s, "1")
def dim(s: str) -> str:         return _c(s, "2")
def red(s: str) -> str:         return _c(s, "31")
def green(s: str) -> str:       return _c(s, "32")
def yellow(s: str) -> str:      return _c(s, "33")
def blue(s: str) -> str:        return _c(s, "34")
def magenta(s: str) -> str:     return _c(s, "35")
def cyan(s: str) -> str:        return _c(s, "36")


# Translation --------------------------------------------------------

_openai_client = None


def _client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def translate(text: str, cache_path: Path) -> dict:
    """Detect language + translate to EN. Returns {detected_lang, translation_en}.

    Caches result on disk; re-runs are free.
    """
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    sys_msg = (
        "Detect the language of the user message and translate it to "
        "English. Reply with strict JSON: "
        '{"detected_language": "<ISO 639-1 code>", "translation": "<english text>"} '
        "and nothing else. If the text is already in English, set "
        '"translation" to the original text and "detected_language" to "en".'
    )
    resp = _client().chat.completions.create(
        model=TRANSLATOR_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    parsed = json.loads(raw)
    out = {
        "detected_language": parsed.get("detected_language", "?"),
        "translation_en": parsed.get("translation", ""),
        "translator_model": TRANSLATOR_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


# Discovery ----------------------------------------------------------

def discover(args) -> list[Path]:
    root = LLM_RESPONSES_DIR
    if args.model:
        candidates = [root / args.model]
    else:
        candidates = [p for p in root.iterdir() if p.is_dir()]
    files: list[Path] = []
    for model_dir in candidates:
        event_dir = model_dir / args.event
        if not event_dir.is_dir():
            continue
        for p in sorted(event_dir.iterdir()):
            if p.suffix != ".json":
                continue
            files.append(p)
    return files


def matches(rec: dict, args) -> bool:
    if args.qid and args.qid not in rec.get("qid", ""):
        return False
    if args.lang and rec.get("language") != args.lang:
        return False
    if args.refusals_only and not is_refusal(rec):
        return False
    return True


# Display ------------------------------------------------------------

def render_record(idx: int, total: int, rec: dict, src: Path,
                  translation: dict | None) -> None:
    print()
    print(cyan("═" * 72))
    refusal_tag = red("  ⚠ REFUSAL") if is_refusal(rec) else ""
    print(
        f"{bold(f'[{idx + 1}/{total}]')}  "
        f"{magenta(rec['model'])}  "
        f"{rec['event']}  "
        f"{yellow(rec['qid'])}  "
        f"prompt-lang={blue(rec['language'])}  "
        f"r{rec.get('repeat', 0):02d}{refusal_tag}"
    )
    print(dim(f"finish_reason: {rec.get('finish_reason')}"))
    print(dim(f"file: {src}"))
    print(cyan("═" * 72))

    print()
    print(bold(f"PROMPT ({rec['language']}):"))
    print(f"  {rec.get('prompt_text', '')}")

    response = rec.get("response_text", "")
    if translation:
        det = translation.get("detected_language", "?")
        det_tag = blue(det)
    else:
        det_tag = blue(rec["language"])
    print()
    print(bold(f"RESPONSE (detected: {det_tag}):"))
    for line in (response or "").splitlines() or [""]:
        print(f"  {line}")

    if translation and translation.get("detected_language") != "en":
        print()
        print(bold(green("TRANSLATION → en:")))
        for line in (translation.get("translation_en") or "").splitlines() or [""]:
            print(f"  {line}")
    elif rec["language"] != "en" and not translation:
        print()
        print(dim("(translation skipped)"))


# Main ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", help="Filter to one model (otherwise: all)")
    ap.add_argument("--event", default="ru_uk_core")
    ap.add_argument("--qid", help="Substring filter on qid (e.g. 'q01')")
    ap.add_argument("--lang", choices=["en", "ru", "uk"],
                    help="Filter on prompt language")
    ap.add_argument("--refusals-only", action="store_true",
                    help="Only show records flagged as content-filter refusals")
    ap.add_argument("--start", type=int, default=0,
                    help="Skip the first N matching records (resume after quit)")
    ap.add_argument("--no-translate", action="store_true",
                    help="Skip the translation API call (offline mode)")
    args = ap.parse_args()

    files = discover(args)
    records: list[tuple[Path, dict]] = []
    for p in files:
        with open(p) as f:
            rec = json.load(f)
        if matches(rec, args):
            records.append((p, rec))

    total = len(records)
    if total == 0:
        print("No records matched.")
        return
    print(f"Matched {total} record(s). Press Enter to step, q + Enter to quit.")

    flagged: list[Path] = []
    i = args.start
    while i < total:
        src, rec = records[i]
        translation = None
        text = rec.get("response_text", "")
        # Always detect+translate when there's text — prompt language ≠ response
        # language (e.g. YandexGPT refusing an EN prompt with a Russian
        # boilerplate). The cache makes this free on re-runs.
        if not args.no_translate and text:
            cache_path = (
                TRANSLATIONS_DIR / rec["model"] / rec["event"] / src.name
            )
            try:
                translation = translate(text, cache_path)
            except Exception as e:
                print(red(f"  translation error: {e}"))

        render_record(i, total, rec, src, translation)

        try:
            cmd = input(dim("\n[Enter] next  [q] quit  [f] flag  [b] back  > ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd == "q":
            break
        if cmd == "b" and i > 0:
            i -= 1
            continue
        if cmd == "f":
            flagged.append(src)
            print(yellow(f"  flagged: {src.name}"))
        i += 1

    if flagged:
        print()
        print(bold(yellow(f"Flagged {len(flagged)} record(s):")))
        for p in flagged:
            print(f"  {p}")


if __name__ == "__main__":
    main()
