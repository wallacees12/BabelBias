"""Interactive CLI inspector for LLM response JSONs.

Groups records by (qid, repeat) and shows all languages on one screen,
one line per language. Press `n` for next, `p` for previous, `q` to quit.

Examples:
  uv run python Code/src/inspect_responses.py --model yandexgpt --event israel_palestine
  uv run python Code/src/inspect_responses.py --model yandexgpt --event taiwan_strait --only answered
  uv run python Code/src/inspect_responses.py --model yandexgpt --event india_pakistan --qid q01
  uv run python Code/src/inspect_responses.py --model yandexgpt --event ru_uk_imaginary --wrap 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import termios
import tty
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "Code" / "src" / ".translate_cache"

# ANSI colour codes
C_DIM = "\033[2m"
C_BOLD = "\033[1m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_MAGENTA = "\033[35m"
C_CYAN = "\033[36m"
C_OFF = "\033[0m"
CLEAR = "\033[2J\033[H"


def is_refusal(rec: dict) -> bool:
    if rec.get("refusal"):
        return True
    finish = str(rec.get("finish_reason", "")).upper()
    return "FILTER" in finish or "BLOCKED" in finish


def status(rec: dict) -> tuple[str, str]:
    text = (rec.get("response_text") or "").strip()
    if is_refusal(rec):
        return "REFUSED ", C_RED
    if len(text) < 20:
        return "EMPTY   ", C_YELLOW
    return "ANSWERED", C_GREEN


def collapse(s: str) -> str:
    """Single-line representation: newlines → space, runs of space → one."""
    return " ".join((s or "").split())


def translate_to_en(text: str, source_lang: str) -> str:
    """DeepL translate non-EN response to EN, with on-disk cache."""
    if not text or source_lang.startswith("en"):
        return text
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{source_lang}::{text}".encode("utf-8")).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{key}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    try:
        from babelbias.translate import translate as deepl_translate
        translated = deepl_translate(text, target_lang="en", source_lang=source_lang)
    except Exception as e:                                              # noqa: BLE001
        return f"[translate failed: {e}]"
    cache_file.write_text(translated, encoding="utf-8")
    return translated


def render(qid, langs, group, idx, total, translate):
    sys.stdout.write(CLEAR)
    sample = next(iter(group.values()))
    theme = sample.get("theme", "")
    event = sample.get("event", "")
    model = sample.get("model", "")

    print(f"{C_CYAN}═══════════════════════════════════════════════════════════════════════{C_OFF}")
    print(f"{C_BOLD}{model} × {event}{C_OFF}   "
          f"{C_DIM}({idx + 1} / {total} questions){C_OFF}")
    print(f"{C_BOLD}{qid}{C_OFF}  ·  theme: {C_MAGENTA}{theme}{C_OFF}")
    print(f"{C_CYAN}═══════════════════════════════════════════════════════════════════════{C_OFF}")

    for lang in langs:
        if lang not in group:
            print(f"\n{C_DIM}── [{lang}] (missing) ──{C_OFF}")
            continue
        rec = group[lang]
        lbl, col = status(rec)
        prompt = (rec.get("prompt_text") or "").strip()
        resp = (rec.get("response_text") or "").strip()
        rep = rec.get("repeat", "?")

        print(f"\n{col}━━━ [{lbl}] {C_BOLD}{lang}{C_OFF}{col}  ·  repeat={rep}  ━━━{C_OFF}")
        print(f"{C_BLUE}prompt:{C_OFF}    {prompt}")
        print(f"{C_BLUE}response:{C_OFF}")
        if resp:
            print(resp)
        else:
            print(f"{C_DIM}<empty>{C_OFF}")
        if translate and resp and not lang.startswith("en"):
            en = translate_to_en(resp, lang)
            print(f"{C_DIM}─── DeepL → EN ───{C_OFF}")
            print(f"{C_DIM}{en}{C_OFF}")

    print()
    print(f"{C_DIM}n/space=next  ·  p=prev  ·  r=next repeat  ·  q=quit{C_OFF}")
    sys.stdout.flush()


def getkey() -> str:
    """Read a single keypress without requiring Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--event", required=True)
    ap.add_argument("--qid", default=None, help="substring filter on qid")
    ap.add_argument("--only", choices=["refused", "answered", "any"], default="any",
                    help="show only groups containing ≥1 record of this kind")
    ap.add_argument("--wrap", type=int, default=None,
                    help="max chars per response line (default: terminal width − 20)")
    ap.add_argument("-t", "--translate", action="store_true",
                    help="show DeepL translation of non-EN responses to EN "
                         "(cached on disk under Code/src/.translate_cache/)")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    in_dir = args.data_root / args.event / "llm_responses" / args.model / args.event
    if not in_dir.is_dir():
        sys.exit(f"No directory: {in_dir}")

    files = sorted(in_dir.glob("*.json"))
    if not files:
        sys.exit(f"No JSON files in {in_dir}")

    # First pass: collect every (qid, lang, repeat) record
    all_recs: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    langs_seen: set[str] = set()
    for f in files:
        try:
            rec = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if args.qid and args.qid not in rec.get("qid", ""):
            continue
        all_recs[rec.get("qid", "?")][rec.get("language", "?")].append(rec)
        langs_seen.add(rec.get("language", "?"))

    # Second pass: one representative repeat per (qid, lang). Prefer
    # the user-selected repeat (default 0); if that one is refused or
    # missing, fall back to the first ANSWERED if any, then anything.
    def pick(reps: list[dict], want_rep: int) -> dict | None:
        if not reps:
            return None
        reps_sorted = sorted(reps, key=lambda r: r.get("repeat", 0))
        # exact match on requested repeat
        for r in reps_sorted:
            if r.get("repeat") == want_rep:
                return r
        # fall back to first answered
        for r in reps_sorted:
            if status(r)[0] == "ANSWERED":
                return r
        return reps_sorted[0]

    rep_idx = {q: 0 for q in all_recs}                                  # per-qid current repeat

    def build_group(qid: str) -> dict[str, dict]:
        return {lang: pick(all_recs[qid][lang], rep_idx[qid])
                for lang in all_recs[qid]
                if pick(all_recs[qid][lang], rep_idx[qid]) is not None}

    # Filter by only-refused / only-answered (any record in the qid)
    if args.only != "any":
        wanted = "REFUSED " if args.only == "refused" else "ANSWERED"
        all_recs = {q: lr for q, lr in all_recs.items()
                    if any(status(r)[0] == wanted
                           for reps in lr.values() for r in reps)}

    if not all_recs:
        sys.exit(f"{C_DIM}No matching questions.{C_OFF}")

    keys = sorted(all_recs.keys())
    langs = sorted(langs_seen, key=lambda l: ("en" != l, l))

    idx = 0
    while True:
        render(keys[idx], langs, build_group(keys[idx]), idx, len(keys), args.translate)
        ch = getkey()
        if ch in ("q", "Q", "\x03"):  # q or Ctrl-C
            sys.stdout.write("\n")
            return
        if ch in ("n", "N", " ", "\r", "j"):
            idx = min(idx + 1, len(keys) - 1)
        elif ch in ("p", "P", "k"):
            idx = max(idx - 1, 0)
        elif ch in ("r", "R"):                                          # next repeat
            qid = keys[idx]
            max_rep = max((r.get("repeat", 0) for reps in all_recs[qid].values() for r in reps), default=0)
            rep_idx[qid] = (rep_idx[qid] + 1) % (max_rep + 1)


if __name__ == "__main__":
    main()
