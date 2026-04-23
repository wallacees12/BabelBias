"""Wikipedia fetch + cache utilities shared by embed_*.py and fetch_*.py."""

import json
import time
from functools import cache
from pathlib import Path

import wikipediaapi

from .config import WIKI_USER_AGENT


@cache
def get_wiki(lang: str) -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(user_agent=WIKI_USER_AGENT, language=lang)


_FILENAME_REPLACEMENTS = str.maketrans({
    " ": "_", "/": "_", "'": "", "(": "", ")": "", "–": "-", ":": "",
})


def safe_name(title: str, max_len: int | None = None) -> str:
    """Filesystem-safe form of a Wikipedia title."""
    s = title.translate(_FILENAME_REPLACEMENTS)
    return s[:max_len] if max_len else s


def resolve_langlinks(en_title: str) -> dict[str, str] | None:
    """Look up an English Wikipedia page and return {lang: title} including 'en'.
    Returns None if the EN page doesn't exist."""
    page_en = get_wiki("en").page(en_title)
    if not page_en.exists():
        return None
    titles = {lang: link.title for lang, link in page_en.langlinks.items()}
    titles["en"] = page_en.title
    return titles


def extract_lead(content: str) -> str:
    """Wikipedia leads are everything before the first '==' section header."""
    if not content:
        return ""
    return content.split("==")[0].strip()


def fetch_with_cache(
    lang: str,
    title: str,
    raw_path: Path,
    extra_meta: dict | None = None,
    sleep_s: float = 0.1,
) -> str | None:
    """Read cached raw JSON if present, else fetch from Wikipedia and save it.
    Returns the article text, or None if the page doesn't exist."""
    raw_path = Path(raw_path)
    if raw_path.exists():
        with open(raw_path) as f:
            return json.load(f).get("content")

    page = get_wiki(lang).page(title)
    if not page.exists():
        return None

    payload = {"title": title, "content": page.text, "url": page.fullurl}
    if extra_meta:
        payload.update(extra_meta)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    time.sleep(sleep_s)
    return page.text
