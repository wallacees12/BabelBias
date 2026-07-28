"""Event-bank loader (post-exp_006 refactor).

A *prompt bank* lives at `Code/prompts/<event>.json` and now declares:
- `event`: the event slug (must match the filename stem)
- `description`: one-paragraph human description
- `languages`: list of ISO 639-1 codes for which the event has prompts
  and Wikipedia anchors (replaces the global DEFAULT_LANGS for code that
  loads from a bank)
- `prompts`: list of question records, each with:
    - `id`              — qNN_short_name, used in output filenames
    - `theme`           — short tag for the question family
    - `wiki_anchor_topic`  — free-text descriptor (legacy field kept for
                              human-readability)
    - `wiki_anchor_slug`   — file stem under processed_leads/{slug}_{lang}.json;
                              this is the canonical anchor identifier the
                              analysis pipeline keys off of
    - `wiki_titles`        — per-language Wikipedia article title used by
                              fetch_anchors when seeding a new event
    - `text`               — per-language prompt text

Every event with the same `wiki_anchor_slug` shares anchor embeddings
(e.g. ru_uk_core's q06/q07/q08 all reference the Crimea-annexation
article).

Backwards compatibility: the legacy `ru_uk_core.json` had no
`languages` field and used `wiki_anchor_topic` as a description-only
field. The migration done as part of exp_006 added the new fields
without breaking the old ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from .paths import PROMPTS_DIR


@dataclass(frozen=True)
class Prompt:
    id: str
    theme: str
    wiki_anchor_slug: str
    wiki_titles: dict[str, str]
    text: dict[str, str]
    wiki_anchor_topic: str = ""  # legacy free-text descriptor


@dataclass(frozen=True)
class EventBank:
    event: str
    description: str
    languages: tuple[str, ...]
    prompts: tuple[Prompt, ...]

    @property
    def anchor_slugs(self) -> dict[str, str]:
        """qid → wiki_anchor_slug, the dict analyze_bias has historically
        called ANCHOR_SLUGS."""
        return {p.id: p.wiki_anchor_slug for p in self.prompts}

    def prompt_by_id(self, qid: str) -> Prompt:
        for p in self.prompts:
            if p.id == qid:
                return p
        raise KeyError(f"No prompt {qid!r} in {self.event!r} bank")


@cache
def load_bank(event: str) -> EventBank:
    """Load the prompt bank for `event` from Code/prompts/<event>.json.

    Cached: re-reading the JSON on every call is wasteful and the file
    is treated as immutable for the lifetime of a run.
    """
    path = PROMPTS_DIR / f"{event}.json"
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    languages = raw.get("languages")
    if languages is None:
        # Legacy bank — infer languages from the first prompt's text dict.
        # Preserves compat with any prompt bank that hasn't migrated yet.
        if not raw["prompts"]:
            raise ValueError(f"{path}: no prompts and no `languages` field")
        languages = list(raw["prompts"][0]["text"].keys())

    prompts = tuple(
        Prompt(
            id=p["id"],
            theme=p.get("theme", ""),
            wiki_anchor_slug=p.get("wiki_anchor_slug")
                              or _infer_slug(p.get("wiki_anchor_topic", p["id"])),
            wiki_titles=p.get("wiki_titles", {}),
            text=p["text"],
            wiki_anchor_topic=p.get("wiki_anchor_topic", ""),
        )
        for p in raw["prompts"]
    )

    return EventBank(
        event=raw["event"],
        description=raw.get("description", ""),
        languages=tuple(languages),
        prompts=prompts,
    )


def _infer_slug(topic: str) -> str:
    """Fallback slug derivation if `wiki_anchor_slug` is missing.

    Replaces spaces and slashes with underscores; strips anything after a
    parenthesis (so 'Foo (Bar context)' → 'Foo'). Used only for prompt
    banks that haven't migrated to the explicit `wiki_anchor_slug` field.
    """
    head = topic.split("(")[0].strip()
    head = head.split(" / ")[0].strip()
    return head.replace(" ", "_")
