"""Project paths anchored to the BabelBias root, regardless of cwd.

The pre-exp_006 layout pinned everything to a single Russo-Ukrainian
data root (`data/Russia-Ukraine/`). exp_006 generalises the pipeline
to multiple event families (Israel-Palestine, Taiwan-strait, Falklands,
…); event-relative paths now live behind `event_root(event)` and the
per-subdirectory helpers below. The legacy `RU_*` constants are kept
for backwards compatibility with the ~30 RU-UK-scoped scripts
(experiment analysers, figure renderers) that pre-date the refactor.
"""

from pathlib import Path

# This file lives at Code/src/babelbias/paths.py — three parents up = project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CODE_ROOT = PROJECT_ROOT / "Code"
SRC_ROOT = CODE_ROOT / "src"
PROMPTS_DIR = CODE_ROOT / "prompts"
ENV_PATH = CODE_ROOT / ".env"

DATA_ROOT = PROJECT_ROOT / "data"


# ── Per-event resolver (post-exp_006) ─────────────────────────────────────
# Going forward: every event has its own data subtree at `data/<event>/`.
# `data/ru_uk_core` is a symlink to `data/Russia-Ukraine` for backwards
# compatibility — the historical directory name is preserved on disk so
# every existing path reference still works, but new code addresses the
# event by its slug.

def event_root(event: str) -> Path:
    """Return the on-disk root for an event family.

    >>> event_root("ru_uk_core")
    PosixPath('.../data/ru_uk_core')   # symlink → data/Russia-Ukraine
    >>> event_root("israel_palestine")
    PosixPath('.../data/israel_palestine')
    """
    return DATA_ROOT / event


def raw_dir(event: str) -> Path:
    return event_root(event) / "raw"


def processed_dir(event: str) -> Path:
    return event_root(event) / "processed"


def processed_leads_dir(event: str) -> Path:
    return event_root(event) / "processed_leads"


def llm_responses_dir(event: str) -> Path:
    return event_root(event) / "llm_responses"


def llm_embeddings_dir(event: str) -> Path:
    return event_root(event) / "llm_embeddings"


def analysis_dir(event: str) -> Path:
    return event_root(event) / "analysis"


def analysis_alt_dir(event: str) -> Path:
    return event_root(event) / "analysis_alt"


def llm_embeddings_alt_dir(event: str) -> Path:
    return event_root(event) / "llm_embeddings_alt"


def processed_leads_alt_dir(event: str) -> Path:
    return event_root(event) / "processed_leads_alt"


# ── Legacy RU-UK constants (pre-exp_006) ──────────────────────────────────
# Kept for backwards compatibility. New code should call the event-aware
# helpers above. Existing RU-UK-scoped scripts continue to work unchanged
# because the underlying directory is identical (data/Russia-Ukraine/).

RU_ROOT = DATA_ROOT / "Russia-Ukraine"
RAW_DIR = RU_ROOT / "raw"
PROCESSED_DIR = RU_ROOT / "processed"
PROCESSED_LEADS_DIR = RU_ROOT / "processed_leads"
LLM_RESPONSES_DIR = RU_ROOT / "llm_responses"
LLM_EMBEDDINGS_DIR = RU_ROOT / "llm_embeddings"
ANALYSIS_DIR = RU_ROOT / "analysis"
ANALYSIS_FULL_DIR = RU_ROOT / "analysis_full"
ANALYSIS_ALT_DIR = RU_ROOT / "analysis_alt"
LLM_EMBEDDINGS_ALT_DIR = RU_ROOT / "llm_embeddings_alt"
PROCESSED_LEADS_ALT_DIR = RU_ROOT / "processed_leads_alt"
METADATA_DIR = RU_ROOT / "metadata"

LINKS_TO_REVIEW_CSV = RU_ROOT / "1hop_links_to_review.csv"
REVIEWED_LINKS_CSV = RU_ROOT / "reviewed_links.csv"
CONFLICT_EXCEL = METADATA_DIR / "Conflict List Collectively biased.xlsx"
