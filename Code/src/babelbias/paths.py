"""Project paths anchored to the BabelBias root, regardless of cwd."""

from pathlib import Path

# This file lives at Code/src/babelbias/paths.py — three parents up = project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CODE_ROOT = PROJECT_ROOT / "Code"
SRC_ROOT = CODE_ROOT / "src"
PROMPTS_DIR = CODE_ROOT / "prompts"
ENV_PATH = CODE_ROOT / ".env"

DATA_ROOT = PROJECT_ROOT / "data"

RU_ROOT = DATA_ROOT / "Russia-Ukraine"
RAW_DIR = RU_ROOT / "raw"
PROCESSED_DIR = RU_ROOT / "processed"
PROCESSED_LEADS_DIR = RU_ROOT / "processed_leads"
LLM_RESPONSES_DIR = RU_ROOT / "llm_responses"
LLM_EMBEDDINGS_DIR = RU_ROOT / "llm_embeddings"
ANALYSIS_DIR = RU_ROOT / "analysis"
ANALYSIS_FULL_DIR = RU_ROOT / "analysis_full"
METADATA_DIR = RU_ROOT / "metadata"

LINKS_TO_REVIEW_CSV = RU_ROOT / "1hop_links_to_review.csv"
REVIEWED_LINKS_CSV = RU_ROOT / "reviewed_links.csv"
CONFLICT_EXCEL = METADATA_DIR / "Conflict List Collectively biased.xlsx"
