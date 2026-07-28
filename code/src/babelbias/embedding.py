"""Embedding helpers shared by every embed_*.py script.

Supports a primary OpenAI baseline and four alternative embedders for
the `exp_015_embedding_robustness_sweep` methodology defence — all
**API-only** (local sentence-transformers / torch loading is out of
scope per memory/feedback_no_local_ml.md):

    openai_te3s   · text-embedding-3-small         (🇺🇸 US, OpenAI training)
    cohere_ml_v3  · embed-multilingual-v3.0        (🇨🇦 Canada, multilingual)
    alibaba_v3    · text-embedding-v3 (DashScope)  (🇨🇳 China, multilingual)
    gemini_001    · gemini-embedding-001           (🇺🇸 US, Google training)
    yandex_doc    · text-search-doc/latest         (🇷🇺 Russia, Russian-trained)
"""

import os
from functools import cache
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from .config import EMBEDDING_MODEL
from .paths import ENV_PATH

load_dotenv(ENV_PATH)

MAX_EMBEDDING_TOKENS = 8000
_encoding = tiktoken.get_encoding("cl100k_base")

ALT_EMBEDDERS = ("cohere_ml_v3", "alibaba_v3", "gemini_001", "yandex_doc")
ALL_EMBEDDERS = ("openai_te3s", *ALT_EMBEDDERS)


@cache
def get_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chunk_text(text: str, max_tokens: int = MAX_EMBEDDING_TOKENS) -> list[str]:
    tokens = _encoding.encode(text)
    return [
        _encoding.decode(tokens[i : i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def embed(text: str) -> list[float] | None:
    """Embed a string. Chunks long text and averages the chunk embeddings."""
    if not text:
        return None
    client = get_client()
    chunks = chunk_text(text)
    vecs = [
        client.embeddings.create(input=c, model=EMBEDDING_MODEL).data[0].embedding
        for c in chunks
    ]
    if len(vecs) == 1:
        return vecs[0]
    return np.mean(vecs, axis=0).tolist()


def embed_short(text: str) -> list[float] | None:
    """Embed a short string with the OpenAI baseline. Truncates above the
    token limit (don't chunk + average — too lossy for short responses).

    Truncation must be token-aware: character-truncation breaks on
    Chinese / Japanese / Hebrew / Arabic where one character can map to
    1.5–2 tokens, so 8000 chars of Chinese can encode to 12k+ tokens
    and trip the 8192-token API cap.
    """
    if not text:
        return None
    tokens = _encoding.encode(text)
    if len(tokens) > MAX_EMBEDDING_TOKENS:
        text = _encoding.decode(tokens[:MAX_EMBEDDING_TOKENS])
    return get_client().embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding


# ── Alternative embedders (exp_015 robustness sweep) ──────────────────────
# All API-based — no local model loading. Local sentence-transformers
# pushes RAM to bricking levels on this machine; see
# memory/feedback_no_local_ml.md.


@cache
def _cohere_session():
    import requests
    sess = requests.Session()
    sess.headers["Authorization"] = f"Bearer {os.getenv('COHERE_API_KEY')}"
    sess.headers["Content-Type"] = "application/json"
    return sess


@cache
def _alibaba_session():
    """DashScope international endpoint — OpenAI-compatible. Same key as the
    Qwen LLM provider (QWEN_API_KEY); DashScope hosts both LLMs and embedders."""
    import requests
    sess = requests.Session()
    sess.headers["Authorization"] = f"Bearer {os.getenv('QWEN_API_KEY')}"
    sess.headers["Content-Type"] = "application/json"
    return sess


def _embed_cohere(text: str) -> list[float] | None:
    """Cohere embed-multilingual-v3.0 via REST. ~$0.10 per 1M tokens."""
    if not text:
        return None
    text = text[:100_000]  # soft cap; Cohere truncates server-side too
    resp = _cohere_session().post(
        "https://api.cohere.com/v2/embed",
        json={
            "model": "embed-multilingual-v3.0",
            "texts": [text],
            "input_type": "search_document",
            "embedding_types": ["float"],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]["float"][0]


def _embed_alibaba(text: str) -> list[float] | None:
    """Alibaba text-embedding-v3 via DashScope. Chinese-trained multilingual
    embedder — the non-Western anchor in the exp_015 robustness panel.
    ~$0.05 per 1M tokens. DashScope's tokenizer is denser than cl100k_base
    (8000 cl100k tokens can exceed Alibaba's 8192-token hard cap), so we
    truncate to 5500 cl100k tokens for safety across Latin and Cyrillic."""
    if not text:
        return None
    tokens = _encoding.encode(text)
    if len(tokens) > 5500:
        text = _encoding.decode(tokens[:5500])
    resp = _alibaba_session().post(
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/embeddings",
        json={"model": "text-embedding-v3", "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _embed_gemini(text: str) -> list[float] | None:
    """Google gemini-embedding-001 via Generative Language API. 3072-dim,
    multilingual. ~$0.15 per 1M tokens."""
    if not text:
        return None
    import requests
    text = text[:50_000]
    key = os.getenv("GOOGLE_API_KEY")
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-embedding-001:embedContent?key={key}",
        headers={"Content-Type": "application/json"},
        json={"content": {"parts": [{"text": text}]}},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def _embed_yandex(text: str) -> list[float] | None:
    """Yandex Foundation Models text-search-doc. Russian-trained 256-dim
    multilingual embedder — the regional anchor for Russo-Ukrainian content.
    Token limit ~2,048 (truncate to 1800 cl100k); aggressive 429 rate-limiting
    under high concurrency, so we retry with exponential backoff."""
    if not text:
        return None
    import time as _time
    import requests
    tokens = _encoding.encode(text)
    if len(tokens) > 1800:
        text = _encoding.decode(tokens[:1800])
    folder = os.getenv("YANDEX_FOLDER_ID")
    key = os.getenv("YANDEX_API_KEY")
    body = {
        "modelUri": f"emb://{folder}/text-search-doc/latest",
        "text": text,
    }
    headers = {
        "Authorization": f"Api-Key {key}",
        "x-folder-id": folder,
        "Content-Type": "application/json",
    }
    for attempt in range(6):
        resp = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
            headers=headers,
            json=body,
            timeout=60,
        )
        if resp.status_code == 429:
            # Exponential backoff: 0.5, 1, 2, 4, 8, 16 sec — bounded total ~31s.
            _time.sleep(0.5 * (2 ** attempt))
            continue
        resp.raise_for_status()
        return resp.json()["embedding"]
    # All retries exhausted — surface the last error to the caller's try/except.
    resp.raise_for_status()


def embed_short_alt(text: str, embedder: str) -> list[float] | None:
    """Dispatch a short-text embedding call to the chosen embedder."""
    if embedder == "openai_te3s":
        return embed_short(text)
    if embedder == "cohere_ml_v3":
        return _embed_cohere(text)
    if embedder == "alibaba_v3":
        return _embed_alibaba(text)
    if embedder == "gemini_001":
        return _embed_gemini(text)
    if embedder == "yandex_doc":
        return _embed_yandex(text)
    raise ValueError(f"Unknown embedder: {embedder}. Expected one of {ALL_EMBEDDERS}")


def embedder_out_dir(base_root: Path, embedder: str) -> Path:
    """Resolve the on-disk output directory for a given base root and embedder.

    `openai_te3s` writes to base_root unchanged (backward-compatible). All
    other embedders route to `{base_root}_alt/{embedder}/` so the OpenAI
    baseline tree stays clean.
    """
    if embedder == "openai_te3s":
        return base_root
    return base_root.parent / f"{base_root.name}_alt" / embedder


def embedder_label(embedder: str) -> str:
    """Human-readable label stamped into output JSON for traceability."""
    return {
        "openai_te3s":  "openai/text-embedding-3-small",
        "cohere_ml_v3": "cohere/embed-multilingual-v3.0",
        "alibaba_v3":   "alibaba/text-embedding-v3",
        "gemini_001":   "google/gemini-embedding-001",
        "yandex_doc":   "yandex/text-search-doc",
    }[embedder]
