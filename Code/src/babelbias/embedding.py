"""OpenAI embedding helpers shared by every embed_*.py script."""

import os
from functools import cache

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from .config import EMBEDDING_MODEL
from .paths import ENV_PATH

load_dotenv(ENV_PATH)

MAX_EMBEDDING_TOKENS = 8000
_encoding = tiktoken.get_encoding("cl100k_base")


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
    """Embed a short string. Truncates above the token limit instead of chunking
    (use for leads / LLM responses where chunking-then-averaging would be lossy)."""
    if not text:
        return None
    if len(_encoding.encode(text)) > MAX_EMBEDDING_TOKENS:
        text = text[:MAX_EMBEDDING_TOKENS]
    return get_client().embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding
