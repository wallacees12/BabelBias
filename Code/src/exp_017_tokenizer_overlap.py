"""
exp_017 Phase A — Subword-vocabulary overlap across embedders.

Computes pairwise (EN, RU, UK) tokenizer overlap on a stratified sample
of Wikipedia raw text from `data/Russia-Ukraine/raw/` for each
inspectable embedder tokenizer:

    cl100k_base   (OpenAI text-embedding-3-small)        — via tiktoken
    cohere_ml_v3  (Cohere embed-multilingual-v3.0)        — via tokenizers
    qwen2_proxy   (Alibaba text-embedding-v3, Qwen surrogate) — via tokenizers

Gemini-embedding-001 and Yandex text-search-doc tokenizers are not
publicly published; reported as N/A in the output table.

Two metrics, both following Qi et al. 2023 (RankC, EMNLP):
  - vocab_jaccard:    |V_L1 ∩ V_L2| / |V_L1 ∪ V_L2|  (unweighted)
  - corpus_overlap:   Σ min(f_L1, f_L2) / Σ max(f_L1, f_L2)  (Qi et al. primary)

Outputs:
  data/Russia-Ukraine/analysis/exp_017_tokenizer_overlap.csv
  data/Russia-Ukraine/analysis/exp_017_tokenizer_overlap_summary.txt

No torch / no model weights. The `tokenizers` package downloads only
tokenizer.json files (a few MB each). See
memory/feedback_no_local_ml.md.

Usage:
    python -m exp_017_tokenizer_overlap --sample 50
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd
import tiktoken

from babelbias.paths import ANALYSIS_DIR, RAW_DIR

LANGS = ("en", "ru", "uk")
LANG_PAIRS = (("en", "ru"), ("en", "uk"), ("ru", "uk"))

INSPECTABLE_TOKENIZERS = {
    "cl100k_base":   {"label": "OpenAI cl100k_base", "kind": "tiktoken"},
    "cohere_ml_v3":  {
        "label": "Cohere embed-multilingual-v3.0",
        "kind": "hf",
        "repo":  "Cohere/Cohere-embed-multilingual-v3.0",
    },
    "qwen2_proxy":   {
        "label": "Qwen2.5 (Alibaba text-embedding-v3 proxy)",
        "kind": "hf",
        "repo":  "Qwen/Qwen2.5-7B-Instruct",
    },
}

PROPRIETARY_TOKENIZERS = {
    "gemini_001": "Google gemini-embedding-001",
    "yandex_doc": "Yandex text-search-doc",
}


# ── Tokenizer loaders ─────────────────────────────────────────────────────


def _load_tiktoken(_repo: str | None) -> Callable[[str], list[int]]:
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode


def _load_hf(repo: str) -> Callable[[str], list[int]]:
    """Load an HF fast tokenizer via the `tokenizers` rust package.

    Pure tokenizer download — no torch, no model weights. If `tokenizers`
    is missing we surface a single uv-install hint rather than a deep
    traceback.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise SystemExit(
            "Missing dependency `tokenizers`. Install with:\n"
            "    uv pip install tokenizers\n"
            "(no torch / no model weights pulled — only tokenizer.json files)"
        ) from e
    tok = Tokenizer.from_pretrained(repo)
    return lambda text: tok.encode(text).ids


def load_tokenizer(name: str) -> Callable[[str], list[int]]:
    spec = INSPECTABLE_TOKENIZERS[name]
    if spec["kind"] == "tiktoken":
        return _load_tiktoken(None)
    return _load_hf(spec["repo"])


# ── Corpus sampling ───────────────────────────────────────────────────────


def discover_triplets(raw_dir: Path) -> list[str]:
    """Return slugs that have all three (_en_raw, _ru_raw, _uk_raw) files."""
    by_slug: dict[str, set[str]] = {}
    for f in raw_dir.glob("*_raw.json"):
        stem = f.stem  # e.g. "Foo_Bar_uk_raw"
        if not stem.endswith("_raw"):
            continue
        body = stem[: -len("_raw")]  # "Foo_Bar_uk"
        if len(body) < 4 or body[-3] != "_":
            continue
        lang = body[-2:]
        slug = body[:-3]
        if lang not in LANGS:
            continue
        by_slug.setdefault(slug, set()).add(lang)
    return sorted(s for s, langs in by_slug.items() if set(LANGS) <= langs)


def load_triplet_text(raw_dir: Path, slug: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for lang in LANGS:
        path = raw_dir / f"{slug}_{lang}_raw.json"
        with path.open(encoding="utf-8") as fh:
            obj = json.load(fh)
        out[lang] = obj.get("content", "")
    return out


def sample_corpus(
    raw_dir: Path,
    n: int,
    seed: int,
) -> dict[str, list[str]]:
    """Stratified sample of N triplets. Returns {lang: [text, ...]}."""
    rng = random.Random(seed)
    triplets = discover_triplets(raw_dir)
    if not triplets:
        raise SystemExit(f"No (en,ru,uk) triplets found under {raw_dir}")
    chosen = triplets if n >= len(triplets) else rng.sample(triplets, n)
    corpus: dict[str, list[str]] = {l: [] for l in LANGS}
    for slug in chosen:
        texts = load_triplet_text(raw_dir, slug)
        for lang in LANGS:
            corpus[lang].append(texts[lang])
    return corpus


# ── Overlap metrics ───────────────────────────────────────────────────────


def tokenize_corpus(
    encode: Callable[[str], list[int]],
    docs: list[str],
) -> Counter:
    counter: Counter = Counter()
    for doc in docs:
        if not doc:
            continue
        counter.update(encode(doc))
    return counter


def vocab_jaccard(c1: Counter, c2: Counter) -> float:
    v1, v2 = set(c1), set(c2)
    if not (v1 or v2):
        return 0.0
    return len(v1 & v2) / len(v1 | v2)


def corpus_overlap(c1: Counter, c2: Counter) -> float:
    """Σ min(f_L1, f_L2) / Σ max(f_L1, f_L2). Qi et al. 2023 RankC."""
    keys = set(c1) | set(c2)
    if not keys:
        return 0.0
    total_min = sum(min(c1[k], c2[k]) for k in keys)
    total_max = sum(max(c1[k], c2[k]) for k in keys)
    if total_max == 0:
        return 0.0
    return total_min / total_max


# ── Driver ────────────────────────────────────────────────────────────────


def compute_one_tokenizer(
    name: str,
    corpus: dict[str, list[str]],
) -> list[dict]:
    label = INSPECTABLE_TOKENIZERS[name]["label"]
    print(f"\n[{name}] loading {label} …")
    encode = load_tokenizer(name)
    print(f"[{name}] tokenizing {sum(len(v) for v in corpus.values())} docs …")
    counters = {lang: tokenize_corpus(encode, corpus[lang]) for lang in LANGS}
    for lang, c in counters.items():
        n_tok = sum(c.values())
        n_vocab = len(c)
        print(f"[{name}]   {lang}: {n_tok:>9,} tokens · {n_vocab:>6,} unique")
    rows: list[dict] = []
    for l1, l2 in LANG_PAIRS:
        rows.append({
            "tokenizer": name,
            "tokenizer_label": label,
            "lang_pair": f"{l1}-{l2}",
            "vocab_jaccard": round(vocab_jaccard(counters[l1], counters[l2]), 4),
            "corpus_overlap": round(corpus_overlap(counters[l1], counters[l2]), 4),
        })
    return rows


def render_summary(df: pd.DataFrame) -> str:
    lines = [
        "exp_017 Phase A — subword-vocabulary overlap across embedders",
        "=" * 64,
        "",
        f"Corpus: {df.attrs.get('n_triplets', '?')} EN/RU/UK Wikipedia triplets",
        "Metric A: vocab_jaccard  = |V_L1 ∩ V_L2| / |V_L1 ∪ V_L2|",
        "Metric B: corpus_overlap = Σ min(f) / Σ max(f) (Qi et al. 2023)",
        "Reference: BLOOM RU-UK corpus_overlap ≈ 0.76 (Qi et al. 2023, Tab. 1)",
        "",
    ]
    for tok in df["tokenizer"].unique():
        sub = df[df["tokenizer"] == tok]
        label = sub["tokenizer_label"].iloc[0]
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"  {'pair':<8}  {'vocab_jaccard':>14}  {'corpus_overlap':>15}")
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['lang_pair']:<8}  "
                f"{row['vocab_jaccard']:>14.4f}  "
                f"{row['corpus_overlap']:>15.4f}"
            )
        lines.append("")
    lines.append("## Proprietary (no public tokenizer)")
    lines.append("")
    for k, v in PROPRIETARY_TOKENIZERS.items():
        lines.append(f"  {k:<14}  {v}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=int, default=50,
                   help="Number of EN/RU/UK Wikipedia triplets to sample (default 50)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizers", nargs="+",
                   choices=list(INSPECTABLE_TOKENIZERS),
                   default=list(INSPECTABLE_TOKENIZERS),
                   help="Subset of tokenizers to compute (default: all 3)")
    args = p.parse_args()

    print(f"Sampling {args.sample} EN/RU/UK triplets from {RAW_DIR} (seed={args.seed}) …")
    corpus = sample_corpus(RAW_DIR, args.sample, args.seed)
    actual_n = len(corpus["en"])
    print(f"Sampled {actual_n} triplets.")

    rows: list[dict] = []
    for name in args.tokenizers:
        rows.extend(compute_one_tokenizer(name, corpus))

    df = pd.DataFrame(rows)
    df.attrs["n_triplets"] = actual_n

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS_DIR / "exp_017_tokenizer_overlap.csv"
    txt_path = ANALYSIS_DIR / "exp_017_tokenizer_overlap_summary.txt"
    df.to_csv(csv_path, index=False)

    summary = render_summary(df)
    txt_path.write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
