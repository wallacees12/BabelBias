"""Run analyze_bias.py for every (alt_embedder, model) combination in the
exp_015 sweep. Produces per-embedder 3×3 cosine matrices (raw + debiased)
under data/.../analysis_alt/<embedder>/<model>/<event>/.

Idempotent — analyze_bias.py rewrites CSVs each invocation but it's
deterministic.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ALT_EMBEDDERS = ("alibaba_v3", "gemini_001", "yandex_doc")
MODELS = [
    "claude-haiku-4-5", "gpt-4o-mini", "gemini-2.5-flash", "grok-3-mini",
    "mercury-2", "deepseek-chat", "qwen-plus", "glm-4.5",
    "baidu/ernie-4.5-300b-a47b", "c4ai-aya-expanse-32b",
    "command-r7b-arabic-02-2025", "ollama:allam-7b", "ollama:taide-llama3-8b",
    "jamba-mini-2-2026-01",
]
EVENT = "ru_uk_core"


def run_one(embedder: str, model: str, debias: bool) -> tuple[bool, str]:
    responses_root = ROOT / "data" / "Russia-Ukraine" / "llm_embeddings_alt" / embedder
    wiki_root      = ROOT / "data" / "Russia-Ukraine" / "processed_leads_alt" / embedder
    out_root       = ROOT / "data" / "Russia-Ukraine" / "analysis_alt" / embedder

    cmd = [
        sys.executable, str(ROOT / "Code" / "src" / "analyze_bias.py"),
        "--model", model,
        "--event", EVENT,
        "--responses-root", str(responses_root),
        "--wiki-root", str(wiki_root),
        "--out-root", str(out_root),
    ]
    if debias:
        cmd.append("--debias")

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT / "Code" / "src")
    return (proc.returncode == 0, proc.stderr or proc.stdout[-200:])


def main() -> None:
    n_total = len(ALT_EMBEDDERS) * len(MODELS) * 2
    t0 = time.monotonic()
    n_done = n_failed = 0

    print(f"Running {n_total} analyses ({len(ALT_EMBEDDERS)} embedders × "
          f"{len(MODELS)} models × {{raw, debiased}})")
    print()

    for embedder in ALT_EMBEDDERS:
        print(f"=== {embedder} ===")
        for model in MODELS:
            for debias in (False, True):
                tag = "debiased" if debias else "raw     "
                ok, msg = run_one(embedder, model, debias)
                marker = "✓" if ok else "✗"
                short = model[:28].ljust(28)
                print(f"  [{tag}] {marker} {short}", end="")
                if not ok:
                    print(f"  — {msg.splitlines()[-1][:80]}")
                    n_failed += 1
                else:
                    print()
                n_done += 1

    elapsed = time.monotonic() - t0
    print()
    print(f"Done. {n_done} runs in {elapsed:.0f}s · {n_failed} failed")


if __name__ == "__main__":
    main()
