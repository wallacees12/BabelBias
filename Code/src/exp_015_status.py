"""Live status dashboard for exp_015 embedding-robustness sweep.

Counts embedding files written across all alt-embedder paths, estimates
spend per embedder, and surfaces any error log entries. Runs in <1 sec.

Usage:
    uv run python Code/src/exp_015_status.py
"""

from __future__ import annotations

import re
from pathlib import Path

from babelbias.paths import LLM_EMBEDDINGS_DIR, PROCESSED_LEADS_DIR, PROJECT_ROOT

EMBEDDERS = ("openai_te3s", "cohere_ml_v3", "alibaba_v3", "gemini_001", "yandex_doc")

# Rough per-call cost (assumes ~400 tokens avg per response, ~500 per anchor/control).
# Conservative numbers; actual spend may be lower.
COST_PER_CALL = {
    "openai_te3s":  400 / 1_000_000 * 0.02,
    "cohere_ml_v3": 400 / 1_000_000 * 0.10,
    "alibaba_v3":   400 / 1_000_000 * 0.05,
    "gemini_001":   400 / 1_000_000 * 0.15,
    "yandex_doc":   400 / 1_000_000 * 0.10,  # rough — Yandex pricing unconfirmed
}

EXP_LOG_DIR = PROJECT_ROOT / "Experiments"


def alt_root(embedder: str, base: Path) -> Path:
    if embedder == "openai_te3s":
        return base
    return base.parent / f"{base.name}_alt" / embedder


def count_files(d: Path, glob: str) -> int:
    if not d.exists():
        return 0
    return sum(1 for _ in d.rglob(glob))


def count_log_errors(prefix: str) -> int:
    n = 0
    for log in EXP_LOG_DIR.glob(f"{prefix}*.log"):
        n += sum(1 for line in log.read_text().splitlines() if "ERROR" in line or "Traceback" in line)
    return n


def main() -> None:
    print(f"{'embedder':<14}  {'responses':>10}  {'anchors':>8}  {'controls':>9}  "
          f"{'total':>8}  {'errors':>7}  {'~cost':>8}")
    print("-" * 78)

    grand_total = 0
    grand_cost = 0.0
    for emb in EMBEDDERS:
        emb_dir = alt_root(emb, LLM_EMBEDDINGS_DIR)
        leads_dir = alt_root(emb, PROCESSED_LEADS_DIR)

        # Anchor count: only the 6 unique slug × 3 langs = 18 (exp_015)
        anchor_files = 0
        if leads_dir.exists():
            for slug in ("Little_green_men",
                         "2014_Russian_annexation_of_Crimea",
                         "Revolution_of_Dignity",
                         "2014_Crimean_status_referendum",
                         "Malaysia_Airlines_Flight_17",
                         "Stepan_Bandera"):
                for lang in ("en", "ru", "uk"):
                    if (leads_dir / f"{slug}_{lang}.json").exists():
                        anchor_files += 1

        controls = count_files(leads_dir, "CONTROL_*.json")
        responses = count_files(emb_dir, "*.json")
        total = responses + anchor_files + controls

        # Errors only counted for alt sweeps (logs prefixed exp_015_<embedder>_)
        errors = count_log_errors(f"exp_015_{emb}_") if emb != "openai_te3s" else 0

        cost = total * COST_PER_CALL.get(emb, 0)

        print(f"{emb:<14}  {responses:>10}  {anchor_files:>8}  {controls:>9}  "
              f"{total:>8}  {errors:>7}  ${cost:>6.3f}")

        grand_total += total
        grand_cost += cost

    print("-" * 78)
    print(f"{'TOTAL':<14}  {' ':>10}  {' ':>8}  {' ':>9}  "
          f"{grand_total:>8}  {' ':>7}  ${grand_cost:>6.3f}")
    print()
    print(f"Target full sweep (4 alt embedders, baseline excluded):")
    print(f"  responses: 14 providers × 270 = 3,780 per embedder × 4 = 15,120")
    print(f"  anchors:   18 per embedder × 4 = 72")
    print(f"  controls:  ~1,056 per embedder × 4 = ~4,224")
    print(f"  total:     ~19,416 alt-embedder calls")


if __name__ == "__main__":
    main()
