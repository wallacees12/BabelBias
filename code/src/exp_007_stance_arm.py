"""exp_007 · stance-axis projection arm for the cross-conflict temporal study.

Companion to the cosine temporal-ladder analysis. For each of the three
conflicts that already have a pre-registered paired-edit stance lexicon
(Russia-Ukraine, Israel-Palestine, India-Pakistan), we:

  1. Build the conflict's stance axis ONCE from its lexicon
     (babelbias.stance_lexicons.LEXICONS), reusing
     exp_021_stance_axis_sweep.build_axis — embeds the minimal-edit seed
     pairs with text-embedding-3-small and returns a normalised
     pole-A-minus-pole-B axis vector.
  2. Project each LADDER EVENT's response embeddings onto that single axis,
     reusing exp_021_stance_axis_sweep.project_responses (loads
     data/<ladder_event>/llm_embeddings/<model>/<ladder_event>/*.json,
     1536-dim text-embedding-3-small vectors, same space as the axis).
  3. Per ladder event, compute per-language mean stance projection and the
     cross-language gap = max(per-lang mean) - min(per-lang mean) across that
     conflict's native languages.

A positive projection aligns a response with pole-A framing vocabulary, a
negative projection with pole-B framing vocabulary (see lexicon labels).

IMPORTANT — interpretive caveats baked into this arm:
  * Stance lexicons are tuned to MODERN framing vocabulary (attacked,
    occupied, bombed, escalated ...). We therefore restrict to post-1900
    ladder events only, and the whole arm is flagged EXPLORATORY.
  * Non-English ladder responses are machine-translated prompts; non-EN
    per-language means are flagged as machine-translated/exploratory.

Outputs (under data/exp_007_timeline/):
  stance_ru_uk_core.csv        rows = ladder_event × lang (mean, n)
  stance_israel_palestine.csv
  stance_india_pakistan.csv
  stance_all.csv               combined, with a `conflict` column

Nothing is committed; this is an analysis arm, not a deck asset.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code" / "src"))

# Reuse the exp_021 axis + projection machinery verbatim. Importing the module
# also runs its load_dotenv(ENV_PATH), so OPENAI creds for build_axis are set.
from exp_021_stance_axis_sweep import build_axis, project_responses  # noqa: E402
from babelbias.stance_lexicons import LEXICONS, StanceLexicon  # noqa: E402

OUT_DIR = ROOT / "data" / "exp_007_timeline"

# Each conflict: lexicon key (axis source) → ordered post-1900 ladder events
# and the native language set the cross-language gap is measured over.
CONFLICTS: dict[str, dict] = {
    "ru_uk_core": {
        "langs": ("en", "ru", "uk"),
        "ladder": [
            "ruuk_1933_holodomor",
            "ruuk_1954_crimea_transfer",
            "ruuk_1991_independence",
            "ru_uk_core",
            "ruuk_2022_invasion",
        ],
    },
    "israel_palestine": {
        "langs": ("en", "he", "ar"),
        "ladder": [
            "ilps_1917_balfour",
            "ilps_1948_nakba",
            "ilps_2000_intifada",
            "ilps_2023_gaza",
        ],
    },
    "india_pakistan": {
        "langs": ("en", "hi", "ur"),
        "ladder": [
            "inpk_1947_partition",
            "inpk_1971_bangladesh",
            "inpk_1999_kargil",
            "inpk_2019_article370",
        ],
    },
    # New lexicons added 2026-06-26 (post-1900 events only; 1894/1453/1821/1772
    # are pre-modern → excluded since the lexicons use modern framing verbs).
    "china_japan": {
        "langs": ("en", "zh", "ja"),
        "ladder": [
            "cnjp_1931_manchuria",
            "cnjp_1937_nanjing",
            "cnjp_2012_senkaku",
        ],
    },
    "greece_turkey": {
        "langs": ("en", "el", "tr"),
        "ladder": [
            "grtr_1922_smyrna",
            "grtr_1974_cyprus",
        ],
    },
    "poland_russia": {
        "langs": ("en", "pl", "ru"),
        "ladder": [
            "plru_1920_warsaw",
            "plru_1940_katyn",
            "plru_1944_warsaw_uprising",
        ],
    },
    # Settled-separation baselines — the stance floor (expect near-zero gap).
    "czech_slovak": {
        "langs": ("en", "cs", "sk"),
        "ladder": ["cssk_1993_divorce"],
    },
    "norway_sweden": {
        "langs": ("en", "no", "sv"),
        "ladder": ["nosv_1905_dissolution"],
    },
}

# Non-EN languages are machine-translated prompts → flagged exploratory.
EN = "en"


def lang_means(rows: list[dict], langs: tuple[str, ...]) -> dict[str, dict]:
    """Per-language mean stance projection + n for one ladder event.

    Only languages in `langs` (the conflict's native set) are kept; the
    cross-language gap is defined over exactly these.
    """
    bucket: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["lang"] in langs:
            bucket[r["lang"]].append(r["stance"])
    return {
        l: {"mean": float(np.mean(v)), "n": len(v)}
        for l, v in bucket.items()
        if v
    }


def cross_lang_gap(per_lang: dict[str, dict]) -> float:
    means = [d["mean"] for d in per_lang.values()]
    return (max(means) - min(means)) if len(means) >= 2 else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_conflict(lex_key: str, cfg: dict) -> tuple[list[dict], list[dict]]:
    """Build the axis once, project every ladder event, summarise.

    Returns (per_lang_rows, gap_rows) where:
      per_lang_rows = one row per (ladder_event × lang): mean, n
      gap_rows      = one row per ladder_event: per-lang means + cross-lang gap
    """
    lex: StanceLexicon = LEXICONS[lex_key]
    langs = cfg["langs"]

    axis, sanity = build_axis(lex)
    print(f"\n{'=' * 78}")
    print(f"{lex_key}  ·  {lex.pole_a_label}  ↔  {lex.pole_b_label}")
    print(f"  axis sanity · pole_A {sanity['pole_a_mean']:+.4f} "
          f"pole_B {sanity['pole_b_mean']:+.4f} "
          f"separation {sanity['separation']:+.4f} "
          f"({sanity['n_pairs']} pairs)")
    print(f"  native langs: {', '.join(langs)}  "
          f"(non-EN = machine-translated, exploratory)")
    print(f"{'=' * 78}")

    per_lang_rows: list[dict] = []
    gap_rows: list[dict] = []

    # Header for the printed summary table.
    nonen = [l for l in langs if l != EN]
    hdr = f"  {'ladder_event':<26s} " + "".join(f"{l.upper():>9s}" for l in langs)
    hdr += f"{'gap':>9s}{'n':>7s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for ev in cfg["ladder"]:
        rows = project_responses(ev, axis)
        per_lang = lang_means(rows, langs)

        if not per_lang:
            print(f"  {ev:<26s}  (no responses found)")
            continue

        gap = cross_lang_gap(per_lang)
        n_total = sum(d["n"] for d in per_lang.values())

        # Per-(event × lang) rows.
        for l in langs:
            if l in per_lang:
                per_lang_rows.append({
                    "conflict": lex_key,
                    "ladder_event": ev,
                    "lang": l,
                    "machine_translated": (l != EN),
                    "mean_stance": round(per_lang[l]["mean"], 6),
                    "n": per_lang[l]["n"],
                })

        # Gap row.
        grow = {
            "conflict": lex_key,
            "ladder_event": ev,
            "pole_a_label": lex.pole_a_label,
            "pole_b_label": lex.pole_b_label,
        }
        for l in langs:
            grow[f"mean_{l}"] = (round(per_lang[l]["mean"], 6)
                                 if l in per_lang else None)
        grow["cross_lang_gap"] = round(gap, 6)
        grow["n_total"] = n_total
        gap_rows.append(grow)

        # Printed row.
        cells = ""
        for l in langs:
            cells += (f"{per_lang[l]['mean']:+9.4f}"
                      if l in per_lang else f"{'--':>9s}")
        flag = "  *" if nonen else ""
        print(f"  {ev:<26s} {cells}{gap:+9.4f}{n_total:>7d}{flag}")

    return per_lang_rows, gap_rows


def main() -> None:
    print("#" * 78)
    print("# exp_007 stance-axis arm — EXPLORATORY (lexicons tuned to modern")
    print("# framing vocab; post-1900 ladder events only; non-EN machine-")
    print("# translated). Companion to the cosine temporal-ladder analysis.")
    print("#" * 78)

    all_per_lang: list[dict] = []

    for lex_key, cfg in CONFLICTS.items():
        per_lang_rows, _gap_rows = run_conflict(lex_key, cfg)
        all_per_lang.extend(per_lang_rows)
        write_csv(OUT_DIR / f"stance_{lex_key}.csv", per_lang_rows)

    write_csv(OUT_DIR / "stance_all.csv", all_per_lang)

    print(f"\n{'#' * 78}")
    print("# Files written:")
    for lex_key in CONFLICTS:
        print(f"#   {OUT_DIR / f'stance_{lex_key}.csv'}")
    print(f"#   {OUT_DIR / 'stance_all.csv'}")
    print("#" * 78)


if __name__ == "__main__":
    main()
