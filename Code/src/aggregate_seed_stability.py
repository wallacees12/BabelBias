"""
Aggregate the 5-seed bmcs=200 EVoC sweep into a per-qid stability table.

Reads each seed's layer0_cross_lingual_breakdown.csv from
Presentations/figures/exp_016/seeds/seed{42,0,1,2,3}/, classifies each
cluster's language pattern, and reports for each qid:

  - In how many seeds does qid X form a fully-cross-lingual cluster
    (≥80% qid-pure, all three of EN/RU/UK > 25% of cluster size)?
  - In how many does it produce an EN-vs-(RU+UK) split (one EN-dominant
    cluster + one RU+UK-fused cluster, both ≥80% qid-pure)?
  - In how many does it produce a three-way split (separate EN, RU, UK
    clusters)?

Output: Presentations/figures/exp_016/seeds/stability_summary.csv +
console table.
"""

import csv
from collections import defaultdict
from pathlib import Path

SEEDS = [42, 0, 1, 2, 3]
SEEDS_DIR = (
    Path(__file__).resolve().parents[2]
    / "Presentations" / "figures" / "exp_016" / "seeds"
)
ALL_QIDS = [
    "q01_little_green_men", "q02_crimea_2014", "q03_maidan_revolution",
    "q04_referendum", "q05_mh17", "q06_crimea_belongs",
    "q07_pov_russia", "q08_pov_ukraine", "q09_bandera",
]

# Pattern classification thresholds
QID_PURE = 0.80      # cluster is "about" qid X if ≥80% of its members are X
LANG_DOMINANT = 0.85  # ≥85% of one language → that language dominates
LANG_DUAL_MIN = 0.25  # both langs in a fused pair ≥25% of cluster
LANG_TRI_MIN = 0.20   # all three langs ≥20% of cluster → cross-lingual


def lang_pattern(en, ru, uk):
    n = en + ru + uk
    if n == 0:
        return "empty"
    fe, fr, fu = en/n, ru/n, uk/n
    # Tri-lingual: all three above threshold
    if fe >= LANG_TRI_MIN and fr >= LANG_TRI_MIN and fu >= LANG_TRI_MIN:
        return "cross_lingual"
    # Dominant single language
    if fe >= LANG_DOMINANT:
        return "en_only"
    if fr >= LANG_DOMINANT:
        return "ru_only"
    if fu >= LANG_DOMINANT:
        return "uk_only"
    # RU+UK fusion (no EN)
    if fe < 0.05 and fr >= LANG_DUAL_MIN and fu >= LANG_DUAL_MIN:
        return "ru_uk_fused"
    # EN+RU fusion (no UK)
    if fu < 0.05 and fe >= LANG_DUAL_MIN and fr >= LANG_DUAL_MIN:
        return "en_ru_fused"
    # EN+UK fusion (no RU)
    if fr < 0.05 and fe >= LANG_DUAL_MIN and fu >= LANG_DUAL_MIN:
        return "en_uk_fused"
    return "mixed"


def load_seed(seed):
    path = SEEDS_DIR / f"seed{seed}" / "layer0_cross_lingual_breakdown.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "cluster": int(r["cluster"]),
                "size": int(r["size"]),
                "en": int(r["en"]),
                "ru": int(r["ru"]),
                "uk": int(r["uk"]),
                "qid": r["dominant_qid"],
                "qid_share": float(r["dominant_qid_share"]),
            })
    return rows


def per_qid_pattern(seed_rows):
    """For each qid, emit the list of language-patterns of clusters where
    that qid is dominant (qid_share >= QID_PURE).

    Also captures EN-mixed clusters: at bmcs=200, EN responses to multiple
    qids often share a single cluster (e.g. cluster 0 in seed 42 is
    281/2/3 with 50% q09 + ~14% q07 + others). We separately flag whether
    each qid has substantial EN presence in a non-qid-pure cluster, since
    that's the signature of the EN-vs-(RU+UK) split.
    """
    by_qid_pure = defaultdict(list)
    qid_in_en_mixed = defaultdict(int)  # how many EN responses for qid X live in non-qid-pure EN clusters
    en_responses_per_qid = defaultdict(int)
    for r in seed_rows:
        # Tally EN responses by qid
        if r["qid_share"] >= 0.20:  # qid is at least minor
            en_responses_per_qid[r["qid"]] += r["en"]
        if r["qid_share"] >= QID_PURE:
            by_qid_pure[r["qid"]].append(lang_pattern(r["en"], r["ru"], r["uk"]))
        # If a cluster is EN-dominated but qid-mixed, record the EN size for its top qid
        n = r["en"] + r["ru"] + r["uk"]
        if n > 0 and r["en"] / n >= 0.85 and r["qid_share"] < QID_PURE:
            qid_in_en_mixed[r["qid"]] += r["en"]
    return by_qid_pure, qid_in_en_mixed


def classify_qid_outcome(pure_patterns, en_in_mixed):
    """Three-way classification:
    - 'fully_cross_lingual' : a single qid-pure cluster contains all three langs
    - 'en_vs_ru_uk'         : qid-pure RU+UK-fused cluster + substantial
                              EN presence in EN-dominated mixed clusters
    - 'three_way_split'     : separate qid-pure EN, RU, UK clusters
    - 'partial_split'       : some pure clusters but missing pieces
    - 'no_pure_cluster'     : qid never reaches QID_PURE
    """
    if "cross_lingual" in pure_patterns:
        return "fully_cross_lingual"
    has_en_pure = "en_only" in pure_patterns
    has_ru_pure = "ru_only" in pure_patterns
    has_uk_pure = "uk_only" in pure_patterns
    has_ru_uk = "ru_uk_fused" in pure_patterns
    if has_en_pure and has_ru_pure and has_uk_pure:
        return "three_way_split"
    if has_ru_uk and en_in_mixed >= 100:  # ≥100 EN responses in mixed-qid EN clusters
        return "en_vs_ru_uk"
    if has_ru_uk and en_in_mixed < 100:
        return "ru_uk_fused_only"
    if has_en_pure and has_ru_pure and not has_uk_pure:
        return "en_and_ru_only"
    if not pure_patterns:
        return "no_pure_cluster"
    return "other:" + ",".join(sorted(pure_patterns))


def main():
    print(f"Reading {len(SEEDS)} seeds from {SEEDS_DIR}")
    qid_outcomes = defaultdict(list)  # qid → [outcome per seed]
    cluster_count_per_seed = []
    pattern_count_per_seed = []
    for seed in SEEDS:
        rows = load_seed(seed)
        cluster_count_per_seed.append(len(rows))
        by_qid_pure, en_mixed = per_qid_pattern(rows)
        for q in ALL_QIDS:
            patterns = by_qid_pure.get(q, [])
            outcome = classify_qid_outcome(patterns, en_mixed.get(q, 0))
            qid_outcomes[q].append((seed, outcome, patterns))

    print(f"\nClusters per seed: "
          f"{dict(zip(SEEDS, cluster_count_per_seed))}")

    # Summary table: qid × seed
    print(f"\nPer-qid outcome by seed "
          f"(qid-pure clusters with qid_share ≥ {QID_PURE}):")
    header = f"{'qid':<22} " + " ".join(f"{f'seed{s}':>20}" for s in SEEDS)
    print(header); print("-" * len(header))
    out_rows = ["qid," + ",".join(f"seed{s}" for s in SEEDS) + ",modal_outcome,stability"]
    for qid in ALL_QIDS:
        cells = {s: "—" for s in SEEDS}
        for s, outcome, _ in qid_outcomes[qid]:
            cells[s] = outcome
        outcomes_only = [cells[s] for s in SEEDS]
        modal = max(set(outcomes_only), key=outcomes_only.count)
        stability = outcomes_only.count(modal) / len(SEEDS)
        line = f"{qid:<22} " + " ".join(f"{cells[s]:>20}" for s in SEEDS)
        print(line + f"   modal={modal} ({stability:.0%})")
        out_rows.append(f"{qid}," + ",".join(cells[s] for s in SEEDS)
                        + f",{modal},{stability:.4f}")

    out_path = SEEDS_DIR / "stability_summary.csv"
    out_path.write_text("\n".join(out_rows))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
