"""Assemble per-conflict temporal-gradient tables (exp_007 cross-conflict).

Generalises build_exp_007_gradient.py from RU-UK to any contested-event family.
For each conflict, for each (event, model), reads the row-centred ingroup-pull
matrix (analysis/<model>/<event>/anchor_heatmap_rowcentered.csv), takes the
diagonal (resp_<l> vs wiki_<l>) for that conflict's languages, averages across
the provider set, and writes one tidy row per event to
data/exp_007_timeline/gradient_<conflict>.csv.

The first language in each conflict's `langs` is the EN control; `en_pull`
is reported separately so the cross-conflict figure can compare the shared
control axis. `framing_gap` is the spread of own-language pull across that
conflict's languages.

All non-EN prompts are machine-translated (DeepL), NOT native-reviewed —
exploratory. RU-UK is the validated, deeper spine; IL-PS / IN-PK are the
cross-conflict replication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from babelbias.paths import PROJECT_ROOT, analysis_dir

# 9 providers that completed across the RU-UK full run; reused for every
# conflict so the matrix is comparable. baidu / command-r7b / c4ai-aya are
# excluded (dead endpoint / Cohere trial quota); Yandex is a categorical
# refuser and analysed separately.
MODELS = [
    "claude-haiku-4-5", "deepseek-chat", "gpt-4o-mini", "gemini-2.5-flash",
    "jamba-mini-2-2026-01", "mercury-2", "qwen-plus", "grok-3-mini", "glm-4.5",
]

# conflict -> (langs[0]=EN control, events as (slug, year))
CONFLICTS: dict[str, dict] = {
    "ru_uk": {
        "langs": ["en", "ru", "uk"],
        "events": [
            ("ruuk_0988_baptism", 988), ("ruuk_1654_pereyaslav", 1654),
            ("ruuk_1709_poltava", 1709), ("ruuk_1876_emsukaz", 1876),
            ("ruuk_1933_holodomor", 1933), ("ruuk_1954_crimea_transfer", 1954),
            ("ruuk_1991_independence", 1991), ("ru_uk_core", 2014),
            ("ruuk_2022_invasion", 2022),
        ],
        "live_from": 2022,
    },
    "il_ps": {
        "langs": ["en", "he", "ar"],
        "events": [
            ("ilps_1917_balfour", 1917), ("ilps_1948_nakba", 1948),
            ("ilps_2000_intifada", 2000), ("ilps_2023_gaza", 2023),
        ],
        "live_from": 2023,
    },
    "in_pk": {
        "langs": ["en", "hi", "ur"],
        "events": [
            ("inpk_1947_partition", 1947), ("inpk_1971_bangladesh", 1971),
            ("inpk_1999_kargil", 1999), ("inpk_2019_article370", 2019),
        ],
        "live_from": 2019,
    },
    "cn_jp": {
        "langs": ["en", "zh", "ja"],
        "events": [
            ("cnjp_1894_shimonoseki", 1894), ("cnjp_1931_manchuria", 1931),
            ("cnjp_1937_nanjing", 1937), ("cnjp_2012_senkaku", 2012),
        ],
        "live_from": 2012,
    },
    "gr_tr": {
        "langs": ["en", "el", "tr"],
        "events": [
            ("grtr_1453_constantinople", 1453), ("grtr_1821_independence", 1821),
            ("grtr_1922_smyrna", 1922), ("grtr_1974_cyprus", 1974),
        ],
        "live_from": 1974,
    },
    "pl_ru": {
        "langs": ["en", "pl", "ru"],
        "events": [
            ("plru_1772_partitions", 1772), ("plru_1920_warsaw", 1920),
            ("plru_1940_katyn", 1940), ("plru_1944_warsaw_uprising", 1944),
        ],
        "live_from": 1944,
    },
    # Settled-status baselines: peaceful, mutually-accepted separations.
    # Predicted near-zero framing gap — they anchor the floor of the scale.
    "cs_sk": {
        "langs": ["en", "cs", "sk"],
        "events": [("cssk_1993_divorce", 1993)],
        "live_from": None,
        "settled": True,
    },
    "no_sv": {
        "langs": ["en", "no", "sv"],
        "events": [("nosv_1905_dissolution", 1905)],
        "live_from": None,
        "settled": True,
    },
}

OUT_DIR = PROJECT_ROOT / "data" / "exp_007_timeline"


def gradient_for(conflict: str, cfg: dict) -> pd.DataFrame:
    langs = cfg["langs"]
    rows = []
    for slug, year in cfg["events"]:
        per_lang: dict[str, list[float]] = {l: [] for l in langs}
        models_found = 0
        for m in MODELS:
            f = analysis_dir(slug) / m / slug / "anchor_heatmap_rowcentered.csv"
            if not f.exists():
                continue
            models_found += 1
            rc = pd.read_csv(f, index_col=0)
            for l in langs:
                ri, ci = f"resp_{l}", f"wiki_{l}"
                if ri in rc.index and ci in rc.columns and pd.notna(rc.loc[ri, ci]):
                    per_lang[l].append(float(rc.loc[ri, ci]))
        row = {"conflict": conflict, "event": slug, "year": year,
               "n_models": models_found}
        for l in langs:
            row[f"pull_{l}"] = float(np.mean(per_lang[l])) if per_lang[l] else np.nan
        diags = [row[f"pull_{l}"] for l in langs if not np.isnan(row[f"pull_{l}"])]
        row["en_pull"] = row.get(f"pull_{langs[0]}", np.nan)
        row["mean_pull"] = float(np.mean(diags)) if diags else np.nan
        row["framing_gap"] = (max(diags) - min(diags)) if len(diags) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = []
    for conflict, cfg in CONFLICTS.items():
        df = gradient_for(conflict, cfg)
        out = OUT_DIR / f"gradient_{conflict}.csv"
        df.to_csv(out, index=False)
        cols = ["event", "year", "n_models", "en_pull", "mean_pull", "framing_gap"]
        print(f"\n=== {conflict} ===")
        print(df[cols].round(4).to_string(index=False))
        miss = df[df["n_models"] == 0]["event"].tolist()
        if miss:
            print(f"  ⚠ no analysis yet for: {miss}")
        combined.append(df)
    pd.concat(combined, ignore_index=True).to_csv(
        OUT_DIR / "gradient_all_conflicts.csv", index=False)
    print(f"\nwrote {OUT_DIR}/gradient_<conflict>.csv + gradient_all_conflicts.csv")


if __name__ == "__main__":
    main()
