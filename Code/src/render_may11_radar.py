"""Provider-fingerprint radar chart for the May 11 deck.

Three polar panels (EN / RU / UK), each showing the per-question
ingroup pull (row-centred diagonal) for the 5 May-11 providers
overlaid on the same radar. Lets the eye read each provider's
*bias signature* across the 9 questions in one glance.

Output: Presentations/figures/May 11/10_provider_fingerprint_radar.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from babelbias.palette import MODEL_COLORS
from babelbias.paths import (
    ANALYSIS_DIR, LLM_EMBEDDINGS_DIR, PROCESSED_LEADS_DIR, PROJECT_ROOT,
)


ANCHOR_SLUGS = {
    "q01_little_green_men":  "Little_green_men",
    "q02_crimea_2014":       "2014_Russian_annexation_of_Crimea",
    "q03_maidan_revolution": "Revolution_of_Dignity",
    "q04_referendum":        "2014_Crimean_status_referendum",
    "q05_mh17":              "Malaysia_Airlines_Flight_17",
    "q06_crimea_belongs":    "2014_Russian_annexation_of_Crimea",
    "q07_pov_russia":        "2014_Russian_annexation_of_Crimea",
    "q08_pov_ukraine":       "2014_Russian_annexation_of_Crimea",
    "q09_bandera":           "Stepan_Bandera",
}


OUT = (PROJECT_ROOT / "Presentations" / "figures" / "May 11"
       / "10_provider_fingerprint_radar.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

EVENT = "ru_uk_core"
LANGS = ("en", "ru", "uk")
LANG_LABEL = {"en": "EN", "ru": "RU", "uk": "UK"}
MODELS = [
    "claude-haiku-4-5",
    "gpt-4o-mini",
    "gemini-2.5-flash",
    "grok-3-mini",
    "deepseek-chat",
]
QID_SHORT = {
    "q01_little_green_men":  "q01\nlittle\ngreen men",
    "q02_crimea_2014":       "q02\nCrimea\n2014",
    "q03_maidan_revolution": "q03\nMaidan",
    "q04_referendum":        "q04\nreferendum",
    "q05_mh17":              "q05\nMH17",
    "q06_crimea_belongs":    "q06\nbelongs to",
    "q07_pov_russia":        "q07\npro-RU",
    "q08_pov_ukraine":       "q08\npro-UK",
    "q09_bandera":           "q09\nBandera",
}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_per_question(model: str) -> pd.DataFrame:
    """Compute the 9-column (resp_lang × anchor_lang) cosine matrix per
    qid from the cached response embeddings + Wikipedia anchor
    embeddings. Equivalent shape to the legacy hand-built
    `anchor_per_question.csv` for the 2 models that had one."""
    # Load anchor embeddings (one per qid × lang).
    anchors: dict[tuple[str, str], np.ndarray] = {}
    for qid, slug in ANCHOR_SLUGS.items():
        for lang in LANGS:
            with open(PROCESSED_LEADS_DIR / f"{slug}_{lang}.json") as f:
                anchors[(qid, lang)] = np.asarray(json.load(f)["embedding"])

    # Load response embeddings for this model — group by (qid, lang).
    resp_dir = LLM_EMBEDDINGS_DIR / model / EVENT
    by_cell: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for f in resp_dir.glob("*.json"):
        try:
            rec = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not rec.get("embedding") or rec.get("refusal"):
            continue
        by_cell[(rec["qid"], rec["language"])].append(
            np.asarray(rec["embedding"])
        )

    rows = []
    for qid in ANCHOR_SLUGS:
        row = {"qid": qid}
        for r_lang in LANGS:
            R = by_cell.get((qid, r_lang), [])
            for a_lang in LANGS:
                anchor = anchors[(qid, a_lang)]
                if R:
                    row[f"{r_lang}->{a_lang}"] = float(np.mean([
                        _cos(r, anchor) for r in R
                    ]))
                else:
                    row[f"{r_lang}->{a_lang}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ingroup_pull(df: pd.DataFrame, lang: str) -> np.ndarray:
    """Row-centred diagonal pull for response_lang=lang, per question."""
    others = [l for l in LANGS if l != lang]
    diag  = df[f"{lang}->{lang}"].to_numpy()
    other = df[[f"{lang}->{o}" for o in others]].mean(axis=1).to_numpy()
    return diag - other


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                              subplot_kw=dict(projection="polar"),
                              constrained_layout=True)

    qids = list(QID_SHORT.keys())
    n = len(qids)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])

    # Find shared y-range so panels are comparable.
    all_vals = []
    cached: dict[str, pd.DataFrame] = {m: load_per_question(m) for m in MODELS}
    for m in MODELS:
        for lang in LANGS:
            all_vals.extend(ingroup_pull(cached[m], lang).tolist())
    r_max = max(all_vals) * 1.10
    r_min = min(min(all_vals) * 1.10, -0.05)

    for ax, lang in zip(axes, LANGS):
        for m in MODELS:
            df = cached[m]
            df = df.set_index("qid").reindex(qids).reset_index()
            r = ingroup_pull(df, lang)
            r_closed = np.concatenate([r, r[:1]])
            ax.plot(theta_closed, r_closed,
                    color=MODEL_COLORS[m], linewidth=2.0,
                    marker="o", markersize=5, label=m, zorder=3)
            ax.fill(theta_closed, r_closed,
                    color=MODEL_COLORS[m], alpha=0.08, zorder=2)
        # Reference circle at zero.
        ax.plot(np.linspace(0, 2*np.pi, 200), np.zeros(200),
                color="#0F172A", linewidth=1.0, zorder=1)
        ax.set_xticks(theta)
        ax.set_xticklabels([QID_SHORT[q] for q in qids], fontsize=8)
        ax.set_ylim(r_min, r_max)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3])
        ax.set_yticklabels(["0", ".1", ".2", ".3"], fontsize=8)
        ax.set_title(f"response = {LANG_LABEL[lang]}",
                      fontsize=12, weight="bold", pad=18, color="#0F172A")
        ax.tick_params(axis="x", pad=14)
        ax.grid(color="#cbd5e1", linewidth=0.6)

    # Shared legend at the bottom.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=5, frameon=False, fontsize=10)

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
