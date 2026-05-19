"""Animated provider horse-race for the May 11 deck.

Each frame fixes one question (q01 → q02 → … → q09 → all). Bars
show per-provider EN ingroup pull on that question. The final
frame shows the cumulative average across all 9 questions —
i.e. the headline +0.18 number emerging from the per-question
spread.

Output: Presentations/figures/May 11/13_horse_race.gif
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from babelbias.palette import MODEL_COLORS
from babelbias.paths import (
    LLM_EMBEDDINGS_DIR, PROCESSED_LEADS_DIR, PROJECT_ROOT,
)


OUT = (PROJECT_ROOT / "Presentations" / "figures" / "May 11"
       / "13_horse_race.gif")
OUT.parent.mkdir(parents=True, exist_ok=True)

EVENT = "ru_uk_core"
LANGS = ("en", "ru", "uk")
MODELS = [
    "claude-haiku-4-5", "gpt-4o-mini", "gemini-2.5-flash",
    "grok-3-mini", "deepseek-chat",
]
MODEL_LABEL = {
    "claude-haiku-4-5": "Claude Haiku",
    "gpt-4o-mini":      "GPT-4o mini",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "grok-3-mini":      "Grok-3 mini",
    "deepseek-chat":    "DeepSeek chat",
}
QID_LABELS = {
    "q01_little_green_men":  "q01  little green men",
    "q02_crimea_2014":       "q02  Crimea 2014",
    "q03_maidan_revolution": "q03  Maidan",
    "q04_referendum":        "q04  referendum",
    "q05_mh17":              "q05  MH17",
    "q06_crimea_belongs":    "q06  Crimea belongs to",
    "q07_pov_russia":        "q07  pro-Russia speech",
    "q08_pov_ukraine":       "q08  pro-Ukraine speech",
    "q09_bandera":           "q09  Bandera",
}
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


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_pulls() -> dict[tuple[str, str], float]:
    """Return {(model, qid): EN-ingroup-pull (row-centred)}."""
    anchors_en = {qid: np.asarray(json.load(
        open(PROCESSED_LEADS_DIR / f"{slug}_en.json")
    )["embedding"]) for qid, slug in ANCHOR_SLUGS.items()}
    anchors_ru = {qid: np.asarray(json.load(
        open(PROCESSED_LEADS_DIR / f"{slug}_ru.json")
    )["embedding"]) for qid, slug in ANCHOR_SLUGS.items()}
    anchors_uk = {qid: np.asarray(json.load(
        open(PROCESSED_LEADS_DIR / f"{slug}_uk.json")
    )["embedding"]) for qid, slug in ANCHOR_SLUGS.items()}

    pulls: dict[tuple[str, str], float] = {}
    for model in MODELS:
        rdir = LLM_EMBEDDINGS_DIR / model / EVENT
        responses_en: dict[str, list[np.ndarray]] = defaultdict(list)
        for f in rdir.glob("*_en_*.json"):
            try:
                rec = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not rec.get("embedding") or rec.get("refusal"):
                continue
            responses_en[rec["qid"]].append(np.asarray(rec["embedding"]))

        for qid in ANCHOR_SLUGS:
            R = responses_en.get(qid, [])
            if not R:
                pulls[(model, qid)] = float("nan")
                continue
            cos_en = float(np.mean([_cos(r, anchors_en[qid]) for r in R]))
            cos_ru = float(np.mean([_cos(r, anchors_ru[qid]) for r in R]))
            cos_uk = float(np.mean([_cos(r, anchors_uk[qid]) for r in R]))
            row_mean = (cos_en + cos_ru + cos_uk) / 3
            pulls[(model, qid)] = cos_en - row_mean
    return pulls


def main() -> None:
    pulls = compute_pulls()
    qids = list(ANCHOR_SLUGS.keys())

    # Per-frame state: cumulative mean across qids[0..k]
    cum = []
    for k in range(1, len(qids) + 1):
        per_model = {}
        for m in MODELS:
            vals = [pulls.get((m, q), np.nan) for q in qids[:k]]
            per_model[m] = float(np.nanmean(vals))
        cum.append(per_model)

    fig, ax = plt.subplots(figsize=(11, 6.0), constrained_layout=True)

    bar_w = 0.7
    x = np.arange(len(MODELS))
    bars = ax.bar(x, [0] * len(MODELS), bar_w,
                  color=[MODEL_COLORS[m] for m in MODELS],
                  edgecolor="white", linewidth=0.8)
    value_labels = [
        ax.text(xi, 0.005, "", ha="center", va="bottom",
                fontsize=10.5, weight="700", color="#0F172A")
        for xi in x
    ]
    avg_label = ax.text(len(MODELS) - 0.3, 0.30, "", fontsize=12,
                         color="#0F172A", weight="700",
                         ha="right", va="top")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS],
                       fontsize=10, weight="600", color="#0F172A")
    ax.set_ylim(-0.05, 0.32)
    ax.set_ylabel("EN ingroup pull\n(row-centred cosine, cumulative mean)",
                  fontsize=10.5, color="#0F172A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Top-of-figure question header
    qid_text = ax.text(
        0.5, 0.97, "", transform=ax.transAxes,
        ha="center", va="top", fontsize=14, weight="800",
        color="#0F172A",
    )
    state_text = ax.text(
        0.5, 0.88, "", transform=ax.transAxes,
        ha="center", va="top", fontsize=10, color="#475569",
        style="italic",
    )

    # Smoother playback: 30 fps with ease-in-out tweens.
    FPS = 30
    HOLD_FRAMES = 36         # 1.2 s hold per question
    TWEEN_FRAMES = 24        # 0.8 s smooth interpolation
    total_frames = (HOLD_FRAMES + TWEEN_FRAMES) * (len(qids) - 1) + HOLD_FRAMES * 2

    def ease(t: float) -> float:
        """Cubic ease-in-out: removes the linear-tween mechanical feel."""
        if t < 0.5:
            return 4 * t ** 3
        return 1 - ((-2 * t + 2) ** 3) / 2

    def update(frame):
        # Walk through cum[] with tween between adjacent states
        block = HOLD_FRAMES + TWEEN_FRAMES
        idx = min(frame // block, len(cum) - 1)
        local = frame - idx * block
        if local < HOLD_FRAMES or idx == len(cum) - 1:
            t = 1.0
            stage = idx
        else:
            raw_t = (local - HOLD_FRAMES) / TWEEN_FRAMES
            t = ease(float(np.clip(raw_t, 0, 1)))
            stage = idx

        stage = min(max(stage, 0), len(cum) - 1)
        cur = cum[stage]
        nxt = cum[min(stage + 1, len(cum) - 1)]
        for i, m in enumerate(MODELS):
            v = cur[m] * (1 - t) + nxt[m] * t
            bars[i].set_height(v)
            value_labels[i].set_position((i, v + 0.005))
            value_labels[i].set_text(f"{v:+.2f}")

        active_qid = qids[min(stage + 1, len(qids) - 1) if t > 0.5
                          else stage]
        if stage == len(cum) - 1:
            qid_text.set_text("Average across all 9 questions")
            state_text.set_text("…and the headline +0.18 EN ingroup pull emerges")
        else:
            qid_text.set_text(QID_LABELS[active_qid])
            state_text.set_text(
                f"Cumulative mean over {min(stage + (2 if t > 0.5 else 1), len(qids))} of 9 questions"
            )

        avg_v = float(np.nanmean(list(cur.values())) * (1 - t)
                      + np.nanmean(list(nxt.values())) * t)
        avg_label.set_text(f"average across providers: {avg_v:+.2f}")

        return list(bars) + value_labels + [avg_label, qid_text, state_text]

    ani = FuncAnimation(fig, update, frames=total_frames,
                         interval=1000 / FPS, blit=False)

    print(f"Saving GIF to {OUT}…")
    ani.save(OUT, writer=PillowWriter(fps=FPS))
    plt.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
