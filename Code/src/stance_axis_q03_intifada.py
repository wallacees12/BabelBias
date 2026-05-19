"""Stance-axis projection prototype for q03_intifada (IL-PS).

Builds a 1D axis in OpenAI text-embedding-3-small space from a paired
seed lexicon (Israel-blame ↔ Palestine-blame), then projects every
provider × language q03 response onto it. The signed scalar per
response is the stance score: positive → Israel-blame end, negative →
Palestine-blame end.

This is the methodology Smirnov 2026 uses qualitatively, formalised
into an embedding-axis projection. exp_021 candidate scoped to one
question to keep the prototype fast.

Outputs:
    data/israel_palestine/analysis/q03_intifada_stance_axis.csv
    Presentations/figures/q03_stance_axis_prototype.png
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code" / "src"))

from dotenv import load_dotenv
from babelbias.paths import ENV_PATH
load_dotenv(ENV_PATH)

EVENT = "israel_palestine"
QID = "q03_intifada"
LANGS = ("en", "he", "ar")

EMB_ROOT = ROOT / "data" / EVENT / "llm_embeddings"
OUT_DIR  = ROOT / "data" / EVENT / "analysis"
FIG_OUT  = ROOT / "Presentations" / "figures" / "q03_stance_axis_prototype.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUT.parent.mkdir(parents=True, exist_ok=True)


# ── Paired seed lexicon (v2 — minimal-edit pairs) ────────────────────────
# Each Israel-blame seed has a Palestine-blame twin that differs only in
# the agent noun. Topical vocabulary (violence, occupation, peace, root
# cause) is held constant. Centroid difference therefore isolates the
# agent direction in embedding space, not "vocabulary register".
#
# v1 lexicon (free-form phrases, not paired) is preserved in git history
# for comparison — it showed the failure mode this v2 is designed to fix.
ISRAEL_BLAME_SEEDS = [
    "Israeli military violence caused civilian deaths during the Second Intifada",
    "Israel rejected the Camp David peace offer",
    "Israeli settlement expansion provoked the uprising",
    "Israeli aggression escalated the conflict",
    "Israeli occupation caused the violence",
    "Israeli forces attacked civilians",
    "Israel's actions triggered the Second Intifada",
    "Israeli policies are responsible for the violence",
    "Israel provoked the uprising",
    "Israel violated the peace agreements",
    "Israeli leaders bear responsibility for the bloodshed",
    "Israeli aggression is the root cause of the conflict",
]
PALESTINE_BLAME_SEEDS = [
    "Palestinian violence caused civilian deaths during the Second Intifada",
    "Palestinians rejected the Camp David peace offer",
    "Palestinian terror attacks provoked the uprising",
    "Palestinian aggression escalated the conflict",
    "Palestinian terrorism caused the violence",
    "Palestinian militants attacked civilians",
    "Palestinian actions triggered the Second Intifada",
    "Palestinian policies are responsible for the violence",
    "Palestinians provoked the uprising",
    "Palestinians violated the peace agreements",
    "Palestinian leaders bear responsibility for the bloodshed",
    "Palestinian aggression is the root cause of the conflict",
]


# ── Embedding utility ────────────────────────────────────────────────────
def embed_text(text: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.asarray(resp.data[0].embedding)


def normed(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def load_responses(model: str) -> dict[str, list[tuple[int, np.ndarray]]]:
    """{lang: [(repeat, embedding), ...]} for q03 responses (refusals dropped)."""
    out: dict[str, list[tuple[int, np.ndarray]]] = {l: [] for l in LANGS}
    rdir = EMB_ROOT / model / EVENT
    for f in sorted(rdir.glob(f"{QID}_*.json")):
        try:
            rec = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not rec.get("embedding") or rec.get("refusal"):
            continue
        lang = rec.get("language")
        if lang in out:
            out[lang].append((rec.get("repeat", 0), np.asarray(rec["embedding"])))
    return out


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 80)
    print(f"Stance-axis prototype · {QID}")
    print(f"Israel-blame pole: {len(ISRAEL_BLAME_SEEDS)} seeds")
    print(f"Palestine-blame pole: {len(PALESTINE_BLAME_SEEDS)} seeds")
    print("=" * 80)

    # 1. Embed every seed phrase
    print(f"\nEmbedding {len(ISRAEL_BLAME_SEEDS) + len(PALESTINE_BLAME_SEEDS)} "
          f"seed phrases with text-embedding-3-small…")
    israel_vecs    = np.vstack([embed_text(s) for s in ISRAEL_BLAME_SEEDS])
    palestine_vecs = np.vstack([embed_text(s) for s in PALESTINE_BLAME_SEEDS])

    # 2. Build the axis: difference of centroids, normalised
    israel_centroid    = israel_vecs.mean(axis=0)
    palestine_centroid = palestine_vecs.mean(axis=0)
    axis = normed(israel_centroid - palestine_centroid)
    print(f"  axis built · |Israel - Palestine centroid| = "
          f"{np.linalg.norm(israel_centroid - palestine_centroid):.3f}")

    # 3. Sanity-check: seeds should project to opposite signs
    israel_proj    = israel_vecs @ axis
    palestine_proj = palestine_vecs @ axis
    print(f"\n  Sanity check (seeds projected onto own axis):")
    print(f"    Israel-blame seeds  mean = {israel_proj.mean():+.4f}  "
          f"(min {israel_proj.min():+.3f}, max {israel_proj.max():+.3f})")
    print(f"    Palestine-blame seeds mean = {palestine_proj.mean():+.4f}  "
          f"(min {palestine_proj.min():+.3f}, max {palestine_proj.max():+.3f})")
    print(f"    Separation = {israel_proj.mean() - palestine_proj.mean():+.4f}")

    # 4. Project every (provider, language, repeat) response
    models = sorted({p.name for p in EMB_ROOT.iterdir() if p.is_dir()})
    rows = []
    for m in models:
        resp = load_responses(m)
        for lang in LANGS:
            for repeat, vec in resp[lang]:
                score = float(vec @ axis)
                rows.append({"model": m, "lang": lang, "repeat": repeat,
                              "stance_score": score})

    # 5. Save raw CSV
    csv_path = OUT_DIR / "q03_intifada_stance_axis.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "lang", "repeat", "stance_score"])
        w.writeheader(); w.writerows(rows)
    print(f"\n→ wrote {csv_path}  ({len(rows)} response projections)")

    # 6. Summary: per (model, lang) mean ± std
    print("\n" + "=" * 80)
    print(f"  Stance score per (provider, lang).")
    print(f"  + = Israeli-framing vocabulary cluster (matches what HE responses do)")
    print(f"  - = Palestinian-framing vocabulary cluster (matches what AR responses do)")
    print(f"  AR − HE: negative means AR is further toward the Palestinian-framing pole.")
    print("=" * 80)
    print(f"  {'provider':<35s} {'EN':>10s} {'HE':>10s} {'AR':>10s} {'AR-HE':>10s}")

    summary_rows = []
    for m in models:
        per_lang = {l: [] for l in LANGS}
        for r in rows:
            if r["model"] == m:
                per_lang[r["lang"]].append(r["stance_score"])
        en_m = float(np.mean(per_lang["en"])) if per_lang["en"] else np.nan
        he_m = float(np.mean(per_lang["he"])) if per_lang["he"] else np.nan
        ar_m = float(np.mean(per_lang["ar"])) if per_lang["ar"] else np.nan
        gap = ar_m - he_m if not (np.isnan(ar_m) or np.isnan(he_m)) else np.nan
        summary_rows.append({"model": m, "en": en_m, "he": he_m,
                              "ar": ar_m, "ar_minus_he": gap})

        def fmt(v): return f"{v:+.4f}" if not np.isnan(v) else "  n/a   "
        print(f"  {m:<35s} {fmt(en_m):>10s} {fmt(he_m):>10s} "
              f"{fmt(ar_m):>10s} {fmt(gap):>10s}")

    # Save summary
    sum_path = OUT_DIR / "q03_intifada_stance_axis_summary.csv"
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader(); w.writerows(summary_rows)
    print(f"\n→ wrote {sum_path}")

    # 7. Figure
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="white", context="talk")

        fig, ax = plt.subplots(figsize=(12, 6.8), constrained_layout=True)

        # Drop providers with no HE/AR data so the figure stays clean.
        models_sorted = sorted(
            [r for r in summary_rows
             if not (np.isnan(r["he"]) and np.isnan(r["ar"]) and np.isnan(r["en"]))],
            key=lambda r: -(r["ar_minus_he"] if not np.isnan(r["ar_minus_he"]) else 0.0)
        )
        y_pos = np.arange(len(models_sorted))

        # One horizontal row per provider; all three lang markers on the
        # same y-line. Connecting line stretches from min to max of the
        # three points so the spread reads as one horizontal segment.
        lang_colour = {"en": "#0F172A", "he": "#1F4E79", "ar": "#A50026"}
        lang_marker = {"en": "o", "he": "s", "ar": "^"}
        lang_label  = {"en": "EN", "he": "HE", "ar": "AR"}

        legend_seen = set()
        for i, r in enumerate(models_sorted):
            vals = [r[l] for l in LANGS if not np.isnan(r[l])]
            if len(vals) >= 2:
                ax.plot([min(vals), max(vals)], [i, i],
                         color="#CBD5E1", linewidth=1.6, zorder=1, solid_capstyle="round")
            for lang in LANGS:
                v = r[lang]
                if np.isnan(v):
                    continue
                label = lang_label[lang] if lang not in legend_seen else None
                if label:
                    legend_seen.add(lang)
                ax.scatter(v, i, s=180,
                            color=lang_colour[lang], marker=lang_marker[lang],
                            edgecolor="white", linewidth=1.4,
                            zorder=3, label=label)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([r["model"] for r in models_sorted], fontsize=10)
        ax.axvline(0, color="#475569", linewidth=0.9, linestyle=":", zorder=0)
        ax.set_xlabel(
            "← more Palestinian-framing   ·   stance-axis projection   ·   "
            "more Israeli-framing →",
            fontsize=11, color="#0F172A",
        )
        ax.invert_yaxis()                                                # biggest gap at top
        ax.legend(loc="lower right", fontsize=11, frameon=False,
                   ncol=3, columnspacing=1.2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors="#475569")

        fig.savefig(FIG_OUT, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"→ wrote {FIG_OUT}")
    except ImportError as e:
        print(f"  (skipping figure: {e})")


if __name__ == "__main__":
    main()
