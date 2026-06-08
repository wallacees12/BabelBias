"""exp_021 · stance-axis lexicon ablation (leave-one-pair-out robustness).

Reviewer-defence for the cross-conflict stance-gap headline. The stance axis
per conflict is built from 12 hand-written minimal-edit seed pairs; a sceptic's
objection is that the +0.131…+0.033 ranking could be an artefact of *which*
pairs were chosen. This script removes that freedom empirically.

Method — leave-one-pair-out:
  For each conflict, drop one of the 12 pairs at a time, re-fit the axis from
  the remaining 11 pairs (same centroid-difference construction as the real
  sweep), re-project every already-embedded response, and recompute the mean
  per-provider cross-language gap. 12 pairs × 5 conflicts = 60 re-fits.

If the gap (and the 5-conflict ranking) barely move under every drop, no single
seed pair is load-bearing → the finding is robust to lexicon choice, which is a
stronger claim than pre-registration alone.

Cost: re-uses the exp_006 response embeddings already on disk (no new response
calls). Only the 120 seed sentences are embedded, once each (<$0.01). The axis
math is imported verbatim from ``exp_021_stance_axis_sweep`` so the ablation
re-fits are byte-identical to the committed headline run.

Outputs:
  Presentations/figures/exp_021/lexicon_ablation.csv          (long: one row / drop + baseline)
  Presentations/figures/exp_021/lexicon_ablation_summary.csv  (one row / conflict)
  Presentations/figures/exp_021/lexicon_ablation.png          (title-less figure)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Importing the sweep module loads .env (OpenAI key) + sets sys.path, and gives
# us the *exact* embedding / axis / summary helpers the headline run used.
from exp_021_stance_axis_sweep import (
    ROOT,
    FIG_DIR,
    embed_seeds,
    normed,
    summarise,
)
from babelbias.stance_lexicons import LEXICONS, StanceLexicon

BASE_CONFLICTS = ["ru_uk_core", "india_pakistan", "israel_palestine",
                  "taiwan_strait", "falklands"]


# ── Load response embeddings once per conflict ───────────────────────────
def load_response_matrix(event: str) -> tuple[list[dict], np.ndarray]:
    """Walk the response-embedding tree for ``event`` and return (meta, R).

    Mirrors ``exp_021_stance_axis_sweep.project_responses`` exactly (same
    path, glob, and refusal/empty filters) but returns the raw embedding
    matrix so the 12 leave-one-out axes can be projected as cheap matmuls
    instead of re-reading disk 12 times. ``meta[k]`` describes row ``k`` of R.
    """
    emb_root = ROOT / "data" / event / "llm_embeddings"
    meta: list[dict] = []
    vecs: list[np.ndarray] = []
    if not emb_root.is_dir():
        return meta, np.empty((0, 0))
    for ev_dir in sorted(emb_root.glob(f"**/{event}")):
        if not ev_dir.is_dir():
            continue
        model = str(ev_dir.parent.relative_to(emb_root))
        for f in sorted(ev_dir.glob("*.json")):
            try:
                rec = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not rec.get("embedding") or rec.get("refusal"):
                continue
            meta.append({"model": model,
                         "lang": rec.get("language", "?"),
                         "qid": rec.get("qid", "?")})
            # No explicit dtype — identical to the sweep's np.asarray path
            # (JSON floats already decode to float64), so the projection math
            # is textually byte-identical to exp_021_stance_axis_sweep.
            vecs.append(np.asarray(rec["embedding"]))
    R = np.vstack(vecs) if vecs else np.empty((0, 0))
    return meta, R


def gap_from_axis(meta: list[dict], R: np.ndarray, axis: np.ndarray) -> dict:
    """Project R onto ``axis`` and return the per-provider cross-language gap.

    Replicates the sweep's headline statistic: per (model, lang) mean stance,
    then per provider (max-language − min-language) for providers seen in ≥2
    languages, then the mean and max of those provider spreads.
    """
    scores = R @ axis
    rows = [{"model": m["model"], "lang": m["lang"], "stance": float(s)}
            for m, s in zip(meta, scores)]
    summary = summarise(rows)
    grid: dict[str, dict[str, float]] = defaultdict(dict)
    for r in summary:
        grid[r["model"]][r["lang"]] = r["mean"]
    spreads = [max(v.values()) - min(v.values())
               for v in grid.values() if len(v) >= 2]
    return {
        "mean_provider_spread": float(np.mean(spreads)) if spreads else float("nan"),
        "max_provider_spread": float(np.max(spreads)) if spreads else float("nan"),
        "n_providers": len(grid),
        "n_languages": len({m["lang"] for m in meta}),
    }


def seed_separation(A: np.ndarray, B: np.ndarray, axis: np.ndarray) -> float:
    """Mean pole-A projection minus mean pole-B projection (axis quality)."""
    return float((A @ axis).mean() - (B @ axis).mean())


# ── Ablation per conflict ────────────────────────────────────────────────
def ablate_conflict(event: str) -> dict:
    """Full-lexicon baseline + 12 leave-one-out re-fits for one conflict."""
    lex: StanceLexicon = LEXICONS[event]
    A = embed_seeds(lex.pole_a_seeds)          # (n_pairs, d)
    B = embed_seeds(lex.pole_b_seeds)
    n_pairs = len(lex.pole_a_seeds)

    meta, R = load_response_matrix(event)
    if R.size == 0:
        raise SystemExit(f"No response embeddings found for '{event}' "
                         f"under data/{event}/llm_embeddings")

    full_axis = normed(A.mean(axis=0) - B.mean(axis=0))
    full = gap_from_axis(meta, R, full_axis)
    full["axis_seed_separation"] = seed_separation(A, B, full_axis)

    loo: list[dict] = []
    for i in range(n_pairs):
        Ai = np.delete(A, i, axis=0)
        Bi = np.delete(B, i, axis=0)
        axis_i = normed(Ai.mean(axis=0) - Bi.mean(axis=0))
        g = gap_from_axis(meta, R, axis_i)
        loo.append({
            "dropped_idx": i,
            "dropped_pole_a": lex.pole_a_seeds[i],
            "dropped_pole_b": lex.pole_b_seeds[i],
            "mean_provider_spread": g["mean_provider_spread"],
            "max_provider_spread": g["max_provider_spread"],
            "axis_seed_separation": seed_separation(Ai, Bi, axis_i),
        })
    return {"event": event, "lex": lex, "n_pairs": n_pairs,
            "full": full, "loo": loo}


# ── Ranking-stability check across conflicts ─────────────────────────────
def rank_of(event: str, gap: float, baseline_gaps: dict[str, float]) -> int:
    """1-based rank of ``event`` (gap descending) when only its gap changes."""
    gaps = dict(baseline_gaps)
    gaps[event] = gap
    order = sorted(gaps, key=lambda e: -gaps[e])
    return order.index(event) + 1


# ── Figure (title-less per project figure rule) ──────────────────────────
def render_figure(results: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white", context="talk")

    # Conflicts ordered by full-lexicon gap (largest at top, RU-UK at bottom).
    results = sorted(results, key=lambda r: r["full"]["mean_provider_spread"])
    labels = [r["event"] for r in results]
    fig, ax = plt.subplots(figsize=(11, max(4.0, 0.9 * len(results))),
                           constrained_layout=True)

    for i, r in enumerate(results):
        loo_gaps = [d["mean_provider_spread"] for d in r["loo"]]
        full_gap = r["full"]["mean_provider_spread"]
        lo, hi = min(loo_gaps), max(loo_gaps)
        # leave-one-out range as a soft bar
        ax.plot([lo, hi], [i, i], color="#CBD5E1", linewidth=6,
                solid_capstyle="round", zorder=1)
        # individual leave-one-out re-fits
        ax.scatter(loo_gaps, np.full(len(loo_gaps), i), s=45,
                   color="#64748B", alpha=0.65, zorder=2,
                   label="leave-one-pair-out (12×)" if i == 0 else None)
        # full-lexicon headline value
        ax.scatter([full_gap], [i], s=320, marker="D",
                   color="#0F172A", edgecolor="white", linewidth=1.6,
                   zorder=3, label="full lexicon (12 pairs)" if i == 0 else None)

    ax.set_yticks(np.arange(len(results)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.axvline(0, color="#475569", linewidth=0.9, linestyle=":", zorder=0)
    ax.set_xlabel("mean per-provider cross-language stance gap", fontsize=12,
                  color="#0F172A")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors="#475569")
    ax.legend(loc="lower right", fontsize=11, frameon=False)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── CSV writers ──────────────────────────────────────────────────────────
def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="exp_021 stance-axis lexicon ablation (leave-one-pair-out)")
    ap.add_argument("--event", default=None,
                    help="Comma-separated event slugs (default = 5 base "
                         "conflicts).")
    args = ap.parse_args()
    selected = ([e.strip() for e in args.event.split(",")]
                if args.event else list(BASE_CONFLICTS))
    unknown = [e for e in selected if e not in LEXICONS]
    if unknown:
        raise SystemExit(f"Unknown event(s): {unknown}. Known: {sorted(LEXICONS)}")

    print("=" * 86)
    print(f"exp_021 · lexicon ablation (leave-one-pair-out) · {', '.join(selected)}")
    print("=" * 86)

    results = []
    for event in selected:
        print(f"\n── {event} ──", flush=True)
        r = ablate_conflict(event)
        full_gap = r["full"]["mean_provider_spread"]
        loo_gaps = [d["mean_provider_spread"] for d in r["loo"]]
        print(f"  full-lexicon gap {full_gap:+.4f}  "
              f"({r['full']['n_providers']} providers, "
              f"{r['n_pairs']} pairs)")
        print(f"  leave-one-out gap range [{min(loo_gaps):+.4f}, "
              f"{max(loo_gaps):+.4f}]  std {np.std(loo_gaps):.4f}")
        results.append(r)

    baseline_gaps = {r["event"]: r["full"]["mean_provider_spread"]
                     for r in results}
    baseline_rank = {e: rank_of(e, g, baseline_gaps)
                     for e, g in baseline_gaps.items()}

    # ── long CSV: baseline + every drop, with ranking under each drop ──
    long_rows: list[dict] = []
    summary_rows: list[dict] = []
    for r in results:
        event = r["event"]
        full_gap = r["full"]["mean_provider_spread"]
        long_rows.append({
            "event": event, "dropped_idx": -1, "dropped_pole_a": "(none)",
            "dropped_pole_b": "(none — full lexicon)",
            "mean_provider_spread": full_gap,
            "max_provider_spread": r["full"]["max_provider_spread"],
            "axis_seed_separation": r["full"]["axis_seed_separation"],
            "rank": baseline_rank[event],
        })
        loo_gaps = [d["mean_provider_spread"] for d in r["loo"]]
        ranks_seen = set()
        for d in r["loo"]:
            rk = rank_of(event, d["mean_provider_spread"], baseline_gaps)
            ranks_seen.add(rk)
            long_rows.append({
                "event": event, "dropped_idx": d["dropped_idx"],
                "dropped_pole_a": d["dropped_pole_a"],
                "dropped_pole_b": d["dropped_pole_b"],
                "mean_provider_spread": d["mean_provider_spread"],
                "max_provider_spread": d["max_provider_spread"],
                "axis_seed_separation": d["axis_seed_separation"],
                "rank": rk,
            })
        # most-influential drop = largest |Δ gap| vs full lexicon
        worst = max(r["loo"],
                    key=lambda d: abs(d["mean_provider_spread"] - full_gap))
        summary_rows.append({
            "event": event,
            "n_pairs": r["n_pairs"],
            "n_providers": r["full"]["n_providers"],
            "full_gap": round(full_gap, 4),
            "loo_gap_min": round(min(loo_gaps), 4),
            "loo_gap_max": round(max(loo_gaps), 4),
            "loo_gap_std": round(float(np.std(loo_gaps)), 4),
            "max_abs_delta": round(max(abs(g - full_gap) for g in loo_gaps), 4),
            "rank_stable": len(ranks_seen) == 1 and baseline_rank[event] in ranks_seen,
            "most_influential_drop": worst["dropped_pole_a"],
        })

    rank_stable_all = all(s["rank_stable"] for s in summary_rows)

    out_long = FIG_DIR / "lexicon_ablation.csv"
    out_summary = FIG_DIR / "lexicon_ablation_summary.csv"
    out_fig = FIG_DIR / "lexicon_ablation.png"
    write_csv(out_long, long_rows)
    write_csv(out_summary, summary_rows)
    render_figure(results, out_fig)

    print("\n" + "=" * 86)
    print("Lexicon-ablation summary (leave-one-pair-out)")
    print("=" * 86)
    print(f"  {'event':<20s} {'full':>8s} {'loo_min':>9s} {'loo_max':>9s} "
          f"{'std':>7s} {'maxΔ':>7s} {'rank✓':>6s}")
    for s in summary_rows:
        print(f"  {s['event']:<20s} {s['full_gap']:+.4f}  "
              f"{s['loo_gap_min']:+.4f}  {s['loo_gap_max']:+.4f}  "
              f"{s['loo_gap_std']:.4f}  {s['max_abs_delta']:.4f}  "
              f"{'yes' if s['rank_stable'] else 'NO':>6s}")
    print(f"\n  5-conflict ranking stable under all "
          f"{sum(r['n_pairs'] for r in results)} leave-one-out re-fits: "
          f"{'YES' if rank_stable_all else 'NO'}")
    print(f"\n→ {out_long}\n→ {out_summary}\n→ {out_fig}")


if __name__ == "__main__":
    main()
