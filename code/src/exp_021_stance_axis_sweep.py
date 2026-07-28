"""exp_021 · stance-axis projection sweep across 5 conflicts.

Pipeline (per conflict):
  1. Load pre-registered paired-edit lexicon from babelbias.stance_lexicons.
  2. Embed each seed with text-embedding-3-small, average per pole,
     normalise the centroid-difference vector → axis.
  3. Sanity-check: seeds must project to opposite signs.
  4. Load every existing (provider, lang, qid, repeat) response
     embedding for that conflict, project onto the axis.
  5. Aggregate to per-(provider, lang) means and write per-conflict CSV.

After all 5 conflicts run:
  6. Render per-conflict figures (clean horizontal stripes, no grid).
  7. Render cross-conflict 5-panel small-multiples figure.
  8. Print cross-conflict ranking by mean per-provider cross-lang gap.

Outputs:
  data/<event>/analysis/exp_021_stance_axis.csv          (per response)
  data/<event>/analysis/exp_021_stance_axis_summary.csv  (per provider × lang)
  Presentations/figures/exp_021/<event>_stance_axis.png  (per conflict)
  Presentations/figures/exp_021/cross_conflict_5panel.png
  Presentations/figures/exp_021/cross_conflict_ranking.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code" / "src"))

from dotenv import load_dotenv
from babelbias.paths import ENV_PATH
load_dotenv(ENV_PATH)

from babelbias.stance_lexicons import LEXICONS, StanceLexicon

FIG_DIR = ROOT / "Presentations" / "figures" / "exp_021"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Embedding helpers ────────────────────────────────────────────────────
def embed_text(text: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.asarray(resp.data[0].embedding)


def embed_seeds(seeds: tuple[str, ...]) -> np.ndarray:
    return np.vstack([embed_text(s) for s in seeds])


def normed(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# ── Per-conflict pipeline ────────────────────────────────────────────────
def build_axis(lex: StanceLexicon) -> tuple[np.ndarray, dict]:
    """Returns (axis, sanity_check_dict)."""
    a = embed_seeds(lex.pole_a_seeds)
    b = embed_seeds(lex.pole_b_seeds)
    axis = normed(a.mean(axis=0) - b.mean(axis=0))
    a_proj = a @ axis
    b_proj = b @ axis
    sanity = {
        "pole_a_mean":   float(a_proj.mean()),
        "pole_b_mean":   float(b_proj.mean()),
        "pole_a_min":    float(a_proj.min()),
        "pole_a_max":    float(a_proj.max()),
        "pole_b_min":    float(b_proj.min()),
        "pole_b_max":    float(b_proj.max()),
        "separation":    float(a_proj.mean() - b_proj.mean()),
        "n_pairs":       len(lex.pole_a_seeds),
    }
    return axis, sanity


def project_responses(event: str, axis: np.ndarray) -> list[dict]:
    """Project every response embedding for this event onto the axis."""
    emb_root = ROOT / "data" / event / "llm_embeddings"
    rows = []
    if not emb_root.is_dir():
        return rows
    # Find every `<event>` leaf dir at any depth, so slash-namespaced model
    # IDs (e.g. baidu/ernie-...) that nest into sub-directories are not
    # silently dropped. The model name is the path from emb_root to the leaf.
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
            v = np.asarray(rec["embedding"])
            score = float(v @ axis)
            rows.append({
                "model":   model,
                "qid":     rec.get("qid", "?"),
                "lang":    rec.get("language", "?"),
                "repeat":  rec.get("repeat", 0),
                "stance":  score,
            })
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """Per-(model, lang) mean stance score."""
    bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        bucket[(r["model"], r["lang"])].append(r["stance"])
    out = []
    for (m, l), vals in bucket.items():
        out.append({"model": m, "lang": l,
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "n":    len(vals)})
    return sorted(out, key=lambda r: (r["model"], r["lang"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)


# ── Per-conflict figure ─────────────────────────────────────────────────
def render_conflict_figure(event: str, lex: StanceLexicon,
                            summary: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white", context="talk")

    langs = sorted({r["lang"] for r in summary})
    models = sorted({r["model"] for r in summary})

    # Reshape to {model: {lang: mean}}
    grid: dict[str, dict[str, float]] = defaultdict(dict)
    for r in summary:
        grid[r["model"]][r["lang"]] = r["mean"]

    # Sort by spread (max - min across langs) descending → most-shifted on top
    def spread(m: str) -> float:
        vals = [grid[m].get(l) for l in langs if l in grid[m]]
        return max(vals) - min(vals) if len(vals) >= 2 else 0.0
    models = sorted(models, key=lambda m: -spread(m))

    # Lang colours/markers — extend palette across conflicts
    LANG_COLOUR = {
        "en": "#0F172A", "ru": "#A50026", "uk": "#1F4E79",
        "he": "#1F4E79", "ar": "#A50026", "es": "#A50026",
        "hi": "#A50026", "ur": "#1F4E79", "zh": "#A50026",
        "zh-cn": "#A50026", "zh-tw": "#1F4E79",
    }
    LANG_MARKER = {
        "en": "o", "ru": "^", "uk": "s",
        "he": "s", "ar": "^", "es": "^",
        "hi": "^", "ur": "s", "zh": "^",
        "zh-cn": "^", "zh-tw": "s",
    }

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(models))),
                            constrained_layout=True)
    y_pos = np.arange(len(models))

    legend_seen = set()
    for i, m in enumerate(models):
        vals = [grid[m][l] for l in langs if l in grid[m]]
        if len(vals) >= 2:
            ax.plot([min(vals), max(vals)], [i, i],
                     color="#CBD5E1", linewidth=1.6, zorder=1,
                     solid_capstyle="round")
        for lang in langs:
            v = grid[m].get(lang)
            if v is None:
                continue
            label = lang.upper() if lang not in legend_seen else None
            if label:
                legend_seen.add(lang)
            ax.scatter(v, i, s=170,
                        color=LANG_COLOUR.get(lang, "#475569"),
                        marker=LANG_MARKER.get(lang, "o"),
                        edgecolor="white", linewidth=1.4,
                        zorder=3, label=label)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.axvline(0, color="#475569", linewidth=0.9, linestyle=":", zorder=0)
    ax.set_xlabel(
        f"← more {lex.pole_b_label}   ·   stance-axis projection   ·   "
        f"more {lex.pole_a_label} →",
        fontsize=11, color="#0F172A",
    )
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=11, frameon=False,
               ncol=len(legend_seen), columnspacing=1.2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors="#475569")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_stance_panel(ax, event: str, lex: StanceLexicon,
                        summary: list[dict], is_lhs: bool) -> None:
    """Draw one stance-axis panel into the supplied axes.

    `is_lhs` toggles the larger marker / label sizes used on the headline
    left-hand panel; the four RHS comparison panels use compact sizing.
    """
    LANG_COLOUR = {
        "en": "#0F172A", "ru": "#A50026", "uk": "#1F4E79",
        "he": "#1F4E79", "ar": "#A50026", "es": "#A50026",
        "hi": "#A50026", "ur": "#1F4E79", "zh": "#A50026",
        "zh-cn": "#A50026", "zh-tw": "#1F4E79",
    }
    LANG_MARKER = {
        "en": "o", "ru": "^", "uk": "s", "he": "s",
        "ar": "^", "es": "^", "hi": "^", "ur": "s", "zh": "^",
        "zh-cn": "^", "zh-tw": "s",
    }
    label_fs = 16 if is_lhs else 11
    tick_fs = 13 if is_lhs else 9
    marker_s = 180 if is_lhs else 75
    title_fs = 20 if is_lhs else 14

    langs = sorted({r["lang"] for r in summary})
    grid: dict[str, dict[str, float]] = defaultdict(dict)
    for r in summary:
        grid[r["model"]][r["lang"]] = r["mean"]
    models = sorted(grid.keys())

    def spread(m: str) -> float:
        vals = [grid[m].get(l) for l in langs if l in grid[m]]
        return max(vals) - min(vals) if len(vals) >= 2 else 0.0
    models = sorted(models, key=lambda m: -spread(m))

    for i, m in enumerate(models):
        vals = [grid[m][l] for l in langs if l in grid[m]]
        if len(vals) >= 2:
            ax.plot([min(vals), max(vals)], [i, i],
                     color="#CBD5E1",
                     linewidth=1.6 if is_lhs else 1.0, zorder=1,
                     solid_capstyle="round")
        for lang in langs:
            v = grid[m].get(lang)
            if v is None:
                continue
            ax.scatter(v, i, s=marker_s,
                        color=LANG_COLOUR.get(lang, "#475569"),
                        marker=LANG_MARKER.get(lang, "o"),
                        edgecolor="white",
                        linewidth=0.9 if is_lhs else 0.6, zorder=3)

    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontsize=tick_fs)
    ax.invert_yaxis()
    ax.axvline(0, color="#475569",
                linewidth=0.9 if is_lhs else 0.6,
                linestyle=":", zorder=0)
    ax.set_xlabel(f"{lex.pole_b_label[:5]} ← • → {lex.pole_a_label[:5]}",
                   fontsize=label_fs, color="#0F172A")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors="#475569", labelsize=tick_fs)
    # Panel title (event slug) inside top-left
    ax.text(0.02, 1.02, event, transform=ax.transAxes,
             fontsize=title_fs, color="#0F172A", weight="700", va="bottom")


def render_5panel_figure(per_conflict: dict[str, tuple[StanceLexicon, list[dict]]],
                          out_path: Path) -> None:
    """Cross-conflict figure: 1 large headline panel on the LHS + 2x2 RHS grid.

    The LHS panel is the Russo-Ukrainian pair (the project's primary case study
    and the smallest-gap surprise from the cross-conflict ranking). The four
    comparison conflicts (India-Pakistan, Israel-Palestine, Taiwan strait,
    Falklands) sit in a 2x2 grid on the right. Replaces the previous five
    equally-sized small-multiples layout, which was too cramped to read at
    deck or thesis-page scale.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as mgs
    import seaborn as sns
    sns.set_theme(style="white", context="talk", font_scale=0.75)

    LHS_KEY = "ru_uk_core"
    RHS_ORDER = ["india_pakistan", "israel_palestine",
                  "taiwan_strait", "falklands"]
    # Fall back gracefully if LHS_KEY missing.
    lhs_key = LHS_KEY if LHS_KEY in per_conflict else next(iter(per_conflict))
    rhs_keys = [k for k in RHS_ORDER if k in per_conflict and k != lhs_key]

    fig = plt.figure(figsize=(22, 12), constrained_layout=True)
    gs = mgs.GridSpec(2, 3, figure=fig, width_ratios=[2.3, 1.0, 1.0])

    ax_lhs = fig.add_subplot(gs[:, 0])
    lex, summary = per_conflict[lhs_key]
    _draw_stance_panel(ax_lhs, lhs_key, lex, summary, is_lhs=True)

    rhs_positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for key, (r, c) in zip(rhs_keys, rhs_positions):
        ax = fig.add_subplot(gs[r, c])
        lex, summary = per_conflict[key]
        _draw_stance_panel(ax, key, lex, summary, is_lhs=False)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="exp_021 stance-axis projection sweep")
    ap.add_argument(
        "--event", default=None,
        help="Comma-separated event slugs to project (default = all "
             "registered lexicons). With a subset, the cross-conflict "
             "5-panel + ranking CSV are skipped so the committed "
             "5-conflict deck assets are not clobbered.")
    args = ap.parse_args()

    base_conflicts = {"ru_uk_core", "india_pakistan", "israel_palestine",
                      "taiwan_strait", "falklands"}
    if args.event:
        selected = [e.strip() for e in args.event.split(",")]
        unknown = [e for e in selected if e not in LEXICONS]
        if unknown:
            raise SystemExit(
                f"Unknown event(s): {unknown}. Known: {sorted(LEXICONS)}")
    else:
        selected = list(LEXICONS)

    print("=" * 86)
    print(f"exp_021 · Stance-axis projection · events: {', '.join(selected)}")
    print("=" * 86)

    per_conflict: dict[str, tuple[StanceLexicon, list[dict]]] = {}
    cross_rank = []

    for event in selected:
        lex = LEXICONS[event]
        print(f"\n── {event} ── {lex.pole_a_label} ↔ {lex.pole_b_label} "
              f"({len(lex.pole_a_seeds)} pairs)")

        # Build axis + sanity check
        axis, sanity = build_axis(lex)
        print(f"  axis · pole_A mean {sanity['pole_a_mean']:+.4f}  "
              f"pole_B mean {sanity['pole_b_mean']:+.4f}  "
              f"separation {sanity['separation']:+.4f}")

        # Project all existing response embeddings
        rows = project_responses(event, axis)
        n_models = len({r["model"] for r in rows})
        n_langs = len({r["lang"] for r in rows})
        print(f"  projected {len(rows):>5d} responses across "
              f"{n_models} providers × {n_langs} languages")

        summary = summarise(rows)
        ana_dir = ROOT / "data" / event / "analysis"
        write_csv(ana_dir / "exp_021_stance_axis.csv", rows)
        write_csv(ana_dir / "exp_021_stance_axis_summary.csv", summary)

        # Per-conflict figure
        fig_path = FIG_DIR / f"{event}_stance_axis.png"
        render_conflict_figure(event, lex, summary, fig_path)
        print(f"  → {fig_path}")

        per_conflict[event] = (lex, summary)

        # Cross-conflict gap = mean per-provider (max_lang_mean - min_lang_mean)
        from collections import defaultdict as _dd
        grid = _dd(dict)
        for r in summary:
            grid[r["model"]][r["lang"]] = r["mean"]
        spreads = []
        for m, lang_means in grid.items():
            if len(lang_means) >= 2:
                spreads.append(max(lang_means.values()) - min(lang_means.values()))
        cross_rank.append({
            "event": event,
            "pole_a_label": lex.pole_a_label,
            "pole_b_label": lex.pole_b_label,
            "n_providers": len(grid),
            "n_languages": n_langs,
            "mean_provider_spread": float(np.mean(spreads)) if spreads else float("nan"),
            "max_provider_spread":  float(np.max(spreads))  if spreads else float("nan"),
            "axis_seed_separation": sanity["separation"],
        })

    # Cross-conflict 5-panel + ranking CSV — only on a full base-set run, so a
    # subset run (e.g. --event china_us) does not clobber the committed
    # 5-conflict deck assets.
    cross_rank.sort(key=lambda r: -r["mean_provider_spread"])
    if base_conflicts.issubset(set(selected)):
        panel_path = FIG_DIR / "cross_conflict_5panel.png"
        render_5panel_figure(per_conflict, panel_path)
        print(f"\n→ {panel_path}")

        rank_path = FIG_DIR / "cross_conflict_ranking.csv"
        write_csv(rank_path, cross_rank)
        print(f"→ {rank_path}")
    else:
        print("\n(subset run — skipped cross_conflict 5-panel + ranking CSV)")

    print("\n" + "=" * 86)
    print("Cross-conflict mean per-provider cross-language gap")
    print("=" * 86)
    print(f"  {'event':<22s} {'gap':>8s} {'max':>8s} {'sep':>8s} {'providers':>10s}")
    for r in cross_rank:
        print(f"  {r['event']:<22s} {r['mean_provider_spread']:+.4f}  "
              f"{r['max_provider_spread']:+.4f}  "
              f"{r['axis_seed_separation']:+.4f}  {r['n_providers']:>10d}")


if __name__ == "__main__":
    main()
