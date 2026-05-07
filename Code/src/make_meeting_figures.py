"""
Build cross-model comparison figures for the next advisor meeting.

Reads the per-model analysis CSVs under data/Russia-Ukraine/analysis/<model>/
and writes presentation-ready PNGs to Presentations/figures/.

Two figures:
    1. Ingroup-pull bar chart — diagonal of the row-centered matrix, with 95%
       CI error bars, grouped by response language across the three models.
       Headline result in one glance.
    2. Side-by-side row-centered heatmaps — full 3x3 pattern for each model,
       shared color scale.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from babelbias.config import DEFAULT_LANGS
from babelbias.palette import MODEL_COLORS, figure_palette_legend
from babelbias.paths import (
    ANALYSIS_DIR,
    ANALYSIS_FULL_DIR,
    LLM_EMBEDDINGS_DIR,
    PROJECT_ROOT,
)

sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=0.75,
    rc={
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#333333",
        "axes.linewidth":    0.8,
        "grid.color":        "#e6e6e6",
        "grid.linewidth":    0.6,
        "font.family":       "sans-serif",
        "savefig.dpi":       180,
        "savefig.bbox":      "tight",
    },
)

MODELS = [
    # 🇺🇸 Western (5)
    "claude-haiku-4-5",
    "gpt-4o-mini",
    "gemini-2.5-flash",
    "grok-3-mini",
    "mercury-2",
    # 🇨🇦 Western multilingual (2)
    "c4ai-aya-expanse-32b",
    "command-r7b-arabic-02-2025",
    # 🇨🇳 China — CAC-regulated (4)
    "deepseek-chat",
    "qwen-plus",
    "glm-4.5",
    "baidu/ernie-4.5-300b-a47b",
    # 🇸🇦 Gulf state-research (1)
    "ollama:allam-7b",
    # 🇹🇼 Taiwan state-counter (1)
    "ollama:taide-llama3-8b",
    # 🇮🇱 Israel commercial (1)
    "jamba-mini-2-2026-01",
    # NOTE: yandexgpt deliberately excluded — 270/270 refusal rate; covered by
    # a separate "categorical refusal" slide rather than a cosine bar.
]
EVENT = "ru_uk_core"
LANGS = list(DEFAULT_LANGS)
OUT_DIR = PROJECT_ROOT / "Presentations" / "figures" / "May 25"


def load_model_matrices(model: str, event: str = EVENT, root: Path = ANALYSIS_DIR):
    d = root / model / event
    rc = pd.read_csv(d / "anchor_heatmap_rowcentered.csv", index_col=0).to_numpy()
    ci = pd.read_csv(d / "anchor_heatmap_ci95.csv",        index_col=0).to_numpy()
    return rc, ci


def figure_ingroup_bars(out_path: Path):
    """Grouped bar chart: diagonal pull per (response language, model) with 95% CIs."""
    n = len(MODELS)
    # Cap width — figsize stays slide-friendly even at n=14.
    fig_w = min(16.0, max(8.0, 1.0 * n + 4))
    fig, ax = plt.subplots(figsize=(fig_w, 5.4))

    x = np.arange(len(LANGS))
    group_w = 0.82
    width = group_w / n

    for i, model in enumerate(MODELS):
        rc, ci = load_model_matrices(model)
        diag_vals = np.diag(rc)
        diag_cis  = np.diag(ci)
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, diag_vals, width, yerr=diag_cis,
               color=MODEL_COLORS[model],
               capsize=2, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"response = {l.upper()}" for l in LANGS])
    ax.set_ylabel("Row-centered cosine\n(+ = pulled toward own-language Wikipedia)")
    ax.set_title("Ingroup bias: each language's response pulls toward its own-language Wikipedia\n"
                 f"Row-centered diagonal, 95% CI across 9 questions × 10 samples · {n} providers",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _draw_heatmap_panel(ax, rc, ci, abs_max, *, cmap, title,
                        show_ylabel=False, show_values=True):
    """Render one model's 3x3 row-centered heatmap with the punchy treatment.
    Diagonal cells get bold text + a thick outline; off-diagonals are quieter.

    `show_values=False` suppresses the per-cell value/CI text — needed for the
    15-panel grid where the numbers are illegible at slide scale and just
    clutter the colour read.
    """
    im = ax.imshow(rc, cmap=cmap, vmin=-abs_max, vmax=abs_max, aspect="equal")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(range(3)); ax.set_xticklabels([l.upper() for l in LANGS], fontsize=11)
    ax.set_yticks(range(3)); ax.set_yticklabels([l.upper() for l in LANGS], fontsize=11)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xlabel("Wikipedia anchor", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Response language", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # White separator grid for the "tile" feel
    for k in range(4):
        ax.axhline(k - 0.5, color="white", lw=2.5, zorder=3)
        ax.axvline(k - 0.5, color="white", lw=2.5, zorder=3)

    if show_values:
        for i in range(3):
            for j in range(3):
                val = rc[i, j]
                txt_color = "white" if abs(val) > 0.55 * abs_max else "#1a1a1a"
                on_diag = (i == j)
                ax.text(j, i - 0.06,
                        f"{val:+.3f}",
                        ha="center", va="center",
                        fontsize=14 if on_diag else 11,
                        fontweight="bold" if on_diag else "regular",
                        color=txt_color, zorder=4)
                ax.text(j, i + 0.30,
                        f"±{ci[i,j]:.3f}",
                        ha="center", va="center",
                        fontsize=8, color=txt_color, alpha=0.85, zorder=4)

    # Thick outline on the diagonal — the story cells.
    for i in range(3):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="#111111",
                                   lw=2.2, zorder=5))
    return im


def figure_heatmap_grid(out_path: Path):
    """Row-centered heatmaps for every model, shared color scale.

    Reflows into a 2D grid (auto chooses ncols≈7 to give ~16:9 slide aspect).
    Empty slots are blanked out cleanly.
    """
    matrices = {m: load_model_matrices(m) for m in MODELS}
    abs_max = max(float(np.max(np.abs(rc))) for rc, _ in matrices.values())
    abs_max = round(abs_max + 0.04, 2)
    cmap = sns.color_palette("vlag", as_cmap=True)

    n = len(MODELS)
    # Pick a 2D grid: prefer ~7 cols for a slide-friendly aspect.
    if n <= 5:
        nrows, ncols = 1, n
    elif n <= 10:
        ncols = (n + 1) // 2
        nrows = 2
    else:
        ncols = 7
        nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.9 * ncols, 3.6 * nrows + 0.4),
        constrained_layout=True,
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    im = None
    for idx, model in enumerate(MODELS):
        ax = axes_flat[idx]
        rc, ci = matrices[model]
        im = _draw_heatmap_panel(ax, rc, ci, abs_max,
                                 cmap=cmap, title=model,
                                 show_ylabel=(idx % ncols == 0),
                                 show_values=False)

    # Blank any unused panels.
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    cbar = fig.colorbar(im, ax=axes_flat[:n], shrink=0.65, pad=0.02,
                        label="Row-centered cosine\n(+ = pulled toward own-language Wikipedia)")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)

    fig.suptitle(f"Diagonal pull = ingroup bias — consistent across all {n} providers",
                 fontsize=14, fontweight="bold")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def figure_heatmap_focus(out_path: Path, model: str):
    """Single-model 'zoom' view of the row-centered heatmap.

    Useful for slide animations that sweep from the 5-panel grid into one
    provider — emit `figures/02_heatmap_focus_<model>.png` and let the
    slide tool cross-fade between them.
    """
    rc, ci = load_model_matrices(model)
    # Use the same shared scale so colors don't shift across the sweep.
    abs_max = max(float(np.max(np.abs(load_model_matrices(m)[0]))) for m in MODELS)
    abs_max = round(abs_max + 0.04, 2)

    cmap = sns.color_palette("vlag", as_cmap=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    im = _draw_heatmap_panel(ax, rc, ci, abs_max,
                             cmap=cmap, title="", show_ylabel=True)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02,
                        label="Row-centered cosine\n(+ = pulled toward own-language Wikipedia)")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    fig.suptitle(f"{model} — diagonal is the ingroup-pull signal",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(out_path, dpi=180)
    plt.close()


def figure_ingroup_bars_raw_vs_debiased(out_path: Path):
    """Two-panel bar chart: raw vs debiased ingroup pull.

    Shows that the ingroup-bias effect survives after the language
    subspace (estimated from neutral-topic controls) is projected out
    of both responses and anchors — i.e. it is not a lexical artefact.
    """
    n = len(MODELS)
    # Cap width — 18 inches is plenty even at n=14.
    fig_w = min(18.0, max(12.0, 1.3 * n + 4))
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 5.4), sharey=True)
    x = np.arange(len(LANGS))
    group_w = 0.84
    width = group_w / n

    panels = [
        (axes[0], EVENT,              "Raw embeddings"),
        (axes[1], f"{EVENT}_debiased", "After debiasing (language subspace removed)"),
    ]

    y_max = 0.0
    for ax, event, title in panels:
        for i, model in enumerate(MODELS):
            rc, ci = load_model_matrices(model, event)
            diag_vals = np.diag(rc)
            diag_cis  = np.diag(ci)
            offset = (i - (n - 1) / 2) * width
            ax.bar(x + offset, diag_vals, width, yerr=diag_cis,
                   color=MODEL_COLORS[model],
                   capsize=2, edgecolor="white", linewidth=0.5)
            y_max = max(y_max, float(np.max(diag_vals + diag_cis)))
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"response = {l.upper()}" for l in LANGS])
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Row-centered cosine\n(+ = pulled toward own-language Wikipedia)")
    for ax in axes:
        ax.set_ylim(top=y_max * 1.15)

    fig.suptitle("Ingroup bias: effect survives after projecting out the language subspace",
                 fontsize=12)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def figure_asymmetry_slope(out_path: Path):
    """Slope graph: raw → debiased diagonal value, per response language,
    per provider. Replaces the 'asymmetry' bullet list with a single visual
    that shows EN survives, RU partially survives, UK collapses."""
    raw = {m: load_model_matrices(m, EVENT)[0] for m in MODELS}
    deb = {m: load_model_matrices(m, f"{EVENT}_debiased")[0] for m in MODELS}

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6), sharey=True,
                             constrained_layout=True)
    for ax, lang_idx, lang in zip(axes, range(3), LANGS):
        for model in MODELS:
            raw_v = raw[model][lang_idx, lang_idx]
            deb_v = deb[model][lang_idx, lang_idx]
            ax.plot([0, 1], [raw_v, deb_v],
                    marker="o", color=MODEL_COLORS[model],
                    linewidth=2.4, markersize=8, alpha=0.95,
                    clip_on=False, zorder=3)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Raw", "Debiased"], fontsize=11)
        ax.set_xlim(-0.18, 1.18)
        ax.set_title(f"response = {lang.upper()}", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Diagonal pull (own-language Wiki)\nrow-centered cosine",
                       fontsize=10)

    fig.suptitle("Debiasing — EN survives (content/framing), UK collapses (lexical)",
                 fontsize=14, fontweight="bold")
    plt.savefig(out_path, dpi=180)
    plt.close()


def _normed(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / (n + 1e-12)


def figure_provider_agreement(out_path: Path):
    """5×5 matrix of mean response cosine *between providers* on the same
    (question, language) pair. Confirms providers cluster as one phenomenon
    regardless of training origin (US / China / X-Twitter).

    Metric: for every (qid, lang), pair provider A's 10 samples with
    provider B's 10 samples → 100 cosines → mean. Then average over the 27
    (qid, lang) cells. This holds question/language constant so what's left
    is genuine inter-provider agreement."""
    by_mql: dict = defaultdict(list)
    for model in MODELS:
        d = LLM_EMBEDDINGS_DIR / model / EVENT
        for p in sorted(d.iterdir()):
            if p.suffix != ".json":
                continue
            rec = json.loads(p.read_text())
            by_mql[(model, rec["qid"], rec["language"])].append(
                _normed(np.asarray(rec["embedding"]))
            )

    n = len(MODELS)
    qids = sorted({k[1] for k in by_mql.keys()})
    M_sum = np.zeros((n, n))
    cells = 0
    for qid in qids:
        for lang in LANGS:
            Ms = []
            ok = True
            for m in MODELS:
                bucket = by_mql.get((m, qid, lang))
                if not bucket:
                    ok = False; break
                Ms.append(np.stack(bucket))
            if not ok:
                continue
            for i in range(n):
                for j in range(n):
                    M_sum[i, j] += float((Ms[i] @ Ms[j].T).mean())
            cells += 1
    M = M_sum / max(cells, 1)

    fig, ax = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True)
    cmap = sns.color_palette("crest", as_cmap=True)
    vmin = float(M.min()) - 0.02
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=1.0, aspect="equal")

    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(range(n)); ax.set_xticklabels(MODELS, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(MODELS, fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", lw=2.0)
        ax.axvline(k - 0.5, color="white", lw=2.0)

    # Per-cell numbers are dropped at n=14 — 196 labels are illegible at slide
    # scale and steal attention from the colour read. The colorbar carries
    # the magnitude; the diagonal-1.0 sanity check is implied by the colormap.

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02,
                        label="Mean response cosine\n(averaged over 9 questions × 10 samples × 3 languages)")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)

    fig.suptitle("Cross-provider response agreement — every model agrees with every other",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(out_path, dpi=180)
    plt.close()


def figure_lead_vs_full_slope(out_path: Path):
    """Slope graph: diagonal pull computed against lead-section vs full-page
    Wikipedia anchors. Pre-empts Dr Urman's lead-section methodology question
    with empirical evidence: pattern preserved across all 5 providers, but
    magnitudes shift — EN dampened, UK amplified, RU roughly unchanged."""
    lead = {m: load_model_matrices(m, EVENT, ANALYSIS_DIR)[0] for m in MODELS}
    full = {m: load_model_matrices(m, EVENT, ANALYSIS_FULL_DIR)[0] for m in MODELS}

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6), sharey=True,
                             constrained_layout=True)
    for ax, lang_idx, lang in zip(axes, range(3), LANGS):
        for model in MODELS:
            lead_v = lead[model][lang_idx, lang_idx]
            full_v = full[model][lang_idx, lang_idx]
            ax.plot([0, 1], [lead_v, full_v],
                    marker="o", color=MODEL_COLORS[model],
                    linewidth=2.4, markersize=8, alpha=0.95,
                    clip_on=False, zorder=3)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Lead-only", "Full page"], fontsize=11)
        ax.set_xlim(-0.18, 1.18)
        ax.set_title(f"response = {lang.upper()}", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Diagonal pull (own-language Wiki)\nrow-centered cosine",
                       fontsize=10)

    fig.suptitle("Sensitivity check — anchor choice (lead vs full page) shifts magnitudes; pattern survives",
                 fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=180)
    plt.close()


def _safe_model_name(m: str) -> str:
    """Filesystem-safe version of model id (handles slashes, colons, dots)."""
    return m.replace("/", "_").replace(":", "_").replace(".", "_")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_palette_legend(OUT_DIR / "00_palette_legend.png")
    figure_ingroup_bars(OUT_DIR / "01_ingroup_bars.png")
    figure_heatmap_grid(OUT_DIR / "02_heatmap_grid.png")
    figure_ingroup_bars_raw_vs_debiased(OUT_DIR / "03_ingroup_bars_raw_vs_debiased.png")
    figure_asymmetry_slope(OUT_DIR / "06_asymmetry_slope.png")
    figure_provider_agreement(OUT_DIR / "07_provider_agreement.png")

    # Lead-vs-full slope was a 11 May figure; for the 25 May matrix we
    # don't have analysis_full data on the new providers (would require
    # re-anchoring every response against full-page Wikipedia bodies).
    # Skip if any model is missing — render only when complete.
    have_full = all(
        (ANALYSIS_FULL_DIR / m / EVENT / "anchor_heatmap_rowcentered.csv").exists()
        for m in MODELS
    )
    if have_full:
        figure_lead_vs_full_slope(OUT_DIR / "09_lead_vs_full_slope.png")
    else:
        missing = [m for m in MODELS
                   if not (ANALYSIS_FULL_DIR / m / EVENT / "anchor_heatmap_rowcentered.csv").exists()]
        print(f"Skipping lead-vs-full slope figure — analysis_full missing for "
              f"{len(missing)} model(s): {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}")

    # Per-model focus heatmaps for slide-tool zoom animations.
    for model in MODELS:
        figure_heatmap_focus(OUT_DIR / f"02_heatmap_focus_{_safe_model_name(model)}.png", model)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
