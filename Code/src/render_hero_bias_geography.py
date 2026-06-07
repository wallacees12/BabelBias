"""Hero figure 1 — bias geography matrix.

Rows: 14 providers grouped by training-ecosystem, plus Yandex as a special
refusal row.  Columns: 5 conflicts ordered by mean EN-ingroup-pull
magnitude (RU-UK > IL-PS > IP > Taiwan > Falklands).  Cell value: the
EN ingroup pull — `topic_lift(EN) − mean(topic_lift across the conflict's
event-native languages)` from `exp_006_topic_vs_language.csv`.  Higher
cell = stronger EN-favouring ingroup pull.

Yandex did not return cosine-eligible responses on any conflict.  Its row
is hatched and annotated with per-conflict refusal rates from the
`exp_022_yandex_refusal_cross_conflict` findings entry.

Per BabelBias figure rule: no in-image titles or footers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pubfig

from babelbias.palette import ORDERED_MODELS, PALETTE_GROUPS

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_CSV = ROOT / "data" / "Russia-Ukraine" / "analysis" / "exp_006_topic_vs_language.csv"
OUT_DIR = ROOT / "Presentations" / "figures" / "hero"

# Column order — recreated from the Experiments.md 2026-05-07 headline.
CONFLICT_ORDER = [
    ("ru_uk_core",        "RU-UK",        ["ru", "uk"]),
    ("israel_palestine",  "IL-PS",        ["he", "ar"]),
    ("india_pakistan",    "IP",           ["hi", "ur"]),
    ("taiwan_strait",     "Taiwan",       ["zh"]),
    ("falklands",         "Falklands",    ["es"]),
]

# Per `2026-05-18` Yandex × 5-conflict sweep — refusal rates as percent.
YANDEX_REFUSAL = {
    "ru_uk_core":       100,
    "israel_palestine":  77,
    "india_pakistan":    52,
    "taiwan_strait":      6,
    "falklands":         22,
}

# Short display names — pubfig auto-truncation works better with short ticks.
DISPLAY_NAME = {
    "claude-haiku-4-5":             "Claude Haiku 4.5",
    "gpt-4o-mini":                  "GPT-4o-mini",
    "gemini-2.5-flash":             "Gemini 2.5 Flash",
    "grok-3-mini":                  "Grok-3-mini",
    "mercury-2":                    "Mercury 2",
    "deepseek-chat":                "DeepSeek V3",
    "qwen-plus":                    "Qwen Plus",
    "glm-4.5":                      "GLM-4.5",
    "baidu/ernie-4.5-300b-a47b":    "ERNIE 4.5 300B",
    "c4ai-aya-expanse-32b":         "Aya Expanse 32B",
    "command-r7b-arabic-02-2025":   "Command-R 7B AR",
    "ollama:allam-7b":              "ALLaM 7B",
    "ollama:taide-llama3-8b":       "TAIDE Llama3 8B",
    "jamba-mini-2-2026-01":         "Jamba Mini 2",
}


def build_cell_matrix() -> tuple[np.ndarray, list[str]]:
    """Compute the (provider, conflict) cell matrix of EN ingroup pull.

    Returns the matrix and the list of provider keys aligned to its rows.
    Yandex is appended last with NaN cells (rendered as hatched refusal).
    """
    df = pd.read_csv(ANALYSIS_CSV)

    row_keys: list[str] = list(ORDERED_MODELS) + ["yandexgpt"]
    matrix = np.full((len(row_keys), len(CONFLICT_ORDER)), np.nan, dtype=float)

    for ci, (event_key, _, native_langs) in enumerate(CONFLICT_ORDER):
        sub = df[df["event"] == event_key]
        for ri, model in enumerate(ORDERED_MODELS):
            cell = sub[sub["model"] == model]
            if cell.empty:
                continue
            en_row = cell[cell["lang"] == "en"]
            nat_rows = cell[cell["lang"].isin(native_langs)]
            if en_row.empty or nat_rows.empty:
                continue
            en_lift = float(en_row["topic_lift"].mean())
            nat_lift = float(nat_rows["topic_lift"].mean())
            matrix[ri, ci] = en_lift - nat_lift
        # Yandex row stays NaN — handled in the annotation pass.

    return matrix, row_keys


def ecosystem_index(model: str) -> int:
    """Return the index of the ecosystem group this model belongs to."""
    for gi, (_, members) in enumerate(PALETTE_GROUPS):
        if model in members:
            return gi
    return len(PALETTE_GROUPS)  # Yandex / other → last bucket


def render(out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix, row_keys = build_cell_matrix()
    col_labels = [label for _, label, _ in CONFLICT_ORDER]
    row_labels = [DISPLAY_NAME.get(k, "YandexGPT") for k in row_keys]

    # Diverging scale anchored on 0 with a symmetric clip.
    finite = matrix[np.isfinite(matrix)]
    bound = float(np.nanmax(np.abs(finite))) if finite.size else 0.4
    bound = max(0.15, round(bound + 0.02, 2))

    fig, ax = plt.subplots(figsize=(7.4, 8.6), dpi=150)

    # Mask NaN cells so pubfig doesn't render an `nan` annotation; we
    # paint Yandex-style hatches over them in the overlay pass.
    plot_matrix = np.ma.masked_invalid(matrix)

    pubfig.heatmap(
        plot_matrix.filled(0.0),
        x_label=" ",
        y_label=" ",
        colorscale="RdBu_r",
        zmin=-bound,
        zmax=bound,
        annotate=True,
        annotate_fmt=".2f",
        cell_border_line_width=0.6,
        cell_border_color="white",
        cbar=True,
        cbar_label="EN ingroup pull  (topic_lift EN − mean event-native)",
        cbar_shrink=0.55,
        tick_rotation=0.0,
        ax=ax,
    )

    # Suppress pubfig default axis labels and apply our own ticks.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9.5)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.5, family="DejaVu Sans")

    # Overlay neutral grey for missing-data cells (no-sweep providers).
    for (ri, ci), val in np.ndenumerate(matrix):
        if np.isnan(val) and ri != len(row_keys) - 1:
            ax.add_patch(
                plt.Rectangle(
                    (ci - 0.5, ri - 0.5),
                    1, 1,
                    facecolor="#f0f0f0",
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=2,
                )
            )
            ax.text(
                ci, ri,
                "—",
                ha="center", va="center",
                fontsize=10, color="#999", zorder=3,
            )

    # Mute annotation text for cells whose colour is too pale to support
    # white text and too dark for default black-on-light.
    for t in ax.texts:
        try:
            x, y = t.get_position()
            ri = int(round(y))
            ci = int(round(x))
        except Exception:
            continue
        if 0 <= ri < matrix.shape[0] and 0 <= ci < matrix.shape[1]:
            v = matrix[ri, ci]
            if np.isnan(v):
                t.set_visible(False)
                continue
            mag = abs(v) / bound if bound else 0
            t.set_color("white" if mag > 0.55 else "#222")
            t.set_fontsize(8.0)

    # Yandex row hatching + refusal-rate overlay.
    yandex_ri = len(row_keys) - 1
    for ci, (event_key, _, _) in enumerate(CONFLICT_ORDER):
        ax.add_patch(
            plt.Rectangle(
                (ci - 0.5, yandex_ri - 0.5),
                1, 1,
                facecolor="#dddddd",
                edgecolor="white",
                hatch="///",
                linewidth=0.6,
                zorder=2,
            )
        )
        ax.text(
            ci, yandex_ri,
            f"{YANDEX_REFUSAL[event_key]}%\nrefused",
            ha="center", va="center",
            fontsize=7.0, color="#444", zorder=3,
        )

    # Faint horizontal dividers between ecosystems.
    last_group = ecosystem_index(row_keys[0])
    for ri, key in enumerate(row_keys[1:], start=1):
        gi = ecosystem_index(key)
        if gi != last_group:
            ax.axhline(ri - 0.5, color="#888", linewidth=0.55, alpha=0.6, zorder=4)
        last_group = gi

    # Column-mean staircase strip above the heatmap.
    col_means = np.nanmean(matrix, axis=0)
    twin = ax.twiny()
    twin.set_xlim(ax.get_xlim())
    twin.set_xticks(np.arange(len(col_labels)))
    twin.set_xticklabels(
        [f"μ = {v:+.2f}" for v in col_means],
        fontsize=8.2, color="#555",
    )
    twin.tick_params(axis="x", length=0, pad=2)
    for s in twin.spines.values():
        s.set_visible(False)

    plt.tight_layout()

    pubfig.batch_export(
        fig,
        out_dir / "fig1_bias_geography",
        formats=("png", "pdf"),
        spec="nature",
        width="double",
        dpi=300,
    )
    plt.close(fig)
    return out_dir / "fig1_bias_geography.png"


if __name__ == "__main__":
    path = render()
    print(f"wrote {path}")
