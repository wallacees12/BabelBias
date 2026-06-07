"""Hero figure 2 — triangulated staircase.

Three vertically-stacked panels sharing a 5-conflict x-axis, ordered by
mean EN-ingroup-pull magnitude on cosine.  The argument: three orthogonal
measurement modalities all stair-step together with external contestation
intensity.

Panel A — mean cosine EN-ingroup pull per conflict from
`exp_006_topic_vs_language.csv`; the headline scaling result.
Panel B — between-language stance-axis gap per conflict.  RU-UK from
`exp_021_stance_axis_summary.csv` and IL-PS from
`q03_intifada_stance_axis_summary.csv`.  IP / Taiwan / Falklands are
hatched "exp_021 pending" because the cross-conflict stance sweep has
not landed yet.
Panel C — YandexGPT refusal rate per conflict from
`exp_022_yandex_refusal_cross_conflict` (2026-05-18 findings entry).

No in-image titles or footers per BabelBias figure rule.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pubfig

ROOT = Path(__file__).resolve().parents[2]
EXP006_CSV = ROOT / "data" / "Russia-Ukraine" / "analysis" / "exp_006_topic_vs_language.csv"
EXP021_RUUK_CSV = ROOT / "data" / "Russia-Ukraine" / "analysis" / "exp_021_stance_axis_summary.csv"
EXP021_ILPS_CSV = ROOT / "data" / "israel_palestine" / "analysis" / "q03_intifada_stance_axis_summary.csv"
OUT_DIR = ROOT / "Presentations" / "figures" / "hero"

# Locked ordering — matches Fig 1's column order.
CONFLICTS = [
    ("ru_uk_core",        "RU-UK",      ["ru", "uk"]),
    ("israel_palestine",  "IL-PS",      ["he", "ar"]),
    ("india_pakistan",    "IP",         ["hi", "ur"]),
    ("taiwan_strait",     "Taiwan",     ["zh"]),
    ("falklands",         "Falklands",  ["es"]),
]
LABELS = [label for _, label, _ in CONFLICTS]

YANDEX_REFUSAL = {  # per 2026-05-18 sweep
    "ru_uk_core":       100,
    "israel_palestine":  77,
    "india_pakistan":    52,
    "taiwan_strait":      6,
    "falklands":         22,
}


def panel_a_cosine() -> np.ndarray:
    df = pd.read_csv(EXP006_CSV)
    out = []
    for event_key, _, native in CONFLICTS:
        sub = df[df["event"] == event_key]
        en_lift = sub[sub["lang"] == "en"]["topic_lift"].mean()
        nat_lift = sub[sub["lang"].isin(native)]["topic_lift"].mean()
        out.append(en_lift - nat_lift)
    return np.array(out)


def panel_b_stance() -> tuple[np.ndarray, np.ndarray]:
    """Return (gap_values, has_data_mask) for the 5 conflicts."""
    gaps = np.full(len(CONFLICTS), np.nan)

    ru_uk = pd.read_csv(EXP021_RUUK_CSV)
    piv = ru_uk.pivot_table(index="model", columns="lang", values="mean")
    if {"en", "ru", "uk"}.issubset(piv.columns):
        gap_ru_uk = (piv["en"] - piv[["ru", "uk"]].mean(axis=1)).abs().mean()
        gaps[0] = gap_ru_uk

    il_ps = pd.read_csv(EXP021_ILPS_CSV)
    gap_il_ps = il_ps["ar_minus_he"].abs().mean()
    gaps[1] = gap_il_ps

    mask = ~np.isnan(gaps)
    return gaps, mask


def panel_c_refusal() -> np.ndarray:
    return np.array([YANDEX_REFUSAL[k] for k, _, _ in CONFLICTS], dtype=float)


def _strip_default_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")


def render(out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    cos = panel_a_cosine()
    stance, stance_has = panel_b_stance()
    refusal = panel_c_refusal()

    fig, (axA, axB, axC) = plt.subplots(
        3, 1, figsize=(8.4, 8.0), dpi=150, sharex=True,
        gridspec_kw={"hspace": 0.42, "left": 0.13, "right": 0.97,
                     "top": 0.97, "bottom": 0.09},
    )

    # ---- Panel A ----------------------------------------------------------
    pubfig.bar(
        cos,
        category_names=LABELS,
        x_label=" ",
        y_label="EN ingroup pull",
        color_palette=["#1F4E79"],
        legend_show=False,
        category_spacing=1.0,
        ax=axA,
    )
    _strip_default_axes(axA)
    axA.set_ylabel("EN ingroup pull\n(cos)", fontsize=9)
    axA.axhline(0, color="#999", linewidth=0.6)
    for i, v in enumerate(cos):
        axA.text(i, v + 0.012, f"{v:+.2f}", ha="center", va="bottom",
                 fontsize=8.5, color="#222")
    axA.set_ylim(0, max(cos) * 1.25)

    # ---- Panel B ----------------------------------------------------------
    stance_safe = np.where(stance_has, stance, 0.0)
    pubfig.bar(
        stance_safe,
        category_names=LABELS,
        x_label=" ",
        y_label="|stance gap|",
        color_palette=["#A50026"],
        legend_show=False,
        category_spacing=1.0,
        ax=axB,
    )
    _strip_default_axes(axB)
    axB.set_ylabel("Stance gap (abs)\n(exp_021 axis)", fontsize=9)
    axB.axhline(0, color="#999", linewidth=0.6)
    ylim_top = max(0.18, float(np.nanmax(stance) * 1.25))
    axB.set_ylim(0, ylim_top)
    for i, ok in enumerate(stance_has):
        if ok:
            axB.text(i, stance[i] + 0.008, f"{stance[i]:.2f}",
                     ha="center", va="bottom", fontsize=8.5, color="#222")
        else:
            axB.add_patch(
                plt.Rectangle(
                    (i - 0.35, 0), 0.7, ylim_top * 0.92,
                    facecolor="#e8e8e8", edgecolor="#bbb",
                    hatch="///", linewidth=0.5, zorder=2,
                )
            )
            axB.text(i, ylim_top * 0.5, "exp_021\npending",
                     ha="center", va="center", fontsize=7.5,
                     color="#666", zorder=3)

    # ---- Panel C ----------------------------------------------------------
    pubfig.bar(
        refusal,
        category_names=LABELS,
        x_label="Conflict (ordered by contestation intensity)",
        y_label="YandexGPT refusal (%)",
        color_palette=["#444444"],
        legend_show=False,
        category_spacing=1.0,
        ax=axC,
    )
    _strip_default_axes(axC)
    axC.set_xlabel("Conflict (ordered by contestation intensity)", fontsize=9.5)
    axC.set_ylabel("Yandex refusal\nrate (%)", fontsize=9)
    axC.set_ylim(0, 110)
    for i, v in enumerate(refusal):
        axC.text(i, v + 3, f"{int(v)}%", ha="center", va="bottom",
                 fontsize=8.5, color="#222")

    for ax in (axA, axB, axC):
        ax.tick_params(axis="x", which="both", length=0)
        ax.set_xlim(-0.6, len(LABELS) - 0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#666")
        ax.spines["bottom"].set_color("#666")

    # Override pubfig.bar's internal tight_layout (axes already laid out).
    pubfig.batch_export(
        fig,
        out_dir / "fig2_triangulated_staircase",
        formats=("png", "pdf"),
        spec="nature",
        width="double",
        dpi=300,
        trim=False,
    )
    plt.close(fig)
    return out_dir / "fig2_triangulated_staircase.png"


if __name__ == "__main__":
    print(f"wrote {render()}")
