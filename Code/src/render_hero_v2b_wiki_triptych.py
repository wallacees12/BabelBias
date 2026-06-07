"""Hero figure v2-B — Wikipedia Triptych.

Five ternaries arranged left-to-right by contestation intensity.  Each
ternary's three corners are the Wikipedia anchors of that conflict's
languages.  Points are per-(provider, response-language) centroids
projected to barycentric coordinates from the
`anchor_heatmap_mean.csv` cosines.

For two-language conflicts (Taiwan: EN/ZH, Falklands: EN/ES) the
third corner is rendered as `—` and the points geometrically collapse
to the EN-native edge — the visual punchline: when a conflict has no
opposing native-language narrative, the bias triangle has nowhere to
spread.

Per BabelBias figure rule: no in-image titles or captions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from babelbias.hero_assets import CONFLICTS, ECOSYSTEM_COLOR, PROVIDERS

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Presentations" / "figures" / "hero"

LANG_LABEL = {
    "en": "EN", "ru": "RU", "uk": "UK",
    "he": "HE", "ar": "AR",
    "hi": "HI", "ur": "UR",
    "zh": "ZH", "es": "ES",
}

LANG_COLOR = {
    "en": "#1F4E79",
    "ru": "#A50026", "uk": "#F2B701",
    "he": "#5B9BD5", "ar": "#1B9E77",
    "hi": "#F46D43", "ur": "#1B9E77",
    "zh": "#A50026",
    "es": "#984EA3",
}

# Equilateral triangle in unit coordinates.
TRI_CORNERS = np.array([
    [0.0, 0.0],          # left-bottom
    [1.0, 0.0],          # right-bottom
    [0.5, np.sqrt(3) / 2],  # apex
])


def conflict_data_dir(event_key: str) -> Path:
    return {
        "ru_uk_core":       ROOT / "data" / "Russia-Ukraine" / "analysis",
        "israel_palestine": ROOT / "data" / "israel_palestine"  / "analysis",
        "india_pakistan":   ROOT / "data" / "india_pakistan"    / "analysis",
        "taiwan_strait":    ROOT / "data" / "taiwan_strait"     / "analysis",
        "falklands":        ROOT / "data" / "falklands"         / "analysis",
    }[event_key]


def folder_name_for(event_key: str) -> str:
    return {
        "ru_uk_core":       "ru_uk_core",
        "israel_palestine": "israel_palestine",
        "india_pakistan":   "india_pakistan",
        "taiwan_strait":    "taiwan_strait",
        "falklands":        "falklands",
    }[event_key]


def load_cells(event_key: str, native: list[str]) -> list[dict]:
    """Return list of dicts with provider, resp_lang, w_en/w_n1/w_n2 weights."""
    rows = []
    base = conflict_data_dir(event_key)
    sub = folder_name_for(event_key)

    # Model directory naming differs slightly: baidu/ernie-... uses 'baidu/ernie-4.5-300b-a47b'
    # in CSVs but on disk lives in the 'baidu/' folder.  We treat both the same
    # by using the canonical key but writing to disk-friendly form.
    for prov in PROVIDERS:
        if prov.key == "yandexgpt":
            continue
        # Map provider key to its on-disk subdir
        disk_key = prov.key
        if disk_key.startswith("baidu/"):
            disk_key = "baidu"
        csv = base / disk_key / sub / "anchor_heatmap_mean.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, index_col=0)

        langs = ["en"] + native  # row order we expect

        for lang in langs:
            row = f"resp_{lang}"
            if row not in df.index:
                continue
            cos_en = float(df.at[row, "wiki_en"])
            cos_n1 = float(df.at[row, f"wiki_{native[0]}"]) if f"wiki_{native[0]}" in df.columns else 0.0
            if len(native) >= 2:
                cos_n2 = float(df.at[row, f"wiki_{native[1]}"]) if f"wiki_{native[1]}" in df.columns else 0.0
            else:
                cos_n2 = 0.0
            # Softmax for barycentric weights (same tau as v1 ternary).
            z = np.array([cos_en, cos_n1, cos_n2]) * 12.0
            z -= z.max()
            e = np.exp(z)
            w = e / e.sum()
            rows.append({
                "provider": prov.key,
                "ecosystem": prov.ecosystem,
                "resp_lang": lang,
                "w_en": w[0], "w_n1": w[1], "w_n2": w[2],
            })
    return rows


def project_bary(w_en: float, w_n1: float, w_n2: float) -> tuple[float, float]:
    """Barycentric weights → 2D cartesian in the unit equilateral triangle."""
    pos = w_en * TRI_CORNERS[0] + w_n1 * TRI_CORNERS[1] + w_n2 * TRI_CORNERS[2]
    return float(pos[0]), float(pos[1])


def draw_triangle(ax: plt.Axes, native: list[str], rows: list[dict],
                  conflict_label: str, mean_pull: float) -> None:
    """Render one ternary panel."""
    degenerate = (len(native) == 1)

    # Triangle frame
    tri = np.vstack([TRI_CORNERS, TRI_CORNERS[:1]])
    ax.plot(tri[:, 0], tri[:, 1],
            color="#bbb" if degenerate else "#555",
            linewidth=0.9 if degenerate else 1.3,
            linestyle=":" if degenerate else "-",
            zorder=1)
    # If degenerate, also bold the EN→native edge to highlight where dots actually land.
    if degenerate:
        ax.plot(
            [TRI_CORNERS[0, 0], TRI_CORNERS[1, 0]],
            [TRI_CORNERS[0, 1], TRI_CORNERS[1, 1]],
            color="#333", linewidth=2.2, zorder=2,
        )

    # Internal grid (medians to centroid)
    centroid = TRI_CORNERS.mean(axis=0)
    for i in range(3):
        mid = (TRI_CORNERS[(i + 1) % 3] + TRI_CORNERS[(i + 2) % 3]) / 2
        ax.plot(
            [TRI_CORNERS[i, 0], mid[0]],
            [TRI_CORNERS[i, 1], mid[1]],
            color="#ddd", linewidth=0.45, linestyle=":", zorder=1,
        )

    # Points
    for r in rows:
        x, y = project_bary(r["w_en"], r["w_n1"], r["w_n2"])
        ax.scatter(
            x, y, s=22,
            color=LANG_COLOR.get(r["resp_lang"], "#444"),
            edgecolor="white", linewidth=0.4,
            alpha=0.78, zorder=3,
        )

    # Per-language centroid rings
    by_lang: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        x, y = project_bary(r["w_en"], r["w_n1"], r["w_n2"])
        by_lang.setdefault(r["resp_lang"], []).append((x, y))
    for lang, pts in by_lang.items():
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        ax.scatter(
            cx, cy, s=180,
            facecolor="none",
            edgecolor=LANG_COLOR.get(lang, "#444"),
            linewidth=1.8, zorder=4,
        )

    # Corner labels
    labels = ["en"] + native + (["—"] if degenerate else [])
    corner_offsets = [(-0.04, -0.05), (+0.04, -0.05), (0.0, 0.05)]
    corner_ha = ["right", "left", "center"]
    corner_va = ["top", "top", "bottom"]
    label_colors = ["#1F4E79"] + [LANG_COLOR.get(n, "#444") for n in native] + (["#bbb"] if degenerate else [])
    for i, lab in enumerate(labels):
        text = LANG_LABEL.get(lab, lab) if lab != "—" else "—"
        ax.text(
            TRI_CORNERS[i, 0] + corner_offsets[i][0],
            TRI_CORNERS[i, 1] + corner_offsets[i][1],
            text,
            ha=corner_ha[i], va=corner_va[i],
            fontsize=10.5, fontweight="bold",
            color=label_colors[i] if i < len(label_colors) else "#444",
        )

    # Conflict label + headline pull below the triangle
    ax.text(
        0.5, -0.20,
        conflict_label,
        ha="center", va="top",
        fontsize=13, fontweight="bold", color="#222",
    )
    ax.text(
        0.5, -0.28,
        f"mean EN ingroup pull = +{mean_pull:.2f}",
        ha="center", va="top",
        fontsize=8.6, color="#666",
    )

    # Degenerate note
    if degenerate:
        ax.text(
            0.5, 0.52,
            "no contested\n2nd native language\n→ bias geometry collapses",
            ha="center", va="center",
            fontsize=7.6, color="#999", style="italic",
        )

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.36, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")


def column_mean_pulls() -> dict[str, float]:
    """Per-conflict mean EN ingroup pull from exp_006_topic_vs_language.csv."""
    df = pd.read_csv(ROOT / "data" / "Russia-Ukraine" / "analysis" / "exp_006_topic_vs_language.csv")
    out: dict[str, float] = {}
    for event_key, _, native in CONFLICTS:
        sub = df[df["event"] == event_key]
        en = sub[sub["lang"] == "en"]["topic_lift"].mean()
        nat = sub[sub["lang"].isin(native)]["topic_lift"].mean()
        out[event_key] = float(en - nat)
    return out


def render(out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pulls = column_mean_pulls()

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), dpi=160,
                             gridspec_kw={"wspace": 0.05})
    for ax, (event_key, label, native) in zip(axes, CONFLICTS):
        rows = load_cells(event_key, native)
        draw_triangle(ax, native, rows, label, pulls[event_key])

    # Shared legend below all 5 panels.
    handle_text = []
    used = set()
    for _, _, native in CONFLICTS:
        for lang in ["en"] + native:
            if lang in used:
                continue
            used.add(lang)
            handle_text.append((LANG_LABEL[lang], LANG_COLOR[lang]))

    for i, (lab, col) in enumerate(handle_text):
        x = 0.10 + i * 0.10
        fig.text(x, 0.04, "●", fontsize=14, color=col, ha="center", va="center")
        fig.text(x + 0.013, 0.04, f" {lab}", fontsize=10,
                 color="#222", ha="left", va="center")
    fig.text(0.06, 0.04, "Response language:",
             fontsize=10, color="#222", ha="right", va="center")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.13)

    base = out_dir / "fig_v2b_wiki_triptych"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base.with_suffix(".png")


if __name__ == "__main__":
    print(f"wrote {render()}")
