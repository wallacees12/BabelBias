"""Canonical provider palette — colour encoding for every multi-provider figure
in the deck and the thesis.

`MODEL_COLORS` is the source of truth. `PALETTE_GROUPS` defines the
ecosystem grouping used by `figure_palette_legend`. If you add a provider
to the sweep, update both, re-render the legend card, and update the table
in `Thesis_Document_Ideas.md` § "Canonical model palette".

Encoding principle: hue carries *training-data ecosystem*; within an
ecosystem the hue is graded so individual providers stay distinguishable.
Read the cluster first, then the individual bar.

Yandex caveat: YandexGPT is excluded by design — covered by a separate
"categorical refusal" slide rather than a cosine bar. Add a neutral grey
here if a future refusal-rate figure needs to plot it.
"""

from pathlib import Path

import matplotlib.pyplot as plt


MODEL_COLORS: dict[str, str] = {
    # 🇺🇸 US blues (claude → mercury, dark → light)
    "claude-haiku-4-5":             "#1F4E79",
    "gpt-4o-mini":                  "#2E75B6",
    "gemini-2.5-flash":             "#5B9BD5",
    "grok-3-mini":                  "#8FAADC",
    "mercury-2":                    "#B4C7E7",
    # 🇨🇳 China red→orange
    "deepseek-chat":                "#A50026",
    "qwen-plus":                    "#D73027",
    "glm-4.5":                      "#F46D43",
    "baidu/ernie-4.5-300b-a47b":    "#FDAE61",
    # 🇨🇦 Cohere teals
    "c4ai-aya-expanse-32b":         "#1B9E77",
    "command-r7b-arabic-02-2025":   "#66C2A5",
    # 🇸🇦 Gulf — amber
    "ollama:allam-7b":              "#F2B701",
    # 🇹🇼 Taiwan — green
    "ollama:taide-llama3-8b":       "#4DAF4A",
    # 🇮🇱 Israel — purple
    "jamba-mini-2-2026-01":         "#984EA3",
}


PALETTE_GROUPS: list[tuple[str, list[str]]] = [
    ("US",                          ["claude-haiku-4-5", "gpt-4o-mini",
                                     "gemini-2.5-flash", "grok-3-mini",
                                     "mercury-2"]),
    ("China (CAC-regulated)",       ["deepseek-chat", "qwen-plus", "glm-4.5",
                                     "baidu/ernie-4.5-300b-a47b"]),
    ("Cohere (multilingual)",       ["c4ai-aya-expanse-32b",
                                     "command-r7b-arabic-02-2025"]),
    ("Gulf (Saudi state-research)", ["ollama:allam-7b"]),
    ("Taiwan (state-counter)",      ["ollama:taide-llama3-8b"]),
    ("Israel (commercial)",         ["jamba-mini-2-2026-01"]),
]


# Canonical ordered list of providers used by figure scripts that loop over
# every model in the sweep. Sister scripts that work on a deck-specific
# subset (e.g. May-11 5-model views) define their own MODELS list locally.
ORDERED_MODELS: list[str] = [m for _, members in PALETTE_GROUPS for m in members]


def figure_palette_legend(out_path: Path) -> None:
    """Render the standalone palette card to `out_path`.

    Used in place of an in-figure legend on every multi-provider chart.
    """
    rows = []  # (kind, text, color); kind ∈ {"header", "swatch"}
    for region, models in PALETTE_GROUPS:
        rows.append(("header", region, None))
        for m in models:
            rows.append(("swatch", m, MODEL_COLORS[m]))

    n_rows = len(rows)
    fig_w = 6.4
    fig_h = 0.6 + 0.32 * n_rows

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()

    swatch_x, swatch_w = 0.5, 0.65
    label_x = 1.5
    hex_x   = 7.6

    for i, (kind, text, color) in enumerate(rows):
        y = i + 0.5
        if kind == "header":
            ax.text(0.2, y, text,
                    fontsize=12, fontweight="bold", color="#222",
                    va="center", ha="left")
        else:
            ax.add_patch(plt.Rectangle((swatch_x, i + 0.18), swatch_w, 0.64,
                                       facecolor=color, edgecolor="white",
                                       linewidth=0.5))
            ax.text(label_x, y, text,
                    fontsize=10.5, va="center",
                    family="monospace", color="#222")
            ax.text(hex_x, y, color,
                    fontsize=9.5, va="center",
                    family="monospace", color="#888")

    fig.suptitle("Provider palette — colour encodes training-data ecosystem",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.text(0.5, 0.01,
             "Within-ecosystem hues are graded so individual providers stay distinguishable. Read the cluster, then the bar.",
             ha="center", fontsize=8.5, color="#666", style="italic")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
