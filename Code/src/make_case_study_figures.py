"""Figures for the q01 'little green men' case study.

Two figures written to Presentations/figures/:

    04_per_question_cross_lingual.png
        Per-question cross-lingual cosine similarity between response
        embeddings, pooled across all three providers. Highlights how
        much of an outlier q01 is relative to the other eight questions.

    05_q01_response_pca.png
        2D PCA of the 90 q01 response embeddings (30 per provider),
        coloured by response language, with the three Wikipedia
        'Little green men' anchors overlaid as stars. Visual evidence
        that EN responses sit in a different region of the space from
        RU/UK responses.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from babelbias.config import DEFAULT_LANGS
from babelbias.paths import LLM_EMBEDDINGS_DIR, PROCESSED_LEADS_DIR, PROJECT_ROOT

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
    "gpt-4o-mini",
    "claude-haiku-4-5",
    "gemini-2.5-flash",
    "deepseek-chat",
    "grok-3-mini",
]
EVENT = "ru_uk_core"
LANGS = list(DEFAULT_LANGS)
OUT_DIR = PROJECT_ROOT / "Presentations" / "figures" / "May 11"

QUESTION_LABELS = {
    "q01_little_green_men":   "little green men",
    "q02_crimea_2014":        "Crimea 2014",
    "q03_maidan_revolution":  "Maidan",
    "q04_referendum":         "referendum",
    "q05_mh17":               "MH17",
    "q06_crimea_belongs":     "Crimea ownership",
    "q07_pov_russia":         "pro-RU speech",
    "q08_pov_ukraine":        "pro-UK speech",
    "q09_bandera":            "Bandera",
}
Q01 = "q01_little_green_men"

# Coordinated with the Carto-Bold model palette in make_meeting_figures.
# EN = teal (neutral control), RU = rose, UK = blue.
LANG_COLOR = {"en": "#11A579", "ru": "#E73F74", "uk": "#3969AC"}
MODEL_MARKER = {
    "gpt-4o-mini":      "o",
    "claude-haiku-4-5": "s",
    "gemini-2.5-flash": "^",
    "deepseek-chat":    "D",
    "grok-3-mini":      "v",
}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def ci95(values) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))


def load_responses_all_models(event: str = EVENT) -> dict:
    """Return {(qid, lang): [vec, ...]} pooled across all MODELS."""
    out: dict = defaultdict(list)
    for model in MODELS:
        d = LLM_EMBEDDINGS_DIR / model / event
        for p in sorted(d.iterdir()):
            if p.suffix != ".json":
                continue
            rec = json.loads(p.read_text())
            out[(rec["qid"], rec["language"])].append(np.asarray(rec["embedding"]))
    return out


def load_q01_embeddings_per_model(event: str = EVENT) -> dict:
    """Return {model: {(lang): [vec, ...]}} for q01 only, keeping model identity."""
    out: dict = {m: defaultdict(list) for m in MODELS}
    for model in MODELS:
        d = LLM_EMBEDDINGS_DIR / model / event
        for p in sorted(d.iterdir()):
            if p.suffix != ".json" or not p.name.startswith(Q01):
                continue
            rec = json.loads(p.read_text())
            out[model][rec["language"]].append(np.asarray(rec["embedding"]))
    return out


def load_little_green_men_anchors() -> dict:
    """Wiki anchors for 'Little green men' in each language."""
    out = {}
    for lang in LANGS:
        p = PROCESSED_LEADS_DIR / f"Little_green_men_{lang}.json"
        out[lang] = np.asarray(json.loads(p.read_text())["embedding"])
    return out


# ---- Figure 4: per-question cross-lingual similarity ---------------------

def figure_per_question_similarity(out_path: Path):
    responses = load_responses_all_models()
    qids = sorted({qid for qid, _ in responses.keys()})
    pairs = list(combinations(LANGS, 2))
    # Carto Bold subset — coordinated with the model palette.
    pair_colors = {("en", "ru"): "#E73F74",  # rose  (en/ru shared theme)
                   ("en", "uk"): "#3969AC",  # blue  (en/uk shared theme)
                   ("ru", "uk"): "#7F3C8D"}  # plum  (ru/uk shared theme)

    means = {pair: [] for pair in pairs}
    cis   = {pair: [] for pair in pairs}
    for qid in qids:
        for la, lb in pairs:
            A = responses.get((qid, la), [])
            B = responses.get((qid, lb), [])
            sims = [cosine(a, b) for a in A for b in B]
            means[(la, lb)].append(float(np.mean(sims)) if sims else np.nan)
            cis[(la, lb)].append(ci95(sims))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(qids))
    width = 0.26

    for i, (la, lb) in enumerate(pairs):
        offset = (i - 1) * width
        ax.bar(x + offset, means[(la, lb)], width,
               yerr=cis[(la, lb)], capsize=3,
               color=pair_colors[(la, lb)], edgecolor="white", linewidth=0.7,
               label=f"{la.upper()}–{lb.upper()}")

    q01_idx = qids.index(Q01)
    ymin, ymax = ax.get_ylim()
    ax.axvspan(q01_idx - 0.5, q01_idx + 0.5,
               color="gold", alpha=0.18, zorder=0)
    ax.annotate("case study", xy=(q01_idx, ymax * 0.97),
                ha="center", va="top", fontsize=10,
                color="#7a5b00", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([QUESTION_LABELS.get(q, q) for q in qids],
                       rotation=25, ha="right")
    ax.set_ylabel("Response-to-response cosine similarity\n(mean over 3 models × 100 sample pairs per cell)")
    ax.set_title("Cross-lingual response agreement per question "
                 "— q01 is the outlier", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="language pair", frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ---- Figure 5: PCA of q01 responses --------------------------------------

def figure_q01_pca(out_path: Path):
    per_model = load_q01_embeddings_per_model()
    anchors = load_little_green_men_anchors()

    vecs, tags = [], []
    for model in MODELS:
        for lang in LANGS:
            for v in per_model[model][lang]:
                vecs.append(v); tags.append(("resp", model, lang))
    for lang in LANGS:
        vecs.append(anchors[lang]); tags.append(("wiki", None, lang))

    X = np.asarray(vecs, dtype=np.float64)
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for lang in LANGS:
        for model in MODELS:
            idx = [i for i, t in enumerate(tags)
                   if t[0] == "resp" and t[1] == model and t[2] == lang]
            if not idx:
                continue
            ax.scatter(Y[idx, 0], Y[idx, 1],
                       s=55, alpha=0.75,
                       color=LANG_COLOR[lang],
                       marker=MODEL_MARKER[model],
                       edgecolor="white", linewidth=0.6,
                       label=None)

    for lang in LANGS:
        i = [k for k, t in enumerate(tags) if t[0] == "wiki" and t[2] == lang][0]
        ax.scatter(Y[i, 0], Y[i, 1],
                   s=380, marker="*",
                   color=LANG_COLOR[lang],
                   edgecolor="black", linewidth=1.2, zorder=5)
        ax.annotate(f"Wiki {lang.upper()}", (Y[i, 0], Y[i, 1]),
                    textcoords="offset points", xytext=(10, 6),
                    fontsize=10, fontweight="bold")

    lang_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                markerfacecolor=LANG_COLOR[l], markersize=10,
                                label=f"response = {l.upper()}")
                    for l in LANGS]
    lang_handles.append(plt.Line2D([0], [0], marker="*", color="w",
                                    markerfacecolor="grey",
                                    markeredgecolor="black", markersize=16,
                                    label="Wikipedia anchor"))
    model_handles = [plt.Line2D([0], [0], marker=MODEL_MARKER[m], color="grey",
                                 linestyle="", markersize=9, label=m)
                     for m in MODELS]

    leg1 = ax.legend(handles=lang_handles, loc="upper left",
                      title="response language", frameon=False, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, loc="lower left",
              title="provider", frameon=False, fontsize=9)

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
    ax.set_title("q01 'little green men' — responses pooled across 3 providers\n"
                 "EN responses separate cleanly from RU/UK responses", fontsize=12)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def figure_q01_sample_card(out_path: Path):
    """3×3 grid of representative q01 responses: 3 providers × 3 languages.

    The contrast is richer than the single-provider version: gpt-4o-mini in
    EN locks to sci-fi alone, but DeepSeek and Grok in EN explicitly cover
    *both* meanings (sci-fi first, Crimea second). All three providers
    correctly identify Crimea 2014 in RU and UK. The figure shows that the
    EN sci-fi default is real but provider-dependent.
    Hand-picked from sample r00. Edit the SAMPLES dict to swap quotes."""
    import textwrap

    SAMPLES = {
        ("gpt-4o-mini", "en"): dict(
            text=('"Little green men" typically refers to fictionalized '
                  'extraterrestrial beings — small, green-skinned aliens. '
                  'The phrase became widely known in the mid-20th century '
                  'in relation to UFO sightings and science fiction. '
                  'Popularized by the 1950s UFO craze.'),
            gloss=None,
        ),
        ("gpt-4o-mini", "ru"): dict(
            text=('«Зелёные человечки» — российские военнослужащие без '
                  'опознавательных знаков, участвующие в военных '
                  'конфликтах, особенно в ситуации на Украине в 2014 году, '
                  'когда они захватывали здания и объекты в Крыму…'),
            gloss=('Russian servicemen without insignia, in armed conflicts, '
                   'especially Ukraine 2014 — seized buildings in Crimea.'),
        ),
        ("gpt-4o-mini", "uk"): dict(
            text=('«Зелені чоловічки» — неофіційна назва для російських '
                  'військовослужбовців, які без розпізнавальних знаків '
                  'брали участь у анексії Криму в 2014 році…'),
            gloss=('Unofficial name for Russian troops who, without identifying '
                   'marks, took part in the annexation of Crimea in 2014.'),
        ),
        ("deepseek-chat", "en"): dict(
            text=('The term "little green men" has two very distinct and '
                  'important meanings — one from popular culture '
                  '(extraterrestrial aliens) and one from modern military '
                  'and political history.'),
            gloss=None,
        ),
        ("deepseek-chat", "ru"): dict(
            text=('«Зелёные человечки» — вооружённые люди в военной форме '
                  'без опознавательных знаков, появившиеся во время '
                  'событий в Крыму в 2014 году. Путин признал, что это '
                  'были российские военные…'),
            gloss=('Armed men in uniform without insignia, appeared during '
                   'the Crimea events of 2014. Putin later admitted they '
                   'were Russian military.'),
        ),
        ("deepseek-chat", "uk"): dict(
            text=('«Зелені чоловічки» (або «ввічливі зелені чоловічки») — '
                  'російські спецпризначенці та морські піхотинці без '
                  'розпізнавальних знаків під час анексії Криму 2014 року. '
                  'Спочатку Путін заперечував…'),
            gloss=('Russian special forces and marines without insignia '
                   'during the 2014 annexation of Crimea. Putin denied at '
                   'first, later confirmed.'),
        ),
        ("grok-3-mini", "en"): dict(
            text=('"Little green men" can refer to different concepts — '
                  'in popular culture and UFO lore (extraterrestrial '
                  'beings), and in the context of the 2014 annexation of '
                  'Crimea by Russia.'),
            gloss=None,
        ),
        ("grok-3-mini", "ru"): dict(
            text=('«Зелёные человечки» — неофициальное ироничное название, '
                  'появившееся в 2014 году для обозначения российских '
                  'военнослужащих, действовавших в Крыму во время '
                  'аннексии полуострова Россией.'),
            gloss=('Informal ironic name from 2014 for Russian servicemen '
                   'operating in Crimea during the annexation of the '
                   'peninsula by Russia.'),
        ),
        ("grok-3-mini", "uk"): dict(
            text=('«Зелені чоловічки» — неформальний термін для '
                  'неідентифікованих озброєних чоловіків у зеленій '
                  'військовій формі без розпізнавальних знаків, що став '
                  'популярним під час анексії Криму Росією у 2014 році.'),
            gloss=('Informal term for unidentified armed men in green uniforms '
                   'without insignia — popularized during the 2014 Russian '
                   'annexation of Crimea.'),
        ),
    }

    providers = ["gpt-4o-mini", "deepseek-chat", "grok-3-mini"]
    n_rows, n_cols = len(providers), len(LANGS)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(15, 9.5),
                             constrained_layout=True)

    for r, provider in enumerate(providers):
        for c, lang in enumerate(LANGS):
            ax = axes[r, c]
            s = SAMPLES[(provider, lang)]
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.add_patch(plt.Rectangle(
                (0.03, 0.04), 0.94, 0.92,
                facecolor="#fafafa",
                edgecolor=LANG_COLOR[lang],
                linewidth=2.0,
                transform=ax.transAxes,
            ))
            # Provider name as ylabel on the leftmost column
            if c == 0:
                ax.set_ylabel(provider,
                              fontsize=12, fontweight="bold", color="#222",
                              rotation=90, labelpad=14)
            # Language label as a clean title on the top row
            if r == 0:
                ax.set_title(f"response = {lang.upper()}",
                             fontsize=12, fontweight="bold",
                             color=LANG_COLOR[lang], pad=10)

            body = textwrap.fill(s["text"], width=40)
            body_y = 0.92 if s["gloss"] else 0.92
            ax.text(0.07, body_y, body,
                    ha="left", va="top", transform=ax.transAxes,
                    fontsize=9.8, color="#1a1a1a")

            if s["gloss"]:
                gloss = textwrap.fill(s["gloss"], width=46)
                ax.text(0.07, 0.32, "ENGLISH GLOSS",
                        ha="left", va="top", transform=ax.transAxes,
                        fontsize=7.8, fontweight="bold", color="#888")
                ax.text(0.07, 0.275, gloss,
                        ha="left", va="top", transform=ax.transAxes,
                        fontsize=9.0, fontstyle="italic", color="#555")

    fig.suptitle('q01 "little green men" — same prompt across 3 providers, 3 languages',
                 fontsize=14, fontweight="bold")
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_per_question_similarity(OUT_DIR / "04_per_question_cross_lingual.png")
    figure_q01_pca(OUT_DIR / "05_q01_response_pca.png")
    figure_q01_sample_card(OUT_DIR / "08_q01_sample_card.png")
    print(f"Wrote case-study figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
