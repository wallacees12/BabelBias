# State of Thesis: BabelBias

**Project Title:** BabelBias — Investigating Linguistic and Geopolitical Bias in Multilingual Large Language Models
**Author:** Sam Wallace · UZH Social Computing Group · Supervisor: Dr Iva Urman
**Last updated:** 2026-05-07

---

## 1. Core research question

> When LLMs are prompted about the same geopolitically contested event in different languages, do they give substantively different answers — and in what direction do those answers lean?

The thesis is best understood as a **rigorous, multi-regulatory-regime extension** of three published findings — Durmus et al. 2023 (Linguistic Prompting null on opinions), Buyl et al. 2024 (creator-region ideology on political persons), and Urman & Makhortykh 2025 (cross-lingual guardrail audit on Western chatbots) — rather than a fresh discovery. The contribution is the corpus, the methodology stack, and the methodological controls that the published partial findings did not have.

## 2. Literature foundation

The lit review has identified five lineages BabelBias inherits from. Twenty paper notes live under `Sources/Papers/` in the project KB; full chapter-2 prose is in `Report/tex/chapters/2_background.tex`.

- **Wikipedia ingroup bias + ecosystem framing** — Oeberst 2020 (anchor of thesis), Samoilenko 2017, Yasseri 2026, Hecht & Gergle 2010, Massa & Scrinzi 2013, Pfister 2011, Urman 2022.
- **Embedding-space bias measurement** — Bolukbasi 2016 → Caliskan 2017 → May 2019 → Kurita 2019 → Azizov 2024.
- **LLM political / ideological bias** — Hartmann 2023, Rozado 2024, Durmus 2023, Buyl 2024, Pacheco 2025.
- **LLM cultural bias** — Atari 2023 (WEIRD), Naous 2024 (CAMeL), Ramezani 2023 (moral norms), Zhao 2025 (MAKIEval — *English Activation* effect, candidate mechanism for the EN-pull).
- **Cross-lingual factuality + refusal** — Qi 2023 (RankC; tokenizer-overlap challenge), Urman 2025 (silence; closest precedent for exp_011), Li 2024 (BORDERLINES; Crimea EN/RU/UK precedent).

## 3. Case study and corpus

- **Primary event family:** 2014 Russian annexation of Crimea (Maidan, MH17, Bandera, referendum).
- **Languages:** EN (control / dominant-corpus), RU, UK.
- **Provider matrix (14):** Anthropic, OpenAI, Google, xAI, Inception (US 🇺🇸); DeepSeek, Alibaba, Zhipu, Baidu (China 🇨🇳); Cohere Aya, Cohere Command-R-Arabic (Canada 🇨🇦); YandexGPT (Russia 🇷🇺); Jamba (Israel 🇮🇱); ALLaM (Saudi Arabia 🇸🇦); TAIDE (Taiwan 🇹🇼). Six regulatory regimes — **no prior published study has the breadth**.
- **Embedders (4):** OpenAI text-embedding-3-small (primary), Alibaba GTE, Google Gemini, Yandex (Russian-trained, methodologically essential).
- **Prompt design:** 9 questions × 3 languages × 10 samples = 270 calls per model (≈4,049 response embeddings in the analysis set).

## 4. Findings to date (with literature-aware framing)

1. **Anchor-cosine pulls EN-ward (+0.18, RU +0.05, UK +0.04).** Sign is method-robust across all four embedders (exp_015). This is the cross-lingual narrative-level **replication** of Durmus 2023's Linguistic Prompting null.
2. **EN ingroup pull survives debiasing (0.18 → 0.16); UK collapses to ≈0.** Genuine framing-driven content for EN; mostly lexical for UK.
3. **EVoC unsupervised clustering recovers (qid × language) cell structure** with frontier-only purity 0.98 / 0.99 (exp_016). Convergent validity from a method orthogonal to anchor-cosine — the methodology defence is load-bearing, not an appendix robustness check.
4. **YandexGPT refused 270/270 prompts (exp_011).** Categorically domain-specific filtering. Reframed (post lit review) as the **limit case of structural inheritance** of Russian-state censorship templates documented by Urman 2025 in Western chatbots, with embedding-space signature visible via EVoC clustering.
5. **Sonnet stance-judge (exp_004 M1) tilts UK-ward overall** (mean stance −0.084) — different direction from the EN-ward cosine pull. Combined with Li 2024's RU-ward MCQ flip on Crimea, this gives a **three-way metric contradiction** (cosine vs stance vs MCQ) that is itself a discussion-chapter contribution.
6. **Imaginary-conflict pilot (exp_004) survives in free-form prose but not in direct attribution.** Bias has two shapes: content-divergence-without-direction (q05 MH17) and directional-hedging-without-content-divergence (q01).

## 5. Acknowledged challenges to the headline numbers

- **Qi 2023 tokenizer overlap (BLOOM RU-UK = 0.76)** predicts the UK-clusters-with-RU finding (82–87%) as tokenizer leakage rather than narrative fusion. **Blocks the discussion-chapter UK-fusion claim** until exp_017 lands.
- **Yasseri 2026 bimodal alignment (34% / 66%)** implies the +0.18 average is misleading; reporting must be by-question, not headline-aggregated.
- **Naous 2024 finding that Arabic Wikipedia is the most Western-centric of six pre-training corpora** generalises a caveat: BabelBias's RU/UK Wikipedia anchors are not culturally-neutral baselines.
- **Zhao 2025 English Activation effect** offers a competing mechanistic story for the EN-pull (training-corpus density rather than ideological alignment).

## 6. What this thesis does *not* claim

- Does *not* claim to discover that prompting in RU/UK fails to elicit RU/UK framing (Durmus 2023).
- Does *not* claim to discover creator-region ideology (Buyl 2024).
- Does *not* claim to discover Western-chatbot cross-lingual censorship inheritance (Urman 2025).

These are replications and rigorous extensions, not discoveries. The honest framing is in `chapters/2_background.tex` § "Position of this thesis".

## 7. Active workstream

- **exp_017 tokenizer-overlap control** — running in another session as of 2026-05-07. Non-optional; blocks the discussion chapter.
- **8 June Urman supervisor deck** — exp_015 already landed; exp_010 Oeberst replication is the remaining queue item. Deck assembly pending.

## 8. Backlog (priority-sorted)

1. **exp_010 Oeberst replication** — original 8 June queue item; embedding-space replication of the four-dimension content-coding rubric.
2. **exp_018 BORDERLINES replication** on the BabelBias provider matrix — completes the three-way metric reconciliation (cosine vs stance vs MCQ).
3. **exp_019 silence reframe** — re-analyse exp_011 alongside Urman 2025 findings; cheap reframe of writeup.
4. **exp_005 lead-section-only** — Urman 27 April directive.
5. **exp_006 ≥15 contested events** — Urman directive; turns the thesis from case study to pattern.
6. **exp_007 time ablation** — vary event date (modern vs historical / past-training-cutoff).
7. **exp_020 CAMeL-style stereotype probe** — port Naous's adjective-odds-ratio to RU/UK named characters.
8. **exp_008 imaginary control set** — replace 1,049 hand-picked controls with LLM-generated neutral set.
9. **exp_013 Grok max_tokens 1200** — Grok-only re-run to address the 27% truncation rate.
10. **exp_012 GigaChat sweep** — blocked on SberID + credentials.

## 9. Repo structure (active)

```
BabelBias/
├── Code/                                   (analysis pipeline; partially refactored)
├── data/Russia-Ukraine/                    (~360 MB; gitignored)
├── Experiments/                            (per-experiment reports; gitignored)
├── Experiments.md                          (tracker; gitignored)
├── Presentations/                          (decks 27 Apr / 11 May / 25 May / 8 Jun; gitignored)
├── Report/
│   ├── tex/                                (Overleaf-managed nested repo; gitignored from BabelBias)
│   │   ├── chapters/
│   │   │   ├── 1_introduction.tex
│   │   │   ├── 2_background.tex            (lit review prose, ~3500 words; just rewritten)
│   │   │   ├── 3_method.tex
│   │   │   ├── 4_experiments.tex
│   │   │   ├── 5_discussion.tex
│   │   │   └── 6_appendix.tex
│   │   └── bibtex/references.bib           (24 entries, one per paper)
│   └── State_of_Thesis.md                  (this file)
└── .claude/project-memory/babelbias.md     (KB binding to Obsidian vault)
```

## 10. Conventions

- **Language codes:** lowercase ISO 639-1 (`en`, `ru`, `uk`).
- **Citations:** fully-qualified keys throughout (e.g. `oeberst2020wikipedia`, not `oeberst2020`). Only `may2019` remains short-keyed pending paper note.
- **Bold key numbers** in all written material (+0.18, 270/270, etc.). Never italics, never colour.
- **Dates:** `YYYY-MM-DD` in this document and in the Experiments tracker; `DD.MM.YY` in supervisor decks (Swiss/UZH convention).
