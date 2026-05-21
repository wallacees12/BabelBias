# State of Thesis: BabelBias

**Project Title:** BabelBias — Investigating Linguistic and Geopolitical Bias in Multilingual Large Language Models
**Author:** Sam Wallace · UZH Social Computing Group · Supervisor: Dr Iva Urman
**Last updated:** 2026-05-21

---

## 1. Core research question

> When LLMs are prompted about the same geopolitically contested event in different languages, do they give substantively different answers — and in what direction do those answers lean?

The thesis is best understood as a **rigorous, multi-regulatory-regime extension** of four published findings — Durmus et al. 2023 (Linguistic Prompting null on opinions), Buyl et al. 2024 (creator-region ideology on political persons), Urman & Makhortykh 2025 (cross-lingual guardrail audit on Western chatbots), and Smirnov 2026 (RU/UK framing flip on a single Ukrainian civil-society document) — rather than a fresh discovery. The contribution is the corpus, the methodology stack, and the methodological controls that the published partial findings did not have.

## 2. Literature foundation

Lit review complete. ~20 paper notes live under `Sources/Papers/` in the project KB; chapter-2 prose (3,712 words) is in `Report/tex/chapters/2_background.tex`.

- **Wikipedia ingroup bias + ecosystem framing** — Oeberst 2020 (anchor of thesis), Samoilenko 2017, Yasseri 2026, Hecht & Gergle 2010, Massa & Scrinzi 2013, Pfister 2011, Urman 2022.
- **Embedding-space bias measurement** — Bolukbasi 2016 → Caliskan 2017 → May 2019 → Kurita 2019 → Azizov 2024.
- **LLM political / ideological bias** — Hartmann 2023, Rozado 2024, Durmus 2023, Buyl 2024, Pacheco 2025.
- **LLM cultural bias** — Atari 2023 (WEIRD), Naous 2024 (CAMeL), Ramezani 2023 (moral norms), Zhao 2025 (MAKIEval — *English Activation* effect, candidate mechanism for the EN-pull).
- **Cross-lingual factuality + refusal** — Qi 2023 (RankC; tokenizer-overlap challenge), Urman 2025 (silence; closest precedent for exp_011/exp_022), Li 2024 (BORDERLINES; Crimea EN/RU/UK precedent), Smirnov 2026 (closest published parallel — see §7b).

## 3. Case study and corpus

- **Primary event family:** 2014 Russian annexation of Crimea (Maidan, MH17, Bandera, referendum).
- **Generalisation events (exp_006, ✅ landed):** Israel–Palestine, India–Pakistan, Taiwan Strait, Falklands. Five conflict families total.
- **Languages:** EN (control / dominant-corpus), RU, UK on Russo-Ukrainian; per-event native pairs (HE/AR, HI/UR, EN/ZH, EN/ES) on generalisation events.
- **Provider matrix (14):** Anthropic, OpenAI, Google, xAI, Inception (US 🇺🇸); DeepSeek, Alibaba, Zhipu, Baidu (China 🇨🇳); Cohere Aya, Cohere Command-R-Arabic (Canada 🇨🇦); YandexGPT (Russia 🇷🇺); Jamba (Israel 🇮🇱); ALLaM (Saudi Arabia 🇸🇦); TAIDE (Taiwan 🇹🇼). Six regulatory regimes — **no prior published study has the breadth**.
- **Embedders (4):** OpenAI text-embedding-3-small (primary), Alibaba GTE, Google Gemini, Yandex (Russian-trained, methodologically essential).
- **Prompt design (RU-UK):** 9 questions × 3 languages × 10 samples = 270 calls per model (≈4,049 response embeddings in the analysis set).
- **Total corpus (post exp_006):** ≈13,000 LLM responses across 5 conflicts; universal-control corpus of 1,000 tuples × 10 languages (6,872 embeddings).

## 4. Findings to date (with literature-aware framing)

1. **Anchor-cosine pulls EN-ward on Russo-Ukrainian (+0.18, RU +0.05, UK +0.04).** Sign is method-robust across all four embedders (exp_015). This is the cross-lingual narrative-level **replication** of Durmus 2023's Linguistic Prompting null.
2. **EN ingroup pull survives debiasing (0.18 → 0.16); UK collapses to ≈0.** Genuine framing-driven content for EN; mostly lexical for UK.
3. **EVoC unsupervised clustering recovers (qid × language) cell structure** with frontier-only purity 0.98 / 0.99 (exp_016). Convergent validity from a method orthogonal to anchor-cosine.
4. **The cosine ingroup-pull pattern generalises across all 5 conflicts (exp_006).** EN ingroup pull: RU-UK +0.18 → IL-PS +0.24 → IP +0.32 → Taiwan +0.11 → Falklands +0.05. Falklands near-null **confirms Oeberst's falsification anchor** (low-contestation event ⇒ low cross-lingual divergence). Under language-axis debiasing, EN survives in every event (5–12% collapse) but HE collapses 96%, UK 66%, ES 50% — lexical artefacts; AR (20%), HI (37%), UR (30%) partially survive. Two independent methods (debiasing + off-topic same-language baseline) converge on the HE-is-language-axis story.
5. **Stance-axis projection captures what cosine misses (exp_021, q03 prototype 2026-05-18).** On the Israel–Palestine intifada question, 11 of 11 IL-PS providers shift HE → Israeli-framing / AR → Palestinian-framing, mean gap ~0.13 stance points (~30% of seed pole separation). This is the qualitative framing bias the 11 May audience saw in the responses, formalised into a single signed scalar — invisible to cosine row-centred analysis. Lexicons pre-registered at `Code/src/babelbias/stance_lexicons.py` (5 conflicts × 12 minimal-edit sentence pairs).
6. **YandexGPT refusal is graded by Russia-state interest AND typed by question form (exp_022).** Per-event refusal: ru_uk_imaginary 100% → IL-PS 77% → IP 52% → Falklands 22% → Taiwan 6%. Within events, POV / one-sided-defence questions (q07/q08) are filtered at the highest rate — at Falklands and Taiwan they are the *only* filtered slots. AR (IL-PS) and UR (IP) filtered ~half as often as same-question EN/HE/HI — script-coverage asymmetry, itself a silence-as-bias finding. **Refines exp_011 from "categorically distinct" to "categorical for high-interest events, question-typed for medium-interest, vestigial for low-interest"** — the limit case of structural inheritance of Russian-state censorship templates documented by Urman 2025 in Western chatbots.
7. **Sonnet stance-judge (exp_004 M1) tilts UK-ward overall** (mean stance −0.084) — different direction from the EN-ward cosine pull. Combined with Li 2024's RU-ward MCQ flip on Crimea, this gives a **three-way metric contradiction** (cosine vs stance vs MCQ) that is itself a discussion-chapter contribution.
8. **Imaginary-conflict pilot (exp_004) survives in free-form prose but not in direct attribution.** Bias has two shapes: content-divergence-without-direction (q05 MH17) and directional-hedging-without-content-divergence (q01).

## 5. Acknowledged challenges to the headline numbers

- **Qi 2023 tokenizer overlap** predicts UK-clusters-with-RU finding as tokenizer leakage. **Partially resolved by exp_017:** all 3 inspectable tokenizers below BLOOM's 0.76 reference; only q05 MH17 and q09 Bandera fuse robustly across embedders. UK-fusion claim partially unblocked.
- **Yasseri 2026 bimodal alignment (34% / 66%)** implies the +0.18 average is misleading; reporting must be by-question, not headline-aggregated.
- **Naous 2024 finding that Arabic Wikipedia is the most Western-centric of six pre-training corpora** generalises a caveat: BabelBias's RU/UK Wikipedia anchors are not culturally-neutral baselines.
- **Zhao 2025 English Activation effect** offers a competing mechanistic story for the EN-pull (training-corpus density rather than ideological alignment).

## 6. What this thesis does *not* claim

- Does *not* claim to discover that prompting in RU/UK fails to elicit RU/UK framing (Durmus 2023).
- Does *not* claim to discover creator-region ideology (Buyl 2024).
- Does *not* claim to discover Western-chatbot cross-lingual censorship inheritance (Urman 2025).
- Does *not* claim to discover that prompt language alone flips framing on a single contested document (Smirnov 2026).

These are replications and rigorous extensions, not discoveries. The honest framing is in `chapters/2_background.tex` § "Position of this thesis".

## 7. Active workstream

- **8 June Urman supervisor deck — pivoted 2026-05-18 to stance-axis headline.** exp_021 q03_intifada prototype landed; cross-conflict sweep + figures pending. exp_015 + exp_017 demoted to closing methodology-supporting slides.
- **25 May Urman supervisor deck — all experiments landed.** exp_011 + exp_014 + exp_016 + exp_002_followup + exp_003 + exp_004 (M1+M2+qual). Only deck assembly + speaker notes remain.
- **22 June deck — exp_006 5-conflict scale-up ✅ landed** (2026-05-07, $5.62 / 9,023 calls); **exp_022 Yandex cross-conflict ✅ landed** (2026-05-18, 945 calls via UZH VPN). Figure rendering for exp_022 pending.
- **6 July deck — planned.** exp_023 Mistral + French control + exp_007 historical timing ablation.
- **20 July deck — planned.** exp_024 Iran event pilot (EN + HE + AR + FA + RU, 3-way framing).

## 7a. Smirnov 2026 — closest published parallel

Discovered during lit review on 2026-05-07 (arXiv:2601.12164). Single Ukrainian civil-society document, semantically equivalent RU/UK prompts, qualitative discourse analysis showing identical model produces Russian-state-discourse vocabulary in RU and Western-liberal-democratic vocabulary in UK. Identical model, identical content, prompt language alone flips the framing.

Chapter 2 § "Position of this thesis" frames BabelBias as the multi-event, multi-provider, embedding-quantified, English-controlled, imaginary-falsified scaling of Smirnov's existence proof. The pre-registered stance-axis projection (exp_021) is the operationalised follow-up.

## 7b. Chapter writing progress

| Ch | File | Words | Status |
|---|---|---|---|
| 1 | `1_introduction.tex` | 1,577 | ✅ full draft — 4 precedents + 7-item contribution list |
| 2 | `2_background.tex` | 3,712 | ✅ full draft — Smirnov 2026 integrated, 24-entry bibtex |
| 3 | `3_method.tex` | 3,219 | ✅ full draft + 6 figures (pipeline, conflict atlas, provider map, cosine schematic, debiasing geometry, topic-vs-language) |
| 4 | `4_experiments.tex` | 1,905 | ✅ thematic rewrite landed, compressed; cross-conflict figure inlined |
| 5 | `5_discussion.tex` | 496 | 🟡 stub — needs the three-way metric contradiction + Smirnov-scaling framing + EN-pull mechanism discussion |
| 6 | `6_appendix.tex` | 177 | 🟡 stub — prompt bank section started |

**Total drafted:** ~11,400 words across 6 chapters. **Critical path: Ch 5 Discussion** — this is the bottleneck for a complete first draft.

## 8. Backlog — scoped for masters thesis

Backlog filtered to what earns a slot in the thesis. Cut items remain in `Experiments.md` for reference but are not load-bearing for the writeup.

**In thesis scope (active):**

1. **exp_021 stance-axis projection** — 8 June headline; cross-conflict sweep + figures pending. Native metric for §4 finding 5.
2. **exp_023 mistral_french_control** — Urman 11 May ask, 6 July deck. Generalises the EN-ingroup story to a French-trained provider with French as a control language. Methods-chapter contribution.
3. **exp_024 iran_event_pilot** — Urman 11 May ask, 20 July deck. 3-way framing case (Iran-state ↔ Israel/Arab-state ↔ US-state), breaks the binary-framing assumption.
4. **exp_007 time_ablation** — Urman 11 May ask, 6 July deck. Tests whether ingroup-pull magnitude depends on training-data density / event recency.
5. **exp_018 borderlines_replication** — Li 2024 MCQ on the 14-provider matrix. Closes the three-way metric reconciliation (cosine vs stance vs MCQ); Ch 5 discussion contribution.
6. **exp_019 silence_reframe** — re-analyse exp_011 + exp_022 alongside Urman & Makhortykh 2025. Pure writeup work; no new data collection. Cheap, high writeup value.

**Cut from thesis scope** (defer to follow-on work / not load-bearing):

- ~~exp_005 lead-section-only~~ — superseded by exp_017's lead-vs-full-page finding; footnote in §3.3, not its own experiment.
- ~~exp_008 imaginary control set~~ — the existing 1,049 hand-picked controls are not the bottleneck.
- ~~exp_009 authorship graph~~ — already deferred indefinitely; orthogonal to the bias measurement story.
- ~~exp_010 oeberst_replication~~ — already reclassified as lit-review-chapter material rather than a new experiment.
- ~~exp_012 gigachat_full_sweep~~ — blocked on SberID + YandexGPT already covers the Russia-regime ecosystem.
- ~~exp_013 grok_max_tokens_1200~~ — single-model truncation re-run; does not change the thesis story.
- ~~exp_020 camel_stereotype_probe~~ — add-on probe; not load-bearing for the headline contributions.

## 9. Repo structure (active)

```
BabelBias/
├── Code/                                   (analysis pipeline; partially refactored)
├── data/{Russia-Ukraine, israel_palestine, india_pakistan, taiwan_strait, falklands}/
│                                            (~360 MB+; gitignored)
├── data/universal_controls/                (1,000 tuples × 10 langs; gitignored)
├── Experiments/                            (per-experiment reports; gitignored)
├── Experiments.md                          (tracker; gitignored)
├── Presentations/                          (decks 27 Apr / 11 May / 25 May / 8 Jun / 22 Jun / 6 Jul / 20 Jul; gitignored)
├── Report/
│   ├── tex/                                (Overleaf-managed nested repo)
│   │   ├── chapters/
│   │   │   ├── 1_introduction.tex
│   │   │   ├── 2_background.tex
│   │   │   ├── 3_method.tex
│   │   │   ├── 4_experiments.tex
│   │   │   ├── 5_discussion.tex            (stub — critical path)
│   │   │   └── 6_appendix.tex              (stub)
│   │   └── bibtex/references.bib           (24 entries)
│   └── State_of_Thesis.md                  (this file)
└── .claude/project-memory/babelbias.md     (KB binding to Obsidian vault)
```

## 10. Conventions

- **Language codes:** lowercase ISO 639-1 (`en`, `ru`, `uk`, `he`, `ar`, `hi`, `ur`, `zh`, `es`, `fr`, `fa`).
- **Citations:** fully-qualified keys throughout (e.g. `oeberst2020wikipedia`, not `oeberst2020`).
- **Bold key numbers** in all written material (+0.18, 270/270, etc.). Never italics, never colour.
- **Dates:** `YYYY-MM-DD` in this document and in the Experiments tracker; `DD.MM.YY` in supervisor decks (Swiss/UZH convention).
- **No exp_NNN in prose** that ships to the reader (thesis / decks / paper). Internal labels (`\label{sec:exp_021}`) and filenames are exempt. Describe the experiment by what it *is*.
