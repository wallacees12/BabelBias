# State of Thesis: BabelBias

**Project Title:** BabelBias - Investigating Linguistic and Geopolitical Bias in Multilingual Large Language Models (LLMs)
**Author:** Sam Wallace
**Date:** April 2026

---

## 1. Core Research Objective
The goal of this thesis is to investigate whether and how LLMs (ChatGPT, Gemini, Claude) exhibit "Ingroup Bias" when prompted in different languages about geopolitically contested events. 

While Wikipedia's "ingroup bias" (the tendency of a language edition to favor its own national or cultural perspective) is well-documented (Oeberst et al., 2020), this research shifts the focus to **LLMs as the next generation of knowledge curators.**

## 2. Key Theoretical Pillars (The "Literature Review" Foundation)
- **Ingroup Bias (Oeberst et al., 2020):** Wikipedia articles consistently present their "ingroup" more favorably and portray the "outgroup" as more responsible for conflict, especially in recent events.
- **Geopolitical Clustering (Samoilenko et al., 2017):** National histories on Wikipedia cluster by geopolitical blocs (e.g., former Soviet states share similar historical "focal points" in time).
- **Recency Bias:** Both human editors and potentially LLMs are exponentially more focused on events from the last 100 years, making modern conflicts (like 2014 Crimea) the most "data-rich" environments for bias research.

## 3. Case Study: 2014 Crimea Annexation/Reunification
We are using the 2014 Crimea event as the primary test case because it features:
- **Divergent Naming:** "Annexation" (EN/UK) vs. "Joining/Accession" (RU).
- **High Stakes:** A recent conflict where "ingroup" identity (Russian vs. Ukrainian) is a direct predictor of framing.

## 4. Current Progress & Methodology
### **Step 1: Establishing the "Ground Truth" (Wikipedia)**
We have fetched and embedded the Wikipedia articles for the 2014 Crimea event in English (EN), Russian (RU), and Ukrainian (UK).
- **Initial Findings (Cosine Similarity):**
  - **RU vs. UK:** 0.81 (Highest similarity - shared context, but divergent framing).
  - **RU vs. EN:** 0.74.
  - **UK vs. EN:** 0.68 (The "Control" language is furthest from the Ukrainian perspective).

### **Step 2: LLM Prompting (Upcoming)**
We will prompt various LLMs (GPT-4, Gemini Pro, Claude 3) in EN, RU, and UK with neutral questions about the 2014 Crimea event.
- **Analysis:** We will embed the LLM responses and calculate their distance from the "source" Wikipedia articles.
- **Hypothesis:** LLM responses in Russian will cluster closer to the Russian Wikipedia article, while responses in Ukrainian will cluster closer to the Ukrainian/English Wikipedia articles.

## 5. Next Steps
1.  **Framework Setup:** Build a robust Python script to query LLMs in multiple languages and store their responses.
2.  **Multilingual Embedding:** Embed these responses into the same vector space as the Wikipedia articles.
3.  **Visualization:** Create PCA/t-SNE plots to visually demonstrate the "alignment" or "divergence" of LLMs based on the language of the prompt.
