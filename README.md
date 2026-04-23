# BabelBias

Masters thesis investigating whether LLMs exhibit **ingroup bias** when prompted in different languages about geopolitically contested events.

![PCA of Wikipedia article embeddings across languages](assets/pca_space.gif)

*PCA of Wikipedia article embeddings (EN / RU / UK) for Russo-Ukrainian conflict topics: same event, different language editions drift apart in the semantic space.*

## Pipeline

1. **Wikipedia baseline** — fetch + embed contested-event articles in EN/RU/UK to measure cross-lingual divergence.
2. **LLM probing** — query GPT, Claude, and Gemini in each language with the same prompts.
3. **Comparison** — embed responses into the same space; measure whether each language's response drifts toward its own Wikipedia framing.

See [`Report/State_of_Thesis.md`](Report/State_of_Thesis.md) for methodology and [`Code/`](Code/) for the pipeline.
