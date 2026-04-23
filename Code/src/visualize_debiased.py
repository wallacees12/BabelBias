import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import os
    import json
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity

    return PCA, cosine_similarity, go, json, mo, np, os, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # BabelBias: Debiased Similarity Analysis

    Raw RU/UK embeddings share a "just-the-language" direction — grammar, script, vocab frequency — that has nothing to do with framing.
    We estimate that direction from the **neutral control articles** and project it out of every embedding.

    - If the conflict-vs-control similarity gap **widens** after removal → residual is framing-driven.
    - If it **collapses** → the original gap was mostly language noise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What are we measuring?

    An **embedding** is a function that maps a document $d$ to a vector $\mathbf{e}(d) \in \mathbb{R}^{1536}$. OpenAI's `text-embedding-3-small` is trained so that documents with similar meaning have similar vectors, including across languages.

    For each contested topic $t$ (e.g. *2014 annexation of Crimea*) we have a Russian article $d_t^{\text{ru}}$ and a Ukrainian article $d_t^{\text{uk}}$. Their embeddings are

    $$\mathbf{r}_t \;=\; \mathbf{e}(d_t^{\text{ru}}), \qquad \mathbf{u}_t \;=\; \mathbf{e}(d_t^{\text{uk}}).$$

    We measure agreement via **cosine similarity**:

    $$\operatorname{cos}(\mathbf{r}_t, \mathbf{u}_t) \;=\; \frac{\mathbf{r}_t \cdot \mathbf{u}_t}{\lVert \mathbf{r}_t \rVert \, \lVert \mathbf{u}_t \rVert} \;\in\; [-1,\,1].$$

    Values near $1$ mean the two articles sit in nearly the same *direction* of embedding space (angle ≈ 0°). Values far below $1$ mean the articles diverge.

    ### The hypothesis

    If RU and UK Wikipedia framings of a contested event genuinely differ, we expect:

    $$\underbrace{\overline{\cos(\mathbf{r}_t, \mathbf{u}_t)}}_{t \,\in\, \text{conflict}} \;\;<\;\; \underbrace{\overline{\cos(\mathbf{r}_t, \mathbf{u}_t)}}_{t \,\in\, \text{control}}$$

    i.e. conflict pairs are **less** cross-lingually similar than neutral pairs. A gap in this direction is evidence of framing divergence; the size of the gap is an effect size we can test statistically.
    """)
    return


@app.cell(hide_code=True)
def _(json, np, os, pd):
    def load_data(path="data/Russia-Ukraine/processed_leads"):
        embs, meta = [], []
        for fn in os.listdir(path):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(path, fn)) as f:
                d = json.load(f)
            embs.append(d["embedding"])
            meta.append({
                "conflict": d.get("conflict") or fn.split("_")[0],
                "language": d["language"],
                "title": d.get("title", "Unknown"),
                "type": d.get("type", "conflict"),
                "filename": fn,
            })
        return np.array(embs), pd.DataFrame(meta)

    embeddings_arr, meta_df = load_data()
    return embeddings_arr, meta_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now hold a data matrix $X \in \mathbb{R}^{n \times d}$ where $n \approx 1350$ is the number of articles and $d = 1536$ is the embedding dimension. Each row of $X$ is one article. `meta_df` records, for each row, the topic, language (`en`/`ru`/`uk`), and whether the topic is a *conflict* or a *control*.

    From here on everything is linear algebra on $X$.

    ---

    ## 2. Why raw cosine similarity is a noisy measurement

    Think of each embedding as a **superposition** of semantically distinct components:

    $$\mathbf{e}(d) \;=\; \underbrace{\boldsymbol{\tau}_t}_{\text{topic}} \;+\; \underbrace{\boldsymbol{\ell}_L}_{\substack{\text{language} \\ \text{signature}}} \;+\; \underbrace{\boldsymbol{\varphi}_{t,L}}_{\substack{\text{framing} \\ \text{(what we want)}}} \;+\; \boldsymbol{\epsilon}.$$

    - $\boldsymbol{\tau}_t$ — **topic** direction. Same for the RU and UK articles of the same topic.
    - $\boldsymbol{\ell}_L$ — **language signature**. A direction shared by *every* document written in language $L$ (Cyrillic stems, morphology, function-word frequencies). It has nothing to do with what the article is *about*.
    - $\boldsymbol{\varphi}_{t,L}$ — **framing**. The subtle, topic-specific, language-specific signal we actually want to quantify.
    - $\boldsymbol{\epsilon}$ — noise.

    Now look at the cross-lingual gap for a single topic:

    $$\mathbf{r}_t - \mathbf{u}_t \;=\; \big(\boldsymbol{\ell}_{\text{ru}} - \boldsymbol{\ell}_{\text{uk}}\big) \;+\; \big(\boldsymbol{\varphi}_{t,\text{ru}} - \boldsymbol{\varphi}_{t,\text{uk}}\big) \;+\; \boldsymbol{\epsilon}'.$$

    The topic term $\boldsymbol{\tau}_t$ cancels (both articles share it), which is great. But the remaining difference contains **two** things: a constant language-pair offset and the framing difference we care about.

    **If $\lVert \boldsymbol{\ell}_{\text{ru}} - \boldsymbol{\ell}_{\text{uk}} \rVert$ is comparable to (or larger than) the framing term, it swamps the signal.** Every topic, conflict or control, looks roughly equally "different" just because one is in Russian and one is in Ukrainian. That's the contamination.

    ---

    ## 3. Estimating and removing the language axis

    **Key idea:** on *neutral* topics the framing term is ≈ 0 by construction. There is no geopolitical axe to grind in an article about *Photosynthesis* or *Bach fugues*. So

    $$\mathbf{r}_t^{\text{ctrl}} - \mathbf{u}_t^{\text{ctrl}} \;\approx\; \boldsymbol{\ell}_{\text{ru}} - \boldsymbol{\ell}_{\text{uk}} \;+\; \text{noise}.$$

    Averaging over all control topics reduces noise (at least variance) and gives a direct estimate:

    $$\hat{\boldsymbol{\ell}} \;=\; \bar{\mathbf{r}}^{\text{ctrl}} \;-\; \bar{\mathbf{u}}^{\text{ctrl}}, \qquad \hat{\mathbf{l}} \;=\; \frac{\hat{\boldsymbol{\ell}}}{\lVert \hat{\boldsymbol{\ell}} \rVert}.$$

    $\hat{\mathbf{l}}$ is a **unit vector in $\mathbb{R}^{1536}$** that points from "Ukrainian style" to "Russian style" in embedding space. It is estimated *only* from control articles — the conflict articles play no role here, so we cannot accidentally "explain away" the signal we're trying to measure.

    ### The projection step

    For every embedding $\mathbf{e}$, we subtract its component along $\hat{\mathbf{l}}$:

    $$\mathbf{e}' \;=\; \mathbf{e} \;-\; \bigl(\mathbf{e} \cdot \hat{\mathbf{l}}\bigr)\,\hat{\mathbf{l}}.$$

    Geometrically this is the **orthogonal projection of $\mathbf{e}$ onto the hyperplane perpendicular to $\hat{\mathbf{l}}$**. It is a rank-$(d{-}1)$ linear operator that deletes *exactly one direction* from the embedding space. By construction,

    $$\mathbf{e}' \cdot \hat{\mathbf{l}} \;=\; 0, \qquad \lVert \mathbf{e}' \rVert^{2} \;=\; \lVert \mathbf{e} \rVert^{2} \;-\; (\mathbf{e} \cdot \hat{\mathbf{l}})^{2}.$$

    After this operation two debiased embeddings agree **only** on topic + framing + noise. The pure-language contribution is gone.

    ### Why this is safe

    - The projection is **linear**, so it distributes neatly over the decomposition above.
    - It removes **one dimension out of $1536$** — ~$0.07\%$ of the representational capacity.
    - The axis is **learned from controls only**, so conflict articles cannot bias the estimator.
    - It is **idempotent**: applying it twice gives the same result as applying it once.
    """)
    return


@app.cell
def _(embeddings_arr, meta_df, np):
    def compute_lang_axis(X, df):
        ctrl_ru = X[((df["type"] == "control") & (df["language"] == "ru")).values].mean(axis=0) # X values with control type and language Russian
        ctrl_uk = X[((df["type"] == "control") & (df["language"] == "uk")).values].mean(axis=0) # X values with control type and language Ukrainian
        axis = ctrl_ru - ctrl_uk # Difference between means
        unit = axis / np.linalg.norm(axis) # normalise
        projections = X @ unit 
        debiased = X - np.outer(projections, unit)
        return unit, projections, debiased

    lang_axis_unit, projections, debiased_arr = compute_lang_axis(embeddings_arr, meta_df)
    return debiased_arr, lang_axis_unit, projections


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now have three new objects:

    - **`lang_axis_unit`** $= \hat{\mathbf{l}}$. A unit vector in $\mathbb{R}^{1536}$ pointing roughly UK → RU.
    - **`projections`** — a length-$n$ vector of scalars $\mathbf{e}_i \cdot \hat{\mathbf{l}}$, one per article. Positive means the article leans RU on the language axis; negative means UK.
    - **`debiased_arr`** — the $n \times 1536$ matrix where each row has had its language component removed.

    ---

    ## 4. PCA — a 3D window into 1536-dim space

    **Principal Component Analysis** finds an orthonormal basis $\{\mathbf{v}_1, \mathbf{v}_2, \dots\}$ that captures the data's variance in decreasing order. Concretely it computes the eigendecomposition of the centered sample covariance

    $$\Sigma \;=\; \frac{1}{n-1}(X - \bar X)^{\top}(X - \bar X), \qquad \Sigma\,\mathbf{v}_k \;=\; \lambda_k\,\mathbf{v}_k, \quad \lambda_1 \geq \lambda_2 \geq \dots \geq 0.$$

    We keep the top 3 eigenvectors and plot each article at its coordinates $(\mathbf{x} \cdot \mathbf{v}_1,\; \mathbf{x} \cdot \mathbf{v}_2,\; \mathbf{x} \cdot \mathbf{v}_3)$.

    The fraction of total variance captured by the first $k$ components is

    $$\text{EVR}_k \;=\; \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}.$$

    **Caveat.** With ~450 topics, $\text{PC}_1$–$\text{PC}_3$ almost certainly encode *which topic* (Crimea vs. Holodomor vs. Chechnya), not *which framing*. The framing signal lives in much higher principal components — probably rank 20+. So a 3D PCA scatter is a useful qualitative picture but a very lossy one: most of the signal we care about is sitting in the other $1533$ dimensions.
    """)
    return


@app.cell(hide_code=True)
def _(PCA, cosine_similarity, pd):
    def build_view(X, df):
        pca = PCA(n_components=3)
        coords = pca.fit_transform(X)
        full = df.copy()
        full["PC1"] = coords[:, 0]
        full["PC2"] = coords[:, 1]
        full["PC3"] = coords[:, 2]

        recs = []
        for (topic, ty), g in full.groupby(["conflict", "type"]):
            ru = g[g["language"] == "ru"]
            uk = g[g["language"] == "uk"]
            if not ru.empty and not uk.empty:
                s = cosine_similarity(
                    X[ru.index[0]].reshape(1, -1),
                    X[uk.index[0]].reshape(1, -1),
                )[0][0]
                recs.append({"topic": topic, "type": ty, "similarity": s})

        sims = pd.DataFrame(recs)
        plot = full[full["language"] != "en"].copy()
        return full, plot, sims, pca.explained_variance_ratio_

    return (build_view,)


@app.cell(hide_code=True)
def _(build_view, debiased_arr, embeddings_arr, meta_df):
    full_raw, plot_raw, sim_raw, evr_raw = build_view(embeddings_arr, meta_df)
    full_db, plot_db, sim_db, evr_db = build_view(debiased_arr, meta_df)
    return evr_db, evr_raw, plot_db, plot_raw, sim_db, sim_raw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Reading the headline table

    The table below reports, for both the raw and debiased embeddings:

    1. **Control mean cosine similarity** — cross-lingual agreement on neutral topics (our baseline).
    2. **Conflict mean cosine similarity** — cross-lingual agreement on contested topics.
    3. **Gap** $\;=\; \overline{\cos}_{\text{ctrl}} \,-\, \overline{\cos}_{\text{conf}}$ — our hypothesis predicts this is **positive**.

    The Δ column shows how each quantity shifts after removing the language axis.

    **Interpretation rubric:**

    | Observation after debiasing | What it tells us |
    |---|---|
    | Gap **widens** (Δ > 0) | Residual divergence is framing-driven — language noise was *hiding* the signal. |
    | Gap **stays the same** | Language axis and framing were orthogonal to begin with — removal doesn't matter. |
    | Gap **collapses** (Δ < 0) | The original gap was mostly language noise, not framing. Bad news for the hypothesis. |
    | Both means **rise** | Expected: removing the language offset makes RU and UK articles look more similar overall. |

    The PCA variance-explained line at the bottom tells you how much of the total embedding variance the first 3 principal components account for. If this number is low (say, < 0.3), the 3D scatter is going to miss a lot of structure.
    """)
    return


@app.cell(hide_code=True)
def _(evr_db, evr_raw, mo, sim_db, sim_raw):
    def _means(df):
        return (
            df[df["type"] == "conflict"]["similarity"].mean(),
            df[df["type"] == "control"]["similarity"].mean(),
            (df["type"] == "conflict").sum(),
            (df["type"] == "control").sum(),
        )

    r_conf, r_ctrl, n_conf, n_ctrl = _means(sim_raw)
    d_conf, d_ctrl, _, _ = _means(sim_db)

    mo.md(f"""
    ## Summary

    |                          | Raw     | Debiased | Δ         |
    |--------------------------|:-------:|:--------:|:---------:|
    | Control mean sim (n={n_ctrl}) | {r_ctrl:.4f} | {d_ctrl:.4f} | {d_ctrl - r_ctrl:+.4f} |
    | Conflict mean sim (n={n_conf}) | {r_conf:.4f} | {d_conf:.4f} | {d_conf - r_conf:+.4f} |
    | **Gap (control − conflict)** | {r_ctrl - r_conf:+.4f} | {d_ctrl - d_conf:+.4f} | {(d_ctrl - d_conf) - (r_ctrl - r_conf):+.4f} |

    PCA variance explained — raw top-3: {evr_raw[:3].sum():.3f} · debiased top-3: {evr_db[:3].sum():.3f}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Distribution view

    Means are summaries — distributions are the real thing. The histograms below show, for every topic we have an RU+UK pair on, the full distribution of the cosine-similarity measurement, split by type. The inset box plots show median, quartiles, and outliers.

    Reading guide:

    - **Separation.** Are the conflict (orange) and control (green) histograms cleanly resolved, or do they overlap heavily? Large overlap means the effect size is small relative to within-group variance.
    - **Shift after debiasing.** Does the control distribution move *more* to the right (higher similarity) than the conflict distribution? That's the signature of language-noise removal — controls gain, conflicts don't.
    - **Tails.** The leftmost topics on the debiased histogram are the ones where RU and UK disagree most even after stripping the language axis. These are the natural case-study candidates for qualitative inspection.
    - **Skewness & multimodality.** A bimodal conflict distribution would suggest two distinct regimes (e.g. hot-button topics vs. merely contested ones).

    The statistically honest follow-up is a **Mann–Whitney U test** or a **two-sample permutation test** on `conflict` vs. `control` similarities, both raw and debiased, which I can add next if you want an actual p-value.
    """)
    return


@app.cell(hide_code=True)
def _(mo, px, sim_db, sim_raw):
    def _hist(df, title):
        return px.histogram(
            df,
            x="similarity",
            color="type",
            marginal="box",
            barmode="overlay",
            title=title,
            labels={"similarity": "Cosine Similarity (RU vs UK)"},
            color_discrete_map={"control": "green", "conflict": "orange"},
            template="plotly_white",
        )

    hist_raw = _hist(sim_raw, "Raw RU-UK Similarity")
    hist_db = _hist(sim_db, "Debiased RU-UK Similarity (language axis removed)")
    mo.hstack([mo.ui.plotly(hist_raw), mo.ui.plotly(hist_db)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 3D PCA scatter

    Each RU article is a red point, each UK article a blue point. **Circles** are conflict topics, **diamonds** are controls. A thin grey line connects the RU and UK articles of the same topic — its 3D length is correlated with (but not identical to) the cosine distance we care about.

    **What to expect:**

    - **Tight pair lines.** Same-topic RU/UK pairs should sit close to each other — the topic direction $\boldsymbol{\tau}_t$ dominates the position along PC1–PC3.
    - **Global topical structure.** Articles about WWII form a region; articles about Soviet arts form another. Conflict and control occupy somewhat different regions simply because the topic distributions differ.

    **What to NOT over-interpret:**

    - **RU vs UK separation.** If the raw scatter shows RU and UK forming two visibly separated "blobs" that *disappear* in the debiased scatter, that's direct visual confirmation that PC1–PC3 were picking up the language signature. Conversely, if the raw scatter *already* shows RU and UK mixed, the language signature was living in later components — the projection removed it regardless, it just doesn't show dramatically here.
    - **A tight cluster ≠ a small effect.** We are throwing away $1533$ dimensions. A clean 3D blob can still hide meaningful separation in PC4, PC5, … . The similarity histograms above are the primary evidence; the scatter is a sanity check.
    """)
    return


@app.cell(hide_code=True)
def _(go, mo, plot_db, plot_raw):
    def _scatter3d(plot_df, title):
        color_map = {"uk": "blue", "ru": "red"}
        fig = go.Figure()

        lx, ly, lz = [], [], []
        for (_, _), g in plot_df.groupby(["conflict", "type"]):
            if len(g) >= 2:
                r = g[g["language"] == "ru"]
                u = g[g["language"] == "uk"]
                if r.empty or u.empty:
                    continue
                r = r.iloc[0]; u = u.iloc[0]
                lx.extend([r["PC1"], u["PC1"], None])
                ly.extend([r["PC2"], u["PC2"], None])
                lz.extend([r["PC3"], u["PC3"], None])

        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="lines",
            line=dict(color="rgba(150,150,150,0.2)", width=1),
            name="Pair link", hoverinfo="none",
        ))

        for ty in ["control", "conflict"]:
            sym = "circle" if ty == "conflict" else "diamond"
            for lg in ["ru", "uk"]:
                sub = plot_df[(plot_df["type"] == ty) & (plot_df["language"] == lg)]
                fig.add_trace(go.Scatter3d(
                    x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
                    mode="markers",
                    marker=dict(size=4, color=color_map[lg], symbol=sym, opacity=0.7),
                    name=f"{ty.capitalize()} ({lg})",
                    text=sub["conflict"], hoverinfo="text",
                ))

        fig.update_layout(
            template="plotly_white", height=700, title=title,
            margin=dict(l=0, r=0, b=0, t=40),
        )
        return fig

    fig_raw = _scatter3d(plot_raw, "Raw PCA (RU vs UK)")
    fig_db = _scatter3d(plot_db, "Debiased PCA (language axis removed)")
    mo.hstack([mo.ui.plotly(fig_raw), mo.ui.plotly(fig_db)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Sanity check — does the language axis actually capture language?

    This is the diagnostic plot for the debiasing procedure. We take every article, compute the scalar projection onto the axis

    $$p_i \;=\; \mathbf{e}_i \cdot \hat{\mathbf{l}} \;\in\; \mathbb{R},$$

    and histogram the $p_i$ values separated by language (RU/UK) and topic type (control/conflict).

    **Expectations:**

    - **Sign separation.** RU articles should sit at positive $p_i$, UK articles at negative $p_i$. If the two distributions cleanly separate with a gap, $\hat{\mathbf{l}}$ is a strong language discriminator.
    - **Same separation on both strata.** The RU-vs-UK gap should be **the same magnitude for control and conflict topics**. This is the crucial check: the axis was *learned* from controls, so it obviously separates them. The question is whether it also separates held-out conflict articles by language alone.
    - **Overlap region.** Articles landing near $p_i = 0$ are ones where the language signature is weakest — short articles, articles dominated by shared proper nouns, or articles with unusual prose.

    **Red flag:** if the conflict panel shows *much larger* RU-vs-UK separation than the control panel, it would mean the axis picked up framing differences, not just language — making the subsequent debiasing overcorrect. In our case this is unlikely because the axis was estimated only from controls, but it's worth visually confirming.
    """)
    return


@app.cell(hide_code=True)
def _(meta_df, mo, projections, px):
    proj_df = meta_df.copy()
    proj_df["lang_projection"] = projections

    fig_proj = px.histogram(
        proj_df[proj_df["language"].isin(["ru", "uk"])],
        x="lang_projection",
        color="language",
        facet_row="type",
        barmode="overlay",
        nbins=60,
        title="Sanity check: projection onto language axis (RU should lean +, UK should lean −)",
        template="plotly_white",
        color_discrete_map={"ru": "red", "uk": "blue"},
    )
    mo.ui.plotly(fig_proj)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Per-topic drill-down

    Every row is one topic with both an RU and a UK article. Columns:

    - `sim_raw` — cosine similarity on the original embeddings, $\cos(\mathbf{r}_t, \mathbf{u}_t)$.
    - `sim_debiased` — cosine similarity after projecting out $\hat{\mathbf{l}}$, $\cos(\mathbf{r}_t', \mathbf{u}_t')$.
    - `delta` $\;=\;$ `sim_debiased` $-$ `sim_raw`.

    Sorted by `sim_debiased` ascending, so the rows at the top are the topics where RU and UK agree *least* even after removing the general linguistic differences. These are the natural qualitative case studies — load both articles, read them side by side, and you'll often find visibly different framings (e.g. *invasion* vs. *reunification*, *annexation* vs. *referendum*).

    **Reading `delta`:**

    - **`delta` strongly positive** → raw similarity *understated* the agreement. The language axis was *exaggerating* the gap. After debiasing, the pair looks closer than before.
    - **`delta` near zero** → the pair's RU and UK embeddings were roughly already orthogonal to $\hat{\mathbf{l}}$. Debiasing didn't change much.
    - **`delta` negative** → the pair's language component was *contributing positively* to the raw cosine (they happened to be offset in the same direction along $\hat{\mathbf{l}}$). Removing it exposes a real gap. Rare but possible.

    You can also sort by `delta` to find the topics most affected by the debiasing — those are the cases where language was most confounding the measurement.
    """)
    return


@app.cell(hide_code=True)
def _(mo, sim_db, sim_raw):
    mo.md("## Topic-wise similarity (sorted by debiased similarity ascending)")
    merged = sim_raw.rename(columns={"similarity": "sim_raw"}).merge(
        sim_db.rename(columns={"similarity": "sim_debiased"}),
        on=["topic", "type"],
    )
    merged["delta"] = merged["sim_debiased"] - merged["sim_raw"]
    mo.ui.table(merged.sort_values("sim_debiased"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 10. Visualising the projection, geometrically

    Everything above was numbers. Here is what the operation *looks like*.

    We pick a carefully chosen orthonormal 3D basis of the full $1536$-dim embedding space:

    - **Axis 1** $= \hat{\mathbf{l}}$ — the language axis itself.
    - **Axes 2 & 3** = $\mathbf{v}_1, \mathbf{v}_2$, the top two principal components **of the debiased data** — i.e. the two highest-variance content directions that are orthogonal to $\hat{\mathbf{l}}$ by construction.

    Every article $\mathbf{e}_i$ is then plotted at the three scalar coordinates $(\mathbf{e}_i \cdot \hat{\mathbf{l}},\; \mathbf{e}_i \cdot \mathbf{v}_1,\; \mathbf{e}_i \cdot \mathbf{v}_2)$.

    **Left panel — raw embeddings.** Look along Axis 1: RU articles (red) lean positive, UK articles (blue) lean negative. The separation you see along that axis is pure language signature — script, morphology, function words. The content information is in the other two axes.

    **Right panel — debiased embeddings.** Every point has been mapped by $\mathbf{e}' = \mathbf{e} - (\mathbf{e}\cdot\hat{\mathbf{l}})\hat{\mathbf{l}}$. In this basis that's literally the map $(a, b, c) \mapsto (0, b, c)$: **every article is flattened onto the grey plane $\hat{\mathbf{l}} = 0$.** The Axes-2&3 content coordinates are untouched, so topic structure is preserved; the language-only direction is gone.

    The grey plane is the image of the projection: every point in $\mathbb{R}^{1536}$ ends up on it. This is the "one dimension deleted out of 1536" — visualised in the only three dimensions that matter for understanding the operation.
    """)
    return


@app.cell(hide_code=True)
def _(PCA, debiased_arr, embeddings_arr, go, lang_axis_unit, meta_df, mo, np):
    def _build_projection_view():
        pca_residual = PCA(n_components=2)
        pca_residual.fit(debiased_arr)
        v1 = pca_residual.components_[0]
        v2 = pca_residual.components_[1]

        basis = np.stack([lang_axis_unit, v1, v2], axis=0)  # (3, d)
        coords_raw = embeddings_arr @ basis.T               # (n, 3)
        coords_db = debiased_arr @ basis.T                  # (n, 3), col 0 ≈ 0

        rng = np.random.RandomState(42)
        sampled_pairs = []
        for target_type, n_want in [("conflict", 25), ("control", 25)]:
            candidates = []
            for (_topic, t), g in meta_df.groupby(["conflict", "type"]):
                if t != target_type:
                    continue
                ru = g[g["language"] == "ru"]
                uk = g[g["language"] == "uk"]
                if not ru.empty and not uk.empty:
                    candidates.append((ru.index[0], uk.index[0], t))
            if len(candidates) > n_want:
                pick = rng.choice(len(candidates), n_want, replace=False)
                candidates = [candidates[i] for i in pick]
            sampled_pairs.extend(candidates)

        # Common axis ranges so the "flattening" is visually obvious
        x_range = [coords_raw[:, 0].min() * 1.1, coords_raw[:, 0].max() * 1.1]
        y_range = [coords_raw[:, 1].min() * 1.1, coords_raw[:, 1].max() * 1.1]
        z_range = [coords_raw[:, 2].min() * 1.1, coords_raw[:, 2].max() * 1.1]

        # Grey plane at x = 0 (the hyperplane orthogonal to l_hat)
        yy, zz = np.meshgrid(np.linspace(*y_range, 2), np.linspace(*z_range, 2))
        xx_plane = np.zeros_like(yy)

        def make_fig(coords, title, show_axis_arrow):
            fig = go.Figure()

            fig.add_trace(go.Surface(
                x=xx_plane, y=yy, z=zz,
                opacity=0.12, showscale=False,
                colorscale=[[0, "grey"], [1, "grey"]],
                name="ℓ̂ = 0 plane",
                hoverinfo="skip",
            ))

            if show_axis_arrow:
                arrow_len = x_range[1] * 0.9
                fig.add_trace(go.Scatter3d(
                    x=[0, arrow_len], y=[0, 0], z=[0, 0],
                    mode="lines+text",
                    line=dict(color="black", width=8),
                    text=["", "ℓ̂"],
                    textposition="top right",
                    textfont=dict(size=18, color="black"),
                    name="language axis ℓ̂",
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Cone(
                    x=[arrow_len], y=[0], z=[0],
                    u=[arrow_len * 0.15], v=[0], w=[0],
                    sizemode="absolute",
                    sizeref=arrow_len * 0.12,
                    anchor="tip",
                    colorscale=[[0, "black"], [1, "black"]],
                    showscale=False,
                    hoverinfo="skip",
                ))

            for i_ru, i_uk, _t in sampled_pairs:
                fig.add_trace(go.Scatter3d(
                    x=[coords[i_ru, 0], coords[i_uk, 0]],
                    y=[coords[i_ru, 1], coords[i_uk, 1]],
                    z=[coords[i_ru, 2], coords[i_uk, 2]],
                    mode="lines",
                    line=dict(color="rgba(120,120,120,0.35)", width=1),
                    showlegend=False, hoverinfo="skip",
                ))

            color_map = {"ru": "red", "uk": "blue"}
            for ty in ["control", "conflict"]:
                sym = "circle" if ty == "conflict" else "diamond"
                for lg in ["ru", "uk"]:
                    idx = [
                        (i_ru if lg == "ru" else i_uk)
                        for i_ru, i_uk, t in sampled_pairs
                        if t == ty
                    ]
                    if not idx:
                        continue
                    topics = [meta_df.loc[i, "conflict"] for i in idx]
                    fig.add_trace(go.Scatter3d(
                        x=coords[idx, 0], y=coords[idx, 1], z=coords[idx, 2],
                        mode="markers",
                        marker=dict(size=5, color=color_map[lg], symbol=sym, opacity=0.9),
                        name=f"{ty} ({lg})",
                        text=topics, hoverinfo="text",
                    ))

            fig.update_layout(
                template="plotly_white", height=650, title=title,
                scene=dict(
                    xaxis=dict(title="ℓ̂ (language axis)", range=x_range),
                    yaxis=dict(title="v₁ (content PC1)", range=y_range),
                    zaxis=dict(title="v₂ (content PC2)", range=z_range),
                    aspectmode="cube",
                    camera=dict(eye=dict(x=1.6, y=1.3, z=0.9)),
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            return fig

        return (
            make_fig(coords_raw, "BEFORE — raw embeddings", True),
            make_fig(coords_db, "AFTER — ℓ̂ component removed (points flattened to plane)", False),
        )

    fig_before, fig_after = _build_projection_view()
    mo.hstack([mo.ui.plotly(fig_before), mo.ui.plotly(fig_after)])
    return


if __name__ == "__main__":
    app.run()
