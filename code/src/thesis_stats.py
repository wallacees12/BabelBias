"""Inference layer for the thesis headline claims.

Computes, from data on disk (no new API calls):

1. Two-level cluster bootstrap (providers x questions) CIs for the
   matrix-mean row-centred diagonals EN/RU/UK on ru_uk_core.
2. Permutation test of the ingroup-pull association: within every
   (provider, question) cell the response-language rows of the 3x3
   cosine-mean matrix are randomly relabelled; the null distribution of
   the grand mean row-centred diagonal is compared to the observed one.
3. Adjusted mutual information (AMI) between EVoC cluster labels and
   each design axis (language / question / provider), replacing the
   cardinality-confounded raw-purity comparison.
4. Cluster bootstrap CIs (providers x responses) for the per-conflict
   cross-language stance gaps of the five-conflict sweep.

Prints a report; writes nothing outside stdout.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
RNG = np.random.default_rng(42)

LANGS = ["en", "ru", "uk"]
ANCHOR_SLUGS = {
    "q01_little_green_men": "Little_green_men",
    "q02_crimea_2014": "2014_Russian_annexation_of_Crimea",
    "q03_maidan_revolution": "Revolution_of_Dignity",
    "q04_referendum": "2014_Crimean_status_referendum",
    "q05_mh17": "Malaysia_Airlines_Flight_17",
    "q06_crimea_belongs": "2014_Russian_annexation_of_Crimea",
    "q07_pov_russia": "2014_Russian_annexation_of_Crimea",
    "q08_pov_ukraine": "2014_Russian_annexation_of_Crimea",
    "q09_bandera": "Stepan_Bandera",
}


def _language_basis(leads: Path) -> np.ndarray:
    """Controls-only language-subspace basis (mirrors analyze_bias --debias)."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "Code" / "src"))
    from babelbias.debias import language_subspace_basis, load_control_embeddings

    ctrl_x, ctrl_langs = load_control_embeddings(leads, LANGS)
    return language_subspace_basis(ctrl_x, ctrl_langs, LANGS)


def load_cell_matrices(debias: bool = False):
    """M[provider][qid] = 3x3 matrix of mean cosines (resp lang x anchor)."""
    leads = DATA / "Russia-Ukraine" / "processed_leads"
    basis = _language_basis(leads) if debias else None

    def _prep(v: np.ndarray) -> np.ndarray:
        if basis is not None:
            v = v - (v @ basis.T) @ basis
        return v / np.linalg.norm(v)

    anchors = {}
    for slug in set(ANCHOR_SLUGS.values()):
        for lang in LANGS:
            v = np.asarray(json.loads((leads / f"{slug}_{lang}.json")
                                      .read_text())["embedding"])
            anchors[(slug, lang)] = _prep(v)

    emb_root = DATA / "Russia-Ukraine" / "llm_embeddings"
    pdirs = [p for p in sorted(emb_root.iterdir())
             if p.is_dir() and "yandex" not in p.name]
    # Baidu's embeddings sit one level deeper (model id contains a slash)
    pdirs = [p if (p / "ru_uk_core").is_dir() else next(p.iterdir())
             for p in pdirs]
    M = defaultdict(dict)
    for pdir in pdirs:
        cells = defaultdict(list)
        for f in (pdir / "ru_uk_core").glob("*.json"):
            qid, lang = f.stem.rsplit("_", 2)[0], f.stem.rsplit("_", 2)[1]
            d = json.loads(f.read_text())
            if not d.get("embedding"):
                continue
            v = np.asarray(d["embedding"])
            cells[(qid, lang)].append(_prep(v))
        for qid, slug in ANCHOR_SLUGS.items():
            mat = np.full((3, 3), np.nan)
            for i, rl in enumerate(LANGS):
                vecs = cells.get((qid, rl))
                if not vecs:
                    continue
                R = np.vstack(vecs)
                for j, al in enumerate(LANGS):
                    mat[i, j] = float(R @ anchors[(slug, al)]).__abs__() \
                        if False else float((R @ anchors[(slug, al)]).mean())
            if not np.isnan(mat).any():
                M[pdir.name][qid] = mat
    return M


def rc_diag(mat):
    return np.diag(mat) - mat.mean(axis=1)


def write_qclustered_cis(M, out_name="exp_014_qclustered_cis.csv"):
    """Per-provider means and 95% CIs across the nine per-question
    row-centred diagonals (topic-level clustering per §3.8)."""
    rows = []
    for p in sorted(M):
        qm = np.array([rc_diag(M[p][q]) for q in sorted(M[p])])
        mean, ci = qm.mean(0), 1.96 * qm.std(0, ddof=1) / np.sqrt(len(qm))
        rows.append((p, *mean, *ci))
    df = pd.DataFrame(rows, columns=["provider", "en", "ru", "uk",
                                     "ci_en", "ci_ru", "ci_uk"])
    out = DATA / "Russia-Ukraine" / "analysis" / out_name
    df.to_csv(out, index=False)
    print(f"\n[5] wrote question-clustered per-provider CIs -> {out}")
    return df


def headline_bootstrap(M, reps=2000, label="Headline"):
    providers = sorted(M)
    qids = sorted(next(iter(M.values())))
    obs = np.mean([[rc_diag(M[p][q]) for q in qids] for p in providers],
                  axis=(0, 1))
    boots = np.empty((reps, 3))
    for b in range(reps):
        ps = RNG.choice(providers, len(providers))
        qs = RNG.choice(qids, len(qids))
        boots[b] = np.mean([[rc_diag(M[p][q]) for q in qs] for p in ps],
                           axis=(0, 1))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print(f"\n[1] {label} matrix means, two-level cluster bootstrap "
          f"(providers x questions, {reps} reps):")
    for i, l in enumerate(LANGS):
        print(f"    {l.upper()}: {obs[i]:+.3f}  95% CI [{lo[i]:+.3f}, "
              f"{hi[i]:+.3f}]")
    d_en_ru = boots[:, 0] - boots[:, 1]
    d_ru_uk = boots[:, 1] - boots[:, 2]
    print(f"    EN-RU difference CI [{np.percentile(d_en_ru, 2.5):+.3f}, "
          f"{np.percentile(d_en_ru, 97.5):+.3f}]")
    print(f"    RU-UK difference CI [{np.percentile(d_ru_uk, 2.5):+.3f}, "
          f"{np.percentile(d_ru_uk, 97.5):+.3f}]")


def permutation_test(M, reps=10000):
    mats = [M[p][q] for p in M for q in M[p]]
    obs = float(np.mean([rc_diag(m).mean() for m in mats]))
    null = np.empty(reps)
    for b in range(reps):
        tot = 0.0
        for m in mats:
            perm = RNG.permutation(3)
            tot += rc_diag(m[perm]).mean()
        null[b] = tot / len(mats)
    p = float((np.sum(null >= obs) + 1) / (reps + 1))
    print(f"\n[2] Permutation test (response-language labels shuffled "
          f"within each of {len(mats)} (provider, question) cells, "
          f"{reps} reps):")
    print(f"    observed mean row-centred diagonal {obs:+.4f}; "
          f"null max {null.max():+.4f}; p = {p:.2g}")


def evoc_ami():
    import evoc
    from sklearn.metrics import adjusted_mutual_info_score as ami

    emb_root = DATA / "Russia-Ukraine" / "llm_embeddings"
    X, langs, qids, models = [], [], [], []
    pdirs = [p for p in sorted(emb_root.iterdir())
             if p.is_dir() and "yandex" not in p.name]
    pdirs = [p if (p / "ru_uk_core").is_dir() else next(p.iterdir())
             for p in pdirs]
    for pdir in pdirs:
        for f in (pdir / "ru_uk_core").glob("*.json"):
            d = json.loads(f.read_text())
            if not d.get("embedding"):
                continue
            X.append(d["embedding"])
            qid, lang = f.stem.rsplit("_", 2)[0], f.stem.rsplit("_", 2)[1]
            langs.append(lang), qids.append(qid), models.append(pdir.name)
    X = np.asarray(X, dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    clu = evoc.EVoC(base_min_cluster_size=30)
    clu.fit(X)
    layers = clu.cluster_layers_
    best = min(range(len(layers)),
               key=lambda i: abs(len(set(layers[i])) - 27))
    labels = layers[best]
    print(f"\n[3] EVoC AMI on {len(X)} responses (layer with "
          f"{len(set(labels))} clusters):")
    for name, ax in [("language", langs), ("question", qids),
                     ("provider", models)]:
        print(f"    AMI vs {name:9s}: {ami(ax, labels):.3f}")


def stance_gap_bootstrap(reps=2000):
    events = [("Russia--Ukraine", "Russia-Ukraine"),
              ("India--Pakistan", "india_pakistan"),
              ("Israel--Palestine", "israel_palestine"),
              ("Taiwan strait", "taiwan_strait"),
              ("Falklands", "falklands")]
    print(f"\n[4] Five-conflict stance gaps, cluster bootstrap "
          f"(providers x responses, {reps} reps):")
    for label, d in events:
        df = pd.read_csv(DATA / d / "analysis" / "exp_021_stance_axis.csv")
        groups = {m: g for m, g in df.groupby("model")}
        models = sorted(groups)

        def gap_of(sample_models, resample):
            gaps = []
            for m in sample_models:
                g = groups[m]
                means = {}
                for lang, gg in g.groupby("lang"):
                    v = gg["stance"].to_numpy()
                    if resample:
                        v = RNG.choice(v, len(v))
                    means[lang] = v.mean()
                if len(means) >= 2:
                    gaps.append(max(means.values()) - min(means.values()))
            return float(np.mean(gaps))

        obs = gap_of(models, resample=False)
        boots = np.array([gap_of(RNG.choice(models, len(models)), True)
                          for _ in range(reps)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"    {label:20s} gap {obs:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    M = load_cell_matrices()
    print(f"loaded {len(M)} providers x {len(next(iter(M.values())))} questions")
    headline_bootstrap(M)
    permutation_test(M)
    write_qclustered_cis(M)
    Md = load_cell_matrices(debias=True)
    headline_bootstrap(Md, label="Debiased (language-subspace projected)")
    permutation_test(Md)
    stance_gap_bootstrap()
    evoc_ami()
