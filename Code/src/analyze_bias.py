"""
Two bias slices on the LLM response embeddings:

    Slice 1 — Response-to-response cosine similarity per question.
        For each question, compare EN vs RU, EN vs UK, RU vs UK responses.
        Low RU-UK (vs EN-anchored pairs) is the 'contested narrative' signal.

    Slice 2 — Response-to-Wikipedia-anchor 3x3 heatmap.
        Rows = response language. Cols = Wikipedia-article language.
        Cell = mean cosine similarity across the anchored questions.
        If the diagonal dominates, the LLM is drifting toward the ingroup
        framing of whatever language it was prompted in.

Outputs under data/Russia-Ukraine/analysis/<model>/<event>/.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from babelbias.config import DEFAULT_LANGS
from babelbias.paths import ANALYSIS_DIR, LLM_EMBEDDINGS_DIR, PROCESSED_LEADS_DIR

LANGS = list(DEFAULT_LANGS)

# Wikipedia lead slugs used as anchors for each question. All nine triplets
# now exist in processed_leads/ (fetched via fetch_anchors.py).
ANCHOR_SLUGS = {
    "q01_little_green_men": "Little_green_men",
    "q02_crimea_2014":      "2014_Russian_annexation_of_Crimea",
    "q03_maidan_revolution":"Revolution_of_Dignity",
    "q04_referendum":       "2014_Crimean_status_referendum",
    "q05_mh17":             "Malaysia_Airlines_Flight_17",
    "q06_crimea_belongs":   "2014_Russian_annexation_of_Crimea",
    "q07_pov_russia":       "2014_Russian_annexation_of_Crimea",
    "q08_pov_ukraine":      "2014_Russian_annexation_of_Crimea",
    "q09_bandera":          "Stepan_Bandera",
}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_response_embeddings(root: Path, model: str, event: str):
    d = root / model / event
    out = {}
    for fn in sorted(p for p in d.iterdir() if p.suffix == ".json"):
        with open(fn) as f:
            rec = json.load(f)
        out[(rec["qid"], rec["language"])] = np.asarray(rec["embedding"])
    return out


def load_anchor_embeddings(wiki_root: Path, slugs: dict[str, str]):
    out = {}
    for qid, slug in slugs.items():
        for lang in LANGS:
            with open(wiki_root / f"{slug}_{lang}.json") as f:
                rec = json.load(f)
            out[(qid, lang)] = np.asarray(rec["embedding"])
    return out


def slice1_response_similarity(responses: dict) -> pd.DataFrame:
    qids = sorted({qid for qid, _ in responses.keys()})
    rows = []
    for qid in qids:
        row = {"qid": qid}
        for la, lb in combinations(LANGS, 2):
            a, b = responses.get((qid, la)), responses.get((qid, lb))
            row[f"{la}-{lb}"] = cosine(a, b) if a is not None and b is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def slice2_anchor_heatmap(responses: dict, anchors: dict, qids: list[str]):
    per_q_rows = []
    cells = {(rl, al): [] for rl in LANGS for al in LANGS}
    for qid in qids:
        r = {"qid": qid}
        for rl in LANGS:
            for al in LANGS:
                resp, anc = responses.get((qid, rl)), anchors.get((qid, al))
                val = cosine(resp, anc) if resp is not None and anc is not None else np.nan
                r[f"{rl}->{al}"] = val
                if not np.isnan(val):
                    cells[(rl, al)].append(val)
        per_q_rows.append(r)
    mean_matrix = np.array([[np.mean(cells[(rl, al)]) for al in LANGS] for rl in LANGS])
    return pd.DataFrame(per_q_rows), mean_matrix


def plot_heatmap(matrix: np.ndarray, path: Path, title: str, centered: bool = False):
    fig, ax = plt.subplots(figsize=(5, 4.3))
    cmap = "coolwarm" if centered else "viridis"
    im = ax.imshow(matrix, cmap=cmap, aspect="equal",
                   vmin=-np.max(np.abs(matrix)) if centered else None,
                   vmax= np.max(np.abs(matrix)) if centered else None)
    ax.set_xticks(range(3)); ax.set_xticklabels([l.upper() for l in LANGS])
    ax.set_yticks(range(3)); ax.set_yticklabels([l.upper() for l in LANGS])
    ax.set_xlabel("Wikipedia anchor language")
    ax.set_ylabel("Response language")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:+.3f}" if centered else f"{matrix[i, j]:.3f}",
                    ha="center", va="center",
                    color="white" if not centered else "black")
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--event", default="ru_uk_core")
    ap.add_argument("--responses-root", type=Path, default=LLM_EMBEDDINGS_DIR)
    ap.add_argument("--wiki-root",      type=Path, default=PROCESSED_LEADS_DIR)
    ap.add_argument("--out-root",       type=Path, default=ANALYSIS_DIR)
    args = ap.parse_args()

    out_dir = args.out_root / args.model / args.event
    out_dir.mkdir(parents=True, exist_ok=True)

    responses = load_response_embeddings(args.responses_root, args.model, args.event)
    anchors   = load_anchor_embeddings(args.wiki_root, ANCHOR_SLUGS)

    # ---- Slice 1 --------------------------------------------------------
    print("=" * 72)
    print("Slice 1: response-to-response cosine (higher = more similar framing)")
    print("=" * 72)
    df1 = slice1_response_similarity(responses)
    summary = df1.drop(columns=["qid"]).agg(["mean", "std"]).round(4)
    print(df1.round(4).to_string(index=False))
    print("\n-- aggregate --")
    print(summary.to_string())
    df1.to_csv(out_dir / "response_similarity.csv", index=False)

    # ---- Slice 2 --------------------------------------------------------
    print("\n" + "=" * 72)
    print("Slice 2: response vs Wikipedia anchor (mean cosine across "
          f"{len(ANCHOR_SLUGS)} anchored questions)")
    print("=" * 72)
    anchored_qids = sorted(ANCHOR_SLUGS.keys())
    df2, mean_matrix = slice2_anchor_heatmap(responses, anchors, anchored_qids)
    print(df2.round(4).to_string(index=False))

    mm_df = pd.DataFrame(mean_matrix,
                         index=[f"resp_{l}" for l in LANGS],
                         columns=[f"wiki_{l}" for l in LANGS])
    print("\n-- mean matrix (rows=response lang, cols=anchor lang) --")
    print(mm_df.round(4))

    row_centered = mean_matrix - mean_matrix.mean(axis=1, keepdims=True)
    rc_df = pd.DataFrame(row_centered,
                         index=[f"resp_{l}" for l in LANGS],
                         columns=[f"wiki_{l}" for l in LANGS])
    print("\n-- row-centered (which anchor is each response language closest to, "
          "controlling for absolute scale?) --")
    print(rc_df.round(4))

    df2.to_csv(out_dir / "anchor_per_question.csv", index=False)
    mm_df.to_csv(out_dir / "anchor_heatmap_mean.csv")
    rc_df.to_csv(out_dir / "anchor_heatmap_rowcentered.csv")

    plot_heatmap(mean_matrix,
                 out_dir / "anchor_heatmap.png",
                 f"{args.model} | {args.event}\n"
                 f"Response vs Wiki anchor — mean cosine ({len(anchored_qids)} questions)")
    plot_heatmap(row_centered,
                 out_dir / "anchor_heatmap_rowcentered.png",
                 f"{args.model} | {args.event}\n"
                 "Row-centered: ingroup pull (+) vs outgroup (-)",
                 centered=True)
    print(f"\nSaved CSVs + PNGs to {out_dir}")


if __name__ == "__main__":
    main()
