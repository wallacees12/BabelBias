"""
Bias analysis across N independent LLM samples per (question, language).

    Slice 1 — Response-to-response cosine similarity per question.
        For each question q and each pair of languages (la, lb), compute all
        N_la * N_lb pairwise cosines between response embeddings.
        Reports mean ± 95% CI per question, and aggregate across questions.

    Slice 2 — Response vs Wikipedia anchor 3x3 heatmap.
        For each (question, response_lang, anchor_lang), compute all N cosines
        between the N response samples and the 1 Wikipedia lead embedding.
        Aggregates across N*Q samples per cell, row-centers for the "ingroup
        pull" view, and plots both versions with CI overlays.

Outputs under data/Russia-Ukraine/analysis/<model>/<event>/.
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from babelbias.config import DEFAULT_LANGS
from babelbias.debias import (
    language_subspace_basis,
    load_control_embeddings,
    project_out,
)
from babelbias.paths import ANALYSIS_DIR, LLM_EMBEDDINGS_DIR, PROCESSED_LEADS_DIR

LANGS = list(DEFAULT_LANGS)

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


def ci95(values: list[float] | np.ndarray) -> float:
    """Half-width of a 95% CI via normal approximation. Returns 0 for n<2."""
    if len(values) < 2:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))


def load_response_embeddings(root: Path, model: str, event: str,
                              skip_refusals: bool = True) -> tuple[dict, dict]:
    """Return (embeddings, refusal_counts).

    embeddings: {(qid, lang): [vec, ...]} — refusal records excluded when
                skip_refusals (default; matches what the analysis wants).
    refusal_counts: {(qid, lang): int} — count of refusals per cell, for
                the report. Cells that never refused don't appear.
    """
    d = root / model / event
    out = defaultdict(list)
    refusals: defaultdict[tuple[str, str], int] = defaultdict(int)
    for p in sorted(d.iterdir()):
        if p.suffix != ".json":
            continue
        with open(p) as f:
            rec = json.load(f)
        key = (rec["qid"], rec["language"])
        if rec.get("refusal"):
            refusals[key] += 1
            if skip_refusals:
                continue
        out[key].append(np.asarray(rec["embedding"]))
    return out, dict(refusals)


def load_anchor_embeddings(wiki_root: Path, slugs: dict[str, str]) -> dict:
    out = {}
    for qid, slug in slugs.items():
        for lang in LANGS:
            with open(wiki_root / f"{slug}_{lang}.json") as f:
                rec = json.load(f)
            out[(qid, lang)] = np.asarray(rec["embedding"])
    return out


def slice1_per_question(responses: dict) -> pd.DataFrame:
    """Per-question cross-lingual cosine: mean ± CI across all sample pairs."""
    qids = sorted({qid for qid, _ in responses.keys()})
    rows = []
    for qid in qids:
        row = {"qid": qid}
        for la, lb in combinations(LANGS, 2):
            A = responses.get((qid, la), [])
            B = responses.get((qid, lb), [])
            sims = [cosine(a, b) for a in A for b in B]
            row[f"{la}-{lb}_mean"] = float(np.mean(sims)) if sims else np.nan
            row[f"{la}-{lb}_ci95"] = ci95(sims)
            row[f"{la}-{lb}_n"]   = len(sims)
        rows.append(row)
    return pd.DataFrame(rows)


def slice1_aggregate(per_q: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across questions: mean of per-question means, CI across questions."""
    rows = []
    for la, lb in combinations(LANGS, 2):
        col = f"{la}-{lb}_mean"
        vals = per_q[col].dropna().to_numpy()
        rows.append({
            "pair": f"{la}-{lb}",
            "mean": float(np.mean(vals)) if len(vals) else np.nan,
            "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "ci95": ci95(vals),
            "n_questions": len(vals),
        })
    return pd.DataFrame(rows)


def slice2_cells(responses: dict, anchors: dict, qids: list[str]) -> dict:
    """For each (response_lang, anchor_lang) cell, collect all N*Q cosine values."""
    cells = {(rl, al): [] for rl in LANGS for al in LANGS}
    for qid in qids:
        for rl in LANGS:
            R = responses.get((qid, rl), [])
            for al in LANGS:
                anc = anchors.get((qid, al))
                if anc is None:
                    continue
                for r_vec in R:
                    cells[(rl, al)].append(cosine(r_vec, anc))
    return cells


def matrix_from_cells(cells: dict, reducer) -> np.ndarray:
    return np.array([[reducer(cells[(rl, al)]) for al in LANGS] for rl in LANGS])


def plot_heatmap(matrix: np.ndarray, ci_matrix: np.ndarray, path: Path,
                 title: str, centered: bool = False):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cmap = "coolwarm" if centered else "viridis"
    kwargs = {}
    if centered:
        abs_max = float(np.max(np.abs(matrix)))
        kwargs = {"vmin": -abs_max, "vmax": abs_max}
    im = ax.imshow(matrix, cmap=cmap, aspect="equal", **kwargs)

    ax.set_xticks(range(3)); ax.set_xticklabels([l.upper() for l in LANGS])
    ax.set_yticks(range(3)); ax.set_yticklabels([l.upper() for l in LANGS])
    ax.set_xlabel("Wikipedia anchor language")
    ax.set_ylabel("Response language")

    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            ci  = ci_matrix[i, j]
            fmt = f"{val:+.3f}" if centered else f"{val:.3f}"
            txt = f"{fmt}\n±{ci:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
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
    ap.add_argument("--debias", action="store_true",
                    help="Project out the language subspace (estimated from "
                         "controls only) from both responses and anchors.")
    args = ap.parse_args()

    run_tag = f"{args.event}_debiased" if args.debias else args.event
    out_dir = args.out_root / args.model / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    responses, refusals = load_response_embeddings(
        args.responses_root, args.model, args.event
    )
    anchors = load_anchor_embeddings(args.wiki_root, ANCHOR_SLUGS)

    if refusals:
        total_ref = sum(refusals.values())
        n_cells = len({k for k in refusals})
        print(f"\n⚠ Refusals/content-filter detected: {total_ref} samples across "
              f"{n_cells} (qid, lang) cells — excluded from cosine analysis.")
        ref_rows = sorted(
            ({"qid": q, "lang": l, "n_refusals": n} for (q, l), n in refusals.items()),
            key=lambda r: (r["qid"], r["lang"]),
        )
        ref_df = pd.DataFrame(ref_rows)
        print(ref_df.to_string(index=False))
        ref_df.to_csv(out_dir / "refusals_per_cell.csv", index=False)
    else:
        ref_df = pd.DataFrame(columns=["qid", "lang", "n_refusals"])

    if args.debias:
        ctrl_X, ctrl_langs = load_control_embeddings(args.wiki_root, LANGS)
        basis = language_subspace_basis(ctrl_X, ctrl_langs, LANGS)
        print(f"\nDebiasing: projecting out {basis.shape[0]} language "
              f"direction(s) learned from {len(ctrl_X)} control articles "
              f"across {LANGS}.")

        for key in list(anchors.keys()):
            anchors[key] = project_out(anchors[key][None, :], basis)[0]
        for key in list(responses.keys()):
            responses[key] = [project_out(v[None, :], basis)[0] for v in responses[key]]

    sample_counts = {lang: [] for lang in LANGS}
    for (qid, lang), vecs in responses.items():
        sample_counts[lang].append(len(vecs))
    print(f"Samples per language (min/median/max across {len(ANCHOR_SLUGS)} questions):")
    for lang in LANGS:
        arr = sample_counts[lang]
        print(f"  {lang}: min={min(arr)} median={int(np.median(arr))} max={max(arr)}")

    # ---- Slice 1 --------------------------------------------------------
    print("\n" + "=" * 72)
    print("Slice 1: response-to-response cosine per question (mean ± 95% CI)")
    print("=" * 72)
    df1 = slice1_per_question(responses)
    print(df1.round(4).to_string(index=False))

    agg1 = slice1_aggregate(df1)
    print("\n-- aggregate across questions (each question = 1 observation) --")
    print(agg1.round(4).to_string(index=False))

    df1.to_csv(out_dir / "response_similarity_per_question.csv", index=False)
    agg1.to_csv(out_dir / "response_similarity_aggregate.csv", index=False)

    # ---- Slice 2 --------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"Slice 2: response vs Wikipedia anchor ({len(ANCHOR_SLUGS)} questions × N samples)")
    print("=" * 72)
    qids = sorted(ANCHOR_SLUGS.keys())
    cells = slice2_cells(responses, anchors, qids)

    mean_matrix = matrix_from_cells(cells, np.mean)
    ci_matrix   = matrix_from_cells(cells, ci95)
    n_matrix    = matrix_from_cells(cells, len)

    labels_row = [f"resp_{l}" for l in LANGS]
    labels_col = [f"wiki_{l}" for l in LANGS]
    mm_df = pd.DataFrame(mean_matrix, index=labels_row, columns=labels_col)
    ci_df = pd.DataFrame(ci_matrix,   index=labels_row, columns=labels_col)
    n_df  = pd.DataFrame(n_matrix.astype(int), index=labels_row, columns=labels_col)

    print("\n-- mean matrix --")
    print(mm_df.round(4))
    print("\n-- 95% CI half-widths --")
    print(ci_df.round(4))
    print("\n-- n per cell --")
    print(n_df)

    row_centered = mean_matrix - mean_matrix.mean(axis=1, keepdims=True)
    rc_df = pd.DataFrame(row_centered, index=labels_row, columns=labels_col)
    print("\n-- row-centered (ingroup pull +, outgroup -) --")
    print(rc_df.round(4))

    mm_df.to_csv(out_dir / "anchor_heatmap_mean.csv")
    ci_df.to_csv(out_dir / "anchor_heatmap_ci95.csv")
    rc_df.to_csv(out_dir / "anchor_heatmap_rowcentered.csv")
    n_df.to_csv(out_dir / "anchor_heatmap_n.csv")

    debias_tag = "  (debiased)" if args.debias else ""
    plot_heatmap(mean_matrix, ci_matrix,
                 out_dir / "anchor_heatmap.png",
                 f"{args.model} | {args.event}{debias_tag}\n"
                 f"Response vs Wiki anchor — mean cosine ± 95% CI")
    plot_heatmap(row_centered, ci_matrix,
                 out_dir / "anchor_heatmap_rowcentered.png",
                 f"{args.model} | {args.event}{debias_tag}\n"
                 "Row-centered: ingroup pull (+) vs outgroup (-)",
                 centered=True)
    print(f"\nSaved CSVs + PNGs to {out_dir}")


if __name__ == "__main__":
    main()
