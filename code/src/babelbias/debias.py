"""Language-subspace projection for multilingual embeddings.

Generalises the RU/UK procedure in visualize_debiased.py to any number of
languages. Given control (neutral-topic) embeddings in each language,
the "language subspace" is the span of the centered per-language
centroids. We project every embedding onto its orthogonal complement.

Properties:
    - Linear and idempotent.
    - Learned from controls only — conflict / contested data cannot
      bias the estimator and be "explained away".
    - Removes at most L - 1 dimensions for L languages (one per
      language centroid, minus one for the global centering).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def language_subspace_basis(
    control_X: np.ndarray,
    control_langs: Sequence[str],
    languages: Sequence[str],
) -> np.ndarray:
    """Orthonormal basis of the language subspace.

    Parameters
    ----------
    control_X : (n, d)
        Control (neutral-topic) embeddings.
    control_langs : length n
        Language tag for each row of `control_X`.
    languages : ordered list of language codes to include
        (e.g. ("en", "ru", "uk")). Controls for every listed language
        must be present.

    Returns
    -------
    basis : (k, d) with orthonormal rows, k <= len(languages) - 1.
    """
    control_X = np.asarray(control_X, dtype=np.float64)
    langs_arr = np.asarray(control_langs)

    centroids = []
    for lang in languages:
        mask = langs_arr == lang
        if not mask.any():
            raise ValueError(f"no control embeddings for language {lang!r}")
        centroids.append(control_X[mask].mean(axis=0))
    C = np.stack(centroids, axis=0)
    M = C - C.mean(axis=0, keepdims=True)

    _U, s, Vt = np.linalg.svd(M, full_matrices=False)
    tol = max(M.shape) * np.finfo(M.dtype).eps * (s[0] if s.size else 0.0)
    rank = int((s > tol).sum())
    return Vt[:rank]


def project_out(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return X with the subspace spanned by `basis` removed.

    `basis` rows must be orthonormal.
    """
    X = np.asarray(X, dtype=np.float64)
    coords = X @ basis.T
    return X - coords @ basis


def load_control_embeddings(
    processed_leads_dir: Path,
    languages: Iterable[str],
) -> tuple[np.ndarray, list[str]]:
    """Load all control-article embeddings in the given languages.

    Returns (X, langs) where X is (n, d) and langs is length-n.
    """
    langs_set = set(languages)
    vecs: list[list[float]] = []
    tags: list[str] = []
    for p in sorted(processed_leads_dir.iterdir()):
        if p.suffix != ".json":
            continue
        with open(p) as f:
            rec = json.load(f)
        if rec.get("type") != "control":
            continue
        lang = rec.get("language")
        if lang not in langs_set:
            continue
        vecs.append(rec["embedding"])
        tags.append(lang)
    if not vecs:
        raise RuntimeError(
            f"no control embeddings found in {processed_leads_dir} "
            f"for languages {sorted(langs_set)}"
        )
    return np.asarray(vecs, dtype=np.float64), tags
