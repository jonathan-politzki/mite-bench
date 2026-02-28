"""Shared evaluation metrics for MITE tasks.

All functions operate on plain numpy arrays / lists so they stay
independent of any particular model or task implementation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation between two sequences.

    Returns the correlation coefficient (rho). Returns 0.0 when the
    computation is undefined (e.g. constant input).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) < 2 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return 0.0
    rho, _ = stats.spearmanr(x_arr, y_arr)
    return float(rho)


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve.

    Parameters
    ----------
    labels : array-like of {0, 1}
        Binary ground-truth labels.
    scores : array-like of float
        Model-predicted scores (higher = more positive).

    Returns
    -------
    float
        AUROC value, or 0.5 if the metric is undefined (single class).
    """
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels_arr)) < 2:
        return 0.5
    return float(roc_auc_score(labels_arr, scores_arr))


def pairwise_accuracy(
    similarities: Sequence[float], labels: Sequence[int]
) -> float:
    """Fraction of (pos, neg) pairs where the positive has higher similarity.

    For every pair of examples where one is positive (label=1) and the other
    is negative (label=0), check whether the positive example has a strictly
    higher similarity score.  Ties count as 0.5.

    Returns
    -------
    float
        Accuracy in [0, 1], or 0.5 when there are no valid pairs.
    """
    sims = np.asarray(similarities, dtype=np.float64)
    labs = np.asarray(labels)

    pos_sims = sims[labs == 1]
    neg_sims = sims[labs == 0]

    if len(pos_sims) == 0 or len(neg_sims) == 0:
        return 0.5

    correct = 0.0
    total = 0
    for ps in pos_sims:
        for ns in neg_sims:
            total += 1
            if ps > ns:
                correct += 1.0
            elif ps == ns:
                correct += 0.5
    return correct / total if total > 0 else 0.5


def separation_score(
    pos_sims: Sequence[float], neg_sims: Sequence[float]
) -> float:
    """Cohen's d between positive and negative similarity distributions.

    A higher value means the model separates positive and negative pairs
    more clearly.

    Returns 0.0 when the pooled standard deviation is zero.
    """
    pos = np.asarray(pos_sims, dtype=np.float64)
    neg = np.asarray(neg_sims, dtype=np.float64)

    n_pos, n_neg = len(pos), len(neg)
    if n_pos < 2 or n_neg < 2:
        return 0.0

    mean_diff = np.mean(pos) - np.mean(neg)

    pooled_var = ((n_pos - 1) * np.var(pos, ddof=1) + (n_neg - 1) * np.var(neg, ddof=1)) / (
        n_pos + n_neg - 2
    )
    pooled_std = np.sqrt(pooled_var)

    if pooled_std < 1e-12:
        return 0.0

    return float(mean_diff / pooled_std)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Macro-averaged F1 score across all classes.

    Wrapper around ``sklearn.metrics.f1_score`` with ``average='macro'``.
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))


# ---------------------------------------------------------------------------
# Asymmetric / directional metrics
# ---------------------------------------------------------------------------


def directional_accuracy(
    pairs: Sequence[tuple[float, float, int]],
) -> float:
    """For asymmetric tasks: does sim(A->B) differ from sim(B->A) correctly?

    Parameters
    ----------
    pairs : list of (sim_ab, sim_ba, direction_label)
        ``direction_label`` is +1 if A->B should be higher, -1 if B->A
        should be higher, or 0 if they should be equal (ties are ignored).

    Returns
    -------
    float
        Fraction of non-tie pairs where the direction is correct.
        Returns 0.5 when there are no non-tie pairs.
    """
    correct = 0
    total = 0
    for sim_ab, sim_ba, direction in pairs:
        if direction == 0:
            continue
        total += 1
        diff = sim_ab - sim_ba
        if (direction > 0 and diff > 0) or (direction < 0 and diff < 0):
            correct += 1
    return correct / total if total > 0 else 0.5


# ---------------------------------------------------------------------------
# Cross-benchmark comparison
# ---------------------------------------------------------------------------


def rank_correlation_analysis(
    mteb_scores: dict[str, float],
    mite_scores: dict[str, float],
    model_names: Sequence[str] | None = None,
) -> dict[str, float | list[str]]:
    """Compute Spearman between MTEB and MITE rankings across models.

    Parameters
    ----------
    mteb_scores : dict mapping model_name -> MTEB aggregate score
    mite_scores : dict mapping model_name -> MITE aggregate score
    model_names : optional explicit ordering; defaults to the intersection

    Returns
    -------
    dict with keys:
        "spearman"    : float  -- Spearman rho
        "p_value"     : float  -- two-sided p-value
        "n_models"    : int    -- number of models compared
        "model_names" : list[str]
    """
    if model_names is None:
        shared = sorted(set(mteb_scores.keys()) & set(mite_scores.keys()))
    else:
        shared = [m for m in model_names if m in mteb_scores and m in mite_scores]

    if len(shared) < 3:
        return {
            "spearman": 0.0,
            "p_value": 1.0,
            "n_models": len(shared),
            "model_names": shared,
        }

    mteb_vals = np.array([mteb_scores[m] for m in shared])
    mite_vals = np.array([mite_scores[m] for m in shared])

    rho, p_value = stats.spearmanr(mteb_vals, mite_vals)

    return {
        "spearman": float(rho),
        "p_value": float(p_value),
        "n_models": len(shared),
        "model_names": shared,
    }
