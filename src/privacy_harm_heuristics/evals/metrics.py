from typing import Dict, List, Literal, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    hamming_loss,
    jaccard_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


def calculate_mlc_metrics(
    y_true: List[Set[str]], y_pred: List[Set[str]], classes: List[str] | None = None
) -> Dict[str, float]:
    """Calculate Multi-Label Classification metrics (Jaccard, F1, Hamming).

    Args:
        y_true: List of sets of true labels.
        y_pred: List of sets of predicted labels.
        classes: Optional list of all possible classes. If None, inferred from data.

    Returns:
        Dictionary of metrics.
    """
    if not y_true and not y_pred:
        # Vacuous case: no samples. sklearn raises ("unknown is not supported")
        # on empty binarized arrays, so return defined values instead: zero
        # losses/overlap scores, perfect exact-match (vacuously true).
        return {
            "instance_jaccard": 0.0,
            "instance_f1": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "hamming_loss": 0.0,
            "exact_match_ratio": 1.0,
        }

    mlb = MultiLabelBinarizer(classes=classes)
    # Fit on both to ensure all seen labels are accounted for, or use provided classes
    if classes:
        mlb.fit([classes])
    else:
        mlb.fit(y_true + y_pred)

    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    # Instance-based metrics (samples average)
    # Jaccard score with average='samples' is the "Instance-Averaged Accuracy"
    instance_jaccard = jaccard_score(y_true_bin, y_pred_bin, average="samples", zero_division=0)
    instance_f1 = f1_score(y_true_bin, y_pred_bin, average="samples", zero_division=0)

    # Label-based metrics
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    h_loss = hamming_loss(y_true_bin, y_pred_bin)

    # Exact Match Ratio (Subset Accuracy)
    exact_match = accuracy_score(y_true_bin, y_pred_bin)

    return {
        "instance_jaccard": float(instance_jaccard),
        "instance_f1": float(instance_f1),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "hamming_loss": float(h_loss),
        "exact_match_ratio": float(exact_match),
    }


def calculate_ranking_metrics(
    y_true: List[List[str]], y_pred: List[List[str]], k: int = 5
) -> Dict[str, float]:
    """Calculate Ranking metrics (nDCG@k, MAP@k, MRR).

    Assumes binary relevance for MAP/MRR (item in y_true is relevant).
    For nDCG, assumes graded relevance based on position in y_true (reverse rank)
    or binary if not specified. Here we use binary relevance for simplicity unless
    scores are provided, but the prompt implies "ordered list of root causes".

    If y_true is ordered by relevance, we should assign scores.
    """
    ndcg_scores = []
    map_scores = []
    mrr_scores = []

    for true_list, pred_list in zip(y_true, y_pred):
        if not true_list:
            ndcg_scores.append(0.0)
            map_scores.append(0.0)
            mrr_scores.append(0.0)
            continue

        # nDCG@k
        # Construct relevance vector for predicted items
        # We assume items in true_list are relevant (score=1)
        # If true_list is ordered by importance, we could assign higher scores.
        # For now, binary relevance: 1 if in true_list, 0 otherwise.
        relevance = [1 if item in true_list else 0 for item in pred_list[:k]]
        # Pad if fewer than k predictions
        if len(relevance) < k:
            relevance += [0] * (k - len(relevance))

        # Ideal relevance: all 1s for the number of true items (up to k)
        ideal_relevance = [1] * min(len(true_list), k)
        if len(ideal_relevance) < k:
            ideal_relevance += [0] * (k - len(ideal_relevance))

        # Use sklearn's ndcg_score (expects 2D array: [n_samples, n_items])
        # But sklearn requires fixed set of items.
        # Manual calculation is often easier for list-wise ranking with variable items.
        dcg = sum((rel / np.log2(idx + 2)) for idx, rel in enumerate(relevance))
        idcg = sum((rel / np.log2(idx + 2)) for idx, rel in enumerate(ideal_relevance))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

        # MAP@k
        # Average Precision for this query
        num_correct = 0
        precisions = []
        for i, item in enumerate(pred_list[:k]):
            if item in true_list:
                num_correct += 1
                precisions.append(num_correct / (i + 1))

        if not precisions:
            ap = 0.0
        else:
            ap = sum(precisions) / min(len(true_list), k)
        map_scores.append(ap)

        # MRR
        # Reciprocal rank of first relevant item
        rr = 0.0
        for i, item in enumerate(pred_list):
            if item in true_list:
                rr = 1.0 / (i + 1)
                break
        mrr_scores.append(rr)

    return {
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
        f"map@{k}": float(np.mean(map_scores)),
        "mrr": float(np.mean(mrr_scores)),
    }


def calculate_ordinal_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    weights: Literal["linear", "quadratic"] | None = "quadratic",
) -> Dict[str, float]:
    """Calculate Ordinal Regression metrics (Kappa, Spearman, Kendall).

    Args:
        y_true: List of true ordinal scores (e.g., 1-5).
        y_pred: List of predicted ordinal scores.
        weights: 'linear' or 'quadratic' for Cohen's Kappa.

    Returns:
        Dictionary of metrics.
    """
    # Cast to numeric arrays for stats functions
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # Quadratic Weighted Kappa. cohen_kappa_score is NaN when both raters use a
    # single (identical) label -- expected agreement is 1 so the correction
    # divides by zero -- but perfect identical agreement should score 1.0, not
    # fall through to the generic NaN -> 0.0 mapping below.
    kappa = cohen_kappa_score(y_true_arr, y_pred_arr, weights=weights)
    if kappa is not None and np.isnan(float(kappa)) and np.array_equal(y_true_arr, y_pred_arr):
        kappa = 1.0

    # Spearman's Rank Correlation
    spearman, _ = spearmanr(y_true_arr, y_pred_arr)

    # Kendall's Tau
    kendall, _ = kendalltau(y_true_arr, y_pred_arr)

    # Accuracy (exact match)
    acc = accuracy_score(y_true_arr, y_pred_arr)

    return {
        "weighted_kappa": float(kappa) if kappa is not None and not np.isnan(float(kappa)) else 0.0,
        "spearman_rho": float(spearman) if spearman is not None and not np.isnan(float(spearman)) else 0.0,  # type: ignore
        "kendall_tau": float(kendall) if kendall is not None and not np.isnan(float(kendall)) else 0.0,  # type: ignore
        "accuracy": float(acc),
    }


def calculate_metrics(
    y_true: Sequence[Sequence[int]], y_pred: Sequence[Sequence[int]]
) -> Dict[str, float]:
    """Calculate Hamming Loss and Macro-F1 for multi-label classification."""
    y_true_arr: NDArray[np.int_] = np.array(y_true, dtype=int)
    y_pred_arr: NDArray[np.int_] = np.array(y_pred, dtype=int)

    h_loss = float(hamming_loss(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro"))

    return {"hamming_loss": h_loss, "macro_f1": macro_f1}


def bootstrap_confidence_interval(
    data: Sequence[float], n_bootstraps: int = 1000, ci: float = 0.95
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for a metric."""
    data_arr = np.array(data, dtype=float)
    if len(data) == 0:
        return 0.0, 0.0

    bootstrapped_means: List[float] = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data_arr, size=len(data_arr), replace=True)
        bootstrapped_means.append(float(np.mean(sample)))

    lower = float(np.percentile(bootstrapped_means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(bootstrapped_means, (1 + ci) / 2 * 100))

    return lower, upper
