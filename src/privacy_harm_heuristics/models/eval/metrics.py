"""Comparison metrics for privacy harm models.

This module centralizes the metrics described in IMPLEMENTATION_SUMMARY_METRICS.md
and is depended on by the training/evaluation pipeline as well as the docs
examples and unit tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _to_list(values: Sequence | Iterable | None) -> list:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if hasattr(values, "tolist"):
        try:
            return list(values.tolist())  # type: ignore[arg-type]
        except Exception:
            pass
    try:
        return list(values)
    except Exception:
        return []


def _validate_equal_length(a: list, b: list) -> bool:
    return bool(a) and bool(b) and len(a) == len(b)


def compute_root_cause_accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Return simple accuracy for predicted root causes."""

    true_list = _to_list(y_true)
    pred_list = _to_list(y_pred)
    if not _validate_equal_length(true_list, pred_list):
        return 0.0
    if not true_list:
        return 0.0
    return float(accuracy_score(true_list, pred_list))


def compute_outcome_accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Return classification accuracy for predicted outcomes."""

    true_list = _to_list(y_true)
    pred_list = _to_list(y_pred)
    if not _validate_equal_length(true_list, pred_list):
        return 0.0
    if not true_list:
        return 0.0
    return float(accuracy_score(true_list, pred_list))


def compute_risk_calibration(
    y_true: Sequence[int | float],
    y_pred: Sequence[int | float],
    *,
    n_bins: int = 10,
) -> float:
    """Compute expected calibration error (ECE)."""

    true_arr = np.asarray(_to_list(y_true), dtype=float)
    pred_arr = np.asarray(_to_list(y_pred), dtype=float)
    if true_arr.size == 0 or pred_arr.size == 0 or true_arr.size != pred_arr.size:
        return 1.0

    pred_arr = np.clip(pred_arr, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = true_arr.size
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (pred_arr >= left) & (pred_arr <= right)
        else:
            mask = (pred_arr >= left) & (pred_arr < right)
        if not mask.any():
            continue
        bin_true = true_arr[mask]
        bin_pred = pred_arr[mask]
        accuracy = bin_true.mean()
        confidence = bin_pred.mean()
        ece += abs(accuracy - confidence) * bin_true.size
    return float(ece / total)


def compute_brier_calibration(
    y_true: Sequence[int | float], y_pred: Sequence[int | float]
) -> float:
    """Compute Brier score for binary probabilities."""

    true_arr = np.asarray(_to_list(y_true), dtype=float)
    pred_arr = np.asarray(_to_list(y_pred), dtype=float)
    if true_arr.size == 0 or pred_arr.size == 0 or true_arr.size != pred_arr.size:
        return 1.0
    pred_arr = np.clip(pred_arr, 0.0, 1.0)
    return float(np.mean((pred_arr - true_arr) ** 2))


def compute_predicted_risk_pct(y_pred: Sequence[int | float]) -> float:
    """Return the average predicted risk expressed as a percentage."""

    pred_arr = np.asarray(_to_list(y_pred), dtype=float)
    if pred_arr.size == 0:
        return 0.0
    return float(pred_arr.mean() * 100.0)


@dataclass
class ComparisonMetrics:
    root_cause_accuracy: float = 0.0
    outcome_accuracy: float = 0.0
    risk_calibration: float = 1.0
    predicted_risk_pct: float = 0.0
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    brier_score: float | None = None
    n_samples: int = 0
    n_correct: int = 0

    def to_dict(self) -> dict[str, float | int]:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


def compute_comprehensive_metrics(
    y_true_causes: Sequence,
    y_pred_causes: Sequence,
    y_true_outcomes: Sequence,
    y_pred_outcomes: Sequence,
    y_true_binary: Sequence[int | float],
    y_pred_proba: Sequence[int | float],
) -> ComparisonMetrics:
    """Compute all comparison metrics in one pass."""

    true_causes = _to_list(y_true_causes)
    pred_causes = _to_list(y_pred_causes)
    true_outcomes = _to_list(y_true_outcomes)
    pred_outcomes = _to_list(y_pred_outcomes)
    true_binary = _to_list(y_true_binary)
    pred_probs = _to_list(y_pred_proba)

    root_acc = compute_root_cause_accuracy(true_causes, pred_causes)
    outcome_acc = compute_outcome_accuracy(true_outcomes, pred_outcomes)
    risk_cal = compute_risk_calibration(true_binary, pred_probs) if true_binary else 1.0
    predicted_pct = compute_predicted_risk_pct(pred_probs)
    brier_score = compute_brier_calibration(true_binary, pred_probs) if true_binary else None

    n_samples = len(true_causes) if true_causes else 0
    n_correct = 0
    if _validate_equal_length(true_causes, pred_causes):
        n_correct = sum(1 for t, p in zip(true_causes, pred_causes) if t == p)
    precision = recall = f1 = None
    if _validate_equal_length(true_causes, pred_causes):
        try:
            precision = float(
                precision_score(true_causes, pred_causes, average="macro", zero_division=0)
            )
            recall = float(recall_score(true_causes, pred_causes, average="macro", zero_division=0))
            f1 = float(f1_score(true_causes, pred_causes, average="macro", zero_division=0))
        except Exception:
            precision = recall = f1 = None

    metrics = ComparisonMetrics(
        root_cause_accuracy=root_acc,
        outcome_accuracy=outcome_acc,
        risk_calibration=risk_cal,
        predicted_risk_pct=predicted_pct,
        accuracy=root_acc,
        precision=precision,
        recall=recall,
        f1_score=f1,
        brier_score=brier_score,
        n_samples=n_samples,
        n_correct=n_correct,
    )
    return metrics


def format_metrics_for_output_contract(
    models_metrics: Mapping[str, ComparisonMetrics],
    frameworks_metrics: Mapping[str, ComparisonMetrics],
) -> dict:
    """Format metrics dictionaries per the Output Contract schema."""

    models_block = [{"name": name, **metrics.to_dict()} for name, metrics in models_metrics.items()]
    frameworks_block = [
        {"name": name, **metrics.to_dict()} for name, metrics in frameworks_metrics.items()
    ]

    combined = []
    for name, metrics in list(models_metrics.items()) + list(frameworks_metrics.items()):
        score = metrics.accuracy if metrics.accuracy is not None else metrics.root_cause_accuracy
        combined.append((name, score))
    combined.sort(key=lambda item: item[1], reverse=True)

    return {
        "models": models_block,
        "expert_frameworks": frameworks_block,
        "ranking": [name for name, _ in combined],
    }


__all__ = [
    "ComparisonMetrics",
    "compute_brier_calibration",
    "compute_comprehensive_metrics",
    "compute_outcome_accuracy",
    "compute_predicted_risk_pct",
    "compute_risk_calibration",
    "compute_root_cause_accuracy",
    "format_metrics_for_output_contract",
]
