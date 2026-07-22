"""Mitigation effectiveness modeling: learn which protections reduce privacy risk.

Trains models to predict mitigation effectiveness based on:
  * Positive sentiment data (users praising privacy features)
  * Absence of incidents when protective features are present
  * Co-occurrence of risk + mitigation features

Target: positive_sentiment = sentiment_score > 0 (or label=='positive')
Features: Both risky features AND protective features present

This complements the root cause model by identifying what PREVENTS harm,
not just what CAUSES it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from .. import ModelResult


def _compute_mitigation_effectiveness(
    coefficients: np.ndarray, feature_names: Sequence[str], top_n: int
) -> Dict[str, Any]:
    """Build mitigation effectiveness mapping from model coefficients.

    Positive coefficients indicate features that correlate with positive sentiment
    (protective mitigations). Negative coefficients indicate features that correlate
    with negative sentiment (still risky even with mitigations).

    Args:
        coefficients: 1D array of logistic regression coefficients
        feature_names: Names corresponding to columns in X
        top_n: Number of top effective / ineffective mitigations to return
    """
    mitigation_records: list[Dict[str, Any]] = []

    for idx, coef in enumerate(coefficients):
        value = float(coef)
        if abs(value) <= 1e-6:  # Skip near-zero
            continue

        feature = feature_names[idx]

        # Determine if this is a protective feature or risk feature
        is_protective = feature.startswith("pf_") and value > 0
        is_risk_marker = feature.startswith("pf_") and value < 0

        mitigation_records.append(
            {
                "feature": feature,
                "coefficient": value,
                "effectiveness_score": float(np.exp(value)),  # odds ratio
                "type": (
                    "protective"
                    if is_protective
                    else "risk_marker" if is_risk_marker else "neutral"
                ),
            }
        )

    # Sort by coefficient (positive first = most effective mitigations)
    mitigation_sorted = sorted(mitigation_records, key=lambda d: d["coefficient"], reverse=True)

    # Split into protective vs risk markers
    protective = [m for m in mitigation_sorted if m["coefficient"] > 0][:top_n]
    risk_markers = [m for m in mitigation_sorted if m["coefficient"] < 0][:top_n]

    return {
        "all_mitigations": mitigation_sorted,
        "effectiveness": {
            "top_protective_features": protective,
            "top_risk_markers": risk_markers,
        },
    }


def train_mitigation_effectiveness(
    X,
    feature_names: Sequence[str],
    sentiment_scores,
    random_state: int = 42,
    checkpoint_dir: Optional[str | Path] = None,
    top_n: int = 25,
) -> ModelResult:
    """Train mitigation effectiveness model on positive sentiment data.

    Unlike the root cause model which predicts negative sentiment from risky features,
    this model predicts POSITIVE sentiment from protective features. This tells us
    which mitigations actually work to build user trust.

    Args:
        X: Feature matrix (N samples × M features)
        feature_names: Feature column names
        sentiment_scores: Sentiment scores for each sample
        random_state: Random seed for reproducibility
        checkpoint_dir: Optional directory to save model checkpoints
        top_n: Number of top mitigations to return

    Returns:
        ModelResult with mitigation effectiveness analysis
    """
    # Create binary target: positive sentiment = 1, else 0
    y = np.array(
        [1 if (score is not None and float(score) > 0) else 0 for score in sentiment_scores]
    )

    # Handle degenerate case
    unique_classes = np.unique(y)
    if (
        unique_classes.size < 2
        or y.shape[0] < 4
        or min((y == cls_val).sum() for cls_val in unique_classes) < 2
    ):
        fallback_extra: Dict[str, Any] = {
            "all_mitigations": [],
            "effectiveness": {
                "top_protective_features": [],
                "top_risk_markers": [],
            },
            "tree_rules": "",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "positive_rate": float(y.mean()),
            "checkpoint_dir": None,
            "note": "Skipped training: insufficient class diversity for mitigation model",
        }
        fallback_metrics = {
            "precision_pos": 0.0,
            "recall_pos": 0.0,
            "f1_pos": 0.0,
            "support_pos": float((y == 1).sum()),
            "f1_weighted": 0.0,
        }
        return ModelResult(
            model_type="mitigation_effectiveness",
            metrics=fallback_metrics,
            artifacts={},
            extra=fallback_extra,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y if y.sum() > 0 else None,
    )

    ckpt_path: Optional[Path] = None
    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

    # Logistic Regression (L1 regularized)
    logreg = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        random_state=random_state,
        max_iter=500,
    )
    logreg.fit(X_train, y_train)
    if ckpt_path is not None:
        joblib.dump(logreg, ckpt_path / "logreg_mitigation_checkpoint.pkl")

    # Shallow decision tree for interpretability
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=random_state)
    tree.fit(X_train, y_train)
    if ckpt_path is not None:
        joblib.dump(tree, ckpt_path / "tree_mitigation_checkpoint.pkl")

    y_pred = logreg.predict(X_test)
    report: Dict[str, Any] = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )  # type: ignore[assignment]

    mitigation_bundle = _compute_mitigation_effectiveness(logreg.coef_[0], feature_names, top_n)
    tree_rules = export_text(tree, feature_names=list(feature_names))
    effectiveness_summary = mitigation_bundle["effectiveness"]

    if ckpt_path is not None:
        with open(ckpt_path / "effectiveness.json", "w", encoding="utf-8") as fh:
            json.dump(effectiveness_summary, fh, ensure_ascii=False, indent=2)
        with open(ckpt_path / "tree_rules_mitigation.txt", "w", encoding="utf-8") as fh:
            fh.write(tree_rules)

    artifacts: Dict[str, Any] = {"logreg": logreg, "decision_tree": tree}
    extra: Dict[str, Any] = {
        "all_mitigations": mitigation_bundle["all_mitigations"],
        "effectiveness": effectiveness_summary,
        "tree_rules": tree_rules,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "checkpoint_dir": str(ckpt_path) if ckpt_path else None,
    }

    cls1: Dict[str, Any] = report.get("1", {}) if isinstance(report.get("1", {}), dict) else {}
    wavg: Dict[str, Any] = (
        report.get("weighted avg", {}) if isinstance(report.get("weighted avg", {}), dict) else {}
    )
    metrics = {
        "precision_pos": float(cls1.get("precision", 0.0)),
        "recall_pos": float(cls1.get("recall", 0.0)),
        "f1_pos": float(cls1.get("f1-score", 0.0)),
        "support_pos": float(cls1.get("support", 0.0)),
        "f1_weighted": float(wavg.get("f1-score", 0.0)),
    }

    return ModelResult(
        model_type="mitigation_effectiveness",
        metrics=metrics,
        artifacts=artifacts,
        extra=extra,
    )
