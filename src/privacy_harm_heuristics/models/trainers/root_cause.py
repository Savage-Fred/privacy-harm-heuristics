"""Root cause modeling: link product feature indicators to negative sentiment.

CRITICAL DESIGN PRINCIPLE:
This model identifies CAUSAL FACTORS (product features, design choices) that lead
to privacy harms, NOT outcomes (penalty amounts, breach sizes). The goal is to
answer questions like:
  - "If we add always-on microphone, what's the privacy risk?"
  - "Using 3rd party data brokers - what harm might this cause?"
  - "Misleading privacy notice - what's the trust impact?"

Trains two interpretable models:
  * Logistic Regression (L1) -> coefficients indicate directionality
  * Shallow Decision Tree -> simple rules

Target: binary label negative_sentiment = sentiment_score < 0 (or label=='negative').
Features: Product features (pf_*), root cause semantics (rc_*), keyword flags (kw_*).
         EXCLUDES outcome variables like penalty_amount, individuals_affected.

Returns ModelResult with combined artifacts and explanation metadata.
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

# Use relative import to avoid dependency on runtime sys.path configuration
from .. import ModelResult


def _compute_directionality(
    coefficients: np.ndarray, feature_names: Sequence[str], top_n: int
) -> Dict[str, Any]:
    """Build sorted coefficient mapping and directionality summaries.

    Args:
        coefficients: 1D array of logistic regression coefficients.
        feature_names: Names corresponding to columns in X.
        top_n: Number of top positive / negative features to retain.
    """
    coef_records: list[Dict[str, Any]] = []
    for idx, coef in enumerate(coefficients):
        value = float(coef)
        if abs(value) <= 1e-6:  # skip effectively zero coefficients
            continue
        coef_records.append(
            {
                "feature": feature_names[idx],
                "coefficient": value,
                "odds_ratio": float(np.exp(value)),
                "direction": "risk_increasing" if value > 0 else "risk_mitigating",
            }
        )
    coef_sorted = sorted(coef_records, key=lambda d: abs(d["coefficient"]), reverse=True)
    top_positive = [c for c in coef_sorted if c["coefficient"] > 0][:top_n]
    top_negative = [c for c in coef_sorted if c["coefficient"] < 0][:top_n]
    return {
        "coefficients": coef_sorted,
        "directionality": {
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
        },
    }


def train_root_cause(
    X,
    feature_names: Sequence[str],
    sentiment_scores,
    random_state: int = 42,
    checkpoint_dir: Optional[str | Path] = None,
    top_n: int = 25,
) -> ModelResult:
    y = np.array(
        [1 if (score is not None and float(score) < 0) else 0 for score in sentiment_scores]
    )
    # Handle degenerate case: only a single class present. Return placeholder result
    # allowing downstream pipeline stages (unsupervised models) to proceed.
    unique_classes = np.unique(y)
    # If only one class OR insufficient samples per class to perform a stratified split / meaningful training, skip.
    if (
        unique_classes.size < 2
        or y.shape[0] < 4
        or min((y == cls_val).sum() for cls_val in unique_classes) < 2
    ):
        fallback_extra: Dict[str, Any] = {
            "coefficients": [],
            "tree_rules": "",
            "directionality": {
                "top_positive_features": [],
                "top_negative_features": [],
            },
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "negative_rate": float(y.mean()),
            "checkpoint_dir": None,
            "note": "Skipped training: insufficient class diversity / sample counts for interpretable model",
        }
        fallback_metrics = {
            "precision_neg": 0.0,
            "recall_neg": 0.0,
            "f1_neg": 0.0,
            "support_neg": float((y == 1).sum()),
            "f1_weighted": 0.0,
        }
        return ModelResult(
            model_type="root_cause",
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

    # Logistic Regression (sparse friendly)
    logreg = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        random_state=random_state,
        max_iter=500,
    )
    logreg.fit(X_train, y_train)
    if ckpt_path is not None:
        joblib.dump(logreg, ckpt_path / "logreg_checkpoint.pkl")

    # Shallow tree
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=random_state)
    tree.fit(X_train, y_train)
    if ckpt_path is not None:
        joblib.dump(tree, ckpt_path / "tree_checkpoint.pkl")

    y_pred = logreg.predict(X_test)
    report: Dict[str, Any] = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )  # type: ignore[assignment]

    coef_bundle = _compute_directionality(logreg.coef_[0], feature_names, top_n)
    tree_rules = export_text(tree, feature_names=list(feature_names))
    directionality = coef_bundle["directionality"]
    if ckpt_path is not None:
        with open(ckpt_path / "directionality.json", "w", encoding="utf-8") as fh:
            json.dump(directionality, fh, ensure_ascii=False, indent=2)
        with open(ckpt_path / "tree_rules.txt", "w", encoding="utf-8") as fh:
            fh.write(tree_rules)

    artifacts: Dict[str, Any] = {"logreg": logreg, "decision_tree": tree}
    extra: Dict[str, Any] = {
        "coefficients": coef_bundle["coefficients"],
        "tree_rules": tree_rules,
        "directionality": directionality,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "negative_rate": float(y.mean()),
        "checkpoint_dir": str(ckpt_path) if ckpt_path else None,
    }
    cls1: Dict[str, Any] = report.get("1", {}) if isinstance(report.get("1", {}), dict) else {}
    wavg: Dict[str, Any] = (
        report.get("weighted avg", {}) if isinstance(report.get("weighted avg", {}), dict) else {}
    )
    metrics = {
        "precision_neg": float(cls1.get("precision", 0.0)),
        "recall_neg": float(cls1.get("recall", 0.0)),
        "f1_neg": float(cls1.get("f1-score", 0.0)),
        "support_neg": float(cls1.get("support", 0.0)),
        "f1_weighted": float(wavg.get("f1-score", 0.0)),
    }
    return ModelResult(
        model_type="root_cause",
        metrics=metrics,
        artifacts=artifacts,
        extra=extra,
    )
