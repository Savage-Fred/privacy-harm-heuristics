"""Explainable Boosting Machine (EBM) trainer.

Uses interpret.glassbox.ExplainableBoostingClassifier to produce an additive,
globally interpretable model.

Exports:
  train_ebm(...) -> ModelResult

Extra metadata includes per-feature importance and (optionally) term scores for first N features.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ..metrics_utils import choose_average

from .. import ModelResult

try:  # Lazy import; interpret can be heavy
    from interpret.glassbox import ExplainableBoostingClassifier  # type: ignore
except Exception:  # pragma: no cover
    ExplainableBoostingClassifier = None  # type: ignore


def train_ebm(
    X_train,
    y_train,
    X_test,
    y_test,
    max_bins: int = 256,
    interactions: int = 0,
    learning_rate: float = 0.01,
    max_leaves: int = 3,
    random_state: int = 42,
    feature_importance_top_terms: int = 15,
) -> ModelResult:
    if ExplainableBoostingClassifier is None:
        raise ImportError("interpret package not available; cannot train EBM")
    clf = ExplainableBoostingClassifier(
        max_bins=max_bins,
        interactions=interactions,
        learning_rate=learning_rate,
        max_leaves=max_leaves,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    avg, pos_label = choose_average(y_train, y_test)
    if avg == "binary":
        f1 = f1_score(y_test, preds, zero_division=0, average="binary", pos_label=pos_label)
        prec = precision_score(
            y_test, preds, zero_division=0, average="binary", pos_label=pos_label
        )
        rec = recall_score(y_test, preds, zero_division=0, average="binary", pos_label=pos_label)
    else:
        f1 = f1_score(y_test, preds, zero_division=0, average="macro")
        prec = precision_score(y_test, preds, zero_division=0, average="macro")
        rec = recall_score(y_test, preds, zero_division=0, average="macro")
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
    }
    # EBM exposes feature_importances_ (aligned with training feature order)
    importances: List[float] = getattr(clf, "feature_importances_", [])
    # Collect term scores for top features for quick inspection (avoid huge payloads)
    top_indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[
        :feature_importance_top_terms
    ]
    term_scores: Dict[str, Any] = {}
    try:
        # ebm_model_.attribute_sets maps terms; we limit to single attribute terms here
        for idx in top_indices:
            if idx < len(clf.term_scores_):  # type: ignore[attr-defined]
                # term_scores_[idx] is array of bin scores. We summarize with min/max/mean.
                scores = clf.term_scores_[idx]  # type: ignore[attr-defined]
                if hasattr(scores, "tolist"):
                    import numpy as np

                    arr = np.asarray(scores)
                    term_scores[str(idx)] = {
                        "min": float(arr.min()),
                        "max": float(arr.max()),
                        "mean": float(arr.mean()),
                        "n_bins": int(getattr(arr, "shape", [len(arr)])[0]),
                    }
    except Exception:  # pragma: no cover - defensive
        pass

    extra = {
        "feature_importances": importances[:feature_importance_top_terms],
        "n_features": X_train.shape[1],
        "interactions": interactions,
        "term_score_summaries": term_scores,
    }
    return ModelResult(
        model_type="ebm",
        metrics=metrics,
        artifacts={"model": clf},
        extra=extra,
    )
