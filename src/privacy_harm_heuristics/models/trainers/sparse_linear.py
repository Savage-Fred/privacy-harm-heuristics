"""Sparse linear (L1 Logistic Regression) trainer."""

from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import ModelResult
from ..metrics_utils import choose_average


def train_sparse_linear(
    X_train,
    y_train,
    X_test,
    y_test,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    class_weight: str | None = None,
) -> ModelResult:
    # Determine if binary classification (influences solver choice)
    y_unique = set(y_train)
    solver = "liblinear" if len(y_unique) == 2 else "saga"
    penalty = "l1"
    clf = Pipeline(
        [
            (
                "scaler",
                (
                    StandardScaler(with_mean=False)
                    if hasattr(X_train, "toarray")
                    else StandardScaler()
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    penalty=penalty,
                    C=C,
                    solver=solver,
                    max_iter=max_iter,
                    random_state=random_state,
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    # Centralized averaging logic (handles missing class in test split gracefully)
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
    lr = clf.named_steps["lr"]
    coefs = lr.coef_
    intercept = lr.intercept_.tolist()
    # Feature names may be lost in dense/ndarray ops; caller can provide mapping if needed.
    extra = {
        "coefficients": coefs.tolist(),
        "intercept": intercept,
        "n_features": X_train.shape[1],
        "penalty": penalty,
        "C": C,
    }
    return ModelResult(
        model_type="sparse_linear",
        metrics=metrics,
        artifacts={"model": clf},
        extra=extra,
    )
