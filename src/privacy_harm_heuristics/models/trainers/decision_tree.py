"""Decision Tree trainer."""

from __future__ import annotations

from typing import Any, Dict, Literal

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ..metrics_utils import choose_average
from sklearn.tree import DecisionTreeClassifier

from .. import ModelResult


def train_decision_tree(
    X_train,
    y_train,
    X_test,
    y_test,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    class_weight: Dict[int, float] | Dict[str, float] | Literal["balanced"] | None = None,
) -> ModelResult:
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight=class_weight,
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
    importances = getattr(clf, "feature_importances_", None)
    extra = {
        "feature_importances": (importances.tolist() if importances is not None else None),
        "n_features": X_train.shape[1],
        "depth": clf.get_depth(),
        "n_leaves": clf.get_n_leaves(),
    }
    return ModelResult(
        model_type="decision_tree",
        metrics=metrics,
        artifacts={"model": clf},
        extra=extra,
    )
