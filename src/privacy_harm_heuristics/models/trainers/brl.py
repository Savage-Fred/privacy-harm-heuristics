"""Bayesian Rule List (BRL) trainer.

Wraps imodels BayesianRuleListClassifier (if available) to produce an interpretable
set of decision rules. Falls back with a clear ImportError if the dependency is
missing so upstream callers can skip gracefully in tests.

Returned ModelResult.extra fields:
  rules: List[ {text: str} ]  (stringified rules for now)
  n_rules: int
  support_estimates: Optional list of support values if obtainable

Future enhancements:
  - Parse rule antecedents into structured predicates
  - Compute precision/recall per rule against training data
  - Export rule weights / posterior probabilities
"""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ..metrics_utils import choose_average

from .. import ModelResult

try:  # Lightweight guard; imodels can be relatively heavy
    from imodels import BayesianRuleListClassifier  # type: ignore
except Exception:  # pragma: no cover - handled at call time
    BayesianRuleListClassifier = None  # type: ignore


def _extract_rules(clf) -> List[Dict[str, Any]]:
    """Best-effort extraction of rules.

    Different imodels versions expose internal attributes differently.
    We attempt a few common attribute names and fall back to str(model).
    """
    candidates = []
    for attr in ("rules_", "rule_list_", "model_", "estimators_"):
        obj = getattr(clf, attr, None)
        if obj is None:
            continue
        if isinstance(obj, (list, tuple)):
            for r in obj:
                candidates.append({"text": str(r)})
        else:
            candidates.append({"text": str(obj)})
        if candidates:
            break
    if not candidates:
        candidates.append({"text": str(clf)})
    return candidates


def train_brl(
    X_train,
    y_train,
    X_test,
    y_test,
    max_rule_length: int = 3,
    n_steps: int = 500,
    n_chains: int = 1,
    random_state: int = 42,
) -> ModelResult:
    """Train a Bayesian Rule List classifier and return a ModelResult.

    Falls back gracefully if ``imodels`` is unavailable by raising ImportError.
    Uses a conservative subset of parameters for cross-version compatibility.
    """
    if BayesianRuleListClassifier is None:  # pragma: no cover - depends on optional dep
        raise ImportError("imodels package not available; cannot train Bayesian Rule List")

    # BRL currently only robust for binary classification in imodels; skip gracefully.
    unique_classes = sorted(set(y_train))
    if len(unique_classes) != 2:
        return ModelResult(
            model_type="brl",
            metrics={
                "status": "skipped",
                "reason": "brl_binary_only",
                "n_classes": len(unique_classes),
            },
            artifacts={},
            extra={
                "rules": [],
                "n_rules": 0,
                "max_rule_length": max_rule_length,
                "n_steps": n_steps,
                "skipped": True,
            },
        )
    # Parameter names may vary slightly between versions; use conservative subset.
    try:
        # Some versions accept these hyperparameters; others have differing names.
        clf = BayesianRuleListClassifier(  # type: ignore[call-arg]
            max_rule_length=max_rule_length,  # type: ignore[arg-type]
            n_steps=n_steps,  # type: ignore[arg-type]
            n_chains=n_chains,  # type: ignore[arg-type]
            random_state=random_state,
        )
    except TypeError:
        try:  # Retry with reduced argument set
            clf = BayesianRuleListClassifier(random_state=random_state)  # type: ignore[call-arg]
        except TypeError:  # Final fallback
            clf = BayesianRuleListClassifier()  # type: ignore[call-arg]
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
    rules = _extract_rules(clf)
    extra = {
        "rules": rules,
        "n_rules": len(rules),
        "max_rule_length": max_rule_length,
        "n_steps": n_steps,
    }
    return ModelResult(
        model_type="brl",
        metrics=metrics,
        artifacts={"model": clf},
        extra=extra,
    )
