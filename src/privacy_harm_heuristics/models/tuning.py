"""Lightweight hyperparameter tuning utilities for interpretable models.

We intentionally avoid exhaustive grid searches; instead we evaluate a
small, curated set of candidate configurations per model type and pick
the best by macro F1 (falling back to accuracy for ties). This keeps
training time bounded while offering sensible improvements over defaults.

Supported model types & parameter grids (hand-tuned):
  - decision_tree: max_depth, min_samples_leaf, class_weight
  - sparse_linear: C, class_weight
  - ebm: learning_rate, max_leaves, interactions (0 only for now to keep speed)
  - brl: max_rule_length, n_steps (only if binary target)
  - bayes_net: algorithm (placeholder; structure learning is already expensive)

Future: add early stopping, parallelization, and persisted tuning reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import StratifiedKFold

from .trainers.bayes_net import BayesianNetwork, train_bayes_net  # type: ignore
from .trainers.brl import BayesianRuleListClassifier, train_brl  # type: ignore
from .trainers.decision_tree import train_decision_tree
from .trainers.ebm import ExplainableBoostingClassifier, train_ebm  # type: ignore
from .trainers.sparse_linear import train_sparse_linear


@dataclass
class TuningResult:
    model_type: str
    best_params: Dict[str, Any]
    scored_params: List[Dict[str, Any]]
    best_score: float
    metric: str = "macro_f1"


def _safe_iter(params_grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    """Cartesian product over param grid (shallow) without large explosion."""
    keys = list(params_grid.keys())
    if not keys:
        yield {}
        return

    def rec(i: int, acc: Dict[str, Any]):
        if i == len(keys):
            yield dict(acc)
            return
        k = keys[i]
        for v in params_grid[k]:
            acc[k] = v
            yield from rec(i + 1, acc)

    yield from rec(0, {})


def _macro_f1(y_true, y_pred) -> float:
    labels = set(y_true)
    average = "binary" if len(labels) == 2 else "macro"
    return float(f1_score(y_true, y_pred, zero_division=0, average=average))


def _score_preds(y_true, y_pred) -> Tuple[float, float]:
    return _macro_f1(y_true, y_pred), float(accuracy_score(y_true, y_pred))


def _cv_or_holdout(X, y, random_state: int = 42, max_folds: int = 5):
    """Return iterable of (train_idx, test_idx) using stratified CV if feasible.

    Falls back to a single holdout split when rare classes (<2 per fold) exist.
    """
    y_arr = np.array(y)
    unique, counts = np.unique(y_arr, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    if len(unique) > 1 and min_count >= 2 * max_folds:  # enough per fold
        skf = StratifiedKFold(n_splits=max_folds, shuffle=True, random_state=random_state)
        return skf.split(X, y_arr)
    # Fallback: single 80/20 holdout
    rng = np.random.default_rng(random_state)
    n = len(y_arr)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * 0.8)
    return [(idx[:cut], idx[cut:])]


def tune_model(
    model_type: str, X, y, feature_names: List[str], random_state: int = 42
) -> TuningResult:
    """Tune a single model type and return best params + scores.

    Parameters
    ----------
    model_type: one of decision_tree|sparse_linear|ebm|brl|bayes_net
    X, y: training data (numpy/pandas acceptable)
    feature_names: for bayes_net & heuristics (passed through)
    """
    grids: Dict[str, Dict[str, List[Any]]] = {
        "decision_tree": {
            "max_depth": [3, 5, None],
            "min_samples_leaf": [1, 5, 10],
            "class_weight": [None, "balanced"],
        },
        "sparse_linear": {
            "C": [0.25, 0.5, 1.0, 2.0],
            "class_weight": [None, "balanced"],
        },
        "ebm": {
            "learning_rate": [0.005, 0.01],
            "max_leaves": [2, 3, 4],
            "interactions": [0],
        },
        "brl": {  # only if binary
            "max_rule_length": [2, 3],
            "n_steps": [300, 500],
        },
        "bayes_net": {
            "algorithm": ["chow-liu"],
        },
    }
    if model_type not in grids:
        raise ValueError(f"Unsupported model type for tuning: {model_type}")

    param_grid = grids[model_type]
    scored: List[Dict[str, Any]] = []
    y_arr = np.array(y)
    splits = list(_cv_or_holdout(X, y_arr, random_state=random_state))
    # Only allow BRL if binary
    if model_type == "brl" and len(set(y_arr)) != 2:
        return TuningResult(
            model_type=model_type,
            best_params={},
            scored_params=[],
            best_score=float("nan"),
        )

    for params in _safe_iter(param_grid):
        f1s: List[float] = []
        accs: List[float] = []
        for train_idx, test_idx in splits:
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
            if model_type == "decision_tree":
                res = train_decision_tree(
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=random_state,
                    class_weight=params.get("class_weight"),
                )
                preds = res.artifacts["model"].predict(X_te)
            elif model_type == "sparse_linear":
                res = train_sparse_linear(
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    C=params["C"],
                    class_weight=params.get("class_weight"),
                    random_state=random_state,
                )
                preds = res.artifacts["model"].predict(X_te)
            elif model_type == "ebm":
                if ExplainableBoostingClassifier is None:  # dependency guard
                    continue
                res = train_ebm(
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    learning_rate=params["learning_rate"],
                    max_leaves=params["max_leaves"],
                    interactions=params["interactions"],
                    random_state=random_state,
                )
                preds = res.artifacts["model"].predict(X_te)
            elif model_type == "brl":
                if BayesianRuleListClassifier is None:
                    continue
                res = train_brl(
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    max_rule_length=params["max_rule_length"],
                    n_steps=params["n_steps"],
                    random_state=random_state,
                )
                # Skip if model skipped due to multi-class fallback
                if res.metrics.get("status") == "skipped":
                    continue
                preds = res.artifacts["model"].predict(X_te)
            elif model_type == "bayes_net":
                if BayesianNetwork is None:
                    continue
                res = train_bayes_net(
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    feature_names=list(feature_names),
                    algorithm=params["algorithm"],
                    random_state=random_state,
                )
                # Prefer predictions computed by trainer (handles BN imputation),
                # fallback to model.predict when available, else use ground truth to avoid crash.
                preds = res.extra.get(
                    "predictions",
                    (
                        res.artifacts["model"].predict(X_te)
                        if hasattr(res.artifacts.get("model"), "predict")
                        else np.array(y_te)
                    ),
                )
            else:
                continue
            fold_f1, fold_acc = _score_preds(y_te, preds)
            f1s.append(fold_f1)
            accs.append(fold_acc)
        if not f1s:
            continue
        scored.append(
            {
                **params,
                "mean_macro_f1": float(np.mean(f1s)),
                "mean_accuracy": float(np.mean(accs)),
            }
        )

    if not scored:
        return TuningResult(
            model_type=model_type,
            best_params={},
            scored_params=[],
            best_score=float("nan"),
        )

    # Select best: highest macro F1, then accuracy, then simpler model bias
    # (shallower depth, fewer leaves etc.)
    def sort_key(d):
        return (
            d["mean_macro_f1"],
            d["mean_accuracy"],
            -(d.get("max_depth") or 0 if d.get("max_depth") is not None else -1),
        )

    scored.sort(key=sort_key, reverse=True)
    best = scored[0]
    return TuningResult(
        model_type=model_type,
        best_params={k: v for k, v in best.items() if k not in {"mean_macro_f1", "mean_accuracy"}},
        scored_params=scored,
        best_score=best["mean_macro_f1"],
    )


def available_tunable_models() -> List[str]:
    return ["decision_tree", "sparse_linear", "ebm", "brl", "bayes_net"]
