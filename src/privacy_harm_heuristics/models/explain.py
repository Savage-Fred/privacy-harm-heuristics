"""Explanation utilities (LIME & SHAP).

Goals:
  - Provide lightweight wrapper functions to compute local explanations for a single record.
  - Avoid heavy re-initialization cost by accepting pre-fit model and background data.

Functions:
  make_shap_explainer(model, background) -> explainer
  shap_explain(explainer, instance, feature_names) -> dict
  lime_explain(model, instance_df, feature_names, class_names=None, num_features=10) -> dict

Returned dict schema (minimal & JSON serializable):
  {
    "method": "shap|lime",
    "feature_contributions": [{"feature": name, "value": value, "contribution": float}],
    "expected_value": float | None,
    "model_output": float | None,
  }

These utilities intentionally avoid writing to disk; callers may decide persistence strategy.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

try:  # optional heavy deps
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None  # type: ignore

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except Exception:  # pragma: no cover
    LimeTabularExplainer = None  # type: ignore


def make_shap_explainer(model, background, approximate: bool = False):
    """Create a SHAP explainer for the given model and background data.

    Attempts a tree-specific explainer first; falls back to generic Kernel
    explainer when necessary (optionally subsampling background when
    ``approximate`` is True).
    """
    if shap is None:
        raise ImportError("shap not installed")
    # Tree-based models can use TreeExplainer; fallback to KernelExplainer.
    model_name = model.__class__.__name__.lower()
    if (
        "tree" in model_name
        or "forest" in model_name
        or "boost" in model_name
        or "ebm" in model_name
    ):
        try:
            return shap.Explainer(model, background, algorithm="auto")
        except Exception:  # pragma: no cover
            pass
    if approximate:
        # KernelExplainer with a small background sample for speed.
        bg = background
        if isinstance(bg, np.ndarray):
            if bg.shape[0] > 50:
                idx = np.random.choice(bg.shape[0], 50, replace=False)
                bg = bg[idx]
        return shap.KernelExplainer(model.predict_proba, bg)
    return shap.Explainer(model.predict_proba, background)


def shap_explain(explainer, instance, feature_names: Sequence[str]):
    """Return SHAP explanation dict for a single instance.

    Normalizes varied SHAP value tensor shapes into a flat contribution list.
    """
    if shap is None:
        raise ImportError("shap not installed")
    # Ensure 2D instance
    if hasattr(instance, "values"):
        arr = instance.values
    else:
        arr = np.asarray(instance)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    values = explainer(arr)
    exp = values[0]
    row = arr[0]
    v = np.asarray(exp.values)
    feature_count = len(feature_names)
    # Derive contribution vector robustly across shapes
    contrib_vector = None
    if v.ndim == 1 and v.shape[0] == feature_count:
        contrib_vector = v
    elif v.ndim >= 2:
        # Prefer last axis if matches
        if v.shape[-1] == feature_count:
            slicer = (0,) * (v.ndim - 1) + (slice(None),)
            contrib_vector = v[slicer]
        elif v.shape[0] == feature_count:
            # Use first along remaining dims
            # Build slicer selecting first for remaining axes
            slicer = (slice(None),) + tuple(0 for _ in range(v.ndim - 1))
            contrib_vector = v[slicer]
    if contrib_vector is None:
        contrib_vector = v.flatten()[:feature_count]
    contributions = []
    for feat, val, contrib in zip(feature_names, row, contrib_vector):
        try:
            c_val = float(contrib)
        except Exception:
            try:
                c_val = float(np.asarray(contrib).item())
            except Exception:
                c_val = 0.0
        try:
            f_val = float(val)
        except Exception:
            f_val = 0.0
        contributions.append({"feature": feat, "value": f_val, "contribution": c_val})
    expected_value = None
    try:
        expected_value = float(exp.base_values)
    except Exception:
        pass
    model_output = None
    try:
        model_output = float(
            np.sum(contrib_vector) + (exp.base_values if expected_value is not None else 0)
        )
    except Exception:
        pass
    return {
        "method": "shap",
        "feature_contributions": contributions,
        "expected_value": expected_value,
        "model_output": model_output,
    }


def lime_explain(
    model,
    instance_df,
    feature_names: Sequence[str],
    class_names: Sequence[str] | None = None,
    num_features: int = 10,
) -> Dict[str, Any]:
    """Return LIME explanation for the first row of ``instance_df``."""
    if LimeTabularExplainer is None:
        raise ImportError("lime not installed")
    data = instance_df
    if hasattr(data, "values"):
        data_values = data.values
    else:
        data_values = np.asarray(data)
    explainer = LimeTabularExplainer(
        training_data=data_values,
        feature_names=list(feature_names),
        class_names=list(class_names) if class_names else None,
        discretize_continuous=True,
        verbose=False,
        mode="classification",
    )
    instance_row = data_values[0]
    exp = explainer.explain_instance(instance_row, model.predict_proba, num_features=num_features)
    contributions = []
    for feat, weight in exp.as_list():  # list of (feature description, weight)
        contributions.append({"feature": feat, "contribution": float(weight)})
    return {
        "method": "lime",
        "feature_contributions": contributions,
        "expected_value": None,
        "model_output": None,
    }
