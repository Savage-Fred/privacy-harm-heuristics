"""Heuristic extraction and summary utilities.

This module provides:
    - extract_heuristics(...): to produce rule-like items from supported models
    - save_heuristics_json(...): to write items as JSONL
    - save_heuristics_markdown(...): to render a concise, human-readable Markdown
        summary suitable for reports and the repository artifacts.

Explainability-first: outputs focus on simple, interpretable representations
such as decision rules and linear coefficients. Avoids black-box summaries.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _decision_tree_to_nested(model, feature_names: Sequence[str]) -> Dict[str, Any]:
    """Convert a fitted sklearn DecisionTree into a nested dict.

    The structure is suitable for UI rendering as a human-readable tree:
      { "heuristic": str, "weight": float, "left": node|None, "right": node|None }

    Notes:
    - Internal nodes use the predicate "<feature> <= <threshold>" as heuristic text.
    - Leaf nodes use "class=<predicted_class>" as heuristic text.
    - "weight" is the support share at the node (n_node_samples / n_total_samples).
    """
    tree = getattr(model, "tree_", None)
    if tree is None:
        raise ValueError("Provided model does not expose a sklearn tree_.")

    TREE_UNDEFINED = -2
    n_samples_total = float(tree.n_node_samples[0]) if hasattr(tree, "n_node_samples") else 0.0

    def build(node: int) -> Dict[str, Any]:
        feature_index = tree.feature[node]
        threshold = tree.threshold[node]
        node_samples = float(tree.n_node_samples[node]) if hasattr(tree, "n_node_samples") else 0.0
        weight = (node_samples / n_samples_total) if n_samples_total else 0.0
        if feature_index != TREE_UNDEFINED:
            feat = feature_names[feature_index]
            return {
                "heuristic": f"{feat} <= {threshold:.4f}",
                "weight": weight,
                "left": build(tree.children_left[node]),
                "right": build(tree.children_right[node]),
            }
        # Leaf node: derive majority class
        value = tree.value[node][0]
        if float(np.sum(value)) == 0:
            pred_class = 0
        else:
            pred_class = int(np.argmax(value))
        return {
            "heuristic": f"class={pred_class}",
            "weight": weight,
            "left": None,
            "right": None,
        }

    return build(0)


def _decision_tree_rules(model, feature_names: Sequence[str], X, y) -> List[Dict[str, Any]]:
    tree = model.tree_
    TREE_UNDEFINED = -2
    rules: List[Dict[str, Any]] = []
    path: List[str] = []
    n_samples = (
        tree.n_node_samples[0]
        if hasattr(tree, "n_node_samples")
        else (len(y) if y is not None else 1)
    )

    def recurse(node: int):
        feature_index = tree.feature[node]
        threshold = tree.threshold[node]
        if feature_index != TREE_UNDEFINED:
            name = feature_names[feature_index]
            path.append(f"{name} <= {threshold:.4f}")
            recurse(tree.children_left[node])
            path.pop()
            path.append(f"{name} > {threshold:.4f}")
            recurse(tree.children_right[node])
            path.pop()
        else:
            value = tree.value[node][0]
            total = float(np.sum(value))
            if total == 0:
                return
            pred_class = int(np.argmax(value))
            support = total / n_samples if n_samples else 0.0
            pos = float(value[1]) if len(value) > 1 else float(value[0])
            precision = (pos / total) if len(value) > 1 else 1.0
            text = " AND ".join(path) if path else "<root>"
            rules.append(
                {
                    "model_type": "decision_tree",
                    "kind": "rule",
                    "text": f"IF {text} THEN class={pred_class}",
                    "support": support,
                    "precision": precision,
                    "extra": {"leaf_samples": int(total)},
                }
            )

    recurse(0)
    return rules


def _sparse_linear_heuristics(
    model, feature_names: Sequence[str], X=None, y=None
) -> List[Dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression

    lr = model
    if hasattr(model, "named_steps"):
        lr = model.named_steps.get("lr", model)
    if not isinstance(lr, LogisticRegression):
        return []
    coefs = lr.coef_
    items: List[Dict[str, Any]] = []
    baseline_pos_rate = None
    if y is not None:
        # Assume binary for now; if multi, we treat class_idx separately
        try:
            baseline_pos_rate = float(np.mean(y == 1)) if len(set(y)) <= 2 else None
        except Exception:
            baseline_pos_rate = None
    # Convert pandas inputs to numpy arrays for consistent slicing
    if X is not None and hasattr(X, "values"):
        try:
            X_array = X.values
        except Exception:
            X_array = X
    else:
        X_array = X
    if y is not None and hasattr(y, "values"):
        try:
            y_array = y.values
        except Exception:
            y_array = y
    else:
        y_array = y

    for class_idx, row in enumerate(coefs):
        ranked = sorted(zip(feature_names, row), key=lambda t: abs(t[1]), reverse=True)
        for feat, weight in ranked:
            if abs(weight) < 1e-9:
                continue
            direction = ">" if weight > 0 else "<="
            support = None
            precision = None
            lift = None
            if (
                X_array is not None
                and y_array is not None
                and feat in feature_names
                and len(set(y_array)) <= 2
            ):
                # Derive a simple threshold at median to approximate a pseudo-rule
                idx = feature_names.index(feat)
                try:
                    col = X_array[:, idx]
                except Exception:
                    # Fallback: attempt column selection if X_array is structured differently
                    col = X_array[feat] if isinstance(X_array, dict) and feat in X_array else None
                if col is None:
                    continue
                thresh = float(np.median(col))
                if weight > 0:
                    mask = col > thresh
                else:
                    mask = col <= thresh
                matched = np.sum(mask)
                if matched > 0:
                    # shape attribute present for numpy/pandas; fallback to len otherwise
                    # Robustly compute total samples across numpy arrays, pandas, or lists
                    if hasattr(X_array, "shape") and getattr(X_array, "shape"):
                        total_samples = float(getattr(X_array, "shape")[0])
                    else:
                        try:
                            total_samples = float(len(X_array))
                        except Exception:
                            total_samples = 0.0
                    support = matched / total_samples if total_samples else None
                    pos = np.sum(y_array[mask] == 1)
                    precision = (pos / matched) if matched else None
                    if precision is not None and baseline_pos_rate and baseline_pos_rate > 0:
                        lift = precision / baseline_pos_rate
            items.append(
                {
                    "model_type": "sparse_linear",
                    "kind": "coefficient",
                    "text": (
                        f"IF {feat} {direction} median_value THEN class "
                        f"{class_idx} (w={weight:.4f})"
                    ),
                    "support": support,
                    "precision": precision,
                    "lift": lift,
                    "extra": {"weight": float(weight), "class_index": class_idx},
                }
            )
    return items


def _brl_rules(extra: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules_raw = extra.get("rules") or []
    out: List[Dict[str, Any]] = []
    for r in rules_raw:
        text = r.get("text") if isinstance(r, dict) else str(r)
        out.append(
            {
                "model_type": "brl",
                "kind": "rule",
                "text": text,
                "support": None,
                "precision": None,
                "lift": None,
                "extra": {},
            }
        )
    return out


def _bayes_net_edges(extra: Dict[str, Any]) -> List[Dict[str, Any]]:
    edges = extra.get("edges") or []
    out: List[Dict[str, Any]] = []
    for src, dst in edges:
        out.append(
            {
                "model_type": "bayes_net",
                "kind": "edge",
                "text": f"{src} -> {dst}",
                "support": None,
                "precision": None,
                "lift": None,
                "extra": {},
            }
        )
    return out


def _ebm_rules(model, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Extract top interaction terms from EBM."""
    # EBM is additive: score = term1 + term2 + ...
    # We can extract the most important terms.
    if not hasattr(model, "term_importances"):
        return []

    importances = model.term_importances()
    term_names = model.term_names_

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    items: List[Dict[str, Any]] = []
    for idx in indices:
        if importances[idx] == 0:
            continue

        name = term_names[idx]
        score = importances[idx]

        items.append(
            {
                "model_type": "ebm",
                "kind": "term",
                "text": f"Term: {name} (importance={score:.4f})",
                "support": None,
                "precision": None,
                "extra": {"importance": float(score)},
            }
        )

    return items


def _harm_taxonomy_rules() -> List[Dict[str, Any]]:
    """Extract static rules from the harm taxonomy definitions."""
    from ..processing.harm_taxonomy import SENSITIVE_DATA_TERMS, DISTRESS_PATTERNS

    items: List[Dict[str, Any]] = []

    # Sensitive Data Rules
    for sensitivity, terms in SENSITIVE_DATA_TERMS.items():
        for term in terms:
            items.append(
                {
                    "model_type": "harm_taxonomy",
                    "kind": "pattern",
                    "text": f"IF text contains '{term}' THEN sensitivity={sensitivity.name}",
                    "support": None,
                    "precision": 1.0,  # Definitional
                    "extra": {"category": "sensitive_data"},
                }
            )

    # Distress Rules
    # Extract terms from regex if possible, or just the regex itself
    pattern = DISTRESS_PATTERNS.pattern
    # Remove \b and other regex chars for display
    clean_pattern = pattern.replace(r"\b", "").replace("(", "").replace(")", "")
    terms = clean_pattern.split("|")

    for term in terms:
        items.append(
            {
                "model_type": "harm_taxonomy",
                "kind": "pattern",
                "text": f"IF text contains '{term}' THEN harm=DISTRESS",
                "support": None,
                "precision": 1.0,
                "extra": {"category": "distress"},
            }
        )

    return items


def extract_heuristics(
    result,
    feature_names: Sequence[str],
    X_train=None,
    y_train=None,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract heuristic items (rules/coefficients/edges) from a model result.

    Supports different model types via internal strategy functions. Adds a
    shared provenance hash + version for downstream caching and reproducibility.

    Args:
        result: Object with ``model_type``, ``artifacts`` and optional ``extra``.
        feature_names: Ordered feature names corresponding to model inputs.
        X_train: Optional training feature matrix (numpy/pandas) for support
            / precision estimates (sparse linear heuristics).
        y_train: Optional training labels for support / precision calculations.
        top_n: If provided, limit number of heuristics after ranking.

    Returns:
        List of heuristic dicts. Each has keys: model_type, kind, text, support,
        precision, (optionally lift), and ``extra`` metadata.
    """
    mtype = getattr(result, "model_type", None)
    model = getattr(result, "artifacts", {}).get("model") if hasattr(result, "artifacts") else None
    items: List[Dict[str, Any]] = []
    if mtype == "decision_tree" and model is not None:
        items.extend(_decision_tree_rules(model, feature_names, X_train, y_train))
    elif mtype == "sparse_linear" and model is not None:
        items.extend(_sparse_linear_heuristics(model, feature_names, X=X_train, y=y_train))
    elif mtype == "brl":
        items.extend(_brl_rules(getattr(result, "extra", {})))
    elif mtype == "bayes_net":
        items.extend(_bayes_net_edges(getattr(result, "extra", {})))
    elif mtype == "ebm" and model is not None:
        items.extend(_ebm_rules(model, feature_names))
    elif mtype == "harm_taxonomy":
        items.extend(_harm_taxonomy_rules())
    elif mtype == "dnn_scipy":
        # Placeholder for DNN
        items.append(
            {
                "model_type": "dnn_scipy",
                "kind": "info",
                "text": "DNN (SciPy) is a black-box model. No interpretable rules available.",
                "support": None,
                "precision": None,
                "extra": {},
            }
        )

    def score(it):
        if it["kind"] == "rule":
            return (it.get("support") or 0) * (it.get("precision") or 0)
        if it["kind"] == "coefficient":
            return abs(it.get("extra", {}).get("weight", 0))
        return 0

    items.sort(key=score, reverse=True)
    if top_n is not None:
        items = items[:top_n]

    # Add provenance hash & version to each item (deterministic ordering)
    payload = "\n".join(sorted(it["text"] for it in items))
    prov_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    for it in items:
        it.setdefault("extra", {})
        it["extra"]["provenance_hash"] = prov_hash
        it["extra"]["version"] = 1
    return items


def save_heuristics_json(items: List[Dict[str, Any]], out_path: str):
    """Write heuristics list to JSONL (one object per line)."""
    import json

    with open(out_path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    return out_path


def save_heuristics_tree_json(result, feature_names: Sequence[str], out_path: str) -> str:
    """Persist a nested tree representation for Decision Trees as JSON.

    Args:
        result: Model result object with ``model_type`` and artifacts["model"].
        feature_names: Names for input features (aligned with the tree indices).
        out_path: Destination file path for JSON document.

    Returns:
        The output path string.
    """
    import json

    if getattr(result, "model_type", None) != "decision_tree":
        # Only meaningful for decision trees; silently skip for others
        return out_path
    model = getattr(result, "artifacts", {}).get("model") if hasattr(result, "artifacts") else None
    if model is None:
        return out_path
    tree_obj = _decision_tree_to_nested(model, feature_names)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(tree_obj, fh)
    return out_path


def save_heuristics_markdown(
    items: List[Dict[str, Any]],
    out_path: str,
    title: str | None = None,
    context: Optional[Dict[str, Any]] = None,
    top_n: int = 20,
) -> str:
    """Render a concise Markdown summary of heuristics.

    Args:
        items: Heuristic items produced by ``extract_heuristics`` or similar.
        out_path: Destination Markdown file path.
        title: Optional document title. Defaults to "Heuristic Summary".
        context: Optional metadata block (model type, metrics, data provenance).
        top_n: Limit the number of items displayed in the top section.

    Returns:
        The output path string.
    """
    title = title or "Heuristic Summary"
    now = datetime.utcnow().isoformat() + "Z"
    ctx = context or {}

    # Build a short header with key facts
    header_lines: list[str] = [f"# {title}", "", f"_Generated: {now}_", ""]
    if ctx:
        header_lines.append("## Context")
        for k, v in ctx.items():
            header_lines.append(f"- **{k}**: {v}")
        header_lines.append("")

    # Top-N list
    top = items[:top_n]
    lines: list[str] = []
    lines.extend(header_lines)
    lines.append("## Top Heuristics")
    if not top:
        lines.append("No heuristics available.")
    else:
        for i, it in enumerate(top, start=1):
            text = it.get("text") or it.get("rule") or "(unknown)"
            support = it.get("support")
            precision = it.get("precision")
            kind = it.get("kind") or "item"
            meta_bits: list[str] = []
            if isinstance(support, (int, float)):
                # Render both share and absolute when available
                if 0 <= float(support) <= 1:
                    meta_bits.append(f"support={float(support):.3f}")
                else:
                    meta_bits.append(f"support={int(support)}")
            if isinstance(precision, (int, float)):
                meta_bits.append(f"precision={float(precision):.3f}")
            lines.append(f"{i}. ({kind}) {text}  ")
            if meta_bits:
                lines.append(f"   - {' | '.join(meta_bits)}")

    # Tabular appendix for easier scanning
    if top:
        lines.append("")
        lines.append("## Appendix: Heuristics Table")
        lines.append("")
        lines.append("| # | Kind | Text | Support | Precision |")
        lines.append("|:-:|:-----|:-----|:-------:|:---------:|")
        for i, it in enumerate(top, start=1):
            text = (it.get("text") or it.get("rule") or "").replace("|", "\\|")
            kind = it.get("kind") or "item"
            support = it.get("support")
            precision = it.get("precision")
            if isinstance(support, (int, float)) and 0 <= float(support) <= 1:
                support_str = f"{float(support):.3f}"
            elif isinstance(support, (int, float)):
                support_str = str(int(support))
            else:
                support_str = ""
            precision_str = f"{float(precision):.3f}" if isinstance(precision, (int, float)) else ""
            lines.append(f"| {i} | {kind} | {text} | {support_str} | {precision_str} |")

    content = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return out_path
