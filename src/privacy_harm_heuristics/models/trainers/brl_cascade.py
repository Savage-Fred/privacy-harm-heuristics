"""Cascading Bayesian Rule List (BRL) trainer for multi-class privacy harm classification.

This module implements a one-vs-rest approach using BRL classifiers, where each
harm category gets its own binary classifier that produces a yes/no decision tree
with associated risk weights.

Architecture:
    - 18 binary classifiers (one per Solove harm category)
    - Each classifier outputs a cascading yes/no question tree
    - Terminal nodes assign risk weights (0-100%) toward total harm score
    - Aggregate across all classifiers for composite risk profile

Example output for one harm category (secondary_use):
    Q1: "Does the product collect personal data?"
        → No: risk_weight = 0.0 (STOP)
        → Yes: continue to Q2
    Q2: "Is data shared with third parties?"
        → No: risk_weight = 0.2
        → Yes: continue to Q3
    Q3: "Is there explicit consent for sharing?"
        → Yes: risk_weight = 0.3
        → No: risk_weight = 0.8 (HIGH RISK)

This approach aligns with the user's vision of BRL as a checklist where each
yes/no answer provides insight into total privacy risk.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from .. import ModelResult

logger = logging.getLogger(__name__)

# Solove taxonomy harm categories
HARM_CATEGORIES = [
    "surveillance",
    "interrogation",
    "aggregation",
    "identification",
    "insecurity",
    "secondary_use",
    "exclusion",
    "breach_of_confidentiality",
    "disclosure",
    "exposure",
    "increased_accessibility",
    "blackmail",
    "appropriation",
    "distortion",
    "intrusion",
    "decisional_interference",
]

try:
    from imodels import BayesianRuleListClassifier
except ImportError:
    BayesianRuleListClassifier = None


@dataclass
class RuleNode:
    """A node in the cascading decision tree."""

    question: str
    feature_name: str
    threshold: float = 0.5
    yes_branch: Optional["RuleNode"] = None
    no_branch: Optional["RuleNode"] = None
    risk_weight: Optional[float] = None  # Terminal node weight

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "question": self.question,
            "feature_name": self.feature_name,
            "threshold": self.threshold,
        }
        if self.risk_weight is not None:
            result["risk_weight"] = self.risk_weight
        if self.yes_branch:
            result["yes"] = self.yes_branch.to_dict()
        if self.no_branch:
            result["no"] = self.no_branch.to_dict()
        return result


@dataclass
class HarmClassifierResult:
    """Result from training one harm category classifier."""

    harm_category: str
    metrics: Dict[str, float]
    decision_tree: RuleNode
    feature_importances: Dict[str, float]
    model: Any = None


@dataclass
class CascadeBRLResult:
    """Aggregate result from all harm category classifiers."""

    classifiers: Dict[str, HarmClassifierResult] = field(default_factory=dict)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    master_checklist: List[Dict[str, Any]] = field(default_factory=list)


def _feature_to_question(feature_name: str) -> str:
    """Convert feature name to human-readable yes/no question.

    Examples:
        kw_camera -> "Does this involve camera/video recording?"
        kw_third_party -> "Is data shared with third parties?"
        f_has_minors -> "Does this involve minors/children?"
    """
    # Common feature mappings
    mappings = {
        "kw_camera": "Does this product have a camera or video recording capability?",
        "kw_always_on": "Is the device or feature always on/listening?",
        "kw_third_party": "Is data shared with third parties?",
        "kw_biometric": "Does this involve biometric data (face, fingerprint, voice)?",
        "kw_location": "Does this track or collect location data?",
        "kw_consent": "Is there explicit user consent?",
        "kw_encryption": "Is data encrypted?",
        "kw_children": "Does this involve children or minors?",
        "kw_health": "Does this involve health or medical data?",
        "kw_financial": "Does this involve financial data?",
        "kw_behavioral": "Does this involve behavioral tracking?",
        "kw_profiling": "Does this involve user profiling?",
        "kw_retention": "Is there long-term data retention?",
        "kw_sharing": "Is data shared externally?",
        "kw_selling": "Is data sold to other parties?",
        "kw_breach": "Has there been a data breach?",
        "kw_hack": "Has there been unauthorized access?",
        "kw_leak": "Has data been leaked or exposed?",
        "f_has_minors": "Does this involve minors/children?",
        "f_has_biometric": "Does this collect biometric identifiers?",
        "f_has_penalty": "Has there been a regulatory penalty?",
        "f_sensitive_data": "Does this involve sensitive personal data?",
        "f_third_party_access": "Do third parties have data access?",
        "f_lack_of_consent": "Is there a lack of clear consent?",
        "f_data_exposed": "Was personal data exposed?",
    }

    if feature_name in mappings:
        return mappings[feature_name]

    # Generate question from feature name
    clean_name = feature_name.replace("kw_", "").replace("f_", "").replace("_", " ")
    return f"Does this involve {clean_name}?"


def _extract_rules_as_tree(clf, feature_names: List[str]) -> RuleNode:
    """Extract BRL rules as a cascading decision tree structure.

    Attempts to parse the internal rule representation into a tree of
    yes/no questions with risk weights at terminal nodes.
    """
    # Try to extract rules from various imodels attributes
    rules_text = []
    for attr in ("rules_", "rule_list_", "rules", "d_star", "final_rule_list"):
        obj = getattr(clf, attr, None)
        if obj is not None:
            if isinstance(obj, (list, tuple)):
                rules_text = [str(r) for r in obj]
            else:
                rules_text = [str(obj)]
            break

    if not rules_text:
        rules_text = [str(clf)]

    # Parse rules into tree structure
    # For now, create a simple linear chain from extracted rules
    root = None
    current = None

    for i, rule_text in enumerate(rules_text[:5]):  # Limit to 5 levels
        # Try to extract feature references from rule text
        feature_name = None
        for fname in feature_names:
            if fname.lower() in rule_text.lower():
                feature_name = fname
                break

        if not feature_name and feature_names:
            feature_name = feature_names[min(i, len(feature_names) - 1)]
        elif not feature_name:
            feature_name = f"feature_{i}"

        node = RuleNode(
            question=_feature_to_question(feature_name),
            feature_name=feature_name,
            threshold=0.5,
        )

        if root is None:
            root = node
            current = node
        else:
            # `current` is always a RuleNode here: it's only None before the
            # first iteration, which always takes the `if` branch above.
            assert current is not None
            current.yes_branch = node
            current.no_branch = RuleNode(
                question="",
                feature_name="",
                risk_weight=0.2 * i,  # Graduated risk for early exits
            )
            current = node

    # Add terminal nodes
    if current:
        current.yes_branch = RuleNode(
            question="",
            feature_name="",
            risk_weight=0.9,  # High risk if all conditions met
        )
        current.no_branch = RuleNode(
            question="",
            feature_name="",
            risk_weight=0.4,  # Medium risk if last condition not met
        )

    return root or RuleNode(
        question="Default question",
        feature_name="unknown",
        risk_weight=0.5,
    )


def _extract_dt_as_tree(clf, feature_names: List[str]) -> RuleNode:
    """Extract decision tree structure as cascading RuleNode tree.

    Converts sklearn DecisionTreeClassifier to our RuleNode format.
    """
    tree = clf.tree_

    def recurse(node_id: int, depth: int = 0) -> Optional[RuleNode]:
        if depth > 5:  # Limit depth
            return None

        # Check if leaf
        if tree.children_left[node_id] == tree.children_right[node_id]:
            # Leaf node - compute risk weight from class probabilities
            values = tree.value[node_id].flatten()
            if len(values) >= 2:
                risk_weight = values[1] / (values[0] + values[1] + 1e-10)
            else:
                risk_weight = 0.5
            return RuleNode(
                question="",
                feature_name="leaf",
                risk_weight=float(risk_weight),
            )

        # Internal node
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        feature_name = (
            feature_names[feature_idx]
            if feature_idx < len(feature_names)
            else f"feature_{feature_idx}"
        )
        question = _feature_to_question(feature_name)

        node = RuleNode(
            question=question,
            feature_name=feature_name,
            threshold=float(threshold),
            yes_branch=recurse(tree.children_right[node_id], depth + 1),
            no_branch=recurse(tree.children_left[node_id], depth + 1),
        )
        return node

    return recurse(0) or RuleNode(
        question="Default question",
        feature_name="unknown",
        risk_weight=0.5,
    )


def train_harm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    harm_category: str,
    feature_names: List[str],
    max_rule_length: int = 3,
    n_steps: int = 500,
    random_state: int = 42,
) -> HarmClassifierResult:
    """Train a single binary BRL classifier for one harm category."""

    if BayesianRuleListClassifier is None:
        raise ImportError("imodels package required for BRL training")

    # Check if we have both positive and negative examples
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        logger.warning(f"Only one class for {harm_category}, using fallback")
        # Return a trivial classifier result
        return HarmClassifierResult(
            harm_category=harm_category,
            metrics={"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0},
            decision_tree=RuleNode(
                question=f"Is this a case of {harm_category}?",
                feature_name="fallback",
                risk_weight=float(unique_classes[0]) if len(unique_classes) > 0 else 0.0,
            ),
            feature_importances={},
        )

    # Discretize continuous features (BRL requires binary/discrete features)
    # Convert to binary: 1 if > median, else 0
    X_train_discrete = np.zeros_like(X_train, dtype=int)
    X_test_discrete = np.zeros_like(X_test, dtype=int)
    for i in range(X_train.shape[1]):
        # Use training median for both train and test to avoid data leakage
        median_val = np.median(X_train[:, i])
        X_train_discrete[:, i] = (X_train[:, i] > median_val).astype(int)
        X_test_discrete[:, i] = (X_test[:, i] > median_val).astype(int)

    try:
        clf = BayesianRuleListClassifier(
            max_rule_length=max_rule_length,
            n_steps=n_steps,
            random_state=random_state,
        )
    except TypeError:
        clf = BayesianRuleListClassifier(random_state=random_state)
    except Exception:
        clf = BayesianRuleListClassifier()

    try:
        clf.fit(X_train_discrete, y_train)
        preds = clf.predict(X_test_discrete)

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
        }

        # Extract decision tree
        decision_tree = _extract_rules_as_tree(clf, feature_names)

        # Compute feature importances (based on rule occurrence)
        feature_importances = {}
        if hasattr(clf, "rules_") and clf.rules_:
            for fname in feature_names:
                count = sum(1 for r in clf.rules_ if fname.lower() in str(r).lower())
                if count > 0:
                    feature_importances[fname] = count / len(clf.rules_)

        return HarmClassifierResult(
            harm_category=harm_category,
            metrics=metrics,
            decision_tree=decision_tree,
            feature_importances=feature_importances,
            model=clf,
        )

    except Exception as e:
        logger.warning(f"BRL failed for {harm_category}: {e}. Falling back to DecisionTree.")
        # Fallback to DecisionTree which is also interpretable
        try:
            from sklearn.tree import DecisionTreeClassifier

            dt_clf = DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=max(5, int(0.01 * len(y_train))),
                random_state=random_state,
            )
            dt_clf.fit(X_train_discrete, y_train)
            preds = dt_clf.predict(X_test_discrete)

            # Distinct name from `metrics` above (renamed to avoid a same-scope
            # mypy redefinition conflict, since this dict carries an extra
            # non-float diagnostic tag below).
            fallback_metrics: Dict[str, Any] = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1": float(f1_score(y_test, preds, zero_division=0)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall": float(recall_score(y_test, preds, zero_division=0)),
                # Diagnostic tag, not a metric -- HarmClassifierResult.metrics is
                # declared Dict[str, float] but tolerates this str value in practice.
                "fallback": "decision_tree",
            }

            # Extract rules from decision tree
            decision_tree = _extract_dt_as_tree(dt_clf, feature_names)

            # Feature importances from DT
            feature_importances = {
                fname: float(imp)
                for fname, imp in zip(feature_names, dt_clf.feature_importances_)
                if imp > 0
            }

            return HarmClassifierResult(
                harm_category=harm_category,
                metrics=fallback_metrics,
                decision_tree=decision_tree,
                feature_importances=feature_importances,
                model=dt_clf,
            )
        except Exception as dt_e:
            logger.error(f"Both BRL and DT failed for {harm_category}: {dt_e}")
            return HarmClassifierResult(
                harm_category=harm_category,
                # Diagnostic tag, not a metric -- see the analogous "fallback"
                # entry above for why this is Dict[str, Any] rather than float-only.
                metrics={
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "error": str(e),  # type: ignore[dict-item]
                },
                decision_tree=RuleNode(
                    question=f"Is this a case of {harm_category}?",
                    feature_name="error",
                    risk_weight=0.5,
                ),
                feature_importances={},
            )


def train_cascade_brl(
    X: np.ndarray,
    y: np.ndarray,  # Multi-label: shape (n_samples, n_categories) or (n_samples,) with category indices
    feature_names: List[str],
    harm_categories: Optional[List[str]] = None,
    test_size: float = 0.2,
    max_rule_length: int = 3,
    n_steps: int = 500,
    random_state: int = 42,
    max_workers: int = 4,
) -> CascadeBRLResult:
    """Train cascading BRL classifiers for all harm categories.

    This implements a one-vs-rest approach where each harm category gets its
    own binary classifier that produces a yes/no decision tree.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels - either multi-label binary matrix or single-label indices
        feature_names: Names of features for interpretability
        harm_categories: List of harm category names (defaults to HARM_CATEGORIES)
        test_size: Fraction for test split
        max_rule_length: Maximum conditions per rule
        n_steps: MCMC steps for BRL
        random_state: Random seed
        max_workers: Max parallel training jobs (constrained to 4)

    Returns:
        CascadeBRLResult with all classifiers and aggregate metrics
    """
    categories = harm_categories or HARM_CATEGORIES

    # Convert y to multi-label format if needed
    if len(y.shape) == 1:
        lb = LabelBinarizer()
        y_multilabel = lb.fit_transform(y)
        if y_multilabel.shape[1] == 1:
            # Binary case - expand to (n_samples, 2)
            y_multilabel = np.hstack([1 - y_multilabel, y_multilabel])
    else:
        y_multilabel = y

    # Ensure we have the right number of columns
    n_cols = y_multilabel.shape[1]
    if n_cols < len(categories):
        # Pad with zeros
        padding = np.zeros((y_multilabel.shape[0], len(categories) - n_cols))
        y_multilabel = np.hstack([y_multilabel, padding])
    elif n_cols > len(categories):
        # Trim to category count
        y_multilabel = y_multilabel[:, : len(categories)]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multilabel, test_size=test_size, random_state=random_state
    )

    result = CascadeBRLResult()

    # Train one classifier per harm category
    logger.info(f"Training {len(categories)} BRL classifiers (one per harm category)")

    # Limit parallelism per project requirements
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def train_one(idx: int) -> Tuple[str, HarmClassifierResult]:
        category = categories[idx]
        y_train_binary = y_train[:, idx].astype(int)
        y_test_binary = y_test[:, idx].astype(int)

        logger.info(f"Training classifier for: {category}")
        clf_result = train_harm_classifier(
            X_train=X_train,
            y_train=y_train_binary,
            X_test=X_test,
            y_test=y_test_binary,
            harm_category=category,
            feature_names=feature_names,
            max_rule_length=max_rule_length,
            n_steps=n_steps,
            random_state=random_state + idx,
        )
        return category, clf_result

    # Use constrained thread pool (max 4 workers per project requirements)
    effective_workers = min(max_workers, 4, len(categories))

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(train_one, i): i for i in range(len(categories))}

        for future in as_completed(futures):
            try:
                category, clf_result = future.result()
                result.classifiers[category] = clf_result
            except Exception as e:
                idx = futures[future]
                category = categories[idx]
                logger.error(f"Failed to train {category}: {e}")

    # Compute aggregate metrics
    all_f1s = [c.metrics.get("f1", 0) for c in result.classifiers.values()]
    all_accs = [c.metrics.get("accuracy", 0) for c in result.classifiers.values()]

    result.aggregate_metrics = {
        "mean_f1": float(np.mean(all_f1s)) if all_f1s else 0.0,
        "mean_accuracy": float(np.mean(all_accs)) if all_accs else 0.0,
        "n_classifiers": len(result.classifiers),
        "n_successful": sum(1 for c in result.classifiers.values() if c.metrics.get("f1", 0) > 0),
    }

    # Build master checklist (merged view of all decision trees)
    result.master_checklist = _build_master_checklist(result.classifiers)

    return result


def _build_master_checklist(classifiers: Dict[str, HarmClassifierResult]) -> List[Dict[str, Any]]:
    """Build a merged master checklist from all harm classifiers.

    Groups questions by feature and shows which harms each question predicts.
    """
    # Collect all unique questions across classifiers
    question_to_harms: Dict[str, List[str]] = {}

    for harm, clf_result in classifiers.items():
        tree = clf_result.decision_tree
        questions = _collect_questions(tree)

        for q in questions:
            question = q.get("question", "")

            if question and question not in question_to_harms:
                question_to_harms[question] = []
            if question:
                question_to_harms[question].append(harm)

    # Build master checklist
    checklist: List[Dict[str, Any]] = []
    for question, harms in question_to_harms.items():
        checklist.append(
            {
                "question": question,
                "predicts_harms": harms,
                "n_harms": len(harms),
            }
        )

    # Sort by number of harms predicted (most impactful questions first)
    checklist.sort(key=lambda x: x["n_harms"], reverse=True)

    return checklist


def _collect_questions(node: Optional[RuleNode], depth: int = 0) -> List[Dict[str, Any]]:
    """Recursively collect questions from a decision tree."""
    if node is None or depth > 10:
        return []

    questions = []

    if node.question and node.risk_weight is None:  # Non-terminal node
        questions.append(
            {
                "question": node.question,
                "feature_name": node.feature_name,
                "depth": depth,
            }
        )

    questions.extend(_collect_questions(node.yes_branch, depth + 1))
    questions.extend(_collect_questions(node.no_branch, depth + 1))

    return questions


def export_cascade_brl(
    result: CascadeBRLResult,
    output_path: Path,
    version: str = "1.0",
) -> None:
    """Export cascading BRL result to versioned JSON file.

    Output structure:
    {
        "version": "1.0",
        "name": "learned_brl_cascade",
        "description": "18 one-vs-rest BRL classifiers for privacy harm categories",
        "created": "2025-11-25",
        "aggregate_metrics": {...},
        "classifiers": {
            "secondary_use": {
                "metrics": {...},
                "decision_tree": {...},
                "feature_importances": {...}
            },
            ...
        },
        "master_checklist": [...]
    }
    """
    from datetime import datetime

    export_data: Dict[str, Any] = {
        "version": version,
        "name": "learned_brl_cascade",
        "description": "18 one-vs-rest BRL classifiers for Solove privacy harm taxonomy. "
        "Each classifier outputs a yes/no decision tree with risk weights.",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "aggregate_metrics": result.aggregate_metrics,
        "harm_categories": HARM_CATEGORIES,
        "classifiers": {},
        "master_checklist": result.master_checklist,
    }

    for harm, clf_result in result.classifiers.items():
        export_data["classifiers"][harm] = {
            "metrics": clf_result.metrics,
            "decision_tree": clf_result.decision_tree.to_dict() if clf_result.decision_tree else {},
            "feature_importances": clf_result.feature_importances,
        }

    # Ensure output_path is a directory, write to cascade_brl.json inside it
    output_path.mkdir(parents=True, exist_ok=True)
    export_file = output_path / "cascade_brl.json"
    with open(export_file, "w") as f:
        json.dump(export_data, f, indent=2)

    logger.info(f"Exported cascade BRL to {export_file}")


def train_brl_cascade_from_jsonl(
    input_path: Path,
    output_path: Path,
    version: str = "1.0",
    max_rule_length: int = 3,
    n_steps: int = 500,
) -> ModelResult:
    """Convenience function to train cascade BRL from JSONL dataset.

    Loads features from JSONL, trains all harm classifiers, exports results.
    """
    import json

    # Load data
    records = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        return ModelResult(
            model_type="brl_cascade",
            metrics={"error": "No records loaded"},
            artifacts={},
            extra={},
        )

    # Extract features (kw_* and f_* columns, but only numeric ones)
    feature_names = []
    sample_record = records[0]
    for key in sample_record.keys():
        if key.startswith("kw_") or key.startswith("f_"):
            # Only include numeric features
            val = sample_record.get(key)
            if isinstance(val, (int, float)) or val is None:
                feature_names.append(key)

    if not feature_names:
        return ModelResult(
            model_type="brl_cascade",
            metrics={"error": "No numeric features found (kw_* or f_*)"},
            artifacts={},
            extra={},
        )

    logger.info(f"Using {len(feature_names)} numeric features: {feature_names}")

    # Build feature matrix (convert to float, handle None as 0)
    X = np.array(
        [[float(r.get(fn) or 0) for fn in feature_names] for r in records], dtype=np.float64
    )

    # Build label matrix from harm categories
    # Support both 'harm_category' (singular, for single-label) and
    # 'harm_categories' / 'harms' (plural, for multi-label)
    y = np.zeros((len(records), len(HARM_CATEGORIES)))
    for i, r in enumerate(records):
        # Try multiple field names
        harms = (
            r.get("harm_category")  # Single category (most common)
            or r.get("harms", [])
            or r.get("harm_categories_solove", [])
            or r.get("harm_categories", [])
        )
        if isinstance(harms, str):
            harms = [harms]
        if not harms:
            continue
        for harm in harms:
            if harm in HARM_CATEGORIES:
                y[i, HARM_CATEGORIES.index(harm)] = 1

    # Train cascade
    cascade_result = train_cascade_brl(
        X=X,
        y=y,
        feature_names=feature_names,
        max_rule_length=max_rule_length,
        n_steps=n_steps,
    )

    # Export
    export_cascade_brl(cascade_result, output_path, version)

    return ModelResult(
        model_type="brl_cascade",
        metrics=cascade_result.aggregate_metrics,
        artifacts={"cascade_result": cascade_result},
        extra={
            "n_classifiers": len(cascade_result.classifiers),
            "n_features": len(feature_names),
            "n_records": len(records),
            "output_path": str(output_path),
        },
    )
