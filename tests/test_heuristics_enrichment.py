"""Tests for heuristic extraction enrichment (hash, support, precision)."""

from pathlib import Path

import numpy as np
import pandas as pd

from privacy_harm_heuristics.models.data import Dataset
from privacy_harm_heuristics.models.heuristics import extract_heuristics
from privacy_harm_heuristics.models.trainers.decision_tree import train_decision_tree


def build_dummy_dataset(tmp_path: Path):
    """Construct small deterministic dataset with signal in f0."""
    # Construct dataset: feature f0 predictive of class to yield non-zero weights
    X = np.array(
        [
            [0, 0],  # 0
            [1, 0],  # 1
            [1, 1],  # 1
            [0, 1],  # 0
            [1, 0],  # 1
            [0, 1],  # 0
            [1, 1],  # 1
            [0, 0],  # 0
        ],
        dtype=float,
    )
    y = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    # Wrap into DataFrames / Series to satisfy Dataset typing
    X_df = pd.DataFrame(X, columns=["f0", "f1"])
    y_s = pd.Series(y, name="dummy_target")
    split = 6  # more train samples for sparse linear to pick signal
    ds = Dataset(
        X_train=X_df.iloc[:split].copy(),
        X_test=X_df.iloc[split:].copy(),
        y_train=y_s.iloc[:split].copy(),
        y_test=y_s.iloc[split:].copy(),
        feature_names=["f0", "f1"],
        target="dummy_target",
    )
    return ds


def test_decision_tree_heuristics_enriched(tmp_path):
    """Decision tree heuristics should include provenance and metrics."""
    ds = build_dummy_dataset(tmp_path)
    result = train_decision_tree(
        ds.X_train,
        ds.y_train,
        ds.X_test,
        ds.y_test,
        max_depth=2,
        min_samples_leaf=1,
        random_state=42,
    )
    items = extract_heuristics(
        result, ds.feature_names, X_train=ds.X_train, y_train=ds.y_train, top_n=None
    )
    assert items, "No heuristics extracted"
    for it in items:
        assert "extra" in it and "provenance_hash" in it["extra"]
        assert it["extra"].get("version") == 1
        # support/precision may be None for some types but decision_tree rules should have them
        if it["model_type"] == "decision_tree" and it["kind"] == "rule":
            assert it["support"] is not None
            assert it["precision"] is not None


def test_sparse_linear_pseudo_rules(tmp_path):
    """Sparse linear model should yield coefficient heuristics with metadata."""
    # Use decision tree dataset but train a sparse linear model
    from privacy_harm_heuristics.models.trainers.sparse_linear import train_sparse_linear

    ds = build_dummy_dataset(tmp_path)
    result = train_sparse_linear(
        ds.X_train,
        ds.y_train,
        ds.X_test,
        ds.y_test,
        C=0.5,
        max_iter=100,
        random_state=42,
    )
    items = extract_heuristics(result, ds.feature_names, X_train=ds.X_train, y_train=ds.y_train)
    coeff_items = [i for i in items if i["model_type"] == "sparse_linear"]
    assert coeff_items, "Expected coefficient heuristics"
    # At least one should have provenance and possibly support/precision calculated
    assert any(i.get("support") is not None for i in coeff_items), "Expected some support values"
    for i in coeff_items:
        assert "provenance_hash" in i["extra"]
