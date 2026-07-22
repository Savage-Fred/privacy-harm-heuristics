"""Bayesian Network trainer (structure learning + simple classification proxy).

Uses pomegranate.BayesianNetwork to learn a probabilistic model over features + target.
For evaluation we perform *predictive imputation* of the target given features by
setting the target position to None and calling network.predict(...). This yields
point predictions we can score using accuracy/F1, etc., suitable for basic comparative
evaluation against discriminative models.

Assumptions / Simplifications:
    - Features are treated as already discrete / categorical or binary. Continuous
    values will be left as-is; pomegranate can bucket numerics but we skip here.
    - Target must be binary or multiclass encoded as int/str categories.
    - Structure learning algorithm default: 'chow-liu' (tree) for speed + determinism.

Future enhancements:
    - Optional discretization / binning for continuous values.
    - Support alternative structure algorithms (exact, greedy, etc.).
    - Per-edge importance metrics (mutual information, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ..metrics_utils import choose_average

from .. import ModelResult

try:  # Optional dependency (pomegranate changed public path in 1.1.x)
    try:
        from pomegranate import BayesianNetwork  # type: ignore
    except Exception:  # pragma: no cover
        from pomegranate.bayesian_network import BayesianNetwork  # type: ignore
except Exception:  # pragma: no cover
    BayesianNetwork = None  # type: ignore


def _to_data_matrix(df: pd.DataFrame, target_col: str, feature_names: List[str]):
    """Return numpy matrix with features followed by target column.

    Keeps a parallel ordered column list for reconstructing edge names later.
    """
    ordered_cols = feature_names + [target_col]
    return df[ordered_cols].values, ordered_cols


def train_bayes_net(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names: List[str],
    algorithm: str = "chow-liu",
    random_state: int = 42,
    n_bins: int = 5,
    min_unique_to_bin: int = 6,
) -> ModelResult:
    """Train a Bayesian Network using pomegranate and return a ModelResult.

    Discretization:
        pomegranate's discrete BayesianNetwork implementation expects *integer*
        tensors for structure learning. Our pipeline may supply continuous
        numeric features (floats). We therefore perform unsupervised
        quantile-based binning of sufficiently non-constant numeric columns to
        convert them into ordinal integer categories. Columns with fewer than
        ``min_unique_to_bin`` unique values are treated as already discrete and
        simply cast to int (after filling NaNs). Bin edges are stored in the
        returned ``extra`` metadata for transparency / reproducibility.
    """
    if BayesianNetwork is None:  # pragma: no cover - optional dep guard
        raise ImportError("pomegranate package not available; cannot train Bayesian Network")
    import numpy as np

    # Assemble training matrix (features + target last)
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["__target__"] = y_train.values

    # Ensure all columns are numeric / acceptable dtypes for pomegranate tensor conversion.
    # - Label encode non-numeric columns (including target) -> ints
    # - Quantile-bin continuous numeric columns into ordinal ints
    from sklearn.preprocessing import LabelEncoder

    encoders: Dict[str, LabelEncoder] = {}
    bin_edges: Dict[str, List[float]] = {}

    # First, label encode any non-numeric (object/string) columns.
    for col in list(train_df.columns):
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str).fillna("<NA>"))
            encoders[col] = le

    # Now process numeric feature columns (exclude target for discretization decision)
    for col in feature_names:
        series = train_df[col]
        # Replace inf / -inf then fill NaNs
        if not np.isfinite(series).all():
            series = series.replace([np.inf, -np.inf], np.nan)
        if series.isna().any():
            series = series.fillna(series.median() if not series.dropna().empty else 0)
        # Decide whether to bin
        unique_count = series.nunique(dropna=True)
        if unique_count >= min_unique_to_bin:
            # Quantile edges (ensure uniqueness & coverage)
            quantiles = np.linspace(0, 1, n_bins + 1)
            arr = series.to_numpy(dtype=float, copy=False)
            raw_edges = np.quantile(arr, quantiles)
            # Deduplicate strictly monotonic edges
            edge_vals = [float(raw_edges[0])]
            for v in raw_edges[1:]:
                if float(v) > edge_vals[-1]:  # keep strictly increasing
                    edge_vals.append(float(v))
            # If after dedup we have <3 edges (constant-ish), skip binning
            if len(edge_vals) >= 3:
                # np.digitize returns indices 1..len(edges)-1; subtract 1 => 0-based
                binned = np.digitize(arr, edge_vals[1:-1], right=False)
                train_df[col] = binned.astype(int)
                bin_edges[col] = edge_vals
            else:
                train_df[col] = series.round().astype(int)
        else:
            # Treat as already discrete (cast to int for safety)
            train_df[col] = series.round().astype(int)

    # Target: ensure integer encoding
    target_col = "__target__"
    if target_col not in encoders:
        # Even if already numeric, label encode to guarantee category mapping metadata
        le = LabelEncoder()
        train_df[target_col] = le.fit_transform(train_df[target_col].astype(str))
        encoders[target_col] = le

    data_matrix, cols = _to_data_matrix(train_df, target_col, feature_names)
    # Instantiate BN with algorithm (newer API uses attribute .algorithm on instance)
    model = BayesianNetwork(algorithm=algorithm)
    model.fit(data_matrix)

    # Predict on test by masking target and calling predict (sequence of rows)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df[target_col] = y_test.values

    # Apply same preprocessing to test: label encoders first (non-numeric already encoded in train)
    for col, le in encoders.items():
        if col in test_df.columns:
            # Any unseen labels -> transform on string form; if error, map to most frequent class
            try:
                test_df[col] = le.transform(test_df[col].astype(str).fillna("<NA>"))
            except Exception:  # pragma: no cover
                most = le.classes_[0]
                test_df[col] = le.transform([most] * len(test_df[col]))  # fallback uniform label

    # Discretize numeric columns with stored bin edges
    for col, edges_list in bin_edges.items():
        if col in test_df.columns:
            series = test_df[col]
            # Fill NaNs with median then digitize
            if series.isna().any():
                series = series.fillna(series.median() if not series.dropna().empty else 0)
            arr = series.to_numpy(dtype=float, copy=False)
            binned = np.digitize(arr, edges_list[1:-1], right=False)
            test_df[col] = binned.astype(int)
    # Cast any remaining float columns to int if they look discrete
    for col in feature_names:
        if pd.api.types.is_float_dtype(test_df[col]) and set(test_df[col].unique()) <= {
            0.0,
            1.0,
        }:
            test_df[col] = test_df[col].astype(int)
        if test_df[col].isna().any():
            test_df[col] = test_df[col].fillna(0)

    test_matrix, _ = _to_data_matrix(test_df, target_col, feature_names)
    preds = []
    for row in test_matrix:
        row_copy = list(row)
        row_copy[-1] = None
        try:
            filled = model.predict([row_copy])[0]
            preds.append(filled[-1])
        except Exception:  # pragma: no cover
            preds.append(y_train.mode().iloc[0])
    # Inverse transform target predictions if they are encoded ints; if already
    # strings (can occur if library surfaces original labels), skip.
    if target_col in encoders and all(isinstance(p, (int, np.integer)) for p in preds):
        inv = encoders[target_col].inverse_transform([int(p) for p in preds])
        preds_arr = np.array(inv)
    else:
        preds_arr = np.array(preds)
    # Coerce prediction dtype to match y_test dtypes to avoid sklearn type errors
    try:
        if y_test.dtype.kind in {"i", "u"}:
            # y_test is integer encoded; attempt to map preds to ints where possible
            preds_arr_cast = []
            for p in preds_arr:
                try:
                    preds_arr_cast.append(int(p))
                except Exception:
                    # Fallback: if string label present but y_test is ints, try inverse transform
                    preds_arr_cast.append(p)
            preds_arr = pd.Series(preds_arr_cast).to_numpy()
    except Exception:  # pragma: no cover
        pass

    avg, pos_label = choose_average(y_train, y_test)
    if avg == "binary":
        f1 = f1_score(y_test, preds_arr, zero_division=0, average="binary", pos_label=pos_label)
        prec = precision_score(
            y_test, preds_arr, zero_division=0, average="binary", pos_label=pos_label
        )
        rec = recall_score(
            y_test, preds_arr, zero_division=0, average="binary", pos_label=pos_label
        )
    else:
        f1 = f1_score(y_test, preds_arr, zero_division=0, average="macro")
        prec = precision_score(y_test, preds_arr, zero_division=0, average="macro")
        rec = recall_score(y_test, preds_arr, zero_division=0, average="macro")
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, preds_arr)),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
    }
    # Structure as list of edges (parent -> child)
    structure = getattr(model, "structure", []) or []
    edges: List[tuple[str, str]] = []
    for i, parents in enumerate(structure):
        for parent_idx in parents:
            edges.append((cols[parent_idx], cols[i]))
    extra = {
        "n_features": len(feature_names),
        "algorithm": algorithm,
        "edges": edges,
        "n_edges": len(edges),
        "bin_edges": bin_edges,
        "n_bins_requested": n_bins,
        # Provide predictions array for downstream evaluators that prefer
        # explicit outputs over invoking model.predict (improves consistency).
        "predictions": preds_arr,
    }
    return ModelResult(
        model_type="bayes_net",
        metrics=metrics,
        artifacts={"model": model},
        extra=extra,
    )
