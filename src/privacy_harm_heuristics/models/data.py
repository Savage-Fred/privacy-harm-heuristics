"""Data loading & preprocessing utilities for modeling.

Responsibilities:
 - Read feature-enriched JSONL (each line a dict) into pandas DataFrame.
 - Select target column (binary) and feature columns (auto or user-specified).
 - Basic cleaning: drop rows with missing target, optionally impute numerical NaNs with median.
 - Provide train/test split helper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_FEATURE_PREFIXES = ("kw_",)
NON_FEATURE_COLUMNS = {
    "source",
    "type",
    "id",
    "raw",
    "created_date",
    "incident_date",
    "incident_date_canonical",
}

OUTCOME_COLUMNS = {
    "penalty_amount",
    "f_penalty_log",
    "f_penalty_bucket",
    "f_has_penalty",
}


@dataclass
class Dataset:
    """Container for train/test splits and metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    target: str


def load_jsonl(path: str | Path) -> pd.DataFrame:
    """Load a local JSONL file into a DataFrame (ignoring malformed lines).

    The old repo also read ``gs://`` URIs via google-cloud-storage; that cloud
    path was dropped in the practicum extraction (offline-only, no GCP dep).
    """
    local_path = str(path)
    records = []
    with open(local_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(obj)
    return pd.DataFrame(records)


def infer_feature_columns(
    df: pd.DataFrame, include_prefixes: Sequence[str] = DEFAULT_FEATURE_PREFIXES
) -> List[str]:
    """Infer candidate feature columns using prefixes & simple heuristics."""
    cols: List[str] = []
    for c in df.columns:
        if c in NON_FEATURE_COLUMNS or c in OUTCOME_COLUMNS:
            continue
        if any(c.startswith(p) for p in include_prefixes):
            cols.append(c)
        # Add basic numeric engineered fields
        elif pd.api.types.is_numeric_dtype(df[c]):
            # Skip obvious identifiers and numeric outcomes we explicitly drop
            if c in OUTCOME_COLUMNS:
                continue
            if c.endswith("_log") or c.endswith("_bucket"):
                cols.append(c)
    return sorted(set(cols))


def infer_root_cause_features(
    df: pd.DataFrame, include_prefixes: Sequence[str] = ("pf_", "rc_", "kw_")
) -> List[str]:
    """Infer feature columns for ROOT CAUSE analysis - excludes outcome variables.

    This is critical: we want to predict harm based on CAUSES (product features,
    design choices) NOT on outcomes (penalty amounts, breach size).

    For root cause modeling, we should ONLY include:
    - Product feature indicators (pf_*)
    - Root cause semantic features (rc_*)
    - Keyword taxonomy flags (kw_*)
    - Basic description presence flags

    We explicitly EXCLUDE:
    - penalty_amount, f_penalty_log, f_penalty_bucket (outcomes, not causes)
    - individuals_affected, f_individuals_log (outcomes, not causes)
    - harm_score, harm_severity (outcomes, not causes)
    """
    # Outcome variables to explicitly exclude
    OUTCOME_VARIABLES = {
        "individuals_affected",
        "f_individuals_log",
        "num_users",
        "harm_score",
        "harm_severity",
        "harm_category_scores",
    } | OUTCOME_COLUMNS

    cols: List[str] = []
    for c in df.columns:
        # Skip non-feature columns
        if c in NON_FEATURE_COLUMNS:
            continue
        # Skip outcome variables
        if c in OUTCOME_VARIABLES:
            continue
        # Include if matches root cause prefixes
        if any(c.startswith(p) for p in include_prefixes):
            cols.append(c)
        # Include basic binary indicators that are causal
        elif c in ("f_has_description",):
            cols.append(c)

    return sorted(set(cols))


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str] | None = None,
    dropna_target: bool = True,
    impute_numeric: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Return (X, y, feature_names) prepared for modeling.

    Applies target NA filtering, median imputation for numeric columns, simple
    fill for categoricals and one-hot encoding for non-numeric features.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in DataFrame")
    work = df.copy()
    if dropna_target:
        work = work[work[target].notna()]
    y = work[target]
    if features is None:
        features = infer_feature_columns(work)
    X = work[list(features)].copy()
    # Simple numeric imputation (median) with robustness for all-NaN columns
    if impute_numeric:
        for c in X.columns:
            col = X[c]
            if pd.api.types.is_numeric_dtype(col):
                # Replace inf/-inf with NaN first
                if col.replace([float("inf"), float("-inf")], pd.NA).isna().any():
                    col = col.replace([float("inf"), float("-inf")], pd.NA)
                if col.isna().all():
                    # All missing: default to 0 (neutral) to keep feature space aligned
                    X[c] = 0
                elif col.isna().any():
                    X[c] = col.fillna(col.median())
            else:
                # Fill non-numeric with placeholder token
                if col.isna().any():
                    X[c] = col.fillna("<NA>")
    # Final safeguard: if any residual NaNs remain (e.g., object conversion issues),
    # fill numeric with 0
    if X.isna().any().any():
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        for c in numeric_cols:
            if X[c].isna().any():
                X[c] = X[c].fillna(0)
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]) and X[c].isna().any():
                X[c] = X[c].fillna("<NA>")
    # One-hot encode non-numeric
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    return X, y, list(X.columns)


def build_dataset(
    path: str | Path,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    features: Sequence[str] | None = None,
    golden_path: str | Path | None = None,
    golden_verified_only: bool = False,
) -> Dataset:
    """Load JSONL features and create a ``Dataset`` with train/test split.

    If ``golden_path`` is provided:
    1. Loads golden records.
    2. Removes any records with matching IDs from the main dataset (to prevent leakage).
    3. Uses the golden records as the TEST set (ignoring ``test_size`` for the split,
       though we still split the main data for validation if needed, but here we return
       Main -> Train, Golden -> Test).

    ``golden_verified_only`` restricts the golden TEST set to human-verified
    cases (``reviewed`` truthy and ``needs_review`` falsy) so metrics are
    computed only against ground truth a reviewer has confirmed.
    """
    df = load_jsonl(path)

    golden_df = None
    if golden_path:
        golden_df = load_jsonl(golden_path)
        if golden_verified_only and len(golden_df):
            reviewed = (
                golden_df["reviewed"].fillna(False).astype(bool)
                if "reviewed" in golden_df.columns
                else pd.Series(False, index=golden_df.index)
            )
            flagged = (
                golden_df["needs_review"].fillna(False).astype(bool)
                if "needs_review" in golden_df.columns
                else pd.Series(False, index=golden_df.index)
            )
            verified_mask = reviewed & ~flagged
            print(
                f"[build_dataset] Golden verified-only: keeping {int(verified_mask.sum())}"
                f"/{len(golden_df)} human-verified cases."
            )
            golden_df = golden_df[verified_mask]
        # Ensure ID column exists for deduplication
        if "id" in golden_df.columns and "id" in df.columns:
            golden_ids = set(golden_df["id"].astype(str))
            initial_len = len(df)
            df = df[~df["id"].astype(str).isin(golden_ids)]
            print(
                f"[build_dataset] Removed {initial_len - len(df)} records from training data that matched golden test set IDs."
            )

        # Align golden_df columns to df (fill missing features with 0)
        # We do this AFTER prepare_xy usually, but prepare_xy needs a single DF to infer features?
        # Actually, prepare_xy takes a DF and returns X, y.
        # We should probably concat them to ensure consistent dummy encoding, then split back.
        df["_is_golden"] = False
        golden_df["_is_golden"] = True

        # Ensure target exists in golden (if not, we can't evaluate)
        if target not in golden_df.columns:
            # Try to map from 'actual_harm' or similar if needed, but for now assume it's prepared
            pass

        combined = pd.concat([df, golden_df], ignore_index=True)
        X_all, y_all, feat_cols = prepare_xy(combined, target=target, features=features)

        # Split back
        # We need to recover which rows were golden.
        # prepare_xy returns a new X dataframe, index might be reset or preserved?
        # prepare_xy does `work = df.copy()`, then `work = work[work[target].notna()]`.
        # So indices might change.
        # But `combined` has `_is_golden`. `prepare_xy` preserves columns in `work` before selecting `X`.
        # Wait, `prepare_xy` returns `X` which only has feature columns. `_is_golden` is lost.

        # Let's modify prepare_xy to return the full processed dataframe or handle this better.
        # Or just rely on the index if prepare_xy preserves it?
        # prepare_xy does `work = work[work[target].notna()]`.

        # Let's do this:
        # 1. Filter NAs from combined first.
        combined = combined[combined[target].notna()]

        # 2. Call prepare_xy on the filtered combined.
        X_all, y_all, feat_cols = prepare_xy(
            combined, target=target, features=features, dropna_target=False
        )

        # 3. Use the `_is_golden` column from `combined` (which aligns with X_all) to split.
        is_golden = combined["_is_golden"].astype(bool)

        X_train = X_all[~is_golden]
        y_train = y_all[~is_golden]
        X_test = X_all[is_golden]
        y_test = y_all[is_golden]

        print(
            f"[build_dataset] Using Golden Set for Testing: Train={len(X_train)}, Test={len(X_test)}"
        )

        return Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feat_cols,
            target=target,
        )

    X, y, feat_cols = prepare_xy(df, target=target, features=features)
    # Determine whether to stratify. Current heuristic previously stratified when
    # number of unique classes <= 10. However, scikit-learn's stratified split
    # requires at *least* 2 samples per class. Our label distribution can include
    # extremely rare classes (count == 1) generated by heuristic labeling. In that
    # case we fall back to a plain random split to avoid ValueError:
    # "The least populated class in y has only 1 member ...".
    stratify_arg = None
    if y.nunique() <= 10:
        counts = y.value_counts()
        if (counts >= 2).all():
            stratify_arg = y
        else:
            # Optional lightweight notice for transparency (avoids logging deps)
            rare = counts[counts < 2].index.tolist()
            print(
                (
                    "[build_dataset] Disabling stratified split due to rare class(es) with <2 "
                    f"samples: {rare}. "
                )
                + "Proceeding with unstratified train/test split."
            )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )
    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feat_cols,
        target=target,
    )
