"""Evaluation harness to compare interpretable models and LLM/heuristic baselines.

This module provides utilities to:
 - Load a labeled JSONL dataset (local path)
 - Evaluate one or more saved interpretable models (artifact dirs)
 - Optionally evaluate a heuristic baseline and/or an LLM classifier baseline
 - Produce a JSON report (and optionally Markdown) with aggregate metrics

The harness intentionally depends only on existing project utilities and
scikit-learn; it makes no outbound calls unless the LLM baseline is enabled
and a provider is configured. When no LLM provider is configured, the LLM
baseline gracefully skips.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import classification_report as skl_classification_report

from ...labeling.harm_labeler import label_harm_category
from ...llm import provider as llm_provider
from ...utils.rate_limit import PaceLimiter
from ..data import load_jsonl
from ..serialize import load_model_artifacts

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    data_path: str
    target: str = "harm_category"
    # List of model artifact directories (each contains model.joblib, feature_names.json, ...)
    model_dirs: Optional[List[str]] = None
    include_heuristic: bool = True
    include_llm: bool = False
    # Max records to evaluate (None = all). Useful for quick smoke tests and cost control.
    limit: Optional[int] = None


def _coerce_label(val) -> str:
    if val is None:
        return "unknown"
    if isinstance(val, (int, float)):
        return str(val)
    return str(val)


def _extract_text(record: dict) -> str:
    parts: List[str] = []
    for k in ("description", "title", "summary", "body", "text"):
        v = record.get(k)
        if isinstance(v, str) and v:
            parts.append(v)
    raw = record.get("raw")
    if isinstance(raw, dict):
        for k in ("description", "title", "summary", "body", "text"):
            v = raw.get(k)
            if isinstance(v, str) and v:
                parts.append(v)
    return " ".join(parts)


def _predict_with_llm(records: List[dict], labels: List[str]) -> List[str]:
    """Attempt LLM-based classification by prompting for a label.

    Falls back to an empty list when LLM provider is not configured.
    """
    provider_name = llm_provider.select_provider(None)
    if provider_name == "fallback":
        # No online provider available; skip
        return []

    limiter = PaceLimiter(
        per_minute=float(
            # Conservative default unless overridden by env-specific limits
            int(
                float(
                    # Use a small default pace (20 rpm) to avoid rate spikes unless overridden
                    # by provider-specific env vars outside of this module.
                    20
                )
            )
        )
    )

    valid_labels = ", ".join(sorted(set(labels)))
    preds: List[str] = []
    for rec in records:
        text = _extract_text(rec)
        if not text:
            preds.append("unknown")
            continue
        prompt = (
            "Select the best privacy harm category for the following text. "
            f"Valid labels: {valid_labels}. Respond with ONLY the label.\n\n" + text
        )
        limiter.wait()
        try:
            resp = llm_provider.summarize(prompt, provider=None, model=None, max_tokens=32)
            pred = _coerce_label(resp).strip().lower().replace(" ", "_")
        except Exception:
            pred = "unknown"
        preds.append(pred)
    return preds


def _predict_with_model(
    model_dir: str, X: pd.DataFrame, feature_names_dataset: List[str]
) -> List[str] | None:
    art = load_model_artifacts(model_dir)
    model = art.get("model")
    if model is None:
        logger.warning(
            "Skipping model at %s because no estimator artifact was saved (possibly rule-only)",
            model_dir,
        )
        return None
    model_module = getattr(model, "__module__", "")
    if model_module.startswith("pomegranate"):
        logger.warning(
            "Skipping Bayesian Network at %s because preprocessing metadata required for inference "
            "is not saved in artifacts yet",
            model_dir,
        )
        return None
    feat_names_model: List[str] = list(art.get("feature_names", []))
    # Align columns: reindex to training features, fill missing with 0
    X_aligned = X.copy()
    # Add any missing columns present in the model's training features
    for c in feat_names_model:
        if c not in X_aligned.columns:
            X_aligned[c] = 0
    # Ensure same column order as training
    X_aligned = X_aligned[feat_names_model]
    try:
        y_pred = model.predict(X_aligned)
    except Exception:
        # Some sklearn models may return numeric labels; coerce to str
        y_pred = [str(x) for x in model.predict(X_aligned)]
    return [_coerce_label(v) for v in y_pred]


def evaluate(config: EvalConfig) -> Dict[str, dict]:
    """Run evaluations and return a dict keyed by runner name with metrics dicts.

    Returns example structure:
      {
        "heuristic": {"precision": {...}, "recall": {...}, "f1-score": {...}, ...},
        "decision_tree": {...},
        "llm": {...},
      }
    """
    # Load raw to preserve text for LLM baseline; build_dataset for feature matrix
    df = load_jsonl(config.data_path)
    if config.limit is not None and len(df) > config.limit:
        df = df.iloc[: config.limit].copy()

    if config.target not in df.columns:
        raise ValueError(f"Target column '{config.target}' not found in dataset")

    y_true = [_coerce_label(v) for v in df[config.target].tolist()]

    # Build feature matrix with the same inference as training without a split
    from ..data import prepare_xy

    X_all, y_full, feat_cols = prepare_xy(df, target=config.target)
    # Ensure y_true aligns with X_all rows (prepare_xy drops NA targets)
    y_true = [_coerce_label(v) for v in y_full.tolist()]

    # Filter df to match X_all (which has dropped NA targets)
    df_filtered = df.loc[X_all.index]

    report: Dict[str, dict] = {}

    # Heuristic baseline
    if config.include_heuristic:
        preds_h = [
            _coerce_label(label_harm_category(rec, text_fields=None, threshold=1.0))
            for rec in df_filtered.to_dict(orient="records")
        ]
        report["heuristic"] = skl_classification_report(
            y_true, preds_h, output_dict=True, zero_division=0
        )

    # Interpretable models
    if config.model_dirs:
        for mdir in config.model_dirs:
            name = Path(mdir).name or mdir
            preds = _predict_with_model(mdir, X_all, feat_cols)
            if preds is None:
                report[name] = {
                    "note": "Skipped: no estimator artifact (rule-based only).",
                    "status": "skipped",
                }
                continue
            report[name] = skl_classification_report(
                y_true, preds, output_dict=True, zero_division=0
            )

    # LLM baseline
    if config.include_llm:
        unique_labels = sorted(set(y_true))
        preds_llm = _predict_with_llm(df.to_dict(orient="records"), unique_labels)
        if preds_llm:
            report["llm"] = skl_classification_report(
                y_true, preds_llm, output_dict=True, zero_division=0
            )
        else:
            report["llm"] = {
                "note": "LLM provider not configured; skipped.",
            }

    return report


def save_report(report: Dict[str, dict], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "comparison_report.json"
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Lightweight Markdown summary
    md = ["# Model Comparison Report\n"]
    for name, metrics in report.items():
        md.append(f"\n## {name}\n")
        if "accuracy" in metrics:
            md.append(f"- accuracy: {metrics['accuracy']:.4f}")
        if "macro avg" in metrics and isinstance(metrics["macro avg"], dict):
            ma = metrics["macro avg"]
            md.append(
                "- macro avg: precision={:.4f}, recall={:.4f}, f1={:.4f}".format(
                    ma.get("precision", 0.0), ma.get("recall", 0.0), ma.get("f1-score", 0.0)
                )
            )
    (out / "comparison_report.md").write_text("\n".join(md), encoding="utf-8")
    return p
