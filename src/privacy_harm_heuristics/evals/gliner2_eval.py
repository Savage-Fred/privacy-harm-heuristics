"""Evaluation helpers for GLiNER2-based labeling.

This module provides utilities to:

- Run GLiNER2 over a golden test set
- Compare predicted harm labels against expected labels
- Emit JSONL logs that are easy to analyze with DuckDB / pandas
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..nlp.gliner2_labeling import GLiNER2Config, label_texts
from ..labeling.harm_labeler import label_harm_category
from ..labeling.golden import is_verified, load_golden_cases

HARM_PREFIX = "harm_"


@dataclass
class GLiNER2EvalConfig:
    golden_path: Path
    out_path: Path
    text_field: str = "description"
    label_field: str = "harms"
    batch_size: int = 16
    model_name: str = "urchade/gliner2-medium"
    verified_only: bool = False  # restrict ground truth to human-verified cases


def normalize_harm_label(label: Optional[str]) -> Optional[str]:
    """Map a prediction into the golden Solove namespace (strip ``harm_`` prefix)."""
    if isinstance(label, str) and label.startswith(HARM_PREFIX):
        return label[len(HARM_PREFIX) :]
    return label


def canonical_prediction(pred: Optional[str]) -> Optional[str]:
    """Collapse abstention spellings (``None``, ``"unknown"``) to ``None``."""
    return None if pred is None or pred == "unknown" else pred


def expected_harms(case: Dict[str, Any], label_field: str = "harms") -> List[str]:
    """Ground-truth harm labels for a golden case, as a list.

    The human-verified Solove labels live in the list-valued ``harms`` field.
    ``expected_label`` holds an incident-type label (e.g. ``data_breach``) from
    a different taxonomy and is deliberately NOT a fallback — it is consulted
    only when passed explicitly as ``label_field``. A present list is
    authoritative even when empty (a verified "no harms" must not be overridden
    by legacy scalar fields); scalar values become single-element lists. Values
    are normalized like predictions (``harm_`` prefix stripped).
    """
    for field_name in (label_field, "harms", "actual_harm", "harm_category"):
        value = case.get(field_name)
        if isinstance(value, list):
            # Ignore nulls and empty/whitespace-only entries the review UI may
            # carry through verbatim, so they never score as literal labels;
            # drop values that become empty after normalization (e.g. "harm_").
            normalized = [
                normalize_harm_label(v.strip()) for v in value if isinstance(v, str) and v.strip()
            ]
            return [str(lab) for lab in normalized if lab]
        if isinstance(value, str) and value.strip():
            lab = normalize_harm_label(value.strip())
            return [str(lab)] if lab else []
    return []


def harm_prediction_correct(pred: Optional[str], exp: List[str]) -> bool:
    """A top-1 prediction is correct if it names any of the verified harms.

    Abstentions — ``None`` (GLiNER2 emitted no ``harm_`` label) or the
    heuristic labeler's ``"unknown"`` sentinel — are correct exactly when the
    verified harm set is empty (a confirmed "no harms" case).
    """
    canonical = canonical_prediction(pred)
    if canonical is None:
        return not exp
    return canonical in exp


def evaluate_gliner2_on_golden(cfg: GLiNER2EvalConfig) -> Dict[str, Any]:
    """Evaluate GLiNER2 harm predictions on a golden dataset.

    Writes a JSONL file where each line contains:
      - case_id (if present)
      - expected_harms (verified ground-truth harm list)
      - pred_gliner2
      - correct (bool)
      - verified (bool)
      - text_labels (full GLiNER2 label list)
      - text_label_scores (per-label scores)

    Returns a summary dict with accuracy, counts, and review-coverage stats,
    suitable for logging or downstream plotting.
    """

    cases, stats = load_golden_cases(cfg.golden_path, verified_only=cfg.verified_only)

    if not cases:
        return {"n": 0, "accuracy": None, "review": stats.as_dict()}

    texts: List[str] = []
    expected: List[List[str]] = []
    ids: List[Optional[str]] = []
    verified_flags: List[bool] = []

    for case in cases:
        text = str(case.get(cfg.text_field, "") or "").strip()
        texts.append(text)
        expected.append(expected_harms(case, cfg.label_field))
        ids.append(case.get("id") or case.get("case_id"))
        verified_flags.append(is_verified(case))

    # Configure GLiNER2
    gl_cfg = GLiNER2Config(
        model_name=cfg.model_name,
        text_field=cfg.text_field,
        batch_size=cfg.batch_size,
    )

    results = label_texts(texts, config=gl_cfg)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    correct = 0

    with cfg.out_path.open("w", encoding="utf-8") as fout:
        for case_id, exp, text, gl, verified in zip(ids, expected, texts, results, verified_flags):
            harm_labels = [
                lab
                for lab in gl.get("text_labels", [])
                if isinstance(lab, str) and lab.startswith(HARM_PREFIX)
            ]
            pred = normalize_harm_label(harm_labels[0]) if harm_labels else None
            is_correct = harm_prediction_correct(pred, exp)
            n += 1
            if is_correct:
                correct += 1

            row = {
                "case_id": case_id,
                "expected_harms": exp,
                "pred_gliner2": pred,
                "correct": is_correct,
                "verified": verified,
                "text_labels": gl.get("text_labels", []),
                "text_label_scores": gl.get("text_label_scores", {}),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    accuracy = (correct / n) if n else None
    return {
        "n": n,
        "correct": correct,
        "accuracy": accuracy,
        "verified_only": cfg.verified_only,
        "review": stats.as_dict(),
        "log_path": str(cfg.out_path),
    }


def compare_gliner2_vs_heuristic_on_golden(cfg: GLiNER2EvalConfig) -> Dict[str, Any]:
    """Compare GLiNER2 vs heuristic predictions on a golden dataset.

    Writes a JSONL with fields:
      - case_id
      - expected_harms (verified ground-truth harm list)
      - pred_gliner2
      - pred_heuristic
      - correct_gliner2
      - correct_heuristic
      - agree
      - verified
      - text_labels
      - text_label_scores
    """

    cases, stats = load_golden_cases(cfg.golden_path, verified_only=cfg.verified_only)

    if not cases:
        return {
            "n": 0,
            "accuracy_gliner2": None,
            "accuracy_heuristic": None,
            "review": stats.as_dict(),
        }

    texts: List[str] = []
    expected: List[List[str]] = []
    ids: List[Optional[str]] = []
    verified_flags: List[bool] = []
    for case in cases:
        text = str(case.get(cfg.text_field, "") or "").strip()
        texts.append(text)
        expected.append(expected_harms(case, cfg.label_field))
        ids.append(case.get("id") or case.get("case_id"))
        verified_flags.append(is_verified(case))

    gl_cfg = GLiNER2Config(
        model_name=cfg.model_name,
        text_field=cfg.text_field,
        batch_size=cfg.batch_size,
    )
    gl_results = label_texts(texts, config=gl_cfg)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    gl_correct = 0
    heu_correct = 0
    agree_count = 0
    gl_only_correct = 0
    heu_only_correct = 0

    with cfg.out_path.open("w", encoding="utf-8") as fout:
        for case_id, exp, text, gl, verified in zip(
            ids, expected, texts, gl_results, verified_flags
        ):
            harm_labels = [
                lab
                for lab in gl.get("text_labels", [])
                if isinstance(lab, str) and lab.startswith(HARM_PREFIX)
            ]
            pred_gl = normalize_harm_label(harm_labels[0]) if harm_labels else None

            # Heuristic prediction using same text field
            rec = {cfg.text_field: text}
            pred_heu = normalize_harm_label(label_harm_category(rec, text_fields=[cfg.text_field]))

            is_gl_correct = harm_prediction_correct(pred_gl, exp)
            is_heu_correct = harm_prediction_correct(pred_heu, exp)
            # Abstentions agree with each other regardless of spelling
            agree = canonical_prediction(pred_gl) == canonical_prediction(pred_heu)

            n += 1
            if is_gl_correct:
                gl_correct += 1
            if is_heu_correct:
                heu_correct += 1
            if agree:
                agree_count += 1
            if is_gl_correct and not is_heu_correct:
                gl_only_correct += 1
            if is_heu_correct and not is_gl_correct:
                heu_only_correct += 1

            row = {
                "case_id": case_id,
                "expected_harms": exp,
                "pred_gliner2": pred_gl,
                "pred_heuristic": pred_heu,
                "correct_gliner2": is_gl_correct,
                "correct_heuristic": is_heu_correct,
                "agree": agree,
                "verified": verified,
                "text_labels": gl.get("text_labels", []),
                "text_label_scores": gl.get("text_label_scores", {}),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    acc_gl = (gl_correct / n) if n else None
    acc_heu = (heu_correct / n) if n else None

    return {
        "n": n,
        "accuracy_gliner2": acc_gl,
        "accuracy_heuristic": acc_heu,
        "agree": agree_count,
        "gl_only_correct": gl_only_correct,
        "heu_only_correct": heu_only_correct,
        "verified_only": cfg.verified_only,
        "review": stats.as_dict(),
        "log_path": str(cfg.out_path),
    }
