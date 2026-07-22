"""Modeling utilities and trainers.

Modules:
  data.py          - Loading feature JSONL -> DataFrame / matrix
  trainers/        - Individual model trainers (decision tree, sparse linear, ebm, brl, bayes_net)
  explain.py       - LIME/SHAP explanation helpers

Design:
Each trainer exposes a `train_<model>(X, y, **config)` returning a ModelResult dict.
CLI wrappers (added later) call loader -> trainer -> write artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ModelResult:
    model_type: str
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    extra: Dict[str, Any]
