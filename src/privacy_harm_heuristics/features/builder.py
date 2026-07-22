"""Feature construction utilities.

Derives lightweight, deterministic features from normalized records:
 - Individuals affected (log)
 - Source categorical
 - Description presence flag
 - Keyword taxonomy flags (kw_<group>) loaded lazily from ``keywords.json``

Design goals: no external ML dependencies, stable across runs, streaming
friendly (stateless transformation per record).
"""

from __future__ import annotations

import json
from functools import lru_cache
from math import log1p
from pathlib import Path
from typing import Any, Dict

from ..nlp.product_features import extract_product_features
from .feature_config import (
    RISK_INCREASING_FEATURES,
    RISK_MITIGATING_FEATURES,
    TOP_CO_OCCURRENCE_PAIRS,
)


@lru_cache(maxsize=1)
def _load_keyword_cache() -> Dict[str, Any]:
    """Load keyword taxonomy from keywords.json file (cached).

    Returns:
        Dict mapping keyword groups to word lists, or empty dict if file not found.
    """
    try:
        with open(Path(__file__).with_name("keywords.json"), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):  # pragma: no cover
        return {}


def build_features(record: Dict[str, Any]) -> Dict[str, Any]:
    """Derive engineered features from a normalized record.

    Args:
        record: Normalized record dict (see schema for standard fields).

    Returns:
        Dict mapping feature name to value. All features are JSON serializable.
    """
    src = record.get("source")
    individuals = record.get("individuals_affected") or record.get("num_users")
    try:
        individuals = int(individuals) if individuals is not None else None
    except (ValueError, TypeError):
        individuals = None
    feats = {
        "f_source": src,
        "f_individuals_log": (log1p(individuals) if individuals and individuals > 0 else 0.0),
        "f_has_description": 1 if record.get("description") else 0,
    }
    # Keyword taxonomy flags (using cached loader)
    kw_cache = _load_keyword_cache()
    desc = (record.get("description") or "").lower()
    text_blob = " ".join(
        [
            desc,
            str(record.get("raw", {})).lower(),
        ]
    )
    for group, words in kw_cache.items():
        feats[f"kw_{group}"] = 1 if any(w.lower() in text_blob for w in words) else 0

    # Product feature indicators (root cause candidates). We use description + serialized raw.
    detected: list[str] = []
    try:
        pf = extract_product_features([desc, text_blob])
        feats.update(pf)
        detected = pf.get("product_features", []) or []
    except (ValueError, TypeError, AttributeError):  # pragma: no cover - defensive
        detected = []

    # Aggregate risk semantics
    risk_feats = [f for f in detected if f in RISK_INCREASING_FEATURES]
    protective_feats = [f for f in detected if f in RISK_MITIGATING_FEATURES]
    feats["rc_risky_feature_count"] = len(risk_feats)
    feats["rc_protective_feature_count"] = len(protective_feats)
    feats["rc_total_features"] = len(detected)
    feats["rc_risk_ratio"] = (len(risk_feats) / max(1, len(detected))) if detected else 0.0
    feats["rc_net_risk"] = len(risk_feats) - len(protective_feats)

    # Co-occurrence binary indicators for curated high-signal pairs
    for a, b in TOP_CO_OCCURRENCE_PAIRS:
        feats[f"rc_pair_{a}__{b}"] = 1 if a in detected and b in detected else 0

    return feats
