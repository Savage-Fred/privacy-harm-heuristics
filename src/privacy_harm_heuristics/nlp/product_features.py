"""Product feature extraction from unstructured text.

Loads ontology from `product_feature_ontology.json` and applies simple
synonym + regex pattern matching to produce:
  - product_features: list[str] of detected feature slugs
  - pf_<slug>: 0/1 binary indicator for easy modeling

Design goals:
  * Deterministic and lightweight (no heavy ML dependency required)
  * Single-pass over concatenated text fields
  * Resilient to missing/empty text

Future extensions: add phrase embedding similarity or entity extraction.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

ONTOLOGY_FILE = Path(__file__).with_name("product_feature_ontology.json")


@lru_cache(maxsize=1)
def _load_ontology() -> Dict[str, Dict[str, Any]]:
    try:
        with open(ONTOLOGY_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


@lru_cache(maxsize=128)
def _compiled_patterns(slug: str) -> Tuple[re.Pattern, ...]:
    ont = _load_ontology()
    entry = ont.get(slug, {})
    regexes = entry.get("regex", []) if isinstance(entry, dict) else []
    compiled = []
    for pat in regexes:
        try:
            compiled.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            continue
    return tuple(compiled)


def extract_product_features(text_iter: Iterable[str]) -> Dict[str, Any]:
    """Extract product feature indicators from iterable of text fragments.

    Args:
        text_iter: Iterable of strings (e.g., description, raw text fields).

    Returns:
        Dict with keys:
            product_features: list[str]
            pf_<slug>: int
    """
    ont = _load_ontology()
    combined = " ".join(t for t in text_iter if t).lower()
    found: list[str] = []
    for slug, entry in ont.items():
        syns = entry.get("synonyms", []) if isinstance(entry, dict) else []
        hit = False
        # Synonym substring match
        for s in syns:
            if s.lower() in combined:
                hit = True
                break
        # Regex patterns
        if not hit:
            for pat in _compiled_patterns(slug):
                if pat.search(combined):
                    hit = True
                    break
        if hit:
            found.append(slug)
    # Build binary indicators
    result: Dict[str, Any] = {f"pf_{slug}": (1 if slug in found else 0) for slug in ont.keys()}
    result["product_features"] = found
    return result


def extract_causal_phrases(text: str) -> Dict[str, list[str]]:
    """Extract causal phrases that indicate root causes from text.

    Looks for patterns like:
    - "because of X"
    - "due to X"
    - "caused by X"
    - "resulted from X"
    - "X led to"

    Args:
        text: Input text to analyze

    Returns:
        Dict with keys:
            - causal_indicators: list of causal phrases found
            - potential_causes: list of extracted cause descriptions
    """
    if not text:
        return {"causal_indicators": [], "potential_causes": []}

    text_lower = text.lower()
    causal_patterns = [
        (r"because of ([^.,;]+)", "because_of"),
        (r"due to ([^.,;]+)", "due_to"),
        (r"caused by ([^.,;]+)", "caused_by"),
        (r"resulted? from ([^.,;]+)", "resulted_from"),
        (r"([^.,;]+) led to", "led_to"),
        (r"([^.,;]+) resulted in", "resulted_in"),
        (r"as a result of ([^.,;]+)", "as_result_of"),
        (r"triggered by ([^.,;]+)", "triggered_by"),
        (r"following ([^.,;]+)", "following"),
    ]

    indicators = []
    causes = []

    for pattern, indicator_type in causal_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            cause_text = match.group(1).strip()
            if 5 < len(cause_text) < 200:  # reasonable length
                indicators.append(indicator_type)
                causes.append(cause_text)

    return {
        "causal_indicators": list(set(indicators)),
        "potential_causes": causes[:10],  # limit to top 10
    }


def annotate_record_with_features(
    record: Dict[str, Any], text_fields: Tuple[str, ...] = ("description",)
) -> bool:
    """Annotate a record dict in-place with product feature indicators.

    Args:
        record: Mutable mapping representing one normalized record.
        text_fields: Candidate text field names to concatenate.

    Returns:
        True if any feature detected, False otherwise.
    """
    fragments: list[str] = []
    for f in text_fields:
        val = record.get(f)
        if isinstance(val, str) and val.strip():
            fragments.append(val)
    # Optionally include parts of raw
    raw = record.get("raw") or {}
    if isinstance(raw, dict):
        for k in ("text", "body", "content"):
            rv = raw.get(k)
            if isinstance(rv, str) and rv.strip():
                fragments.append(rv)
    if not fragments:
        return False
    feats = extract_product_features(fragments)
    # Geo + spaCy NER enrichment (nlp/geo_entities.py, nlp/spacy_ner.py) were not
    # extracted from the old repo (peripheral to this practicum slice — see
    # STOP RULE note in extraction report). Degrade gracefully: this function
    # still returns product-feature + causal-phrase annotations, just without
    # the 50+ geo/entity binary flags the old repo added here.
    combined_feats = dict(feats)
    record.update(combined_feats)

    # NEW: Also populate root_cause_features list for the schema
    # This is a critical field for predictive modeling of what CAUSES harm
    detected_features = combined_feats.get("product_features", [])
    if detected_features:
        # Store as root_cause_features for causal analysis
        record["root_cause_features"] = detected_features

    # NEW: Extract causal phrases to understand what caused the incident
    combined_text = " ".join(fragments)
    causal_info = extract_causal_phrases(combined_text)
    if causal_info.get("potential_causes"):
        # Store causal context in record for analysis
        record["causal_indicators"] = causal_info["causal_indicators"]
        record["potential_causes"] = causal_info["potential_causes"]

    return bool(feats.get("product_features"))
