"""Weak supervision labeling functions for non-monetary privacy harms.

This module implements evidence-backed labeling functions (LFs) that enrich
privacy incident records with structured harm indicators, following the
Solove taxonomy and supporting multi-dimensional risk modeling.

Design principles:
- No synthetic/invented records; only enrich existing data with derived labels
- Maintain provenance: track which LFs fired and what they matched
- Prefer precision over recall for labeling functions
- All patterns are evidence-based from privacy literature and real incidents
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..schema import DataSensitivity, HarmCategory, Record

# Distress indicators from user sentiment analysis literature
DISTRESS_PATTERNS = re.compile(
    r"\b(anxious|anxiety|afraid|fear|chilling|embarrass|ashamed|panic|"
    r"stalk|harassed|violated|uncomfortable|upset|angry|frustrated|"
    r"stopped\s+posting|stopped\s+using|deleted\s+account|concerned|worried)\b",
    re.IGNORECASE,
)

# Sensitive data types requiring special protection (GDPR Art. 9, HIPAA, etc.)
SENSITIVE_DATA_TERMS: Dict[DataSensitivity, List[str]] = {
    DataSensitivity.SPECIAL: [
        "biometric",
        "genetic",
        "geolocation",
        "precise location",
        "health",
        "medical",
        "fertility",
        "minor",
        "child",
        "racial",
        "ethnic",
        "religious",
        "sexual orientation",
        "political opinion",
        "trade union",
        "ssn",
        "social security",
    ],
    DataSensitivity.HIGH: [
        "financial",
        "credit card",
        "bank account",
        "password",
        "credential",
        "authentication",
        "location history",
        "browsing history",
        "email content",
        "message content",
        "private message",
    ],
    DataSensitivity.MODERATE: [
        "email address",
        "phone number",
        "ip address",
        "device id",
        "user id",
        "profile",
        "purchase history",
        "search query",
    ],
}

# Contextual mismatch patterns (Nissenbaum's Contextual Integrity)
CONTEXT_MISMATCH_PATTERNS = re.compile(
    r"\b(unexpected|without\s+consent|repurposed|secondary\s+use|"
    r"function\s+creep|scope\s+creep|changed\s+terms|sold\s+to|"
    r"shared\s+with\s+third|cross-context|purpose\s+limitation|"
    r"not\s+disclosed|hidden\s+in\s+terms)\b",
    re.IGNORECASE,
)

# Information dissemination harm indicators (Solove)
DISSEMINATION_PATTERNS = re.compile(
    r"\b(publicly\s+posted|leaked|exposed|doxxed|published|"
    r"breach\s+of\s+confidentiality|disclosed|revealed|"
    r"available\s+online|searchable|indexed)\b",
    re.IGNORECASE,
)

# Information collection harm indicators (Solove)
COLLECTION_PATTERNS = re.compile(
    r"\b(surveillance|tracking|monitoring|recorded|collected|"
    r"harvested|scraped|intercepted|captured|observed)\b",
    re.IGNORECASE,
)

# Information processing harm indicators (Solove)
PROCESSING_PATTERNS = re.compile(
    r"\b(profiling|aggregat|correlat|infer|analyz|process|"
    r"algorithm|automated\s+decision|scoring|rating)\b",
    re.IGNORECASE,
)

# Invasion harm indicators (Solove)
INVASION_PATTERNS = re.compile(
    r"\b(intrusi|harass|spam|unwanted|unsolicited|manipulat|"
    r"coerce|dark\s+pattern|deceptive|trick|forced)\b",
    re.IGNORECASE,
)


def apply_labeling_functions(record: Record) -> Record:
    """Apply weak supervision labeling functions to enrich a record.

    Args:
        record: Privacy incident record to enrich

    Returns:
        Enriched record with additional structured harm indicators

    Note:
        - Modifies record in-place
        - Stores provenance in label_provenance field
        - Only adds labels when evidence found in text
    """
    provenance: Dict[str, Any] = {}

    # Combine text fields for analysis
    text_parts = []
    if record.description:
        text_parts.append(record.description)
    if record.harm_summary:
        text_parts.append(record.harm_summary)
    if record.raw and isinstance(record.raw, dict):
        for key in ["body", "text", "summary", "title"]:
            if key in record.raw and isinstance(record.raw[key], str):
                text_parts.append(record.raw[key])

    combined_text = " ".join(text_parts) if text_parts else ""

    # LF_DISTRESS: User distress indicators
    distress_matches = DISTRESS_PATTERNS.findall(combined_text)
    if distress_matches:
        # Normalize and deduplicate
        normalized = list(set(m.lower().strip() for m in distress_matches))
        if record.user_distress_indicators is None:
            record.user_distress_indicators = []
        record.user_distress_indicators.extend(normalized)
        record.user_distress_indicators = list(set(record.user_distress_indicators))
        provenance["LF_DISTRESS"] = {"matches": normalized}

    # LF_SENSITIVE_DATA: Data sensitivity classification
    text_lower = combined_text.lower()
    highest_sensitivity = None
    matched_terms = []

    # Check in order from highest to lowest sensitivity
    for sensitivity in [DataSensitivity.SPECIAL, DataSensitivity.HIGH, DataSensitivity.MODERATE]:
        terms = SENSITIVE_DATA_TERMS.get(sensitivity, [])
        found = [term for term in terms if term.lower() in text_lower]
        if found:
            highest_sensitivity = sensitivity
            matched_terms = found
            break

    if highest_sensitivity:
        record.data_sensitivity_level = highest_sensitivity
        provenance["LF_SENSITIVE_DATA"] = {
            "level": highest_sensitivity.value,
            "matched_terms": matched_terms[:5],  # Limit to first 5 for brevity
        }

    # LF_CONTEXT_MISMATCH: Contextual integrity violations
    if CONTEXT_MISMATCH_PATTERNS.search(combined_text):
        record.contextual_mismatch = True
        provenance["LF_CONTEXT_MISMATCH"] = {"detected": True}

    # LF_HARM_CATEGORIES: Solove taxonomy classification
    if record.harm_categories_solove is None:
        record.harm_categories_solove = []

    harm_detected = []

    if DISSEMINATION_PATTERNS.search(combined_text):
        if HarmCategory.INFORMATION_DISSEMINATION not in record.harm_categories_solove:
            record.harm_categories_solove.append(HarmCategory.INFORMATION_DISSEMINATION)
        harm_detected.append("dissemination")

    if COLLECTION_PATTERNS.search(combined_text):
        if HarmCategory.INFORMATION_COLLECTION not in record.harm_categories_solove:
            record.harm_categories_solove.append(HarmCategory.INFORMATION_COLLECTION)
        harm_detected.append("collection")

    if PROCESSING_PATTERNS.search(combined_text):
        if HarmCategory.INFORMATION_PROCESSING not in record.harm_categories_solove:
            record.harm_categories_solove.append(HarmCategory.INFORMATION_PROCESSING)
        harm_detected.append("processing")

    if INVASION_PATTERNS.search(combined_text):
        if HarmCategory.INVASION not in record.harm_categories_solove:
            record.harm_categories_solove.append(HarmCategory.INVASION)
        harm_detected.append("invasion")

    if harm_detected:
        provenance["LF_HARM_CATEGORIES"] = {"categories": harm_detected}

    # Update provenance
    if record.label_provenance is None:
        record.label_provenance = {}
    record.label_provenance.update(provenance)

    return record


def batch_apply_labeling(records: List[Record]) -> List[Record]:
    """Apply labeling functions to a batch of records.

    Args:
        records: List of records to enrich

    Returns:
        List of enriched records
    """
    return [apply_labeling_functions(record) for record in records]
