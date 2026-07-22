"""Automatic harm category labeling based on Solove's taxonomy and keywords.

This module provides functionality to automatically label privacy incidents with
harm categories based on textual analysis. It uses keyword matching and pattern
recognition to assign labels from Solove's privacy taxonomy.

Categories are based on Daniel Solove's "A Taxonomy of Privacy":
- Information Collection (surveillance, interrogation)
- Information Processing (aggregation, identification, insecurity, secondary use, exclusion)
- Information Dissemination (breach of confidentiality, disclosure, exposure, increased accessibility, blackmail, appropriation, distortion)
- Invasion (intrusion, decisional interference)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, overload

from ..llm.provider import classify_privacy_relevance
from ..nlp.gliner2_labeling import GLiNER2Config, label_texts
from ..nlp.gliner2_projection import project_gliner2_to_record_fields

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_HARM_MODELS = ["gpt-5.1-mini", "gpt-4o-mini"]

# Solove's taxonomy categories with associated keywords
HARM_CATEGORIES = {
    "surveillance": [
        "surveillance",
        "tracking",
        "monitoring",
        "watch",
        "observe",
        "spy",
        "location data",
        "geolocation",
        "gps",
        "camera",
        "video",
        "recording",
        "wiretap",
        "intercept",
        "eavesdrop",
        "stalking",
        "following",
    ],
    "interrogation": [
        "interrogation",
        "questioning",
        "inquiry",
        "probe",
        "investigation",
        "background check",
        "verification",
        "audit",
    ],
    "aggregation": [
        "aggregation",
        "profile",
        "profiling",
        "data mining",
        "analytics",
        "combine data",
        "merge data",
        "linking data",
        "correlate",
        "inference",
        "big data",
        "data warehouse",
        "compilation",
    ],
    "identification": [
        "identification",
        "de-anonymization",
        "re-identification",
        "deanonymize",
        "unmask",
        "reveal identity",
        "pii",
        "personal identification",
        "biometric",
        "fingerprint",
        "facial recognition",
        "voice recognition",
    ],
    "model_inversion": [
        "model inversion",
        "membership inference",
        "reconstruct training data",
        "extract training data",
        "memory leakage",
        "memorization",
    ],
    "training_data_extraction": [
        "training data extraction",
        "scraping",
        "copyrighted data",
        "sensitive training data",
        "pii in training",
    ],
    "synthetic_media_misuse": [
        "deepfake",
        "synthetic media",
        "impersonation",
        "fake voice",
        "fake video",
        "non-consensual deepfake",
    ],
    "provenance_failure": [
        "provenance",
        "watermark",
        "ai disclosure",
        "undisclosed ai",
        "labeling logic",
    ],
    "insecurity": [
        "insecurity",
        "data breach",
        "breach",
        "hack",
        "hacked",
        "cyber attack",
        "ransomware",
        "malware",
        "vulnerability",
        "exploit",
        "leak",
        "leaked",
        "unauthorized access",
        "security incident",
        "stolen data",
        "data theft",
        "exposure",
        "unsecured",
        "unencrypted",
        "misconfigured",
    ],
    "secondary_use": [
        "secondary use",
        "repurpose",
        "unexpected use",
        "without consent",
        "unapproved use",
        "unauthorized use",
        "change of terms",
        "sold to third party",
        "shared with",
        "transfer data",
    ],
    "exclusion": [
        "exclusion",
        "no access",
        "denied access",
        "cannot access",
        "hidden data",
        "opacity",
        "opaque",
        "black box",
        "no transparency",
        "cannot delete",
        "cannot correct",
        "no control",
    ],
    "breach_of_confidentiality": [
        "breach of confidentiality",
        "confidentiality breach",
        "disclosed",
        "revealed",
        "leaked confidential",
        "expose confidential",
        "broken trust",
        "violated privacy",
        "privacy violation",
    ],
    "disclosure": [
        "disclosure",
        "reveal",
        "expose",
        "publish",
        "public",
        "publicize",
        "disseminate",
        "distribute",
        "share",
        "leak",
        "release",
    ],
    "increased_accessibility": [
        "increased accessibility",
        "easier access",
        "searchable",
        "indexed",
        "database",
        "public database",
        "available online",
        "posted online",
    ],
    "blackmail": [
        "blackmail",
        "extortion",
        "ransom",
        "threaten",
        "coerce",
        "intimidate",
        "sextortion",
        "revenge porn",
    ],
    "appropriation": [
        "appropriation",
        "misappropriation",
        "identity theft",
        "impersonation",
        "stolen identity",
        "fraudulent use",
        "unauthorized account",
    ],
    "distortion": [
        "distortion",
        "misinformation",
        "false information",
        "inaccurate",
        "incorrect data",
        "defamation",
        "reputation harm",
        "misrepresent",
    ],
    "intrusion": [
        "intrusion",
        "invasive",
        "unwanted contact",
        "spam",
        "harassment",
        "phishing",
        "unsolicited",
        "trespass",
        "unauthorized entry",
    ],
    "decisional_interference": [
        "decisional interference",
        "manipulation",
        "coercion",
        "dark pattern",
        "deceptive design",
        "trick",
        "misleading",
        "forced consent",
        "no choice",
        "manipulate behavior",
        "nudge",
    ],
}

# Additional category for general data handling issues
HARM_CATEGORIES["data_handling"] = [
    "data retention",
    "retention policy",
    "kept too long",
    "never deleted",
    "excessive collection",
    "overcollection",
    "unnecessary data",
    "data minimization",
    "proportionality",
]

# Regulatory violations
HARM_CATEGORIES["regulatory_violation"] = [
    "gdpr",
    "ccpa",
    "coppa",
    "hipaa",
    "ferpa",
    "glba",
    "fcra",
    "violation",
    "non-compliance",
    "fine",
    "penalty",
    "enforcement",
    "consent violation",
    "notice violation",
    "ftc",
    "dpa",
    "ico",
]


def _score_text_for_category(text: str, keywords: List[str]) -> float:
    """Score text relevance to a category based on keyword matches.

    Args:
        text: Text to analyze (lowercased)
        keywords: List of keywords for the category

    Returns:
        Score representing keyword match density
    """
    if not text:
        return 0.0

    score = 0.0
    text_words = set(re.findall(r"\w+", text))

    for keyword in keywords:
        # Exact phrase match
        if keyword in text:
            score += 2.0
        # Individual word match
        keyword_words = set(re.findall(r"\w+", keyword))
        overlap = len(text_words & keyword_words)
        if overlap > 0:
            score += overlap * 0.5

    return score


@overload
def label_harm_category(
    record: Dict[str, Any],
    text_fields: Optional[List[str]] = None,
    threshold: float = 1.0,
    return_scores: Literal[False] = False,
) -> str: ...


@overload
def label_harm_category(
    record: Dict[str, Any],
    text_fields: Optional[List[str]] = None,
    threshold: float = 1.0,
    return_scores: Literal[True] = True,
) -> Dict[str, float]: ...


def label_harm_category(
    record: Dict[str, Any],
    text_fields: Optional[List[str]] = None,
    threshold: float = 1.0,
    return_scores: bool = False,
) -> str | Dict[str, float]:
    """Automatically label a record with a harm category.

    Args:
        record: Normalized record dict with text fields
        text_fields: List of field names to analyze (default: description, title, raw text)
        threshold: Minimum score required for labeling (default: 1.0)
        return_scores: If True, return dict with scores for all categories

    Returns:
        Harm category label string, or "unknown" if no clear match.
        If return_scores=True, returns dict with category scores.
    """
    if not isinstance(record, dict):
        logger.warning(
            "label_harm_category received non-dict record (%s); returning 'unknown'",
            type(record).__name__,
        )
        return "unknown"

    if text_fields is None:
        text_fields = ["description", "title", "body", "summary"]

    assert text_fields is not None

    # Combine text from specified fields
    text_parts = []
    for field in text_fields:
        value = record.get(field)
        if value and isinstance(value, str):
            text_parts.append(value)

    # Also check raw dict if present
    raw = record.get("raw", {})
    if isinstance(raw, dict):
        for field in text_fields:
            value = raw.get(field)
            if value and isinstance(value, str):
                text_parts.append(value)

    combined_text = " ".join(text_parts).lower()

    # Score each category
    category_scores = {}
    for category, keywords in HARM_CATEGORIES.items():
        score = _score_text_for_category(combined_text, keywords)
        category_scores[category] = score

    if return_scores:
        return category_scores

    # Find best category
    if not category_scores:
        return "unknown"

    best_category = max(category_scores.items(), key=lambda x: x[1])

    if best_category[1] < threshold:
        return "unknown"

    return best_category[0]


def _collect_text_chunks(record: Dict[str, Any], text_fields: List[str]) -> str:
    """Extract and concatenate text from the configured fields (including raw)."""

    text_parts: List[str] = []
    for field in text_fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            text_parts.append(value.strip())

    raw = record.get("raw", {})
    if isinstance(raw, dict):
        for field in text_fields:
            value = raw.get(field)
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())

    return "\n\n".join(text_parts)


def _normalize_models(models: Sequence[str] | str | None) -> List[str]:
    if models is None:
        return []
    if isinstance(models, str):
        candidates = models.split(",")
    else:
        candidates = list(models)
    return [m.strip() for m in candidates if m and m.strip()]


def _llm_label_harms(
    record: Dict[str, Any],
    text_fields: List[str],
    *,
    provider: Optional[str],
    models: Sequence[Optional[str]],
    max_tokens: int,
) -> Optional[Dict[str, Any]]:
    if not provider:
        return None

    text_blob = _collect_text_chunks(record, text_fields)
    if not text_blob.strip():
        return None

    model_candidates: List[Optional[str]] = list(models)
    if not model_candidates and provider.lower() == "openai":
        model_candidates = [m for m in DEFAULT_OPENAI_HARM_MODELS]
    if not model_candidates:
        model_candidates = [None]

    for model_name in model_candidates:
        # Enable Google Search grounding for Gemini Pro/Heavy models to improve accuracy
        tools = None
        if provider and provider.lower() == "gemini":
            m_lower = (model_name or "").lower()
            # Enable search for pro/heavy models or explicit 2.5 versions which support it well
            if "pro" in m_lower or "heavy" in m_lower or "2.5" in m_lower or "ultra" in m_lower:
                tools = "google_search"

        try:
            response = classify_privacy_relevance(
                text_blob,
                provider=provider,
                model=model_name,
                max_tokens=max_tokens,
                tools=tools,
            )
        except Exception as exc:  # pragma: no cover - defensive safeguard
            logger.warning(
                "LLM classification failed (provider=%s, model=%s): %s", provider, model_name, exc
            )
            continue

        harms = response.get("harms") or []
        if not harms:
            continue
        label = harms[0]
        return {
            "label": label,
            "harms": harms,
            "model_used": model_name,
            "confidence": response.get("confidence"),
            "rationale": response.get("rationale"),
        }
    return None


def label_jsonl_file(
    input_path: str,
    output_path: str,
    text_fields: Optional[List[str]] = None,
    overwrite: bool = False,
    llm_provider: Optional[str] = None,
    llm_models: Sequence[str] | str | None = None,
    llm_max_tokens: int = 256,
    use_gliner2: bool = True,
) -> int:
    """Add harm_category labels to a JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file with labels
        text_fields: Fields to analyze for labeling
        overwrite: If True, overwrite existing harm_category field
        llm_provider: Optional provider name (e.g., "openai") to relabel "unknown" rows
        llm_models: Sequence or comma-delimited string of model names to try (in order)
        llm_max_tokens: Max tokens for the LLM JSON classification response

    Returns:
        Number of records processed
    """
    import json

    normalized_models = _normalize_models(llm_models)

    if llm_provider:
        llm_provider = llm_provider.strip().lower()

    if text_fields is None:
        text_fields = ["description", "title", "body", "summary"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # When no LLM provider is requested, prefer GLiNER2 as the primary
    # labeling engine for harm categories and root cause features. LLM
    # + heuristic labeling remain available when llm_provider is set.
    use_gliner2 = bool(use_gliner2) and llm_provider is None
    gl_cfg = GLiNER2Config()
    gl_model = None
    if use_gliner2:
        try:  # Lazy, optional GLiNER2 load
            from privacy_harm_heuristics.nlp import gliner2_labeling as _g_mod  # type: ignore

            gl_model = _g_mod._load_gliner2_model(gl_cfg.model_name)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            gl_model = None

    count = 0

    # When GLiNER2 is enabled and available, process in batches for efficiency.
    if use_gliner2 and gl_model is not None:
        batch_records: list[Dict[str, Any]] = []
        batch_texts: list[str] = []

        def flush_batch() -> None:
            nonlocal count, batch_records, batch_texts
            if not batch_records:
                return
            gl_results = label_texts(batch_texts, config=gl_cfg, model=gl_model)
            # Align results with records
            for rec, gl in zip(batch_records, gl_results):
                harm_category: Optional[str] = None
                # Attach GLiNER2 output and project into schema
                rec["gliner2_labels"] = gl
                project_gliner2_to_record_fields(rec)
                harm_labels = [
                    lab
                    for lab in gl.get("text_labels", [])
                    if isinstance(lab, str) and lab.startswith("harm_")
                ]
                if harm_labels:
                    harm_category = harm_labels[0]
                    rec["harm_category"] = harm_category
                    rec["harm_category_source"] = "gliner2"

                # Heuristic + LLM supplement when GLiNER2 did not assign a harm label
                if harm_category is None:
                    heuristic_label = label_harm_category(rec, text_fields=text_fields)
                    harm_category = heuristic_label
                    rec["harm_category"] = heuristic_label
                    rec["harm_category_source"] = "heuristic"

                    if heuristic_label == "unknown" and llm_provider:
                        llm_result = _llm_label_harms(
                            rec,
                            text_fields,
                            provider=llm_provider,
                            models=normalized_models,
                            max_tokens=llm_max_tokens,
                        )
                        if llm_result:
                            rec["harm_category"] = llm_result["label"]
                            rec["harm_category_source"] = "llm"
                            rec["harm_categories_llm"] = llm_result["harms"]
                            rec["harm_category_confidence_llm"] = llm_result.get("confidence")
                            rec["harm_category_rationale_llm"] = llm_result.get("rationale")
                            rec["harm_category_model_llm"] = llm_result.get("model_used")

                # Write out enriched record
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

            batch_records = []
            batch_texts = []

        with open(input_path, "r", encoding="utf-8") as fin:
            with open(output_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid JSON line in %s", input_path)
                        continue

                    if not isinstance(record, dict):
                        logger.warning(
                            "Skipping non-object record in %s (type=%s)",
                            input_path,
                            type(record).__name__,
                        )
                        continue

                    # Add label if not present or if overwriting
                    if overwrite or "harm_category" not in record:
                        parts: list[str] = []
                        for f_name in text_fields:
                            val = record.get(f_name)
                            if isinstance(val, str) and val.strip():
                                parts.append(val.strip())
                        text_blob = "\n\n".join(parts).strip()

                        if text_blob:
                            batch_records.append(record)
                            batch_texts.append(text_blob)
                            if len(batch_texts) >= gl_cfg.batch_size:
                                flush_batch()
                        else:
                            # No usable text – fall back immediately to heuristic/LLM
                            heuristic_label = label_harm_category(record, text_fields=text_fields)
                            record["harm_category"] = heuristic_label
                            record["harm_category_source"] = "heuristic"
                            if heuristic_label == "unknown" and llm_provider:
                                llm_result = _llm_label_harms(
                                    record,
                                    text_fields,
                                    provider=llm_provider,
                                    models=normalized_models,
                                    max_tokens=llm_max_tokens,
                                )
                                if llm_result:
                                    record["harm_category"] = llm_result["label"]
                                    record["harm_category_source"] = "llm"
                                    record["harm_categories_llm"] = llm_result["harms"]
                                    record["harm_category_confidence_llm"] = llm_result.get(
                                        "confidence"
                                    )
                                    record["harm_category_rationale_llm"] = llm_result.get(
                                        "rationale"
                                    )
                                    record["harm_category_model_llm"] = llm_result.get("model_used")

                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                            count += 1
                    else:
                        # Record already labeled and overwrite=False
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1

                # Flush any remaining GLiNER2 batch
                flush_batch()

        return count

    # Fallback path: GLiNER2 disabled or unavailable – use heuristic + optional LLM only.
    with open(input_path, "r", encoding="utf-8") as fin:
        with open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line in %s", input_path)
                    continue

                if not isinstance(record, dict):
                    logger.warning(
                        "Skipping non-object record in %s (type=%s)",
                        input_path,
                        type(record).__name__,
                    )
                    continue

                if overwrite or "harm_category" not in record:
                    heuristic_label = label_harm_category(record, text_fields=text_fields)
                    record["harm_category"] = heuristic_label
                    record["harm_category_source"] = "heuristic"

                    if heuristic_label == "unknown" and llm_provider:
                        llm_result = _llm_label_harms(
                            record,
                            text_fields,
                            provider=llm_provider,
                            models=normalized_models,
                            max_tokens=llm_max_tokens,
                        )
                        if llm_result:
                            record["harm_category"] = llm_result["label"]
                            record["harm_category_source"] = "llm"
                            record["harm_categories_llm"] = llm_result["harms"]
                            record["harm_category_confidence_llm"] = llm_result.get("confidence")
                            record["harm_category_rationale_llm"] = llm_result.get("rationale")
                            record["harm_category_model_llm"] = llm_result.get("model_used")

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    return count


def analyze_label_distribution(jsonl_path: str) -> Dict[str, int]:
    """Analyze distribution of harm_category labels in a JSONL file.

    Args:
        jsonl_path: Path to JSONL file with harm_category field

    Returns:
        Dict mapping categories to counts
    """
    import json

    category_counts: Counter[str] = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                category = record.get("harm_category", "unknown")
                category_counts[category] += 1
            except json.JSONDecodeError:
                continue

    return dict(category_counts)
