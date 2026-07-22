"""Privacy harms taxonomy constants (Solove-aligned) and offline heuristics.

This module centralizes our taxonomy so prompts and UI stay consistent.

Categories follow Daniel J. Solove's taxonomy at a high level:
 - information_collection
 - information_processing
 - information_dissemination
 - invasion

Each category includes a concise definition and representative subtypes. We also
include a lightweight keyword heuristic for offline relevance gating.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Values are intentionally heterogeneous (nested lists/dicts of keywords, subtypes,
# etc.), so `Any` reflects that -- `object` made every `.get(...)` call site an
# untyped-iterable under mypy without adding any real type safety.
TAXONOMY_SOLOVE: Dict[str, Dict[str, Any]] = {
    "information_collection": {
        "title": "Information Collection",
        "definition": "Gathering data via surveillance or compulsory questioning.",
        "subtypes": [
            {"name": "surveillance", "definition": "Watching/listening/recording people"},
            {"name": "interrogation", "definition": "Compelling disclosures/answers"},
        ],
        "keywords": [
            "surveillance",
            "monitoring",
            "track",
            "tracking",
            "spy",
            "spyware",
            "keylogger",
            "always on",
            "listening",
            "interrogation",
        ],
    },
    "information_processing": {
        "title": "Information Processing",
        "definition": "Handling data in ways that create risk or reduce control.",
        "subtypes": [
            {"name": "aggregation", "definition": "Combining datasets reveals more"},
            {"name": "identification", "definition": "De-anonymization / re-identification"},
            {"name": "insecurity", "definition": "Poor security / breaches of safeguards"},
            {"name": "secondary_use", "definition": "Using data beyond original purpose"},
            {"name": "exclusion", "definition": "No access/correction/notice to individuals"},
        ],
        "keywords": [
            "aggregate",
            "re-identification",
            "deanonym",
            "insecurity",
            "breach",
            "leak",
            "secondary use",
            "repurpose",
            "no consent",
            "no notice",
            "no access",
        ],
    },
    "information_dissemination": {
        "title": "Information Dissemination",
        "definition": "Sharing or exposing data in harmful ways.",
        "subtypes": [
            {
                "name": "breach_of_confidentiality",
                "definition": "Breaking a confidentiality promise",
            },
            {"name": "disclosure", "definition": "Publishing personal data"},
            {"name": "exposure", "definition": "Revealing intimate/sensitive details"},
            {"name": "increased_accessibility", "definition": "Doxxing / indexing private info"},
            {"name": "blackmail", "definition": "Coercion with threatened disclosure"},
            {"name": "appropriation", "definition": "Using likeness/reputation without consent"},
            {"name": "distortion", "definition": "Misleading/false data harming reputation"},
        ],
        "keywords": [
            "dox",
            "doxing",
            "doxxing",
            "expose",
            "exposure",
            "leak",
            "publicly posted",
            "breach of confidentiality",
            "blackmail",
            "deepfake",
            "defamation",
            "distortion",
        ],
    },
    "invasion": {
        "title": "Invasion",
        "definition": "Intruding into private affairs or decisions.",
        "subtypes": [
            {"name": "intrusion", "definition": "Unwanted entry into private space/solitude"},
            {
                "name": "decisional_interference",
                "definition": "Manipulating/interfering with choices",
            },
        ],
        "keywords": [
            "intrusion",
            "intrusive",
            "stalking",
            "swatting",
            "home camera",
            "forced",
            "coerc",
            "manipulat",
            "dark pattern",
        ],
    },
}


def solove_categories() -> List[str]:
    return list(TAXONOMY_SOLOVE.keys())


# Optional registry for additional curated taxonomies.
# If/when authoritative frameworks are added, register them here.
TAXONOMIES: Dict[str, Dict[str, Dict[str, object]]] = {
    # Our canonical
    "solove": TAXONOMY_SOLOVE,
    # Placeholder aliases to allow integration code to run even when the
    # authoritative frameworks aren't bundled yet. These return empty dicts
    # by default to avoid surprising mappings.
    # Populate with real content in future commits under this registry.
    "nist_8062": {},
    "nist_privacy_framework": {},
    "ico_dp_harms": {},
    "iso_29100": {},
    # Academic variants; until curated, point to Solove as a conservative fallback.
    "citron_solove": {},
    "calo": {},
}


def available_taxonomies() -> List[str]:
    return sorted(TAXONOMIES.keys())


def get_taxonomy(name: str) -> Dict[str, Dict[str, object]]:
    """Return taxonomy spec by name.

    Names are lowercase identifiers like 'solove', 'nist_8062', etc.
    If the requested taxonomy isn't available, return an empty mapping.

    For now, academic variants map to curated content when present;
    otherwise they yield empty dicts. Callers should handle empty returns.
    """
    key = str(name).strip().lower()
    if key == "solove":
        return TAXONOMY_SOLOVE
    spec = TAXONOMIES.get(key)
    if spec is None:
        return {}
    return spec


def taxonomy_prompt_block() -> str:
    """Return a compact, model-friendly taxonomy block for prompts."""
    lines: List[str] = []
    for key, spec in TAXONOMY_SOLOVE.items():
        title = spec.get("title", key)
        definition = spec.get("definition", "")
        lines.append(f"- {key} ({title}): {definition}")
        subs = spec.get("subtypes", [])
        if isinstance(subs, list) and subs:
            example_names: List[str] = []
            for sub in subs:
                if not isinstance(sub, dict):
                    continue
                name = sub.get("name", "")
                if isinstance(name, str) and name:
                    example_names.append(name)
            if example_names:
                lines.append(f"  e.g., {', '.join(example_names)}")
    return "\n".join(lines)


def keyword_fallback_categories(text: str) -> List[str]:
    """Heuristic category guess using per-category keywords (lowercase match)."""
    t = text.lower()
    hits: List[str] = []

    # Check each top-level category
    for key, spec in TAXONOMY_SOLOVE.items():
        # Check high-level keywords
        kws = [k.lower() for k in spec.get("keywords", []) if isinstance(k, str)]
        if any(k in t for k in kws):
            # If high-level match, try to find specific subtypes, or default to all subtypes?
            # Better strategy: Only return subtpyes if mentioned?
            # But high-level keywords like "surveillance" are in the parent block in this file?
            # Actually, looking at lines 28+, "surveillance" is a keyword for "information_collection".
            # The subtype "surveillance" is listed in "subtypes".
            # Let's try to map keywords to subtypes if possible, or return subtype keys if parent matches.

            # Simple heuristic: If parent matches, return its "primary" subtypes or all?
            # Or better, let's look for subtype names in text too.

            # Current quick fix:
            # If "surveillance" (keyword) is in text, return "surveillance" (subtype).
            # The current structure has keywords on the PARENT.
            # We should probably return the subtypes that match the keywords.
            # But the mapping isn't explicit in this file.
            pass

    # Alternative: Robust flattening
    # Iterate ALL subtypes across all categories
    for cat_key, spec in TAXONOMY_SOLOVE.items():
        for sub in spec.get("subtypes", []):
            if not isinstance(sub, dict):
                continue
            s_name = sub.get("name")
            if not s_name:
                continue

            # Check if subtype name is in text
            if s_name.replace("_", " ") in t:
                hits.append(s_name)

            # Also check parent keywords... this is tricky without direct mapping.
            # Let's rely on the expanded FALLBACK_RELEVANCE_KEYWORDS and just return broad hits?
            # No, Scorer needs specific keys.

    # For now, let's manually map some common ones based on the parent keywords.
    if "surveillance" in t or "tracking" in t:
        hits.append("surveillance")
    if "interrogation" in t:
        hits.append("interrogation")
    if "aggregate" in t or "profiling" in t:
        hits.append("aggregation")
    if "identif" in t:
        hits.append("identification")
    if "insecurity" in t or "breach" in t or "hack" in t:
        hits.append("insecurity")
    if "secondary" in t or "repurpose" in t:
        hits.append("secondary_use")
    if "blackmail" in t:
        hits.append("blackmail")
    if "exposure" in t or "leak" in t:
        hits.append("exposure")
    if "disclosure" in t:
        hits.append("disclosure")
    if "intrusi" in t:
        hits.append("intrusion")

    # Expanded mappings for better recall
    if "password" in t or "credential" in t:
        hits.append("insecurity")
        hits.append("breach_of_confidentiality")
    if "email" in t or "phone" in t or "address" in t:
        hits.append("identification")
    if "stolen" in t or "theft" in t:
        hits.append("breach_of_confidentiality")
    if "camera" in t or "video" in t or "gps" in t or "location" in t:
        hits.append("surveillance")
    if "biometric" in t or "face" in t:
        hits.append("identification")
        hits.append("biometric_misuse")
    if "deepfake" in t or "synthetic" in t:
        hits.append("synthetic_media_misuse")
    if "scrape" in t or "scraping" in t:
        hits.append("aggregation")
        hits.append("training_data_extraction")

    return list(set(hits))


FALLBACK_RELEVANCE_KEYWORDS: List[str] = [
    # General privacy terms
    "privacy",
    "gdpr",
    "ccpa",
    "hipaa",
    "data protection",
    "personal data",
    "personally identifiable",
    "pii",
    "dox",
    "breach",
    "leak",
    "consent",
    "surveillance",
    "tracking",
    "cookie",
]

COMMON_ROOT_CAUSES: List[str] = [
    "api_abuse",
    "credential_theft",
    "deceptive_practices",
    "inadequate_oversight",
    "insider_threat",
    "lack_of_access_control",
    "lack_of_consent",
    "misconfiguration",
    "phishing",
    "ransomware",
    "scraping",
    "social_engineering",
    "state_sponsored_attack",
    "third_party_breach",
    "unpatched_vulnerability",
    "weak_encryption",
]


def all_solove_terms() -> List[str]:
    """Return all Solove categories and subtypes."""
    terms = list(TAXONOMY_SOLOVE.keys())
    for spec in TAXONOMY_SOLOVE.values():
        for sub in spec.get("subtypes", []):
            if isinstance(sub, dict) and "name" in sub:
                terms.append(sub["name"])
    return sorted(list(set(terms)))
