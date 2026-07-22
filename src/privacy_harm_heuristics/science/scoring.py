import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..llm.provider import classify_privacy_relevance
from ..labeling.harm_labeler import label_harm_category

logger = logging.getLogger(__name__)

# Default weights derived from initial expert heuristic
DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "gdpr_art_83": 0.18,
    "gdpr_art_7": 0.12,
    "gdpr_art_5": 0.10,
    "gdpr_art_9": 0.14,
    "gdpr_art_22": 0.12,
    "gdpr_art_32": 0.16,
    "lack_of_consent": 0.10,
    "privacy_program_gap": 0.06,
    "surveillance_overreach": 0.08,
    "behavioral_manipulation": 0.08,
    "profiling_violation": 0.08,
    "data_minimization_violation": 0.06,
    "transparency_failure": 0.08,
    "penalty_enforcement": 0.10,
    "regulatory_enforcement": 0.10,
    "online_tracking": 0.06,
    "behavioral_tracking": 0.06,
    "unauthorized_sharing": 0.08,
    "ccpa_data_sharing": 0.08,
    "secret_collection": 0.08,
    "inadequate_security": 0.10,
    "encryption_failure": 0.12,
    "bipa_biometric_violation": 0.14,
    "biometric_misuse": 0.12,
    # AI Harms
    "model_inversion": 0.14,
    "training_data_extraction": 0.16,
    "synthetic_media_misuse": 0.18,
    "provenance_failure": 0.08,
}

# Framework mapping for "science provenance"
FRAMEWORK_HINTS: Dict[str, List[str]] = {
    "f_has_penalty": ["GDPR Art. 83", "FTC Enforcement Playbook", "NIST Respond"],
    "kw_privacy": [
        "Nissenbaum Contextual Integrity",
        "Solove Taxonomy (Information Dissemination)",
        "NIST Govern",
        "Westin Personal Autonomy",
    ],
    "kw_location": [
        "GDPR Art. 4(1)",
        "Solove (Surveillance)",
        "OECD Purpose Limitation",
        "Zuboff Behavioral Extraction",
    ],
    "kw_biometric": [
        "GDPR Art. 9",
        "OECD Data Quality",
        "NIST Protect",
        "Solove Identification",
    ],
    "kw_monetary_penalty": ["GDPR Art. 83", "Solove (Enforcement)", "NIST Communicate"],
    "kw_reg_enforcement": ["GDPR Art. 83", "NIST Govern", "ISO/IEC 27701"],
    "f_penalty_bucket": ["GDPR Art. 83", "FTC Penalty Authority"],
    "kw_surveillance": [
        "Solove Surveillance",
        "Zuboff Surveillance Capitalism",
        "Nissenbaum Contextual Integrity",
    ],
    "kw_tracking": [
        "Solove Surveillance",
        "Zuboff Behavioral Extraction",
        "Calo Objective Harm",
    ],
    "kw_behavioral": [
        "Zuboff Behavioral Modification",
        "Calo Subjective Harm",
        "Westin Personal Autonomy",
    ],
    "kw_manipulation": [
        "Zuboff Instrumentarian Power",
        "Calo Dignitary Harm",
        "Westin Emotional Release",
    ],
    "kw_anonymization": [
        "Data Privacy Lab K-Anonymity",
        "Solove Identification",
        "Data Privacy Lab Re-identification Risk",
    ],
    "kw_profiling": ["Solove Aggregation", "Zuboff Behavioral Surplus", "GDPR Art. 22"],
    "kw_consent": ["GDPR Art. 7", "Westin Privacy Pragmatists", "Solove Secondary Use"],
    "kw_transparency": [
        "Nissenbaum Appropriateness",
        "Solove Exclusion",
        "GDPR Art. 12-14",
    ],
    # AI Framework Hints
    "model_inversion": ["NIST Protect", "Solove Identification", "Iso/IEC 27001"],
    "training_data_extraction": ["GDPR Art. 5", "Solove Appropriation", "NIST Protect"],
    "synthetic_media_misuse": ["Calo Dignitary Harm", "Solove Distortion", "NIST Govern"],
    "provenance_failure": ["NIST Govern", "GDPR Art. 13-14", "EU AI Act"],
}


class ScoringEngine:
    """Scientific scoring engine for privacy risk assessment.

    Encapsulates logic for:
    1. Loading calibrated weights.
    2. Calculating risk scores based on features and categories.
    3. Providing framework-aligned explanations (provenance).
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.weights = DEFAULT_CATEGORY_WEIGHTS.copy()
        self.calibrated = False
        self._load_calibration()

    def _load_calibration(self) -> None:
        """Load optimized weights if available."""
        weight_path = self.data_dir / "calibrated_weights.json"
        if weight_path.exists():
            try:
                with open(weight_path, "r") as f:
                    calibrated = json.load(f)
                    # We expect a simple dict {"category_name": weight}
                    self.weights.update(calibrated)
                    self.calibrated = True
                    logger.info(f"Loaded calibrated weights from {weight_path}")
            except Exception as e:
                logger.error(f"Failed to load calibration: {e}")

    def calculate_score(
        self,
        harm_categories: List[str],
        privacy_keywords: List[str],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate a risk score (0-10) and detailed breakdown.

        Args:
            harm_categories: List of detected harm categories (e.g. from rules).
            privacy_keywords: List of detected product features/keywords.

        Returns:
            Tuple of (score, provenance_dict)
        """
        score = 0.0
        details: Dict[str, Any] = {
            "category_contributions": [],
            "keyword_contribution": 0.0,
            "base_score": 0.0,
        }

        # 1. Category-based risk
        for cat in harm_categories:
            weight = self.weights.get(cat, 0.5)  # Default weight for unknown categories
            contribution = weight
            score += contribution
            details["category_contributions"].append(
                {"category": cat, "weight": weight, "contribution": contribution}
            )

        # 2. Key-word based risk (simple heuristic add-on)
        # We cap this contribution to avoid keyword-stuffing skew
        kw_weight = 0.5
        kw_contribution = kw_weight * len(privacy_keywords)
        kw_contribution = min(5.0, kw_contribution)  # Cap at 5.0
        score += kw_contribution
        details["keyword_contribution"] = kw_contribution

        # 3. Clamp final score
        score = max(0.0, min(10.0, score))

        return score, details

    def get_framework_provenance(self, harm_categories: List[str]) -> Dict[str, float]:
        """Aggregate risk contribution by theoretical framework.

        Returns a percentage share for each major framework (Solove, NIST, GDPR).
        """
        framework_counts: Dict[str, float] = {
            "Solove": 0.0,
            "NIST": 0.0,
            "GDPR": 0.0,
            "Zuboff": 0.0,
        }
        total_hits = 0

        # We will iterate through categories and try to match them to hints
        for cat in harm_categories:
            # Check if this category is in our Hint map
            if cat in FRAMEWORK_HINTS:
                hints = FRAMEWORK_HINTS[cat]
                for h in hints:
                    if "Solove" in h:
                        framework_counts["Solove"] += 1
                    if "NIST" in h:
                        framework_counts["NIST"] += 1
                    if "GDPR" in h:
                        framework_counts["GDPR"] += 1
                    if "Zuboff" in h:
                        framework_counts["Zuboff"] += 1
                    total_hits += 1
            else:
                # If category isn't in hints directly, try to match loose keywords
                if "gdpr" in cat:
                    framework_counts["GDPR"] += 1
                    total_hits += 1
                elif "surveillance" in cat:
                    framework_counts["Solove"] += 1
                    framework_counts["Zuboff"] += 1
                    total_hits += 2

        # Normalize to percentages
        if total_hits == 0:
            return {f: 0.0 for f in framework_counts}

        return {k: round((v / total_hits) * 100, 1) for k, v in framework_counts.items()}


class HybridScorer:
    """
    Implements the 'Hybrid (LLM-First)' architecture.
    1. LLM proposes harms (High Recall).
    2. Heuristics validate harms (High Precision).
    """

    def __init__(self, scoring_engine: ScoringEngine):
        self.engine = scoring_engine

    def hybrid_score(self, text: str, provider: str = "gemini") -> Dict[str, Any]:
        """
        Execute Hybrid scoring pipeline.

        Args:
            text: The unstructured text to analyze.
            provider: LLM provider to use (default: gemini).

        Returns:
            Dict containing score, validated harms, and provenance.
        """
        # 1. LLM Proposal
        # We use a lower confidence threshold for the LLM since we have a validation step
        llm_result = classify_privacy_relevance(text, provider=provider)
        proposed_harms = llm_result.get("harms", [])
        rationale = llm_result.get("rationale", "")

        # 2. Heuristic Validation
        # Check against features extracted from text
        # We assume text is the full record content
        record = {"description": text}
        # label_harm_category returns {category: score} when return_scores=True
        heuristic_scores = label_harm_category(
            record, return_scores=True, text_fields=["description"]
        )

        validated_harms = []
        rejected_harms = []

        # We define a lenient threshold for validation.
        # If the LLM says "surveillance", and we find even a weak signal (score > 0.5)
        # in the deterministic rules, we validate it.
        # This differs from "Pure Rules" which might need score > 1.0 or 2.0 to trigger on its own.
        VALIDATION_THRESHOLD = 0.5

        # We also check if specific keywords are present to back up the LLM
        # label_harm_category logic already does keyword density.

        if isinstance(heuristic_scores, dict):
            for harm in proposed_harms:
                h_score = heuristic_scores.get(harm, 0.0)
                if h_score >= VALIDATION_THRESHOLD:
                    validated_harms.append(harm)
                else:
                    rejected_harms.append(harm)
        else:
            # Fallback if label_harm_category returns unexpected type
            # We trust LLM if heuristic fails entirely (rare)
            validated_harms = proposed_harms

        # 3. Calculate Score based on VALIDATED harms
        # We pass empty keyword list for now as we focus on category weights
        score, details = self.engine.calculate_score(validated_harms, [])

        # 4. Get Framework Alignment for the VALIDATED harms
        provenance = self.engine.get_framework_provenance(validated_harms)

        # 5. Construct Result
        return {
            "score": score,
            "risk_score_0_100": score * 10,
            "proposed_harms": proposed_harms,
            "validated_harms": validated_harms,
            "rejected_harms": rejected_harms,
            "rationale": rationale,
            "details": details,
            "frameworks": provenance,
        }
