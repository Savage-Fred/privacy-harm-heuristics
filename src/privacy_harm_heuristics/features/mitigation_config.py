"""Mitigation recommendations for privacy risks.

Maps risky product features to recommended protective mitigations.
Each risky feature has:
- Primary mitigations (most effective)
- Secondary mitigations (additional protection)
- Implementation notes

This configuration is used by the mitigation recommendation system to suggest
concrete protective measures for identified privacy risks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, TypedDict


class EffectivenessDetail(TypedDict):
    feature: str
    effectiveness: float
    combined: float


class MitigationTemplate(TypedDict):
    primary: List[str]
    secondary: List[str]
    description: str
    effectiveness_weight: float


class MitigationRecommendation(MitigationTemplate, total=False):
    risk_feature: str
    primary_effectiveness: List[EffectivenessDetail]
    secondary_effectiveness: List[EffectivenessDetail]


# Comprehensive mitigation mapping: risky feature → protective measures
MITIGATION_MAP: Dict[str, MitigationTemplate] = {
    "always_on_listening": {
        "primary": ["local_processing", "user_consent_granular", "privacy_dashboard"],
        "secondary": ["ephemeral_messages", "data_minimization", "encryption_at_rest"],
        "description": "Always-on listening devices should process locally when possible",
        "effectiveness_weight": 0.85,  # High effectiveness
    },
    "webcam_access": {
        "primary": ["user_consent_granular", "privacy_dashboard", "device_encryption"],
        "secondary": ["encryption_in_transit", "data_minimization"],
        "description": "Webcam access requires explicit consent and encryption",
        "effectiveness_weight": 0.80,
    },
    "location_tracking": {
        "primary": ["data_minimization", "user_consent_granular", "short_data_retention"],
        "secondary": ["data_localization", "anonymization", "encryption_at_rest"],
        "description": "Location tracking should minimize data collection and retention",
        "effectiveness_weight": 0.90,
    },
    "data_broker_usage": {
        "primary": ["user_consent_granular", "ad_transparency", "data_minimization"],
        "secondary": ["privacy_labels", "privacy_dashboard", "data_deletion_user_control"],
        "description": "Data broker usage requires transparent disclosure and user control",
        "effectiveness_weight": 0.75,
    },
    "third_party_sharing": {
        "primary": ["user_consent_granular", "ad_transparency", "privacy_labels"],
        "secondary": ["data_minimization", "anonymization", "privacy_dashboard"],
        "description": "Third-party sharing must be disclosed and user-controllable",
        "effectiveness_weight": 0.80,
    },
    "biometric_collection": {
        "primary": ["end_to_end_encryption", "local_processing", "device_encryption"],
        "secondary": ["data_minimization", "short_data_retention", "user_consent_granular"],
        "description": "Biometric data requires strongest encryption and local processing",
        "effectiveness_weight": 0.95,
    },
    "background_data_collection": {
        "primary": ["user_consent_granular", "privacy_dashboard", "ad_transparency"],
        "secondary": ["data_minimization", "short_data_retention"],
        "description": "Background collection requires explicit disclosure and control",
        "effectiveness_weight": 0.85,
    },
    "sensor_always_active": {
        "primary": ["local_processing", "user_consent_granular", "privacy_dashboard"],
        "secondary": ["device_encryption", "data_minimization"],
        "description": "Always-active sensors should process locally and allow user control",
        "effectiveness_weight": 0.85,
    },
    "forced_consent": {
        "primary": ["user_consent_granular", "privacy_dashboard"],
        "secondary": ["ad_transparency", "privacy_labels"],
        "description": "Provide granular consent options instead of all-or-nothing",
        "effectiveness_weight": 0.90,
    },
    "hidden_data_collection": {
        "primary": ["ad_transparency", "privacy_labels", "privacy_dashboard"],
        "secondary": ["user_consent_granular", "data_minimization"],
        "description": "Make all data collection transparent and visible to users",
        "effectiveness_weight": 0.95,
    },
    "misleading_privacy_notice": {
        "primary": ["privacy_labels", "ad_transparency"],
        "secondary": ["privacy_dashboard", "user_consent_granular"],
        "description": "Use clear, simple language and privacy nutrition labels",
        "effectiveness_weight": 0.80,
    },
    "fourth_party_tracking": {
        "primary": ["user_consent_granular", "ad_transparency", "data_minimization"],
        "secondary": ["privacy_labels", "anonymization"],
        "description": "Disclose all tracking parties and provide opt-out",
        "effectiveness_weight": 0.75,
    },
    "excessive_permissions": {
        "primary": ["data_minimization", "user_consent_granular"],
        "secondary": ["privacy_dashboard", "privacy_labels"],
        "description": "Request only necessary permissions with clear justification",
        "effectiveness_weight": 0.85,
    },
    "cross_context_data_use": {
        "primary": ["user_consent_granular", "data_minimization", "ad_transparency"],
        "secondary": ["privacy_dashboard", "privacy_labels"],
        "description": "Get explicit consent for any secondary use of data",
        "effectiveness_weight": 0.80,
    },
    "no_user_control": {
        "primary": ["privacy_dashboard", "data_deletion_user_control", "user_consent_granular"],
        "secondary": ["ad_transparency", "privacy_labels"],
        "description": "Provide comprehensive privacy dashboard with data controls",
        "effectiveness_weight": 0.90,
    },
    "opaque_algorithms": {
        "primary": ["ad_transparency", "privacy_dashboard"],
        "secondary": ["user_consent_granular", "privacy_labels"],
        "description": "Explain algorithmic decisions and provide transparency",
        "effectiveness_weight": 0.70,
    },
    "profiling": {
        "primary": ["user_consent_granular", "ad_transparency", "data_minimization"],
        "secondary": ["anonymization", "privacy_dashboard", "privacy_labels"],
        "description": "Get consent for profiling and allow users to opt-out",
        "effectiveness_weight": 0.75,
    },
    "fingerprinting": {
        "primary": ["user_consent_granular", "ad_transparency"],
        "secondary": ["data_minimization", "privacy_labels"],
        "description": "Disclose fingerprinting and provide alternatives",
        "effectiveness_weight": 0.70,
    },
    "no_encryption": {
        "primary": ["encryption_at_rest", "encryption_in_transit", "end_to_end_encryption"],
        "secondary": ["device_encryption", "zero_knowledge_encryption"],
        "description": "Implement encryption for data at rest and in transit",
        "effectiveness_weight": 0.95,
    },
    "insecure_storage": {
        "primary": ["encryption_at_rest", "device_encryption"],
        "secondary": ["multi_factor_auth", "data_minimization"],
        "description": "Secure data storage with encryption and access controls",
        "effectiveness_weight": 0.90,
    },
    "dark_patterns": {
        "primary": ["user_consent_granular", "privacy_labels"],
        "secondary": ["ad_transparency", "privacy_dashboard"],
        "description": "Remove manipulative design and use clear language",
        "effectiveness_weight": 0.85,
    },
    "default_opt_in": {
        "primary": ["user_consent_granular", "privacy_dashboard"],
        "secondary": ["ad_transparency", "privacy_labels"],
        "description": "Use opt-in by default for privacy-sensitive features",
        "effectiveness_weight": 0.85,
    },
    "deceptive_interface": {
        "primary": ["ad_transparency", "privacy_labels", "user_consent_granular"],
        "secondary": ["privacy_dashboard"],
        "description": "Design clear, honest interfaces without deception",
        "effectiveness_weight": 0.90,
    },
    "no_audit_trail": {
        "primary": ["privacy_dashboard", "ad_transparency"],
        "secondary": ["data_deletion_user_control"],
        "description": "Maintain audit logs and allow users to review data access",
        "effectiveness_weight": 0.75,
    },
    "unclear_data_retention": {
        "primary": ["short_data_retention", "privacy_labels", "ad_transparency"],
        "secondary": ["data_deletion_user_control", "privacy_dashboard"],
        "description": "Clearly communicate and minimize data retention periods",
        "effectiveness_weight": 0.80,
    },
}

# Effectiveness scores for protective features (learned from positive sentiment data)
MITIGATION_EFFECTIVENESS: Dict[str, float] = {
    "differential_privacy": 0.75,
    "federated_learning": 0.80,
    "local_processing": 0.90,
    "privacy_by_design": 0.85,
    "encryption_at_rest": 0.95,
    "end_to_end_encryption": 0.95,
    "zero_knowledge_encryption": 0.90,
    "device_encryption": 0.90,
    "contextual_ads": 0.70,
    "data_minimization": 0.85,
    "age_verification": 0.75,
    "parental_controls": 0.80,
    "anonymization": 0.80,
    "data_deletion_user_control": 0.85,
    "privacy_dashboard": 0.85,
    "ephemeral_messages": 0.80,
    "short_data_retention": 0.80,
    "multi_factor_auth": 0.75,
    "encryption_in_transit": 0.90,
    "data_localization": 0.70,
    "user_consent_granular": 0.90,
    "ad_transparency": 0.80,
    "privacy_labels": 0.85,
}

# Combined risk-mitigation effectiveness (risk × mitigation → reduction factor).
# Higher values mean better risk reduction.
COMBINED_EFFECTIVENESS: Dict[tuple[str, str], float] = {
    # Always-on listening + mitigations
    ("always_on_listening", "local_processing"): 0.90,
    ("always_on_listening", "user_consent_granular"): 0.75,
    # Location tracking + mitigations
    ("location_tracking", "data_minimization"): 0.85,
    ("location_tracking", "short_data_retention"): 0.80,
    ("location_tracking", "anonymization"): 0.75,
    # Encryption fixes
    ("no_encryption", "encryption_at_rest"): 0.95,
    ("no_encryption", "end_to_end_encryption"): 0.95,
    ("insecure_storage", "encryption_at_rest"): 0.90,
    ("insecure_storage", "device_encryption"): 0.90,
    # Consent and transparency fixes
    ("forced_consent", "user_consent_granular"): 0.90,
    ("hidden_data_collection", "ad_transparency"): 0.85,
    ("misleading_privacy_notice", "privacy_labels"): 0.85,
    ("dark_patterns", "user_consent_granular"): 0.85,
    # Data broker mitigations
    ("data_broker_usage", "user_consent_granular"): 0.80,
    ("data_broker_usage", "ad_transparency"): 0.75,
    ("third_party_sharing", "user_consent_granular"): 0.80,
    # User control mitigations
    ("no_user_control", "privacy_dashboard"): 0.90,
    ("no_user_control", "data_deletion_user_control"): 0.85,
}


def get_mitigations_for_risk(risk_feature: str) -> MitigationRecommendation:
    """Get recommended mitigations for a specific risk feature.

    Args:
        risk_feature: The risky feature slug (e.g., 'always_on_listening')

    Returns:
        Dictionary with primary/secondary mitigations and metadata
    """
    if risk_feature not in MITIGATION_MAP:
        return {
            "risk_feature": risk_feature,
            "primary": [],
            "secondary": [],
            "description": "No specific mitigations mapped for this risk",
            "effectiveness_weight": 0.5,
        }

    template = MITIGATION_MAP[risk_feature]
    primary = list(template["primary"])
    secondary = list(template["secondary"])
    mitigation: MitigationRecommendation = {
        "risk_feature": risk_feature,
        "primary": primary,
        "secondary": secondary,
        "description": template["description"],
        "effectiveness_weight": template["effectiveness_weight"],
    }

    # Add individual effectiveness scores
    mitigation["primary_effectiveness"] = [
        {
            "feature": m,
            "effectiveness": MITIGATION_EFFECTIVENESS.get(m, 0.5),
            "combined": COMBINED_EFFECTIVENESS.get((risk_feature, m), 0.5),
        }
        for m in primary
    ]

    mitigation["secondary_effectiveness"] = [
        {
            "feature": m,
            "effectiveness": MITIGATION_EFFECTIVENESS.get(m, 0.5),
            "combined": COMBINED_EFFECTIVENESS.get((risk_feature, m), 0.5),
        }
        for m in secondary
    ]

    return mitigation


def get_all_mitigations_for_risks(risk_features: List[str]) -> Dict[str, Any]:
    """Get comprehensive mitigation recommendations for multiple risks.

    Args:
        risk_features: List of risky feature slugs

    Returns:
        Dictionary with aggregated mitigation recommendations
    """
    all_primary: Set[str] = set()
    all_secondary: Set[str] = set()
    risk_details: List[MitigationRecommendation] = []

    for risk in risk_features:
        mit = get_mitigations_for_risk(risk)
        risk_details.append(mit)
        all_primary.update(mit["primary"])
        all_secondary.update(mit["secondary"])

    # Prioritize: if a mitigation is primary for any risk, it's primary overall
    all_secondary = all_secondary - all_primary

    # Calculate aggregate effectiveness
    effectiveness_scores: Dict[str, float] = {}
    for mit_feature in all_primary | all_secondary:
        scores: List[float] = []
        for risk in risk_features:
            if risk in MITIGATION_MAP:
                # Check if this mitigation addresses this risk
                if mit_feature in MITIGATION_MAP[risk]["primary"]:
                    scores.append(COMBINED_EFFECTIVENESS.get((risk, mit_feature), 0.7))
                elif mit_feature in MITIGATION_MAP[risk]["secondary"]:
                    scores.append(COMBINED_EFFECTIVENESS.get((risk, mit_feature), 0.5))

        if scores:
            effectiveness_scores[mit_feature] = sum(scores) / len(scores)
        else:
            effectiveness_scores[mit_feature] = MITIGATION_EFFECTIVENESS.get(mit_feature, 0.5)

    # Sort by effectiveness
    sorted_primary = sorted(
        all_primary, key=lambda x: effectiveness_scores.get(x, 0.5), reverse=True
    )
    sorted_secondary = sorted(
        all_secondary, key=lambda x: effectiveness_scores.get(x, 0.5), reverse=True
    )

    return {
        "risk_features": risk_features,
        "risk_count": len(risk_features),
        "primary_mitigations": [
            {
                "feature": m,
                "effectiveness": effectiveness_scores.get(m, 0.5),
                "addresses_risks": [
                    r
                    for r in risk_features
                    if r in MITIGATION_MAP and m in MITIGATION_MAP[r]["primary"]
                ],
            }
            for m in sorted_primary
        ],
        "secondary_mitigations": [
            {
                "feature": m,
                "effectiveness": effectiveness_scores.get(m, 0.5),
                "addresses_risks": [
                    r
                    for r in risk_features
                    if r in MITIGATION_MAP and m in MITIGATION_MAP[r]["secondary"]
                ],
            }
            for m in sorted_secondary
        ],
        "per_risk_details": risk_details,
    }
