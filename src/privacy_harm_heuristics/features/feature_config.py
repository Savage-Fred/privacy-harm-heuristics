"""Configuration for product feature risk semantics.

Defines which product feature ontology slugs are considered risk increasing
vs. risk mitigating for root cause modeling. These lists are intentionally
curated (deterministic) and should be updated with care.
"""

from __future__ import annotations

RISK_INCREASING_FEATURES = {
    "always_on_listening",
    "webcam_access",
    "location_tracking",
    "cookies",
    "third_party_sharing",
    "biometric_collection",
    "behavioral_ads",
    "dark_patterns",
    "tracking_pixel",
    "session_replay",
    "cross_site_tracking",
    # New root cause features
    "data_broker_usage",
    "misleading_privacy_notice",
    "background_data_collection",
    "sensor_always_active",
    "forced_consent",
    "hidden_data_collection",
    "fourth_party_tracking",
    "unclear_data_retention",
    "excessive_permissions",
    "cross_context_data_use",
    "no_user_control",
    "opaque_algorithms",
    "profiling",
    "fingerprinting",
    "no_encryption",
    "insecure_storage",
    "default_opt_in",
    "notification_fatigue",
    "deceptive_interface",
    "no_audit_trail",
}

RISK_MITIGATING_FEATURES = {
    "differential_privacy",
    "federated_learning",
    "local_processing",
    "privacy_by_design",
    "encryption_at_rest",
    "end_to_end_encryption",
    "zero_knowledge_encryption",
    "device_encryption",
    "contextual_ads",
    "data_minimization",
    "age_verification",
    "parental_controls",
    "anonymization",
    "data_deletion_user_control",
    "privacy_dashboard",
    "ephemeral_messages",
    "short_data_retention",
    "multi_factor_auth",
    "encryption_in_transit",
    "data_localization",
    "user_consent_granular",
    "ad_transparency",
    "privacy_labels",
}

TOP_CO_OCCURRENCE_PAIRS = [
    # High-risk combinations based on common privacy incidents
    ("always_on_listening", "third_party_sharing"),
    ("webcam_access", "session_replay"),
    ("location_tracking", "behavioral_ads"),
    ("data_broker_usage", "third_party_sharing"),
    ("background_data_collection", "no_user_control"),
    ("forced_consent", "dark_patterns"),
    ("hidden_data_collection", "no_encryption"),
    ("misleading_privacy_notice", "cross_context_data_use"),
    ("profiling", "opaque_algorithms"),
    ("fingerprinting", "cross_site_tracking"),
]
