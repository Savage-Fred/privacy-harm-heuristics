"""Unified record schema definitions.

Provides a Pydantic model to validate normalized records across sources.

Design notes:
* Keep fields optional where some sources legitimately lack them.
* Preserve `raw` as an unvalidated dict (arbitrary types) but ensure it's present.
* Do not coerce/transform here beyond basic type validation; upstream connectors
  should already format ISO date strings.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

CORE_SOURCE_TYPES = {
    "reddit",
    "hhs_ocr",
    "ftc",
    "sec",
    "hn",
    "amo",
    "wikipedia",
    "mastodon",
    "kaggle",
    "state_ca",
    "state_ny",
    "gdpr_enforcement",
    "global_privacy_action",
    # Voice of Customer sources
    "cfpb",
    "nvd",
    "apple_appstore",
    "google_play",
    "trustpilot",
    "g2",
    "bbb",
    "capterra",
    "bluesky",
    "stackexchange",
    "rss_feed",
    "threads",
    "chrome_webstore",
    "edpb",
    "privacy_intl",
    "hibp",
    "arxiv",
    "databreaches_net",
    "tech_news",
    "eff",
    "krebs",
    "schneier",
    "github_advisory",
    "cisa_kev",
    "urlhaus",
    "phishtank",
    "malwarebazaar",
    "threatfox",
    "ransomlook",
    "noyb",
    "tosdr",
    # Additional state breach portals
    "state_tx",
    "state_ma",
    "state_fl",
    "state_il",
    "state_wa",
    "state_co",
}


class HarmCategory(str, Enum):
    """Solove-aligned harm categories for privacy violations."""

    INFORMATION_COLLECTION = "information_collection"
    INFORMATION_PROCESSING = "information_processing"
    INFORMATION_DISSEMINATION = "information_dissemination"
    INVASION = "invasion"


class DataSensitivity(str, Enum):
    """Data sensitivity classification levels."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SPECIAL = "special"


class Record(BaseModel):
    """Canonical normalized record across heterogeneous privacy sources.

    Only core interoperability / analytical fields are enumerated; connectors
    may supply additional fields (``extra=allow``) without breaking older
    datasets. ``raw`` retains the original source object for transparency.
    """

    model_config = ConfigDict(extra="allow")  # allow domain-specific extension fields

    source: str = Field(..., description="Short source slug")
    type: str = Field(..., description="Domain record type")
    id: str = Field(..., description="Stable record id")
    created_date: Optional[str] = Field(None, description="ISO 8601 timestamp/date string")
    incident_date: Optional[str] = Field(None, description="Original incident date representation")
    incident_date_canonical: Optional[str] = Field(
        None, description="Derived canonical incident date YYYY-MM-DD"
    )
    description: Optional[str] = None
    harm_summary: Optional[str] = None
    penalty_amount: Optional[float] = None
    jurisdiction: Optional[str] = Field(None, description="Jurisdiction code (e.g., US-CA, EU)")
    regulatory_body: Optional[str] = Field(
        None, description="Regulator or governing body (e.g., FTC, SEC, ICO)"
    )
    company: Optional[str] = Field(None, description="Company/entity involved")
    product_name: Optional[str] = Field(None, description="Product or service name if extractable")
    # Sentiment analysis fields
    sentiment_score: Optional[float] = Field(
        None, description="Sentiment score from -1 (negative) to 1 (positive)"
    )
    sentiment_label: Optional[str] = Field(
        None, description="Sentiment label: positive, negative, neutral"
    )
    # Privacy harm classification fields
    incident_type: Optional[str] = Field(None, description="Primary type of privacy incident")
    harm_categories: Optional[list[str]] = Field(
        default_factory=list, description="List of privacy harm categories"
    )
    harm_severity: Optional[str] = Field(
        None, description="Severity level: low, medium, high, critical"
    )
    harm_score: Optional[float] = Field(
        None,
        description="Overall aggregated harm score in [0,1] derived from classifier/confidence",
    )
    harm_category_scores: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Per-category harm confidence scores in [0,1] (multi-label)",
    )
    classification_version: Optional[str] = Field(
        None,
        description="Version identifier for harm classification model/rules used",
    )
    categories_extracted_from: Optional[str] = Field(
        None,
        description="Source text field used for category extraction (e.g., description, raw.text)",
    )
    # Additional temporal and context fields
    discovery_date: Optional[str] = Field(None, description="Date when incident was discovered")
    notification_date: Optional[str] = Field(
        None, description="Date when affected parties were notified"
    )

    # Root cause and causal factor fields - CRITICAL for predictive modeling
    # These capture WHAT caused the harm, not just the outcome
    root_cause_features: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "Specific product features/design choices that caused the harm (e.g., "
            "'always_on_listening', 'third_party_data_broker')"
        ),
    )
    product_components: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "Product components involved in the incident (e.g., 'voice_assistant', "
            "'location_services', 'ad_network')"
        ),
    )
    third_party_services: Optional[list[str]] = Field(
        default_factory=list,
        description="Third-party services, data brokers, or integrations involved",
    )
    ux_design_issues: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "UX/design problems identified (e.g., 'dark_patterns', 'misleading_consent', "
            "'hidden_settings')"
        ),
    )
    data_practices: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "Specific data collection/handling practices (e.g., 'excessive_collection', "
            "'unclear_retention', 'cross_context_use')"
        ),
    )
    technical_implementation_issues: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "Technical implementation problems (e.g., 'always_on_sensor', "
            "'no_local_processing', 'insecure_storage')"
        ),
    )

    # Multi-dimensional harm tracking fields
    harm_categories_solove: Optional[List[HarmCategory]] = Field(
        default_factory=list,
        description="Solove-aligned harm categories identified via weak supervision",
    )
    contextual_mismatch: bool = Field(
        default=False,
        description=(
            "Whether contextual integrity violation detected (collection context != " "use context)"
        ),
    )
    data_sensitivity_level: Optional[DataSensitivity] = Field(
        None,
        description="Classified data sensitivity level based on data types involved",
    )
    user_distress_indicators: Optional[List[str]] = Field(
        default_factory=list,
        description="User distress/emotional harm indicators extracted from text",
    )
    synthetic_flag: bool = Field(
        default=False,
        description="True if record contains synthetic/augmented data for balancing",
    )
    label_provenance: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Provenance metadata tracking which labeling functions were applied",
    )

    raw: Dict[str, Any]

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Ensure ``source`` field is within allowed CORE_SOURCE_TYPES."""
        if v and v not in CORE_SOURCE_TYPES:
            # Allow future expansion but warn via exception to catch typos early
            # Consider relaxing once dynamic registry added.
            raise ValueError(f"Unknown source '{v}' (expected one of {sorted(CORE_SOURCE_TYPES)})")
        return v

    @field_validator("penalty_amount")
    @classmethod
    def non_negative_penalty(cls, v: Optional[float]) -> Optional[float]:
        """Validate penalty is non-negative when provided."""
        if v is not None and v < 0:
            raise ValueError("penalty_amount must be non-negative")
        return v

    @field_validator("sentiment_score")
    @classmethod
    def validate_sentiment_score(cls, v: Optional[float]) -> Optional[float]:
        """Validate sentiment score is between -1 and 1."""
        if v is not None and not (-1.0 <= v <= 1.0):
            raise ValueError("sentiment_score must be between -1.0 and 1.0")
        return v

    @field_validator("sentiment_label")
    @classmethod
    def validate_sentiment_label(cls, v: Optional[str]) -> Optional[str]:
        """Validate sentiment label is one of accepted values."""
        if v is not None and v not in {"positive", "negative", "neutral"}:
            raise ValueError("sentiment_label must be 'positive', 'negative', or 'neutral'")
        return v

    @field_validator("harm_severity")
    @classmethod
    def validate_harm_severity(cls, v: Optional[str]) -> Optional[str]:
        """Validate harm severity is one of accepted levels."""
        if v is not None and v not in {"low", "medium", "high", "critical"}:
            raise ValueError("harm_severity must be 'low', 'medium', 'high', or 'critical'")
        return v

    @field_validator("harm_score")
    @classmethod
    def validate_harm_score(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("harm_score must be between 0.0 and 1.0")
        return v

    @field_validator("harm_category_scores")
    @classmethod
    def validate_harm_category_scores(
        cls, v: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        if v is not None:
            for k, score in v.items():
                if not (0.0 <= float(score) <= 1.0):
                    raise ValueError(f"harm_category_scores[{k}] score {score} not in [0,1]")
        return v


def validate_record_dict(rec: Dict[str, Any]) -> Record:
    """Validate and return a Record model for a normalized record dict.

    Raises pydantic.ValidationError on failure.
    """
    return Record(**rec)


def is_valid_record(rec: Dict[str, Any]) -> bool:
    """Return True if dict validates against ``Record`` schema."""
    try:
        validate_record_dict(rec)
        return True
    except Exception:
        return False
