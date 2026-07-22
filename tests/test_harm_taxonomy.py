"""Tests for weak supervision labeling functions."""

from __future__ import annotations

from privacy_harm_heuristics.processing.harm_taxonomy import apply_labeling_functions
from privacy_harm_heuristics.schema import DataSensitivity, HarmCategory, Record


def test_distress_indicators_detected():
    """Test that user distress indicators are correctly identified."""
    record = Record(  # type: ignore
        source="reddit",
        type="comment",
        id="test-1",
        description="I'm anxious about this breach. I stopped using the service because I felt violated.",
        raw={},
    )

    enriched = apply_labeling_functions(record)

    assert enriched.user_distress_indicators is not None
    assert len(enriched.user_distress_indicators) > 0
    assert any("anxious" in ind or "violated" in ind for ind in enriched.user_distress_indicators)
    assert "LF_DISTRESS" in (enriched.label_provenance or {})


def test_sensitive_data_classification_special():
    """Test classification of special category data."""
    record = Record(  # type: ignore
        source="hhs_ocr",
        type="breach",
        id="test-2",
        description="Breach of health records including biometric data and genetic information.",
        raw={},
    )

    enriched = apply_labeling_functions(record)

    assert enriched.data_sensitivity_level == DataSensitivity.SPECIAL
    prov = enriched.label_provenance or {}
    assert "LF_SENSITIVE_DATA" in prov
    assert "matched_terms" in prov.get("LF_SENSITIVE_DATA", {})


def test_sensitive_data_classification_high():
    """Test classification of high sensitivity data."""
    record = Record(  # type: ignore
        source="ftc",
        type="enforcement",
        id="test-3",
        description="Company exposed credit card numbers and passwords in breach.",
        raw={},
    )

    enriched = apply_labeling_functions(record)

    assert enriched.data_sensitivity_level == DataSensitivity.HIGH
    assert "LF_SENSITIVE_DATA" in (enriched.label_provenance or {})


def test_contextual_mismatch_detected():
    """Test detection of contextual integrity violations."""
    record = Record(  # type: ignore
        source="reddit",
        type="post",
        id="test-4",
        description="They repurposed my data without consent for unexpected secondary use.",
        raw={},
    )

    enriched = apply_labeling_functions(record)

    assert enriched.contextual_mismatch is True
    assert "LF_CONTEXT_MISMATCH" in (enriched.label_provenance or {})


def test_harm_category_dissemination():
    """Test detection of information dissemination harms."""
    record = Record(  # type: ignore
        source="hn",
        type="story",
        id="test-5",
        description="User data was publicly posted and exposed online, leading to doxxing.",
        raw={},
    )

    enriched = apply_labeling_functions(record)

    assert enriched.harm_categories_solove is not None
    assert HarmCategory.INFORMATION_DISSEMINATION in enriched.harm_categories_solove
    assert "LF_HARM_CATEGORIES" in (enriched.label_provenance or {})


def test_harm_category_collection():
    """Test detection of information collection harms."""
    record = Record(  # type: ignore
        source="reddit",
        type="post",
        id="test-6",
        description="Surveillance cameras are tracking and monitoring users without disclosure.",
        raw={},
    )

    enriched = apply_labeling_functions(record)

    assert enriched.harm_categories_solove is not None
    assert HarmCategory.INFORMATION_COLLECTION in enriched.harm_categories_solove


def test_harm_category_processing():
    """Test detection of information processing harms."""
    record = Record(  # type: ignore
        source="ftc",
        type="enforcement",
        id="test-7",
        description="Automated profiling and algorithmic scoring without transparency.",
        raw={},
    )

    enriched = apply_labeling_functions(record)

    assert enriched.harm_categories_solove is not None
    assert HarmCategory.INFORMATION_PROCESSING in enriched.harm_categories_solove


def test_harm_category_invasion():
    """Test detection of invasion harms."""
    record = Record(  # type: ignore
        source="reddit",
        type="complaint",
        id="test-8",
        description="Dark patterns and deceptive design manipulated users into unwanted actions.",
        raw={},
    )

    enriched = apply_labeling_functions(record)

    assert enriched.harm_categories_solove is not None
    assert HarmCategory.INVASION in enriched.harm_categories_solove


def test_multiple_harms_detected():
    """Test that multiple harm categories can be detected simultaneously."""
    record = Record(
        source="ftc",
        type="enforcement",
        id="test-9",
        description=(
            "Company tracked users' health data through surveillance, "
            "then disclosed it publicly without consent causing anxiety."
        ),
        raw={},
    )

    enriched = apply_labeling_functions(record)

    # Should detect multiple harm categories
    assert enriched.harm_categories_solove is not None
    assert len(enriched.harm_categories_solove) >= 2
    # Should detect special category data (health)
    assert enriched.data_sensitivity_level == DataSensitivity.SPECIAL
    # Should detect contextual mismatch (without consent)
    assert enriched.contextual_mismatch is True
    # Should detect user distress
    assert len(enriched.user_distress_indicators) > 0


def test_no_false_positives_on_neutral_text():
    """Test that neutral text doesn't trigger false alarms."""
    record = Record(
        source="reddit",
        type="post",
        id="test-10",
        description="The company announced a new privacy policy update.",
        raw={},
    )

    enriched = apply_labeling_functions(record)

    # Should not detect much with neutral text
    assert len(enriched.user_distress_indicators or []) == 0
    assert enriched.contextual_mismatch is False
    assert len(enriched.harm_categories_solove or []) == 0


def test_provenance_tracking():
    """Test that provenance is properly tracked for all labeling functions."""
    record = Record(
        source="hhs_ocr",
        type="breach",
        id="test-11",
        description="Exposed biometric data was leaked, causing users to feel violated.",
        raw={},
    )

    enriched = apply_labeling_functions(record)

    # Should have provenance for multiple LFs
    assert enriched.label_provenance is not None
    assert len(enriched.label_provenance) > 0

    # Each LF that fires should have metadata
    if "LF_DISTRESS" in enriched.label_provenance:
        assert "matches" in enriched.label_provenance["LF_DISTRESS"]
    if "LF_SENSITIVE_DATA" in enriched.label_provenance:
        assert "level" in enriched.label_provenance["LF_SENSITIVE_DATA"]
        assert "matched_terms" in enriched.label_provenance["LF_SENSITIVE_DATA"]


def test_text_from_raw_field():
    """Test that labeling works with text in raw field."""
    record = Record(
        source="reddit",
        type="post",
        id="test-12",
        description="Short description",
        raw={
            "body": "I'm really worried about my health data being exposed in this breach.",
            "title": "Major privacy concern",
        },
    )

    enriched = apply_labeling_functions(record)

    # Should detect distress from raw.body
    assert len(enriched.user_distress_indicators or []) > 0
    # Should detect special category data from raw.body
    assert enriched.data_sensitivity_level == DataSensitivity.SPECIAL
