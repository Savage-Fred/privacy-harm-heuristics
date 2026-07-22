"""PII / content-scrub guarantees for the committed data samples.

Two kinds of assertion:

1. Schema-level, over EVERY row of both committed samples — the forbidden
   identity/content fields must never reappear. This runs in CI forever and is
   the standing guard that a future regeneration can't silently re-leak handles,
   raw source records, or verbatim posts.
2. Unit tests for the pure ``scrub_row`` function in ``scripts/make_samples.py``
   (field removal + the email/phone/handle regex backstop).

See ``scripts/make_samples.py`` for the full kept/dropped field inventory.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = [
    REPO_ROOT / "data" / "with_features.sample.jsonl",
    REPO_ROOT / "data" / "labeled_gliner2.sample.jsonl",
]

# Identity / raw-content field names that must never appear in a committed sample.
FORBIDDEN_FIELDS = {
    "raw",
    "author",
    "username",
    "account_username",
    "account_display_name",
    "account_id",
    "account_bot",
    "account_url",
    "reporter",
    "submission_id",
    "link_id",
    "parent_id",
    "url",
    "source_url",
    "article_url",
    "permalink",
    "uri",
    "xbrl_archive",
    "logo_path",
    "disclosureurl",
    "selftext",
    "content_html",
    "body",
    "title",
    "description",
    "web_description",
    "summary",
    "harm_summary",
    "content_excerpt",
    "company_response",
    "search_query",
    "location",
    "zip_code",
    "domain",
    "instance",
}


def _load_scrub_module():
    """Import scripts/make_samples.py (not a package) by file path."""
    path = REPO_ROOT / "scripts" / "make_samples.py"
    spec = importlib.util.spec_from_file_location("_make_samples", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_samples = _load_scrub_module()
scrub_row = make_samples.scrub_row


def _iter_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line:
                yield i, json.loads(line)


# ---------------------------------------------------------------------------
# 1. Schema-level guard over the committed samples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sample_path", SAMPLES, ids=lambda p: p.name)
def test_committed_sample_has_no_forbidden_fields(sample_path):
    assert sample_path.exists(), f"missing committed sample: {sample_path}"
    n = 0
    for i, row in _iter_rows(sample_path):
        n += 1
        leaked = FORBIDDEN_FIELDS.intersection(row.keys())
        assert not leaked, f"{sample_path.name} row {i} leaks forbidden field(s): {sorted(leaked)}"
    assert n > 0, f"{sample_path.name} is empty"


@pytest.mark.parametrize("sample_path", SAMPLES, ids=lambda p: p.name)
def test_committed_sample_has_no_pii_regex_hits(sample_path):
    """No email / fediverse handle / reddit permalink / phone anywhere in a row."""
    for i, row in _iter_rows(sample_path):
        blob = json.dumps(row, ensure_ascii=False)
        assert "reddit.com/r/" not in blob, f"{sample_path.name} row {i}: reddit permalink"
        assert not make_samples._value_has_pii(
            row
        ), f"{sample_path.name} row {i}: value matches email/phone/handle regex"


# ---------------------------------------------------------------------------
# 2. Unit tests for the pure scrub function
# ---------------------------------------------------------------------------


def test_scrub_drops_raw_and_identity_fields():
    row = {
        "id": "abc",
        "harm_category": "harm_surveillance",
        "kw_privacy": 1,
        "pf_location_tracking": 1,
        "author": "LaylaRileyThrowaway",
        "selftext": "Location: New Haven, CT ... my divorce ...",
        "permalink": "/r/legaladvice/comments/1os5dkn/divorce_in_ct/",
        "raw": {"author": "LaylaRileyThrowaway", "selftext_html": "<p>...</p>"},
    }
    out = scrub_row(row)
    assert out == {
        "id": "abc",
        "harm_category": "harm_surveillance",
        "kw_privacy": 1,
        "pf_location_tracking": 1,
    }
    # pure: input not mutated
    assert "author" in row and "raw" in row


def test_scrub_keeps_engineered_features_and_labels():
    row = {
        "kw_biometric": 1,
        "pf_biometric_collection": 1,
        "rc_net_risk": 0.5,
        "geo_us_state_CA": 1,
        "f_has_description": 1,
        "harm_category": "harm_insecurity",
        "harm_category_source": "gliner2",
        "source": "reddit_enhanced",
        "source_type": "social",
    }
    assert scrub_row(row) == row  # nothing dropped


def test_scrub_regex_backstop_drops_email_handle_phone():
    row = {
        "harm_category": "unknown",
        "notes": "reach me at jane.doe@example.com",
        "handle_field": "@throws_lemy@lemmy.nz",
        "phone_field": "call 415-555-0132",
        "clean": "no pii here",
        "count": 3,
    }
    out = scrub_row(row)
    assert "notes" not in out
    assert "handle_field" not in out
    assert "phone_field" not in out
    assert out["clean"] == "no pii here"
    assert out["count"] == 3
    assert out["harm_category"] == "unknown"


def test_scrub_regex_backstop_recurses_into_nested_values():
    row = {
        "harm_category": "unknown",
        "nested": {"deep": ["ok", "mail me user@host.org"]},
        "kept": {"deep": ["all", "clean"]},
    }
    out = scrub_row(row)
    assert "nested" not in out
    assert out["kept"] == {"deep": ["all", "clean"]}


def test_scrub_does_not_flag_plain_numbers_or_ids():
    # bare digit runs (ids, counts, years) must NOT trip the phone regex
    row = {"id": "202511010001", "individuals_affected": 12345, "year": "2025"}
    assert scrub_row(row) == row
