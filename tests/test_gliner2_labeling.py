import json

import pytest

# Even with a fake model, building the labeling schema requires the real
# gliner2 package (Schema comes from gliner2.inference.engine). It's an
# optional heavy NLP dependency not present in the CI install surface.
pytest.importorskip("gliner2", reason="optional gliner2 package not installed")

from privacy_harm_heuristics.nlp import gliner2_labeling as g  # noqa: E402
from privacy_harm_heuristics.nlp.gliner2_projection import (
    project_gliner2_to_record_fields,
)  # noqa: E402


class _FakeGLiNER2:
    """Minimal fake GLiNER2 for tests (no HF download)."""

    def __init__(self, results):
        self._results = results

    def batch_extract(
        self,
        texts,
        schema,
        batch_size=8,
        threshold=0.5,
        format_results=True,
        include_confidence=False,
    ):
        # Ignore schema and thresholds; just echo pre-baked results per text.
        return list(self._results)[: len(texts)]


def test_label_texts_with_fake_model(monkeypatch):
    fake_results = [
        {
            "text_labels": [
                {"label": "harm_surveillance", "confidence": 0.9},
                {"label": "rc_no_mitigation", "confidence": 0.8},
            ],
            "entities": {
                "pf_dark_pattern": ["dark pattern"],
                "data_recipient": ["advertisers"],
            },
        }
    ]

    monkeypatch.setattr(g, "_load_gliner2_model", lambda model_name: _FakeGLiNER2(fake_results))

    out = g.label_texts(["Example text about tracking"], schema_spec=None)
    assert len(out) == 1
    rec = out[0]
    assert "harm_surveillance" in rec["text_labels"]
    assert rec["text_label_scores"]["harm_surveillance"] == 0.9
    # Span labels flattened
    span_labels = {(s["label"], s["span"]) for s in rec["span_labels"]}
    assert ("pf_dark_pattern", "dark pattern") in span_labels
    assert ("data_recipient", "advertisers") in span_labels


def test_label_texts_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(g, "_load_gliner2_model", lambda model_name: None)
    out = g.label_texts(["anything"])
    assert out == [
        {"text_labels": [], "text_label_scores": {}, "span_labels": []},
    ]


def test_annotate_jsonl_with_gliner2_streams_and_writes(tmp_path, monkeypatch):
    # Prepare a small JSONL file
    in_path = tmp_path / "input.jsonl"
    records = [
        {"id": "1", "description": "Camera always listening, dark patterns in consent"},
        {"id": "2", "description": "Open database with no password"},
    ]
    with in_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    fake_results = [
        {
            "text_labels": [{"label": "harm_surveillance", "confidence": 0.95}],
            "entities": {"pf_dark_pattern": ["dark pattern"]},
        },
        {
            "text_labels": [{"label": "harm_insecurity", "confidence": 0.9}],
            "entities": {},
        },
    ]

    monkeypatch.setattr(g, "_load_gliner2_model", lambda model_name: _FakeGLiNER2(fake_results))

    out_path = tmp_path / "out.jsonl"
    cfg = g.GLiNER2Config(model_name="fake/model", text_field="description", batch_size=2)
    total, annotated = g.annotate_jsonl_with_gliner2(
        str(in_path), str(out_path), config=cfg, overwrite=True
    )

    assert total == 2
    assert annotated == 2

    # Verify augmented JSONL contents
    with out_path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    for rec in lines:
        assert "gliner2_labels" in rec
        assert rec["gliner2_labels"]["text_labels"]


def test_project_gliner2_to_record_fields_hypothesis_mapping():
    record = {
        "gliner2_labels": {
            "text_labels": ["harm_surveillance", "rc_no_mitigation", "rc_inadequate_security"],
            "span_labels": [
                {"span": "dark patterns", "label": "pf_dark_pattern", "score": None},
                {
                    "span": "third party data brokers",
                    "label": "pf_third_party_sharing",
                    "score": None,
                },
            ],
        }
    }

    project_gliner2_to_record_fields(record)

    # harm_* → harm_categories
    assert "harm_categories" in record
    assert "harm_surveillance" in record["harm_categories"]

    # rc_* and pf_* → root_cause_features
    assert "root_cause_features" in record
    for lab in [
        "rc_no_mitigation",
        "rc_inadequate_security",
        "pf_dark_pattern",
        "pf_third_party_sharing",
    ]:
        assert lab in record["root_cause_features"]

    # pf_* → product_features (stripped prefix)
    assert "product_features" in record
    for slug in ["dark_pattern", "third_party_sharing"]:
        assert slug in record["product_features"]
