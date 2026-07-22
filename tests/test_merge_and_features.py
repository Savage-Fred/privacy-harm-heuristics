import json
from pathlib import Path

from privacy_harm_heuristics.features.builder import build_features
from privacy_harm_heuristics.utils.merge import merge_jsonl


def test_merge_jsonl(tmp_path: Path):
    f1 = tmp_path / "a.jsonl"
    f2 = tmp_path / "b.jsonl"
    f1.write_text('{"source":"x","incident_date":"09/01/2025"}\n', encoding="utf-8")
    f2.write_text('{"source":"y","incident_date":1695000000}\n', encoding="utf-8")
    out = tmp_path / "merged.jsonl"
    n = merge_jsonl([str(f1), str(f2)], str(out))
    assert n == 2
    lines = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert any(r.get("incident_date_canonical") == "2025-09-01" for r in lines)


def test_feature_builder_user_signals():
    rec = {
        "source": "hhs_ocr",
        "penalty_amount": 5000,
        "individuals_affected": 1000,
        "description": "breach",
    }
    feats = build_features(rec)
    assert "f_has_penalty" not in feats
    assert "f_penalty_log" not in feats
    assert "f_penalty_bucket" not in feats
    assert feats["f_individuals_log"] > 0
    assert feats["f_has_description"] == 1
    # Keyword flags should reflect description
    kw_keys = [k for k in feats if k.startswith("kw_")]
    assert any(feats[k] in (0, 1) for k in kw_keys)


def test_keyword_flags_detection(tmp_path):
    rec = {
        "source": "hn",
        "type": "story",
        "id": "42",
        "created_date": "2024-01-01",
        "description": "Facial recognition camera tracking users",
        "raw": {},
    }
    feats = build_features(rec)
    # Expect multiple keyword groups to trigger
    assert feats.get("kw_privacy") == 1
    assert feats.get("kw_video_surveillance") == 1
    assert feats.get("kw_biometric") == 1
