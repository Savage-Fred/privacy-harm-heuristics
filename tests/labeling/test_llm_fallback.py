from __future__ import annotations

import json
from pathlib import Path

from privacy_harm_heuristics.labeling.harm_labeler import label_jsonl_file


def test_label_jsonl_file_uses_llm_for_unknown(tmp_path: Path, monkeypatch):
    sample = {
        "id": 1,
        "description": "marmalade aurora festival with origami sculptures and blueberry tastings",
    }
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "labeled.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(sample) + "\n")

    calls: list[tuple[str | None, str | None]] = []

    # **kwargs absorbs newer call-site kwargs (e.g. tools=...) so the mock
    # doesn't silently break again when harm_labeler grows another parameter.
    def _fake_classify(text: str, *, provider=None, model=None, max_tokens=256, **kwargs):  # type: ignore[override]
        calls.append((provider, model))
        return {
            "is_privacy_relevant": True,
            "harms": ["surveillance"],
            "rationale": "LLM fallback",
            "confidence": 0.91,
        }

    monkeypatch.setattr(
        "privacy_harm_heuristics.labeling.harm_labeler.classify_privacy_relevance",
        _fake_classify,
    )

    processed = label_jsonl_file(
        str(input_path),
        str(output_path),
        overwrite=True,
        llm_provider="openai",
        llm_models=["gpt-5.1-mini"],
        llm_max_tokens=512,
    )
    assert processed == 1
    assert calls and calls[0][0] == "openai"

    with output_path.open("r", encoding="utf-8") as handle:
        labeled = json.loads(handle.readline())

    assert labeled["harm_category"] == "surveillance"
    assert labeled["harm_category_source"] == "llm"
    assert labeled["harm_categories_llm"] == ["surveillance"]
    assert labeled["harm_category_model_llm"] == "gpt-5.1-mini"
