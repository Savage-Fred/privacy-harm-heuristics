from __future__ import annotations

import json
from pathlib import Path

import pytest

from privacy_harm_heuristics.evals.llm_eval import LLMEvalConfig, run_llm_eval


def test_llm_eval_with_fake_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Prepare a tiny input JSONL
    input_path = tmp_path / "input.jsonl"
    records = [
        {
            "id": 1,
            "description": "A serious data breach exposed records",
            "harm_category": "insecurity",
        },
        {
            "id": 2,
            "description": "The app tracks users via geofencing",
            "harm_category": "surveillance",
        },
    ]
    with input_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Fake complete() to output the first word (pretend it's a label)
    def fake_complete(
        prompt: str, *, provider=None, model=None, max_tokens: int = 64
    ) -> str:  # noqa: ARG001
        # simple heuristic: choose label keyword based on prompt content
        if "breach" in prompt.lower():
            return "insecurity"
        if "geofenc" in prompt.lower() or "track" in prompt.lower():
            return "surveillance"
        return "unknown"

    # Patch the complete function used by the evaluator
    monkeypatch.setattr("privacy_harm_heuristics.evals.llm_eval.complete", fake_complete)

    out_path = tmp_path / "judgments.jsonl"
    cfg = LLMEvalConfig(
        in_path=input_path,
        out_path=out_path,
        text_field="description",
        target_field="harm_category",
        labels=["insecurity", "surveillance"],
        provider="fallback",
        model=None,
        sample=None,
        seed=123,
        max_tokens=32,
    )

    metrics = run_llm_eval(cfg)

    # Check outputs
    assert metrics["total"] == 2
    assert pytest.approx(metrics["accuracy"], rel=0) == 1.0
    # Ensure judgments file has two lines
    with out_path.open("r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 2
