"""Model-pinning: resolve_model_name must name the concrete model a run uses.

Dec 2025's runs recorded only ``provider: gemini`` and left the model unpinned
(see data/experiments/rerun_20260721/README.md). resolve_model_name is what
callers persist into results metadata so no future run is ambiguous.
"""

from __future__ import annotations

from privacy_harm_heuristics.llm.provider import (
    DEFAULT_GEMINI_MODEL,
    resolve_model_name,
)


def test_fallback_resolves_to_fallback(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_model_name("fallback") == "fallback"


def test_gemini_default_is_explicit(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert resolve_model_name("gemini") == DEFAULT_GEMINI_MODEL


def test_gemini_env_override(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-pinned")
    assert resolve_model_name("gemini") == "gemini-2.5-flash-pinned"


def test_explicit_model_hint_wins(monkeypatch):
    assert resolve_model_name("gemini", "gemini-3.0-flash") == "gemini-3.0-flash"


def test_openai_default(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert resolve_model_name("openai") == "gpt-4o-mini"
