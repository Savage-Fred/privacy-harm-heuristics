from __future__ import annotations

import os
from typing import List, Optional

__doc__ = """Lightweight Anthropic client wrapper.

Provides optional summarization via Claude with graceful offline fallbacks so
the broader pipeline remains deterministic when the SDK / API key are absent.
"""

try:  # pragma: no cover - optional dependency
    from anthropic import Anthropic  # type: ignore
except Exception:  # broad to avoid any import-time issues
    Anthropic = None  # type: ignore

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")


class AnthropicNotAvailable(Exception):
    pass


def get_client():
    """Return an Anthropic client or raise if library/key missing."""
    if Anthropic is None:
        raise AnthropicNotAvailable(
            "anthropic package not installed. Add to requirements and pip install."
        )
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise AnthropicNotAvailable("ANTHROPIC_API_KEY not set in environment.")
    return Anthropic(api_key=api_key)


def summarize_text(
    text: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 256,
    system: Optional[str] = None,
) -> str:
    """Summarize a block of text using Claude Sonnet.

    Gracefully falls back to returning the first N characters if Anthropic is not configured.
    """
    model_name = model or DEFAULT_MODEL
    try:
        client = get_client()
    except AnthropicNotAvailable:
        # Fallback deterministic simple summary (truncate) so pipeline still works offline
        return (
            (text[:max_tokens].rsplit(" ", 1)[0] + ("…" if len(text) > max_tokens else ""))
            if text
            else ""
        )

    system_prompt = system or (
        "You are a concise assistant that produces a 1-2 sentence privacy harm summary."
    )
    # New Messages API (best-effort compatibility)
    try:
        msg = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize the following incident in 2 sentences focusing on privacy impacts.\n\n{text}",
                }
            ],
            temperature=0.2,
        )
    except Exception:
        # Network/model error fallback deterministic truncation
        return (
            (text[:max_tokens].rsplit(" ", 1)[0] + ("…" if len(text) > max_tokens else ""))
            if text
            else ""
        )
    parts: List[str] = []
    for block in getattr(msg, "content", []):
        if hasattr(block, "text"):
            parts.append(getattr(block, "text", ""))
        elif isinstance(block, dict) and block.get("type") == "text":  # defensive
            parts.append(block.get("text", ""))
    return " ".join(p.strip() for p in parts if p.strip())
