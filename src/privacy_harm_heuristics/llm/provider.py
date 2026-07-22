"""Unified LLM provider utilities (OpenAI, Gemini, LLaMA, fallback).

Selection precedence:
1. Explicit provider argument
2. LLM_PROVIDER env var
3. First available among: openai, gemini (all require package + API key)
4. Offline fallback (deterministic truncation)

Environment variables:
    OPENAI_API_KEY, OPENAI_MODEL (default gpt-4o-mini)
    GEMINI_API_KEY, GEMINI_MODEL (default gemini-2.5-flash)
    GEMINI_HEAVY_MODEL (default gemini-2.5-pro for long/complex prompts)
    LLAMA_MODEL_PATH (filesystem path to GGUF model for llama-cpp)
    LLAMA_THREADS (optional int; defaults to os.cpu_count())
    LLAMA_CTX (optional int context window; default 4096)
    LLAMA_TOP_P (optional float; default 0.9)
    LLAMA_TOP_K (optional int; default 40)
    LLAMA_TEMPERATURE (optional float; default 0.2)
    LLAMA_PROMPT_TEMPLATE (optional; uses {text} placeholder)
    LLM_PROVIDER = openai|gemini|llama|fallback

All summarize functions return a short 1-2 sentence privacy impact summary.
Always safe to call offline (falls back to truncation without raising).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..constants.privacy_taxonomy import (
    FALLBACK_RELEVANCE_KEYWORDS,
    keyword_fallback_categories,
    solove_categories,
    taxonomy_prompt_block,
)

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    pass  # type: ignore

logger = logging.getLogger(__name__)

# Gemini model defaults - can override via GEMINI_MODEL / GEMINI_HEAVY_MODEL env vars
# gemini-2.0-flash has 2,000 RPM but can produce malformed JSON
# gemini-2.5-flash has 1,000 RPM with better output quality (preferred)
# gemini-2.5-pro has 150 RPM but best reasoning (use for complex tasks)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_HEAVY_MODEL = "gemini-2.5-pro"


def _resolve_gemini_model(model_hint: Optional[str], *, heavy: bool = False) -> str:
    """Return the Gemini model name based on hints and environment overrides."""
    if model_hint:
        hint_lower = model_hint.lower()
        if hint_lower in {"heavy", "pro", "gemini-pro"}:
            return os.getenv("GEMINI_HEAVY_MODEL", DEFAULT_GEMINI_HEAVY_MODEL)
        return model_hint
    if heavy:
        return os.getenv("GEMINI_HEAVY_MODEL", DEFAULT_GEMINI_HEAVY_MODEL)
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def resolve_model_name(provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """Resolve the concrete model string a call would use, without calling out.

    This is what pins a run: Dec 2025's live runs recorded only
    ``provider: gemini`` and left the actual model unpinned (see
    ``data/experiments/rerun_20260721/README.md``). Callers persist this value
    into results metadata so every future run names the exact model. Kept in
    sync with the model selection inside ``complete()`` / ``summarize()``.
    """
    provider_name = select_provider(provider)
    if provider_name == "gemini":
        return _resolve_gemini_model(model)
    if provider_name == "openai":
        return model or os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    if provider_name == "llama":
        return model or os.getenv("LLAMA_MODEL_PATH", "llama") or "llama"
    return provider_name  # "fallback" (offline deterministic truncation)


# Lazy import wrappers
def _have_provider(module_name: str, api_key_env: str) -> bool:
    """Check if a provider module is available and has API key configured.

    Args:
        module_name: Python module to import (e.g., "openai", "google.generativeai")
        api_key_env: Environment variable name for API key

    Returns:
        True if module importable and API key is set
    """
    try:  # pragma: no cover - optional
        import importlib

        importlib.import_module(module_name)
        return bool(os.getenv(api_key_env))
    except (ImportError, ModuleNotFoundError):
        return False


def _have_openai() -> bool:
    return _have_provider("openai", "OPENAI_API_KEY")


def _have_gemini() -> bool:
    return _have_provider("google.generativeai", "GEMINI_API_KEY")


# def _have_anthropic() -> bool:
#     return _have_provider("anthropic", "ANTHROPIC_API_KEY")


def _have_llama() -> bool:
    model_path = os.getenv("LLAMA_MODEL_PATH")
    if not model_path:
        return False
    path = Path(model_path).expanduser()
    if not path.exists():
        return False
    try:
        import importlib

        if importlib.util.find_spec("llama_cpp") is None:  # type: ignore[attr-defined]
            return False
    except Exception:
        return False
    return True


_LLAMA_CACHE: dict[str, object] = {}


def _get_llama_instance(model_path: str | None = None):
    path_str = model_path or os.getenv("LLAMA_MODEL_PATH")
    if not path_str:
        return None
    expanded = str(Path(path_str).expanduser())
    if expanded in _LLAMA_CACHE:
        return _LLAMA_CACHE[expanded]
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("llama-cpp-python unavailable: %s", exc)
        return None

    if not Path(expanded).exists():
        logger.warning("LLAMA_MODEL_PATH %s does not exist", expanded)
        return None

    try:
        n_ctx = int(os.getenv("LLAMA_CTX", "4096"))
    except ValueError:  # pragma: no cover - defensive
        n_ctx = 4096
    try:
        threads_env = os.getenv("LLAMA_THREADS")
        if threads_env is not None:
            n_threads = int(threads_env)
        else:
            n_threads = os.cpu_count() or 4
    except ValueError:  # pragma: no cover - defensive
        n_threads = os.cpu_count() or 4

    try:
        instance = Llama(
            model_path=expanded,
            n_ctx=n_ctx,
            n_threads=n_threads,
        )
    except Exception as exc:  # pragma: no cover - initialization failure
        logger.warning("Failed to initialise LLaMA model: %s", exc)
        return None

    _LLAMA_CACHE[expanded] = instance
    return instance


def select_provider(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.lower()
    env_choice = os.getenv("LLM_PROVIDER")
    if env_choice:
        return env_choice.lower()
    if _have_llama():
        return "llama"
    if _have_openai():
        return "openai"
    if _have_gemini():
        return "gemini"
    # if _have_anthropic():
    #     return "anthropic"
    return "fallback"


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = text[:limit]
    if " " in head:
        head = head.rsplit(" ", 1)[0]
    return head.rstrip() + "…"


def summarize(
    text: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 256,
) -> str:
    if not text:
        return ""
    provider_name = select_provider(provider)
    if provider_name == "openai":
        return _summarize_openai(text, model=model, max_tokens=max_tokens)
    if provider_name == "gemini":
        return _summarize_gemini(text, model=model, max_tokens=max_tokens)
    # if provider_name == "anthropic":
    #     if not _have_anthropic():
    #         return _truncate(text, max_tokens)
    #     from .anthropic_client import summarize_text
    #
    #     result = summarize_text(text, model=model, max_tokens=max_tokens)
    #     return _truncate(result, max_tokens)
    if provider_name == "llama":
        return _summarize_llama(text, model=model, max_tokens=max_tokens)
    # fallback deterministic truncation
    return _truncate(text, max_tokens)


def complete(
    prompt: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 256,
    tools: Optional[object] = None,
) -> str:
    """Generic text completion wrapper using the selected provider.

    This mirrors ``summarize`` but accepts a raw prompt. It preserves the
    centralized provider selection and offers an offline fallback that returns
    a truncated echo of the prompt for deterministic behavior.

    Args:
        prompt: The full text prompt to send to the model.
        provider: Optional explicit provider name.
        model: Optional model hint/name for the chosen provider.
        max_tokens: Max output tokens (provider-specific conversion heuristics apply).
        tools: Optional tools configuration (e.g. for Gemini grounding).

    Returns:
        Model text output or a truncated echo in offline fallback mode.
    """
    if not prompt:
        return ""
    provider_name = select_provider(provider)
    if provider_name == "openai":
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return _truncate(prompt, max_tokens)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return _truncate(prompt, max_tokens)
        client = OpenAI(api_key=api_key)  # type: ignore[call-arg]
        model_name: str = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
        try:  # type: ignore[attr-defined]
            resp = client.responses.create(
                model=model_name,
                input=prompt,
                max_output_tokens=max_tokens,
                temperature=0.0,
            )
            pieces = []
            output = getattr(resp, "output", None)
            if isinstance(output, list):
                for block in output:
                    content = getattr(block, "content", None) or getattr(block, "text", None)
                    if isinstance(content, list):
                        for c in content:
                            t = getattr(c, "text", None) or (
                                c.get("text") if isinstance(c, dict) else None
                            )
                            if t:
                                pieces.append(t)
                    elif isinstance(content, str):
                        pieces.append(content)
            if not pieces:
                maybe = getattr(resp, "output_text", None)
                if isinstance(maybe, str) and maybe.strip():
                    pieces.append(maybe.strip())
            text_out = " ".join(p.strip() for p in pieces if isinstance(p, str) and p.strip())
            return text_out or _truncate(prompt, max_tokens)
        except Exception:
            return _truncate(prompt, max_tokens)
    if provider_name == "gemini":
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError:
            logger.error("google-generativeai package not installed.")
            return _truncate(prompt, max_tokens)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # If the user explicitly requested gemini, we should warn them.
            # For now, we log an error to make it visible in logs.
            logger.error("GEMINI_API_KEY not found. Returning truncated prompt.")
            # We raise an error to stop the pipeline from generating bad data
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        # The library auto-configures from GOOGLE_API_KEY.
        # This avoids a direct call to genai.configure which can cause linting errors.
        os.environ["GOOGLE_API_KEY"] = api_key
        model_name = _resolve_gemini_model(model)
        # Log the resolved model at call time: Dec 2025's runs went out unpinned
        # (recorded only "provider: gemini") and later drifted -- see
        # data/experiments/rerun_20260721/README.md.
        logger.info("Gemini completion resolved model=%s", model_name)
        try:
            # Handle string tools if necessary
            final_tools = tools
            if isinstance(tools, str) and tools == "google_search":
                # Convert string shortcut to tool configuration using protos
                try:
                    import google.generativeai.protos as protos

                    # Use the GoogleSearch message nested in Tool if available, or top-level if not
                    gs = getattr(protos.Tool, "GoogleSearch", None)
                    if gs:
                        final_tools = [protos.Tool(google_search=gs())]
                    else:
                        # Fallback or alternative path if structure differs
                        final_tools = [{"google_search": {}}]
                except ImportError:
                    # Fallback to dict if protos not available
                    final_tools = [{"google_search": {}}]

            m = genai.GenerativeModel(model_name, tools=final_tools)  # type: ignore
            # Add timeout to prevent hung API calls (60s default)
            from google.generativeai.types import RequestOptions

            request_opts = RequestOptions(timeout=60)
            gemini_resp = m.generate_content(prompt, request_options=request_opts)
            if hasattr(gemini_resp, "text") and isinstance(gemini_resp.text, str):
                return gemini_resp.text.strip()[: max_tokens * 4]
            candidates = getattr(gemini_resp, "candidates", None)
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                content = getattr(first, "content", None)
                if content and getattr(content, "parts", None):
                    parts = getattr(content, "parts")
                    texts = [getattr(p, "text", "") for p in parts]
                    out = " ".join(t.strip() for t in texts if t.strip())
                    if out:
                        return out[: max_tokens * 4]
        except Exception as e:
            error_str = str(e).lower()
            # Re-raise on timeout/network errors for retry handling
            if any(
                x in error_str
                for x in ["timeout", "deadline", "503", "504", "429", "quota", "rate"]
            ):
                raise RuntimeError(f"Gemini API error: {e}") from e
            print(f"Gemini Error: {e}")
            return _truncate(prompt, max_tokens)
        return _truncate(prompt, max_tokens)
    # if provider_name == "anthropic":
    #     # For now, fallback to deterministic offline behavior unless a
    #     # specialized client adapter is added similar to summarize.
    #     if not _have_anthropic():
    #         return _truncate(prompt, max_tokens)
    #     try:
    #         # Reuse summarize_text helper if available; otherwise truncate.
    #         from .anthropic_client import summarize_text  # type: ignore
    #
    #         result = summarize_text(prompt, model=model, max_tokens=max_tokens)
    #         return _truncate(result, max_tokens)
    #     except Exception:
    #         return _truncate(prompt, max_tokens)
    if provider_name == "llama":
        llm = _get_llama_instance(model)
        if llm is None:
            return _truncate(prompt, max_tokens)
        try:
            temperature = 0.0
            top_p = float(os.getenv("LLAMA_TOP_P", "0.9"))
            top_k = int(os.getenv("LLAMA_TOP_K", "40"))
        except Exception:  # pragma: no cover - defensive
            temperature = 0.0
            top_p = 0.9
            top_k = 40
        try:
            response = llm(  # type: ignore
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=["###", "</s>", "\n\n"],
            )
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("LLaMA completion failed: %s", exc)
            return _truncate(prompt, max_tokens)
        choices = response.get("choices") if isinstance(response, dict) else None
        if choices:
            out = choices[0].get("text", "")
        else:
            out = response
        if not isinstance(out, str):
            return _truncate(prompt, max_tokens)
        return out.strip() or _truncate(prompt, max_tokens)
    return _truncate(prompt, max_tokens)


def _summarize_openai(text: str, *, model: Optional[str], max_tokens: int) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return _truncate(text, max_tokens)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _truncate(text, max_tokens)
    client = OpenAI(api_key=api_key)  # type: ignore[call-arg]
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    prompt = (
        "Summarize the following incident in 1-2 concise sentences focusing on privacy impacts.\n\n"
        + text
    )
    try:
        resp = client.responses.create(
            model=model_name,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=0.2,
        )
        # Defensive parsing across SDK variations
        pieces = []
        output = getattr(resp, "output", None)
        if isinstance(output, list):
            for block in output:
                # block may have .content or be dict-like
                content = getattr(block, "content", None) or getattr(block, "text", None)
                if isinstance(content, list):
                    for c in content:
                        t = getattr(c, "text", None) or (
                            c.get("text") if isinstance(c, dict) else None
                        )
                        if t:
                            pieces.append(t)
                elif isinstance(content, str):
                    pieces.append(content)
        # Fallback: try resp.output_text
        if not pieces:
            maybe = getattr(resp, "output_text", None)
            if isinstance(maybe, str) and maybe.strip():
                pieces.append(maybe.strip())
        text_out = " ".join(p.strip() for p in pieces if isinstance(p, str) and p.strip())
        return text_out or _truncate(text, max_tokens)
    except Exception:
        return _truncate(text, max_tokens)


def _summarize_gemini(text: str, *, model: Optional[str], max_tokens: int) -> str:
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return _truncate(text, max_tokens)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _truncate(text, max_tokens)
    # The library auto-configures from GOOGLE_API_KEY.
    # This avoids a direct call to genai.configure which can cause linting errors.
    os.environ["GOOGLE_API_KEY"] = api_key
    model_name = _resolve_gemini_model(model)
    try:  # type: ignore[attr-defined]
        m = genai.GenerativeModel(model_name)  # type: ignore
        prompt = (
            "Summarize the following incident in 1-2 concise sentences "
            "focusing on privacy impacts.\n\n" + text
        )
        # Add timeout to prevent hung API calls (60s default)
        from google.generativeai.types import RequestOptions

        request_opts = RequestOptions(timeout=60)
        resp = m.generate_content(prompt, request_options=request_opts)
        # Gemini may expose resp.text or resp.candidates
        if hasattr(resp, "text") and isinstance(resp.text, str):
            return resp.text.strip()[: max_tokens * 4]
        candidates = getattr(resp, "candidates", None)
        if isinstance(candidates, list) and candidates:
            first = candidates[0]
            # attempt to extract from content parts
            content = getattr(first, "content", None)
            if content and getattr(content, "parts", None):
                parts = getattr(content, "parts")
                texts = [getattr(p, "text", "") for p in parts]
                out = " ".join(t.strip() for t in texts if t.strip())
                if out:
                    return out[: max_tokens * 4]
    except Exception:
        return _truncate(text, max_tokens)
    return _truncate(text, max_tokens)


def _summarize_llama(text: str, *, model: Optional[str], max_tokens: int) -> str:
    llm = _get_llama_instance(model)
    if llm is None:
        return _truncate(text, max_tokens)
    try:
        temperature = float(os.getenv("LLAMA_TEMPERATURE", "0.2"))
    except ValueError:  # pragma: no cover - defensive
        temperature = 0.2
    try:
        top_p = float(os.getenv("LLAMA_TOP_P", "0.9"))
    except ValueError:  # pragma: no cover - defensive
        top_p = 0.9
    try:
        top_k = int(os.getenv("LLAMA_TOP_K", "40"))
    except ValueError:  # pragma: no cover - defensive
        top_k = 40

    prompt_template = os.getenv(
        "LLAMA_PROMPT_TEMPLATE",
        (
            "### Instruction\n"
            "Summarize the following incident in 1-2 concise sentences "
            "focusing on privacy impacts.\n\n"
            "### Incident\n{text}\n\n"
            "### Summary\n"
        ),
    )
    prompt = prompt_template.format(text=text)

    try:
        response = llm(  # type: ignore
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=["###", "</s>"],
        )
    except Exception as exc:  # pragma: no cover - runtime failure
        logger.warning("LLaMA summarization failed: %s", exc)
        return _truncate(text, max_tokens)

    choices = response.get("choices") if isinstance(response, dict) else None
    if choices:
        summary = choices[0].get("text", "")
    else:
        summary = response  # llama-cpp may return raw string in some modes
    if not isinstance(summary, str):
        return _truncate(text, max_tokens)
    return summary.strip() or _truncate(text, max_tokens)


def classify_privacy_relevance(
    text: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 256,
    tools: Optional[object] = None,
) -> dict:
    """Classify whether text is privacy-relevant and map to Solove categories.

    Contract
    - Input: free-form text
    - Output JSON dict:
        {
          "is_privacy_relevant": bool,
          "harms": [category...],  # subset of solove_categories()
          "rationale": str,
          "confidence": float  # 0..1
        }
    - Offline fallback: keyword-based heuristic with conservative defaults.
    """
    from json import JSONDecodeError

    def default_result(relevant: bool, harms: list[str] | None = None, rationale: str = "") -> dict:
        return {
            "is_privacy_relevant": bool(relevant),
            "harms": harms or [],
            "rationale": rationale,
            "confidence": 0.5 if relevant else 0.3,
        }

    if not text or not text.strip():
        return default_result(False, [], "Empty text")

    # Build compact instruction with taxonomy block and strict JSON constraint
    taxo = taxonomy_prompt_block()
    allowed = ", ".join(solove_categories())
    instruction = (
        "You are a privacy analyst. Decide if the text is about privacy or a privacy harm.\n"
        "If relevant, select ALL applicable high-level categories from this Solove-aligned list:\n"
        f"{taxo}\n\n"
        "Return ONLY a minified JSON object with keys: is_privacy_relevant (bool), "
        "harms (array of strings using the keys above), rationale (short string), "
        "confidence (0..1). Do not include extra text.\n"
        f"Allowed harms: [{allowed}].\n"
        "Text:\n" + text
    )

    # Try the selected provider via generic completion
    raw = complete(instruction, provider=provider, model=model, max_tokens=max_tokens, tools=tools)

    # Attempt robust JSON extraction from various model formatting habits
    candidate = raw.strip()
    # Extract fenced code block if present
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", candidate, re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    # Otherwise, try to find the first {...} block
    if not candidate.startswith("{"):
        brace = re.search(r"\{[\s\S]*\}", candidate)
        if brace:
            candidate = brace.group(0)

    parsed: Optional[dict] = None
    try:
        parsed = json.loads(candidate)
    except JSONDecodeError:
        parsed = None

    # Validate/normalize
    if isinstance(parsed, dict) and parsed:
        is_rel = bool(parsed.get("is_privacy_relevant", False))
        harms_in = parsed.get("harms")
        if not isinstance(harms_in, list):
            harms_list: list[str] = []
        else:
            harms_list = [str(h).strip() for h in harms_in if str(h).strip()]
        # keep only allowed keys
        allowed_set = set(solove_categories())
        harms_norm = [h for h in harms_list if h in allowed_set]
        rationale = str(parsed.get("rationale", "")).strip()
        try:
            conf = float(parsed.get("confidence", 0.7 if is_rel else 0.4))
        except Exception:
            conf = 0.7 if is_rel else 0.4
        conf = max(0.0, min(1.0, conf))
        return {
            "is_privacy_relevant": is_rel,
            "harms": harms_norm,
            "rationale": rationale,
            "confidence": conf,
        }

    # Offline/parse fallback: simple keyword heuristic
    tlow = text.lower()
    relevant = any(k in tlow for k in FALLBACK_RELEVANCE_KEYWORDS)
    harms = keyword_fallback_categories(tlow) if relevant else []
    rationale = (
        "Keyword heuristic match (offline/fallback)" if relevant else "No privacy keywords detected"
    )
    return default_result(relevant, harms, rationale)
