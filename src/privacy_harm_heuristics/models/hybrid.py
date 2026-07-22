"""Hybrid model implementation combining deterministic rules and LLM reasoning."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from ..llm.provider import complete
from ..constants.privacy_taxonomy import COMMON_ROOT_CAUSES, all_solove_terms


class HybridMode(str, Enum):
    BASELINE = "baseline"
    RULES_STATIC = "rules_static"
    RULES_DYNAMIC = "rules_dynamic"  # Per-dataset or continuously tuned
    RAG = "rag"
    HYBRID_DETERMINISTIC_FIRST = "hybrid_deterministic_first"
    HYBRID_LLM_FIRST = "hybrid_llm_first"


class HybridModel:
    """Hybrid model for privacy harm prediction."""

    def __init__(
        self,
        mode: HybridMode = HybridMode.BASELINE,
        rules: Optional[List[str]] = None,
        deterministic_model: Any = None,  # e.g. loaded DecisionTree or EBM
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.mode = mode
        self.rules = rules or []
        self.deterministic_model = deterministic_model
        self.provider = provider
        self.model_name = model_name

    def predict(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Predict privacy harms, ranking, and severity.

        Args:
            text: The scenario description.
            context: Optional context (e.g. retrieved rules for RAG, or pre-computed features).

        Returns:
            Dict containing:
                - root_causes: List[str] (set of labels)
                - ranking: List[str] (ordered by importance)
                - harm_score: int (1-5)
                - rationale: str
        """
        context = context or {}

        # 1. Pre-processing / Deterministic Step
        deterministic_output = None
        if self.mode == HybridMode.HYBRID_DETERMINISTIC_FIRST and self.deterministic_model:
            # Assume deterministic model has a predict method that returns a list of causes
            # This is a placeholder; actual integration depends on the model interface
            try:
                # If the model expects features, they should be in context['features']
                # If it expects text, pass text.
                if hasattr(self.deterministic_model, "predict"):
                    # Simplified assumption for now
                    deterministic_output = self.deterministic_model.predict([text])[0]
            except Exception:
                pass

        # 2. Construct Prompt
        prompt = self._build_prompt(text, deterministic_output, context)

        # 3. Call LLM
        raw_response = complete(
            prompt,
            provider=self.provider,
            model=self.model_name,
            max_tokens=512,  # Enough for JSON output
        )

        # 4. Parse Output
        return self._parse_response(raw_response)

    def _build_prompt(
        self,
        text: str,
        deterministic_output: Any,
        context: Dict[str, Any],
    ) -> str:
        base_instruction = (
            "You are a privacy expert. Analyze the following scenario to identify privacy harms.\n"
            "Output ONLY a valid JSON object. Do not include any other text.\n"
            "The JSON must have the following keys:\n"
            "- root_causes: List of specific root causes/harms present.\n"
            "- ranking: Ordered list of root causes by severity (most critical first).\n"
            "- harm_score: Integer 1-5 (1=Minimal, 5=Severe).\n"
            "- rationale: Brief explanation.\n\n"
            f"Allowed Harms (Solove Taxonomy): {', '.join(all_solove_terms())}\n"
            f"Allowed Root Causes: {', '.join(COMMON_ROOT_CAUSES)}\n"
            "IMPORTANT: You MUST use terms from the above lists for 'root_causes'. Do not invent new terms.\n"
            "You may include both harms and root causes in the 'root_causes' list.\n"
            "\nExample output:\n"
            "{\n"
            '  "root_causes": ["information_collection", "invasion", "lack_of_consent"],\n'
            '  "ranking": ["invasion", "information_collection"],\n'
            '  "harm_score": 4,\n'
            '  "rationale": "The system collects data without consent..."\n'
            "}\n"
        )

        mode_instruction = ""
        if self.mode == HybridMode.RULES_STATIC:
            rules_text = "\n".join(f"- {r}" for r in self.rules)
            mode_instruction = f"\nApply the following privacy rules:\n{rules_text}\n"
        elif self.mode == HybridMode.HYBRID_DETERMINISTIC_FIRST:
            if deterministic_output:
                mode_instruction = (
                    f"\nA deterministic model suggested these potential causes: {deterministic_output}.\n"
                    "Verify these and include them if valid. Add any missing critical harms.\n"
                )

        return f"{base_instruction}{mode_instruction}\nScenario:\n{text}"

    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        default_response = {
            "root_causes": [],
            "ranking": [],
            "harm_score": 0,
            "rationale": "Failed to parse response",
        }

        candidate = raw_text.strip()
        # Extract fenced code block
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", candidate, re.IGNORECASE)
        if fenced:
            candidate = fenced.group(1).strip()
        else:
            # Fallback: find the first { and last }
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1:
                candidate = candidate[start : end + 1]

        try:
            parsed = json.loads(candidate)
            return {
                "root_causes": parsed.get("root_causes", []),
                "ranking": parsed.get("ranking", []),
                "harm_score": int(parsed.get("harm_score", 0)),
                "rationale": parsed.get("rationale", ""),
            }
        except (json.JSONDecodeError, ValueError) as e:
            # Log the failure for debugging
            print(f"JSON Parse Error: {e}")
            print(f"Raw Candidate: {candidate[:500]}...")
            default_response["rationale"] = f"Failed to parse response. Raw: {raw_text[:200]}..."
            return default_response
