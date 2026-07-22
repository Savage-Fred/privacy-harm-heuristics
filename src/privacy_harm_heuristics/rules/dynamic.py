"""Dynamic rule selection and composition utilities.

These helpers let us:
- tag rules by jurisdiction/sector/language/source
- detect lightweight context from an input text or metadata
- select and order rules for LLM prompting/validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Rule:
    text: str
    jurisdictions: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source: Optional[str] = None
    priority: float = 1.0  # higher = rank earlier


def _as_rule(obj: Any) -> Rule:
    """Normalize a Rule/dict/string into a Rule object."""
    if isinstance(obj, Rule):
        return obj
    if isinstance(obj, str):
        return Rule(text=obj)
    if isinstance(obj, dict):
        return Rule(
            text=obj.get("text") or obj.get("rule") or "",
            jurisdictions=obj.get("jurisdictions") or obj.get("countries") or [],
            sectors=obj.get("sectors") or [],
            languages=obj.get("languages") or [],
            tags=obj.get("tags") or [],
            source=obj.get("source"),
            priority=float(obj.get("priority", 1.0)),
        )
    return Rule(text=str(obj))


def detect_context(
    text: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """Very lightweight context detection (non-ML, no extra deps)."""
    metadata = metadata or {}
    text_lower = (text or "").lower()

    jurisdiction = metadata.get("jurisdiction")
    sector = metadata.get("sector")
    language = metadata.get("language")

    # Crude detectors based on regulator names/currencies; meant as hints, not hard matches.
    if jurisdiction is None:
        if any(k in text_lower for k in ["ico", "uk gdpr", "ico.org.uk"]):
            jurisdiction = "UK"
        elif any(k in text_lower for k in ["cnil", "france", "rgpd"]):
            jurisdiction = "FR"
        elif any(k in text_lower for k in ["lgpd", "anpd", "brazil", "brl"]):
            jurisdiction = "BR"
        elif any(k in text_lower for k in ["pdpc", "singapore", "sgd"]):
            jurisdiction = "SG"
        elif any(k in text_lower for k in ["aepd", "spain", "esgpd"]):
            jurisdiction = "ES"
        elif any(k in text_lower for k in ["garante", "italy", "it privacy"]):
            jurisdiction = "IT"
        elif any(k in text_lower for k in ["pipc", "korea", "south korea"]):
            jurisdiction = "KR"
        elif any(k in text_lower for k in ["oaic", "australia"]):
            jurisdiction = "AU"
        elif any(k in text_lower for k in ["pipeda", "opc", "canada"]):
            jurisdiction = "CA"
        elif any(k in text_lower for k in ["gdpr", "eu regulator", "edpb"]):
            jurisdiction = "EU"

    if sector is None:
        if any(k in text_lower for k in ["hospital", "medical", "hipaa", "ehr", "patient"]):
            sector = "health"
        elif any(k in text_lower for k in ["bank", "fintech", "payment", "card", "psd2"]):
            sector = "finance"
        elif any(k in text_lower for k in ["camera", "iot", "smart home", "doorbell", "ring"]):
            sector = "smart_home"
        elif any(k in text_lower for k in ["school", "student", "edtech", "children"]):
            sector = "children_edtech"

    return {"jurisdiction": jurisdiction, "sector": sector, "language": language}


def select_rules(
    rules: Iterable[Rule] | Iterable[Dict[str, Any]] | Iterable[str],
    context: Dict[str, Optional[str]],
    max_rules: int = 50,
) -> List[Rule]:
    """Filter and rank rules by context."""
    jurisdiction = context.get("jurisdiction")
    sector = context.get("sector")
    language = context.get("language")

    selected: List[Rule] = []
    for raw in rules:
        rule = _as_rule(raw)
        score = rule.priority

        # Jurisdiction match boosts, mismatch penalizes
        if rule.jurisdictions:
            if jurisdiction and jurisdiction in rule.jurisdictions:
                score += 1.5
            elif jurisdiction:
                score -= 1.0

        # Sector match boosts, mismatch penalizes
        if rule.sectors:
            if sector and sector in rule.sectors:
                score += 1.0
            elif sector:
                score -= 0.5

        # Language hint
        if rule.languages:
            if language and language in rule.languages:
                score += 0.5

        # Only keep if score is still positive
        if score > 0:
            rule.priority = score
            selected.append(rule)

    # Sort by priority desc and truncate
    selected.sort(key=lambda r: r.priority, reverse=True)
    return selected[:max_rules]


def format_rules(rules: List[Rule]) -> List[Dict[str, Any]]:
    """Return a JSON-serializable view of rules with metadata."""
    out: List[Dict[str, Any]] = []
    for r in rules:
        out.append(
            {
                "text": r.text,
                "jurisdictions": r.jurisdictions,
                "sectors": r.sectors,
                "languages": r.languages,
                "tags": r.tags,
                "source": r.source,
                "priority": r.priority,
            }
        )
    return out


def compose_rule_block(rules: List[Rule]) -> str:
    """Format selected rules for LLM prompting/validation."""
    lines = ["# SELECTED PRIVACY RULES", ""]
    for i, r in enumerate(rules, 1):
        meta_parts = []
        if r.jurisdictions:
            meta_parts.append(f"JURIS={','.join(r.jurisdictions)}")
        if r.sectors:
            meta_parts.append(f"SECTOR={','.join(r.sectors)}")
        if r.tags:
            meta_parts.append(f"TAGS={','.join(r.tags)}")
        if r.source:
            meta_parts.append(f"SRC={r.source}")
        meta = f" ({'; '.join(meta_parts)})" if meta_parts else ""
        lines.append(f"{i}. {r.text}{meta}")
    return "\n".join(lines)
