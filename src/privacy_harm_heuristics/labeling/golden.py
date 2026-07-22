"""Review-aware loading of the golden dataset.

The golden set (``data/golden_cases.jsonl``) is the held-out ground truth for
harm-label evaluation and for the golden-as-test-set training split. Cases go
through LLM normalization (``labels normalize``) and human review (``labels
review`` / ``labels review-web``), which stamp the review fields this module
interprets:

- ``reviewed`` (bool) — a human accepted or edited the case.
- ``needs_review`` (bool) — flagged (low LLM confidence, or explicitly kept
  flagged by a reviewer).
- ``manually_edited`` (bool) — the reviewer changed labels rather than
  accepting them as-is.

A case is *verified* when ``reviewed`` is truthy and ``needs_review`` is not.
Evaluation code should prefer verified cases as ground truth; unverified
cases carry machine-generated labels no human has confirmed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class GoldenReviewStats:
    """Review coverage of a golden dataset file."""

    total: int = 0
    verified: int = 0
    flagged: int = 0  # needs_review truthy
    unreviewed: int = 0  # never touched by the review workflow
    manually_edited: int = 0

    @property
    def verified_fraction(self) -> float:
        return (self.verified / self.total) if self.total else 0.0

    @property
    def fully_verified(self) -> bool:
        return self.total > 0 and self.verified == self.total

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "verified": self.verified,
            "flagged": self.flagged,
            "unreviewed": self.unreviewed,
            "manually_edited": self.manually_edited,
            "verified_fraction": round(self.verified_fraction, 4),
            "fully_verified": self.fully_verified,
        }


def is_verified(case: Dict[str, Any]) -> bool:
    """A case counts as human-verified once reviewed and no longer flagged."""
    return bool(case.get("reviewed")) and not bool(case.get("needs_review"))


def review_stats(cases: List[Dict[str, Any]]) -> GoldenReviewStats:
    stats = GoldenReviewStats(total=len(cases))
    for case in cases:
        if is_verified(case):
            stats.verified += 1
        elif case.get("needs_review"):
            stats.flagged += 1
        else:
            stats.unreviewed += 1
        if case.get("manually_edited"):
            stats.manually_edited += 1
    return stats


def load_golden_cases(
    path: str | Path,
    verified_only: bool = False,
) -> Tuple[List[Dict[str, Any]], GoldenReviewStats]:
    """Load golden cases with review-coverage stats.

    Returns ``(cases, stats)`` where ``stats`` always describes the full file,
    even when ``verified_only`` filters the returned cases. Malformed lines are
    skipped, matching the tolerant readers elsewhere in the eval code.
    """
    cases: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                cases.append(obj)

    stats = review_stats(cases)
    if verified_only:
        cases = [c for c in cases if is_verified(c)]
    return cases, stats
