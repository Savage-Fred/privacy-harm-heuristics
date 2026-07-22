"""Utilities to merge multiple source JSONL files into a unified dataset.

Merges line-delimited JSON records, normalizes `incident_date` via
`canonicalize_incident_date`, and writes out combined JSONL. An optional Parquet
conversion can be performed with existing writers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..storage.writers import write_jsonl
from .dates import canonicalize_incident_date

logger = logging.getLogger(__name__)


def _iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    """Iterate JSON objects from a JSONL file, skipping malformed lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record
            else:
                logger.debug(
                    "Skipping non-object JSON value in %s (type=%s)",
                    path,
                    type(record).__name__,
                )


def merge_jsonl(inputs: List[str], output: str) -> int:
    """Merge multiple JSONL files writing a unified JSONL output.

    Adds ``incident_date_canonical`` when an ``incident_date`` field exists in
    a record. Lines with invalid JSON are skipped silently. Streaming design
    avoids loading entire datasets into memory.

    Args:
        inputs: List of input JSONL file paths.
        output: Destination JSONL path (overwritten if exists).

    Returns:
        Number of records written.
    """

    def gen():
        for p in inputs:
            for rec in _iter_jsonl(p):
                if "incident_date" in rec:
                    rec["incident_date_canonical"] = canonicalize_incident_date(
                        rec.get("incident_date")
                    )
                yield rec

    return write_jsonl(gen(), output)
