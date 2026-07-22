"""Utilities for bulk feature derivation over a JSONL dataset."""

from __future__ import annotations

import json
from pathlib import Path

from ..nlp.product_features import annotate_record_with_features
from ..storage.writers import jsonl_to_parquet
from .builder import build_features


def build_features_for_jsonl(in_path: str, out_path: str, parquet_path: str | None = None) -> int:
    """Read normalized records from JSONL, derive features, write combined rows.

    Output schema: original record fields + each feature key (flat merge).

    NEW: Also applies product feature annotation which populates root_cause_features
    and extracts causal phrases. This is CRITICAL for root cause modeling.

    Returns number of records processed.
    """
    count = 0
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with (
        open(in_path, "r", encoding="utf-8") as src,
        open(out_path, "w", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)

            # Step 1: Build engineered features (keywords, penalty buckets, etc.)
            feats = build_features(rec)
            rec.update(feats)

            # Step 2: NEW - Apply product feature annotation for root cause analysis
            # This populates root_cause_features, causal_indicators, potential_causes
            annotate_record_with_features(
                rec, text_fields=("description", "harm_summary", "web_description")
            )

            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    if parquet_path:
        jsonl_to_parquet(out_path, parquet_path)
    return count
