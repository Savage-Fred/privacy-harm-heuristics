"""Storage helpers for streaming JSONL writing and Parquet conversion.

Functions here are purposely small and side-effect aware (auto-create
directories, JSON-safe coercion). They avoid loading entire datasets in memory
when writing JSONL, enabling large-scale streaming.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _ensure_duckdb_installed() -> bool:
    # Inlined from old repo's storage/duckdb_store.py (not extracted — this was
    # its only consumer, and everything else duckdb_store.py did is out of
    # scope for this practicum slice).
    try:
        import duckdb  # noqa: F401

        return True
    except ImportError:
        return False


def ensure_dir(path: str | os.PathLike) -> None:
    """Create parent directory for the target path if missing."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation for arbitrary objects.

    Recursively converts mappings, sequences, sets and falls back to ``str``
    for unknown objects (after attempting attribute dict extraction).
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(v) for v in value]
    try:
        attrs = vars(value)
    except TypeError:
        return str(value)
    return _json_safe(attrs)


def write_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> int:
    """Write an iterable of dict records to a JSONL file.

    Args:
        records: Iterable producing dictionaries.
        out_path: Destination file path (overwrites if exists).

    Returns:
        Count of records written.
    """
    ensure_dir(out_path)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            safe_rec = _json_safe(rec)
            f.write(json.dumps(safe_rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def _hash_record(record: Dict[str, Any]) -> str:
    """Generate a stable hash for a record used as fallback ID."""

    payload = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_jsonl_unique(
    records: Iterable[Dict[str, Any]],
    out_path: str,
    *,
    source: str,
    key_field: str = "id",
    db_path: Optional[str] = None,
    capture_new_records: bool = False,
) -> Dict[str, Any]:
    """Append records to a JSONL file while deduplicating by ``key_field``.

    Each accepted record is also indexed inside DuckDB so subsequent runs skip
    duplicates efficiently and the raw payloads remain queryable via SQL.

    Args:
        records: Iterable of dictionaries to append.
        out_path: Destination JSONL path (created if missing).
        source: Logical data source name (e.g., ``reddit``) used for indexing.
        key_field: Record field treated as the primary identifier.
        db_path: Optional DuckDB path (defaults to ``data/privacy.duckdb``).

    Returns:
        Dict with counts ``written``, ``skipped`` and ``processed``. When
        ``capture_new_records`` is True, an additional ``new_records`` list
        containing the JSON-safe payloads that were written in this call is
        included.
    """

    if not _ensure_duckdb_installed():
        raise ImportError("duckdb package is required for incremental ingestion")

    import duckdb  # type: ignore

    target = Path(out_path)
    ensure_dir(target)
    mode = "a" if target.exists() and target.stat().st_size > 0 else "w"

    db_path_resolved = db_path or os.environ.get("INGESTION_DB_PATH") or "data/privacy.duckdb"
    Path(db_path_resolved).parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(db_path_resolved)
    conn.execute("CREATE TABLE IF NOT EXISTS ingestion_index (source TEXT, record_id TEXT)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS raw_events (source TEXT, record_id TEXT, payload JSON, inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_ingestion_unique ON ingestion_index(source, record_id)"
    )

    existing_rows = conn.execute(
        "SELECT record_id FROM ingestion_index WHERE source = ?", (source,)
    ).fetchall()
    existing_ids = {row[0] for row in existing_rows}

    processed = 0
    written = 0
    skipped = 0
    new_ids: List[str] = []
    new_payloads: List[str] = []
    captured_records: Optional[List[Dict[str, Any]]] = [] if capture_new_records else None

    with target.open(mode, encoding="utf-8") as handle:
        for record in records:
            processed += 1
            record_id = record.get(key_field)
            if record_id is None:
                record_id = _hash_record(record)
            record_id = str(record_id)
            if record_id in existing_ids:
                skipped += 1
                continue
            existing_ids.add(record_id)
            safe_record = _json_safe(record)
            handle.write(json.dumps(safe_record, ensure_ascii=False) + "\n")
            new_ids.append(record_id)
            new_payloads.append(json.dumps(safe_record, ensure_ascii=False))
            if capture_new_records and captured_records is not None:
                captured_records.append(safe_record)
            written += 1

    if new_ids:
        conn.executemany(
            "INSERT INTO ingestion_index (source, record_id) VALUES (?, ?)",
            [(source, rid) for rid in new_ids],
        )
        conn.executemany(
            "INSERT INTO raw_events (source, record_id, payload) VALUES (?, ?, ?)",
            [(source, rid, payload) for rid, payload in zip(new_ids, new_payloads)],
        )

    conn.close()

    result: Dict[str, Any] = {"processed": processed, "written": written, "skipped": skipped}
    if capture_new_records:
        result["new_records"] = captured_records or []
    return result


def jsonl_to_parquet(in_path: str, out_path: str) -> int:
    """Convert a JSONL file to Parquet, returning number of rows.

    Notes:
        - Columns that contain nested Python objects (dict, list, set, tuple)
          are converted to JSON-encoded strings before writing to Parquet.
          This avoids PyArrow ArrowInvalid errors such as:

              "cannot mix list and non-list, non-null values"

          which can occur when a column mixes scalar and container types.
    """
    df = pd.read_json(in_path, lines=True)

    # Normalize nested object columns into JSON strings so that Parquet
    # sees a consistent scalar type instead of heterogeneous Python objects.
    def _is_null(value: Any) -> bool:
        return value is None or (isinstance(value, float) and pd.isna(value))

    for col in df.columns:
        series = df[col]
        if series.dtype != "object":
            continue

        non_null_values = [val for val in series if not _is_null(val)]

        has_nested = any(isinstance(val, (dict, list, tuple, set)) for val in non_null_values)
        scalar_types = {
            type(val) for val in non_null_values if not isinstance(val, (dict, list, tuple, set))
        }

        def _to_json_str(v: Any) -> Any:
            if _is_null(v):
                return None
            if isinstance(v, (dict, list, tuple, set)):
                return json.dumps(_json_safe(v), ensure_ascii=False)
            return str(v)

        if has_nested or len(scalar_types) > 1:
            # Parquet requires consistent scalar types; serialize nested payloads and
            # mixed scalar columns (e.g., str + int) as strings to avoid ArrowTypeError.
            df[col] = series.map(_to_json_str)
        elif scalar_types and all(issubclass(t, (bytes, bytearray)) for t in scalar_types):
            # Normalize bytes columns (including numpy byte dtypes) to UTF-8 strings.
            df[col] = series.map(
                lambda v: None if _is_null(v) else v.decode("utf-8", errors="ignore")
            )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)
