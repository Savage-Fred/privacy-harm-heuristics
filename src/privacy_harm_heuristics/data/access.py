"""Unified data access layer for the Privacy Heuristics application.

This module abstracts over multiple storage backends so the web app and
pipelines can transparently read incident & feature data from either
PostgreSQL (Render deployment) or local JSONL files (developer mode), with
an optional DuckDB path still available for ad‑hoc analytics.

Selection precedence:
1. Explicit environment variable PRIMARY_DATA_BACKEND=postgres|files|duckdb
2. If DATABASE_URL present (or PG_* env variables) -> postgres
3. Fallback to files

The objective is to *read* path first. Writes (pipelines) will continue to
emit JSONL; a separate import command will sync into Postgres. This keeps
model training reproducible and auditable.

NOTE: Keep this lightweight; avoid ORM complexity for interpretability and
deployment minimalism.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

try:  # Optional dependency: psycopg2 only needed in postgres mode
    import psycopg2  # type: ignore
    from psycopg2.extras import RealDictCursor  # type: ignore
except Exception:  # pragma: no cover - if not installed we simply cannot use postgres
    psycopg2 = None  # type: ignore
    RealDictCursor = None  # type: ignore

# Optional dependency: Cloud SQL Python Connector (used when CLOUD_SQL_INSTANCE is set)
try:  # pragma: no cover - imported only when used at runtime
    from google.cloud.sql.connector import Connector, IPTypes  # type: ignore
except Exception:  # pragma: no cover - optional
    Connector = None  # type: ignore
    IPTypes = None  # type: ignore

DATA_DIR = Path("data")
FEATURE_FILE = DATA_DIR / "with_features.jsonl"
HEURISTICS_FILE = DATA_DIR / "heuristics.jsonl"


def resolve_database_url() -> str | None:
    """Resolve a PostgreSQL connection string from environment variables.

    Preference order:
        1. ``DATABASE_URL`` (direct connection URI)
        2. Individual ``PG_*`` components (``PG_USER``, ``PG_PASSWORD``/``PGPASSWORD``,
           ``PG_HOST``, optional ``PG_PORT``, and ``PG_DB``/``PG_DATABASE``)

    Returns:
        A PostgreSQL connection URI or ``None`` if the environment lacks the
        necessary variables.
    """

    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn

    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD") or os.getenv("PGPASSWORD")
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT")
    database = os.getenv("PG_DB") or os.getenv("PG_DATABASE")

    if not all([user, password, host, database]):
        return None

    # mypy: the guard above ensures these values are non-None
    user_str = quote_plus(str(user))
    password_str = quote_plus(str(password))
    host_str = str(host)
    database_str = str(database)
    host_part = host_str if port is None else f"{host_str}:{port}"
    return f"postgresql://{user_str}:{password_str}@{host_part}/{database_str}"


def _ensure_adc_credentials_file() -> None:
    """Ensure Google ADC credentials are available if provided as JSON.

    If the environment contains ``GOOGLE_APPLICATION_CREDENTIALS_JSON`` but not
    ``GOOGLE_APPLICATION_CREDENTIALS`` (file path), write the JSON to a temp
    file and point ADC to it. This is convenient for platforms like Render
    where mounting a file is less ergonomic than injecting an env var.
    """
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_json and not creds_path:
        tmp_path = "/tmp/gcp-service-account.json"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(creds_json)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_path
        except Exception:  # pragma: no cover - best-effort helper
            pass


def _iter_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file.

    Args:
        path: Path to JSON Lines file
        limit: Optional max number of records
    """
    if not path.exists():
        return
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield obj
            count += 1
            if limit and count >= limit:
                break


# ---------------------------------------------------------------------------
# Backend base + implementations
# ---------------------------------------------------------------------------


class DataBackendProtocol(Protocol):
    """Structural protocol for data backends.

    Using a Protocol avoids tight inheritance coupling while keeping
    type-checking happy across different backend implementations.
    """

    name: str

    def get_features(self, limit: int | None = None) -> list[dict[str, Any]]: ...

    def get_feature_count(self) -> int: ...

    def get_heuristics(self) -> list[dict[str, Any]]: ...

    def health(self) -> dict[str, Any]: ...


class DataBackend:
    """Nominal base class retained for runtime clarity/documentation.

    Concrete backends may inherit from this class, but it is not required.
    Static typing uses ``DataBackendProtocol`` for flexibility.
    """

    name: str = "abstract"

    def get_features(
        self, limit: int | None = None
    ) -> list[dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_feature_count(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def get_heuristics(self) -> list[dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def health(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class FileBackend(DataBackend):
    """File system JSONL backend (current default)."""

    name = "files"

    def get_features(self, limit: int | None = None) -> list[dict[str, Any]]:
        return list(_iter_jsonl(FEATURE_FILE, limit=limit))

    def get_feature_count(self) -> int:
        if not FEATURE_FILE.exists():
            return 0
        with FEATURE_FILE.open("r", encoding="utf-8") as fh:
            return sum(1 for _ in fh if _.strip())

    def get_heuristics(self) -> list[dict[str, Any]]:
        if HEURISTICS_FILE.exists():
            return list(_iter_jsonl(HEURISTICS_FILE))
        return []

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "feature_file_exists": FEATURE_FILE.exists(),
            "heuristics_exists": HEURISTICS_FILE.exists(),
            "feature_count": self.get_feature_count(),
        }


class PostgresBackend(DataBackend):
    """PostgreSQL backend reading from structured tables.

    Expects a DATABASE_URL env var readable by psycopg2. Only *reads* for now.
    """

    name = "postgres"

    def __init__(self, dsn: str):
        if psycopg2 is None:  # pragma: no cover - runtime guard
            raise RuntimeError("psycopg2 not installed; cannot use postgres backend")
        self.dsn = dsn

    def _connect(self):  # context-managed ephemeral connections
        if psycopg2 is None:  # pragma: no cover - defensive
            raise RuntimeError("psycopg2 not available")
        timeout = os.getenv("POSTGRES_CONNECT_TIMEOUT", "3")
        try:
            timeout_val = int(timeout)
        except (TypeError, ValueError):  # pragma: no cover - guard invalid env
            timeout_val = 3
        return psycopg2.connect(self.dsn, connect_timeout=timeout_val)  # type: ignore[union-attr]

    def get_features(self, limit: int | None = None) -> list[dict[str, Any]]:
        # Prefer incidents even if engineered_features is empty; keep a LEFT JOIN
        # for future feature columns
        q = (
            "SELECT i.id AS id, i.company, i.source, i.type, i.harm_severity, "
            "i.penalty_amount, i.individuals_affected, i.sentiment_score, i.description, i.raw "
            "FROM incidents i LEFT JOIN engineered_features ef ON ef.incident_id = i.id "
            "ORDER BY i.created_at DESC"
        )
        if limit:
            q += f" LIMIT {int(limit)}"
        conn = self._connect()  # type: ignore[attr-defined]
        try:
            # Avoid driver-specific cursor factories; build dicts from cursor.description
            cur = conn.cursor()  # type: ignore[arg-type]
            try:
                cur.execute(q)
                # Build column names from description, ensuring strings without None
                columns: list[str] = []
                for idx, desc in enumerate(getattr(cur, "description", []) or []):  # type: ignore[attr-defined]
                    name = desc[0] if isinstance(desc, tuple) else getattr(desc, "name", None)
                    columns.append(str(name) if name is not None else f"col_{idx}")
                fetched = cur.fetchall()
                rows: list[dict[str, Any]] = []
                for r in fetched:
                    # Some drivers may already return mappings; handle both
                    if isinstance(r, dict):
                        rows.append(dict(r))
                    else:
                        rows.append({col: val for col, val in zip(columns, r)})
            finally:
                try:
                    cur.close()
                except Exception:
                    pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return rows

    def get_feature_count(self) -> int:
        # Treat total incident rows as the materializable count
        q = "SELECT COUNT(*) FROM incidents"
        conn = self._connect()  # type: ignore[attr-defined]
        try:
            cur = conn.cursor()  # type: ignore[arg-type]
            try:
                cur.execute(q)
                row = cur.fetchone()
                n = row[0] if row else 0
            finally:
                try:
                    cur.close()
                except Exception:
                    pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return int(n)

    def get_heuristics(self) -> list[dict[str, Any]]:
        # Still read file for heuristics until rules are stored in DB
        if HEURISTICS_FILE.exists():
            return list(_iter_jsonl(HEURISTICS_FILE))
        return []

    def health(self) -> dict[str, Any]:
        try:
            count = self.get_feature_count()
        except Exception as exc:  # pragma: no cover - protective
            message = str(exc)
            hint = None
            if "does not exist" in message:
                hint = (
                    "engineered_features table missing. "
                    "Run database/init_db.py then populate_db.py."
                )
            return {
                "backend": self.name,
                "error": message,
                "hint": hint,
            }
        return {"backend": self.name, "feature_count": count}


class CloudSQLBackend(PostgresBackend):
    """Cloud SQL Postgres backend using the Cloud SQL Python Connector.

    Activate by setting ``CLOUD_SQL_INSTANCE`` ("project:region:instance").
    Expects standard ``PG_USER``, ``PG_PASSWORD`` (or ``PGPASSWORD``), and
    ``PG_DB``/``PG_DATABASE`` env vars for database credentials.

    Authentication to GCP is resolved via ADC. To supply a service account key
    via env var, set ``GOOGLE_APPLICATION_CREDENTIALS_JSON``; this helper will
    write it to ``/tmp/gcp-service-account.json`` and set
    ``GOOGLE_APPLICATION_CREDENTIALS`` accordingly.
    """

    name = "postgres"

    def __init__(self) -> None:
        if Connector is None:  # pragma: no cover - runtime guard
            raise RuntimeError("Cloud SQL connector not available; cannot use Cloud SQL backend")
        # dsn unused in connector path but kept for compatibility in parent
        # Do not call PostgresBackend.__init__ to avoid hard dependency on psycopg2
        self.dsn = ""

    def _connect(self):  # type: ignore[override]
        if Connector is None or IPTypes is None:  # pragma: no cover
            raise RuntimeError("Cloud SQL connector not available")

        instance = os.getenv("CLOUD_SQL_INSTANCE")
        if not instance:
            raise RuntimeError("CLOUD_SQL_INSTANCE not set (expected 'project:region:instance')")

        user = os.getenv("PG_USER") or os.getenv("PGUSERNAME") or "postgres"
        password = os.getenv("PG_PASSWORD") or os.getenv("PGPASSWORD") or ""
        database = os.getenv("PG_DB") or os.getenv("PG_DATABASE") or "postgres"

        _ensure_adc_credentials_file()

        # Always use PUBLIC IP to allow egress from Render without VPC.
        ip_type = IPTypes.PUBLIC

        # Create a new connector per call; connection outlives the context.
        # Use pg8000 driver with the Connector (psycopg2 is not supported by connector).
        # See: https://cloud.google.com/sql/docs/postgres/connect-connectors#python
        with Connector() as connector:  # type: ignore[call-arg]
            conn = connector.connect(
                instance_connection_string=instance,
                driver="pg8000",
                user=user,
                password=password,
                db=database,
                ip_type=ip_type,
            )
        return conn


def select_backend() -> DataBackendProtocol:
    """Select an appropriate backend based on environment variables."""
    explicit = os.getenv("PRIMARY_DATA_BACKEND")

    # Short-circuit: if explicitly requesting files, use FileBackend immediately
    if explicit == "files":
        logger.info("PRIMARY_DATA_BACKEND=files; using FileBackend.")
        return FileBackend()

    # Prefer Cloud SQL connector when instance specified
    # If explicitly requesting postgres AND a Cloud SQL instance is configured,
    # honor Cloud SQL even if the database is currently empty (operator override).
    if os.getenv("CLOUD_SQL_INSTANCE") and explicit in (None, "postgres"):
        try:
            candidate = CloudSQLBackend()
            # If PRIMARY_DATA_BACKEND=postgres explicitly, treat Cloud SQL as authoritative and
            # return it even if the tables are currently empty. This lets operators verify
            # connectivity and avoid unexpected fallback.
            if explicit == "postgres":
                return candidate
            else:
                try:
                    count = candidate.get_feature_count()
                except Exception as exc:  # pragma: no cover - defensive runtime guard
                    logger.warning(
                        "Cloud SQL backend unavailable (%s); falling back to JSONL dataset.",
                        exc,
                    )
                    return FileBackend()
                if count > 0:
                    return candidate
                logger.warning(
                    "Cloud SQL backend has zero engineered_features rows; "
                    "falling back to JSONL dataset."
                )
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning(
                "Failed to initialise Cloud SQL backend (%s); trying standard Postgres dsn...",
                exc,
            )

    resolved_dsn = resolve_database_url()
    if explicit == "postgres" or (explicit is None and resolved_dsn):
        dsn = resolved_dsn
        if not dsn:
            logger.warning(
                "PRIMARY_DATA_BACKEND=postgres but no Postgres connection details found; "
                "falling back to JSONL dataset"
            )
            return FileBackend()
        try:
            # Renamed from `candidate` (used above for the CloudSQLBackend
            # branch) to avoid a same-scope type clash under mypy.
            postgres_candidate = PostgresBackend(dsn)
            try:
                count = postgres_candidate.get_feature_count()
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                logger.warning(
                    "Postgres backend unavailable (%s); falling back to JSONL dataset. "
                    "Run database/populate_db.py to load records.",
                    exc,
                )
                return FileBackend()
            if count > 0:
                return postgres_candidate
            logger.warning(
                "Postgres backend has zero engineered_features rows; "
                "falling back to JSONL dataset. "
                "Run database/populate_db.py to import data before switching back."
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning(
                "Failed to initialise Postgres backend (%s); falling back to JSONL dataset.",
                exc,
            )
        return FileBackend()
    # (duckdb placeholder) - treat as files for now
    return FileBackend()


# Singleton style accessor to avoid repeated env parsing
_BACKEND: DataBackendProtocol | None = None


def get_backend() -> DataBackendProtocol:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = select_backend()
    return _BACKEND


# Convenience re-exported helpers
def load_feature_records(limit: int | None = None) -> list[dict[str, Any]]:
    return get_backend().get_features(limit=limit)


def load_heuristics() -> list[dict[str, Any]]:
    return get_backend().get_heuristics()


def backend_health() -> dict[str, Any]:
    return get_backend().health()


def materialize_features_jsonl(
    out_path: str | Path,
    limit: int | None = None,
    gcs_keep_latest: bool | None = False,
) -> int:
    """Materialize feature records from the active backend into a JSONL file.

    This is used in low-risk mode to keep existing file-based pipelines working
    while the primary storage may be Postgres. If the backend is already file
    based, this becomes a no-op (unless limit truncation requested).

    Args:
        out_path: Destination JSONL path. Supports local filesystem paths and
            GCS URIs (gs://bucket/path).
        limit: Optional maximum number of records to write
        gcs_keep_latest: When writing to GCS, also copy the object to a
            sibling "-latest.jsonl" key for easy discovery. If bucket
            versioning is enabled, this provides a stable pointer while
            retaining history.

    Returns:
        Number of records written.
    """
    backend = get_backend()
    # Always read records from the active backend to ensure correctness, even in files mode.
    # This avoids stale counts if the file was just modified by a test or prior step.
    records = backend.get_features(limit=limit)

    # If destination is a GCS URI, upload via google-cloud-storage
    out_str = str(out_path)
    if out_str.startswith("gs://"):
        try:
            from google.cloud import storage  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime-only
            raise RuntimeError(
                "google-cloud-storage is required to write to GCS. "
                "Install it or use a local file path."
            ) from exc

        # Parse gs://bucket/key
        # Split once on '/' after the scheme to separate bucket and blob name
        without_scheme = out_str[len("gs://") :]
        parts = without_scheme.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid GCS path: {out_str}")
        bucket_name, blob_name = parts[0], parts[1]

        client = storage.Client()  # ADC will be used in Cloud Run
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Stream to a temp file to avoid keeping all content in memory
        with NamedTemporaryFile("w", delete=True, encoding="utf-8") as tmp:
            for rec in records:
                tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp.flush()
            blob.upload_from_filename(tmp.name, content_type="application/x-ndjson")

        # Optionally also copy to a -latest.jsonl key
        if gcs_keep_latest:
            # Build a sibling name with a stable "-latest.jsonl" suffix.
            # Use a robust slice (".jsonl" is 6 chars) to avoid off-by-one errors.
            if blob_name.endswith(".jsonl"):
                base = blob_name[: -len(".jsonl")]  # strip the extension
            else:
                base = blob_name
            latest_name = f"{base}-latest.jsonl"
            latest_blob = bucket.blob(latest_name)
            # Use rewrite to copy within the same bucket efficiently
            latest_blob.rewrite(blob)

        return len(records)

    # Local filesystem write
    out_path_obj = out_path if isinstance(out_path, Path) else Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with out_path_obj.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)
