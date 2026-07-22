"""Model artifact persistence utilities.

Persist trained model directories (zipped) into Postgres when available so the
web layer and downstream evaluation jobs can fetch a consistent snapshot even
if the underlying filesystem changes or the container is ephemeral.

Design goals:
- No new heavy ORM dependency (use psycopg2 directly)
- Safe no-op when Postgres is not configured
- Generic directory archiving (works for sklearn, pytorch, heuristic exports)
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
import hashlib
from pathlib import Path
from typing import Optional

from ..data.access import resolve_database_url
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

try:  # Optional, mirror access.py pattern
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover
    psycopg2 = None  # type: ignore


TABLE_DDL = """
CREATE TABLE IF NOT EXISTS model_artifacts (
    name TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    content BYTEA NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (name, artifact_type)
);
"""

UPSERT_SQL = (
    "INSERT INTO model_artifacts (name, artifact_type, sha256, size_bytes, content) "
    "VALUES (%s, %s, %s, %s, %s) "
    "ON CONFLICT (name, artifact_type) DO UPDATE SET "
    "sha256=EXCLUDED.sha256, size_bytes=EXCLUDED.size_bytes, content=EXCLUDED.content, updated_at=NOW()"
)

CHECK_SQL = "SELECT 1 FROM model_artifacts WHERE name=%s LIMIT 1"

LIST_SQL = "SELECT name, artifact_type, size_bytes, sha256, updated_at FROM model_artifacts ORDER BY updated_at DESC"


def _pg_connect():
    dsn = resolve_database_url()
    # Support a bare host token (e.g. 'postgres' or 'dpg-xxxxx') by composing a URL
    # when the resolved value lacks scheme / key=value structure.
    # If no DSN and no PG_HOST provided, attempt auto-load from common env files.
    if not dsn and not os.getenv("PG_HOST"):
        _auto_load_db_env()
        dsn = resolve_database_url()

    if dsn and "://" not in dsn and "=" not in dsn:
        host = dsn
        domain = os.getenv("PG_DOMAIN", "")
        if "." not in host and domain:
            if not domain.startswith("."):  # normalize
                domain = "." + domain
            host = host + domain
        user = os.getenv("PG_USER", "postgres")
        password = os.getenv("PG_PASSWORD") or os.getenv("PGPASSWORD") or ""
        database = os.getenv("PG_DB") or os.getenv("PG_DATABASE") or "postgres"
        port = os.getenv("PG_PORT", "5432")
        dsn = f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{database}"
        logger.info(
            "Composed Postgres DSN from host token '%s' -> postgresql://%s:***@%s:%s/%s",
            host,
            user,
            host,
            port,
            database,
        )
    elif not dsn:
        # Compose DSN from discrete PG_* vars (fallback)
        host = os.getenv("PG_HOST") or os.getenv("PGHOST")
        if host:
            domain = os.getenv("PG_DOMAIN", "")
            if "." not in host and domain:
                if not domain.startswith("."):
                    domain = "." + domain
                host = host + domain
            user = os.getenv("PG_USER", "postgres")
            password = os.getenv("PG_PASSWORD") or os.getenv("PGPASSWORD") or ""
            database = os.getenv("PG_DB") or os.getenv("PG_DATABASE") or "postgres"
            port = os.getenv("PG_PORT", "5432")
            dsn = f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{database}"
    if not dsn or psycopg2 is None:
        return None
    try:
        conn = psycopg2.connect(dsn)  # type: ignore[return-value]
        logger.info(
            "Model persistence connected to Postgres host=%s db=%s",
            conn.get_dsn_parameters().get("host"),
            conn.get_dsn_parameters().get("dbname"),
        )
        return conn
    except Exception as exc:  # pragma: no cover
        logger.warning("Cannot connect to Postgres for model persistence: %s", exc)
        return None


def _auto_load_db_env() -> None:
    """Best-effort auto loading of database env vars.

    Order:
      1. .env.db.local
      2. .env.db
      3. .env.local (only DB-prefixed keys)
    Does not overwrite existing DB-related vars already in os.environ.
    """
    candidates = [
        Path(".env.db.local"),
        Path(".env.db"),
        Path(".env.local"),
    ]
    db_keys = {
        "DATABASE_URL",
        "PG_USER",
        "PG_PASSWORD",
        "PGPASSWORD",
        "PG_HOST",
        "PGHOST",
        "PG_PORT",
        "PG_DB",
        "PG_DATABASE",
        "PG_DOMAIN",
    }

    for file in candidates:
        if not file.exists():
            continue
        try:
            with file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    if key not in db_keys:
                        continue
                    if key in os.environ:  # do not clobber existing
                        continue
                    # Strip surrounding quotes if present
                    val = val.strip().strip('"').strip("'")
                    os.environ[key] = val
            logger.info("Auto-loaded DB env vars from %s", file)
            # After first successful load, stop; precedence preserved
            break
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed loading %s: %s", file, exc)


def _ensure_table(conn) -> None:
    with conn.cursor() as cur:  # type: ignore[call-arg]
        cur.execute(TABLE_DDL)
    conn.commit()


def _archive_directory(directory: Path) -> tuple[bytes, str]:
    """Create a gzipped tarball of a directory in-memory and return bytes + sha256."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path in directory.rglob("*"):
            if path.is_file():
                # Use relative path inside archive
                arcname = path.relative_to(directory)
                try:
                    tf.add(path, arcname=str(arcname))
                except Exception as exc:  # pragma: no cover
                    logger.debug("Skip file in archive (%s): %s", path, exc)
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    return data, sha


def persist_model_directory(
    name: str, directory: Path, artifact_type: str = "tar.gz"
) -> Optional[str]:
    """Persist a model directory as a single binary blob in Postgres.

    Returns sha256 if stored, None if skipped.
    """
    if not directory.exists() or not directory.is_dir():
        logger.warning("Model directory for %s does not exist (%s)", name, directory)
        return None
    conn = _pg_connect()
    if conn is None:
        logger.info("Postgres not configured; skipping model persistence (%s)", name)
        return None
    try:
        _ensure_table(conn)
        data, sha = _archive_directory(directory)
        size = len(data)
        with conn.cursor() as cur:  # type: ignore[call-arg]
            cur.execute(UPSERT_SQL, (name, artifact_type, sha, size, psycopg2.Binary(data)))  # type: ignore[attr-defined]
        conn.commit()
        logger.info("Persisted model %s (%s bytes, sha=%s) to Postgres", name, size, sha[:10])
        return sha
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to persist model %s: %s", name, exc)
        try:
            conn.rollback()
        except Exception:
            pass
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def model_artifact_exists(name: str) -> bool:
    conn = _pg_connect()
    if conn is None:
        return False
    try:
        _ensure_table(conn)
        with conn.cursor() as cur:  # type: ignore[call-arg]
            cur.execute(CHECK_SQL, (name,))
            row = cur.fetchone()
            return bool(row)
    except Exception:  # pragma: no cover
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def list_model_artifacts() -> list[dict]:
    conn = _pg_connect()
    if conn is None:
        return []
    try:
        _ensure_table(conn)
        with conn.cursor() as cur:  # type: ignore[call-arg]
            cur.execute(LIST_SQL)
            rows = cur.fetchall()
            desc = getattr(cur, "description", []) or []
            cols = []
            for i, d in enumerate(desc):
                if isinstance(d, tuple) and d:
                    cols.append(str(d[0]))
                else:  # pragma: no cover - defensive fallback
                    name = getattr(d, "name", None)
                    cols.append(str(name) if name else f"col_{i}")
            return [dict(zip(cols, r)) for r in rows]
    except Exception:  # pragma: no cover
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def persist_all_models(models_dir: Path) -> dict[str, str | None]:
    # Allow string paths for convenience
    if isinstance(models_dir, str):  # type: ignore[arg-type]
        models_dir = Path(models_dir)
    results: dict[str, str | None] = {}
    if not models_dir.exists():
        return results
    for child in models_dir.iterdir():
        if child.is_dir():
            sha = persist_model_directory(child.name, child)
            results[child.name] = sha
    return results
