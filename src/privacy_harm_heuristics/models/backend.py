"""Runtime model artifact discovery and backend hydration.

This module keeps model storage concerns out of the web layer. Local artifacts
remain the default, while ``PRIMARY_MODEL_BACKEND=backblaze`` can hydrate
``models/`` from Backblaze B2 during startup.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ARTIFACT_MARKERS = (
    "model.joblib",
    "classifier_head.pt",
    "pytorch_model.bin",
    "metrics.json",
    "metadata.json",
    "training_config.json",
)

MODEL_BACKEND_ENV = "PRIMARY_MODEL_BACKEND"
MODEL_PREFIX_ENV = "MODEL_BACKEND_PREFIX"
B2_MODEL_PREFIX_ENV = "B2_MODEL_PREFIX"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_models_dir() -> Path:
    return _repo_root() / "models"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, int | float]:
    return {key: value for key, value in metrics.items() if isinstance(value, (int, float))}


def _model_type(name: str, metadata: dict[str, Any], training_config: dict[str, Any]) -> str:
    explicit = metadata.get("type") or training_config.get("model")
    if isinstance(explicit, str) and explicit:
        return explicit
    base = name.removesuffix("_v2").removesuffix("_v3").removesuffix("_v4")
    if "transformer" in base:
        return "transformer"
    if base in {"decision_tree", "ebm", "sparse_linear", "bayes_net", "brl"}:
        return base
    return "unknown"


def _last_modified_iso(model_dir: Path) -> str | None:
    mtimes: list[float] = []
    for marker in ARTIFACT_MARKERS:
        path = model_dir / marker
        if path.exists():
            mtimes.append(path.stat().st_mtime)
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat()


def model_dir_has_artifacts(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    return any((model_dir / marker).exists() for marker in ARTIFACT_MARKERS)


def discover_model_metadata(models_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Discover first-level model artifact directories.

    The returned shape is intentionally compatible with ``/api/models``.
    Metadata is synthesized from metrics/training files when ``metadata.json``
    is absent so existing trained classical models still appear in the UI.
    """

    root = models_dir or default_models_dir()
    if not root.exists():
        return {}

    candidate_dirs = {child for child in root.iterdir() if child.is_dir()}
    for marker in ARTIFACT_MARKERS:
        candidate_dirs.update(path.parent for path in root.glob(f"**/{marker}"))

    discovered: dict[str, dict[str, Any]] = {}

    def _sort_key(path: Path) -> tuple[int, str]:
        rel = path.relative_to(root)
        return (len(rel.parts), rel.as_posix())

    for model_dir in sorted(candidate_dirs, key=_sort_key):
        if not model_dir_has_artifacts(model_dir):
            continue

        metadata = _read_json(model_dir / "metadata.json")
        metrics = _read_json(model_dir / "metrics.json")
        training_config = _read_json(model_dir / "training_config.json")
        rel_parts = model_dir.relative_to(root).parts
        model_id = model_dir.name
        if model_id in discovered:
            model_id = "__".join(rel_parts)
        accuracy = metadata.get("accuracy")
        if accuracy is None and isinstance(metrics.get("accuracy"), (int, float)):
            accuracy = metrics["accuracy"]

        discovered[model_id] = {
            "model_id": model_id,
            "name": metadata.get("name", model_id),
            "type": _model_type(model_id, metadata, training_config),
            "path": str(model_dir),
            "available": True,
            "metrics": _numeric_metrics(metrics),
            "last_trained": metadata.get("last_trained") or _last_modified_iso(model_dir),
            "accuracy": accuracy if isinstance(accuracy, (int, float)) else None,
            "source": metadata.get("source", "local"),
        }
    return discovered


def has_local_model_artifacts(models_dir: Path | None = None) -> bool:
    return bool(discover_model_metadata(models_dir))


def ensure_models_available(
    models_dir: Path | None = None,
    backend: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Ensure runtime model artifacts are available.

    ``local`` is a no-op that reports discovered artifacts.
    ``backblaze``/``b2`` downloads from B2.
    ``auto`` downloads only when no local artifacts are present.
    """

    root = models_dir or default_models_dir()
    selected = (backend or os.getenv(MODEL_BACKEND_ENV) or "local").strip().lower() or "local"
    local_before = discover_model_metadata(root)
    status: dict[str, Any] = {
        "backend": selected,
        "models_dir": str(root),
        "local_model_count_before": len(local_before),
        "downloaded": [],
        "error": None,
    }

    should_download = selected in {"backblaze", "b2"}
    if selected == "auto":
        should_download = force or not local_before

    if selected in {"local", "files"}:
        should_download = False
    elif selected not in {"auto", "backblaze", "b2"}:
        status["error"] = f"Unknown {MODEL_BACKEND_ENV} value: {selected}"
        status["models"] = local_before
        status["local_model_count_after"] = len(local_before)
        return status

    if should_download:
        try:
            from .downloader import sync_models_from_backblaze

            prefix = os.getenv(MODEL_PREFIX_ENV) or os.getenv(B2_MODEL_PREFIX_ENV)
            root.mkdir(parents=True, exist_ok=True)
            status["downloaded"] = sync_models_from_backblaze(root, prefix=prefix)
        except Exception as exc:
            status["error"] = str(exc)
            if selected in {"backblaze", "b2"}:
                logger.warning("Model Backblaze hydration failed: %s", exc)
            else:
                logger.info("Model Backblaze hydration skipped/failed in auto mode: %s", exc)

    local_after = discover_model_metadata(root)
    status["models"] = local_after
    status["local_model_count_after"] = len(local_after)
    return status
