"""Model artifact serialization utilities.

Artifacts layout (directory):
  metrics.json
  feature_names.json
  training_config.json
  provenance.json
  model.joblib
  extra.json (optional: feature_importances, coefficients, etc.)

Future additions: rules.json, shap_summary.json, lime_examples/.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Dict

import joblib

from . import ModelResult


def _default(o):  # safe JSON fallback
    return getattr(o, "__dict__", str(o))


def compute_file_hash(path: Path, algo: str = "sha256", limit_bytes: int | None = 2_000_000) -> str:
    """Return hexadecimal digest of a file with optional byte cap.

    Args:
        path: File to hash.
        algo: Hash algorithm name accepted by ``hashlib``.
        limit_bytes: When set, only the first ``limit_bytes`` are read to
            bound cost for very large datasets.

    Returns:
        Hex digest string.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        if limit_bytes is None:
            h.update(fh.read())
        else:
            remaining = limit_bytes
            while remaining > 0:
                chunk = fh.read(min(65536, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
    return h.hexdigest()


def save_model_result(
    result: ModelResult,
    out_dir: str | Path,
    feature_names: list[str],
    training_config: Dict[str, Any] | None = None,
    data_path: str | Path | None = None,
) -> Path:
    """Persist a ``ModelResult`` bundle to a directory.

    Creates standard artifact files (metrics, feature names, training config,
    provenance, optional extra metadata, model binary). Computes data file
    hash when provided.

    Returns the output directory path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Core JSON artifacts
    (out / "metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    (out / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    if training_config is None:
        training_config = {}
    (out / "training_config.json").write_text(
        json.dumps(training_config, indent=2), encoding="utf-8"
    )
    # Provenance
    prov = {
        "saved_at": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "model_type": result.model_type,
        "data_path": str(data_path) if data_path else None,
    }
    if data_path and Path(data_path).exists():
        try:
            prov["data_sha256"] = compute_file_hash(Path(data_path))
        except Exception:
            prov["data_sha256"] = None
    (out / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    # Extra
    if result.extra:
        (out / "extra.json").write_text(
            json.dumps(result.extra, default=_default, indent=2), encoding="utf-8"
        )
    # Model binary
    joblib.dump(result.artifacts.get("model"), out / "model.joblib")
    return out


def load_model_artifacts(base_dir: str | Path) -> Dict[str, Any]:
    """Load a serialized model bundle produced by ``save_model_result``.

    Returns dict with keys: model, feature_names, metrics, extra.
    """
    base = Path(base_dir)
    with open(base / "feature_names.json", "r", encoding="utf-8") as fh:
        feature_names = json.load(fh)
    with open(base / "metrics.json", "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    extra = {}
    if (base / "extra.json").exists():
        extra = json.loads((base / "extra.json").read_text(encoding="utf-8"))
    model = joblib.load(base / "model.joblib")
    return {
        "model": model,
        "feature_names": feature_names,
        "metrics": metrics,
        "extra": extra,
    }
