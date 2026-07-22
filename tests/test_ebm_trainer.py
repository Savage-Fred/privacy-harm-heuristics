import json
from pathlib import Path

from privacy_harm_heuristics.models.data import build_dataset
from privacy_harm_heuristics.models.trainers.ebm import train_ebm


def _write_synthetic(path: Path, n: int = 70):
    lines = []
    for i in range(n):
        rec = {
            "id": f"ebm{i}",
            "source": "synthetic",
            "type": "record",
            "created_date": "2024-01-01T00:00:00Z",
            "raw": {},
            "kw_privacy": 1 if i % 2 == 0 else 0,
            "kw_security": 1 if i % 3 == 0 else 0,
            "penalty_amount": float(i % 4),
            "incident_date": "2024-01-01",
            "target_flag": 1 if i % 2 == 0 else 0,
        }
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_train_ebm(tmp_path: Path):
    jsonl = tmp_path / "ebm.jsonl"
    _write_synthetic(jsonl)
    ds = build_dataset(jsonl, target="target_flag")
    result = train_ebm(ds.X_train, ds.y_train, ds.X_test, ds.y_test, interactions=0)
    assert result.model_type == "ebm"
    assert result.metrics["accuracy"] >= 0.5
    assert "feature_importances" in result.extra
