import json
from pathlib import Path

from privacy_harm_heuristics.models.data import build_dataset
from privacy_harm_heuristics.models.trainers.sparse_linear import train_sparse_linear


def _write_synthetic_jsonl(path: Path, n: int = 80):
    lines = []
    for i in range(n):
        rec = {
            "id": f"sl{i}",
            "source": "synthetic",
            "type": "record",
            "created_date": "2024-01-01T00:00:00Z",
            "raw": {},
            "kw_privacy": 1 if i % 2 == 0 else 0,
            "kw_security": 1 if i % 3 == 0 else 0,
            "penalty_amount": float((i * 3) % 7),
            "incident_date": "2024-01-01",
            "target_flag": 1 if i % 2 == 0 else 0,
        }
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_sparse_linear_trainer(tmp_path: Path):
    jsonl = tmp_path / "synthetic_sl.jsonl"
    _write_synthetic_jsonl(jsonl)
    ds = build_dataset(jsonl, target="target_flag")
    result = train_sparse_linear(ds.X_train, ds.y_train, ds.X_test, ds.y_test, C=0.5)
    assert result.model_type == "sparse_linear"
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert "coefficients" in result.extra
    # Expect some learning above chance
    assert result.metrics["accuracy"] >= 0.5
