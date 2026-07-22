import json
from pathlib import Path

from privacy_harm_heuristics.models.data import build_dataset
from privacy_harm_heuristics.models.trainers.decision_tree import train_decision_tree


def _write_synthetic_jsonl(path: Path, n: int = 60):
    # Simple binary target correlated with kw_privacy
    for i in range(n):
        rec = {
            "id": f"r{i}",
            "source": "synthetic",
            "type": "record",
            "created_date": "2024-01-01T00:00:00Z",
            "raw": {},
            "kw_privacy": 1 if i % 2 == 0 else 0,
            "kw_security": 1 if i % 3 == 0 else 0,
            "penalty_amount": float(i % 5),
            "incident_date": "2024-01-01",
            "kw_other": 1 if i % 7 == 0 else 0,
            # target identical to kw_privacy for test determinism
            "target_flag": 1 if i % 2 == 0 else 0,
        }
        path.write_text(
            (path.read_text() if path.exists() else "") + json.dumps(rec) + "\n",
            encoding="utf-8",
        )


def test_decision_tree_trainer_basic(tmp_path: Path):
    jsonl = tmp_path / "synthetic.jsonl"
    _write_synthetic_jsonl(jsonl)
    ds = build_dataset(jsonl, target="target_flag")
    result = train_decision_tree(ds.X_train, ds.y_train, ds.X_test, ds.y_test, max_depth=3)
    assert result.model_type == "decision_tree"
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert "feature_importances" in result.extra
    # Should learn some signal: accuracy > 0.5
    assert result.metrics["accuracy"] >= 0.5
