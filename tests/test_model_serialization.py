import json
from pathlib import Path

from privacy_harm_heuristics.models.data import build_dataset
from privacy_harm_heuristics.models.serialize import load_model_artifacts, save_model_result
from privacy_harm_heuristics.models.trainers.decision_tree import train_decision_tree


def _write_jsonl(path: Path, n: int = 40):
    lines = []
    for i in range(n):
        rec = {
            "id": f"ser{i}",
            "source": "synthetic",
            "type": "record",
            "created_date": "2024-01-01T00:00:00Z",
            "raw": {},
            "kw_privacy": 1 if i % 2 == 0 else 0,
            "kw_security": 1 if i % 3 == 0 else 0,
            "incident_date": "2024-01-01",
            "target_flag": 1 if i % 2 == 0 else 0,
        }
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_save_and_load_round_trip(tmp_path: Path):
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl)
    ds = build_dataset(jsonl, target="target_flag")
    result = train_decision_tree(ds.X_train, ds.y_train, ds.X_test, ds.y_test, max_depth=4)
    out_dir = tmp_path / "artifacts"
    save_model_result(
        result,
        out_dir,
        feature_names=ds.feature_names,
        training_config={"max_depth": 4},
        data_path=jsonl,
    )
    loaded = load_model_artifacts(out_dir)
    # Basic checks
    assert loaded["metrics"]["accuracy"] >= 0.5
    assert len(loaded["feature_names"]) == len(ds.feature_names)
    preds_original = result.artifacts["model"].predict(ds.X_test)
    preds_loaded = loaded["model"].predict(ds.X_test)
    assert (preds_original == preds_loaded).all()
