import json
from pathlib import Path

import pytest

from privacy_harm_heuristics.models.data import build_dataset
from privacy_harm_heuristics.models.explain import lime_explain, make_shap_explainer, shap_explain
from privacy_harm_heuristics.models.trainers.decision_tree import train_decision_tree


def _write_jsonl(path: Path, n: int = 40):
    lines = []
    for i in range(n):
        rec = {
            "id": f"ex{i}",
            "source": "synthetic",
            "type": "record",
            "created_date": "2024-01-01T00:00:00Z",
            "raw": {},
            "kw_privacy": 1 if i % 2 == 0 else 0,
            "kw_security": 1 if i % 3 == 0 else 0,
            "penalty_amount": float(i % 5),
            "incident_date": "2024-01-01",
            "target_flag": 1 if i % 2 == 0 else 0,
        }
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_shap_explain(tmp_path: Path):
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl)
    ds = build_dataset(jsonl, target="target_flag")
    result = train_decision_tree(ds.X_train, ds.y_train, ds.X_test, ds.y_test, max_depth=3)
    # build shap explainer on training subset for speed
    try:
        explainer = make_shap_explainer(result.artifacts["model"], ds.X_train)
    except ImportError:
        pytest.skip("shap not installed")
    inst = ds.X_test.iloc[[0]]
    exp = shap_explain(explainer, inst, feature_names=list(inst.columns))
    assert exp["method"] == "shap"
    assert len(exp["feature_contributions"]) == inst.shape[1]


def test_lime_explain(tmp_path: Path):
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl)
    ds = build_dataset(jsonl, target="target_flag")
    result = train_decision_tree(ds.X_train, ds.y_train, ds.X_test, ds.y_test, max_depth=3)
    inst_df = ds.X_test.iloc[[0]]
    try:
        exp = lime_explain(
            result.artifacts["model"],
            inst_df,
            feature_names=list(inst_df.columns),
            num_features=5,
        )
    except ImportError:
        pytest.skip("lime not installed")
    assert exp["method"] == "lime"
    assert len(exp["feature_contributions"]) <= 5
