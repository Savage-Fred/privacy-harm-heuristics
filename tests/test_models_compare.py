from __future__ import annotations

import json
from pathlib import Path

from privacy_harm_heuristics.models.data import build_dataset
from privacy_harm_heuristics.models.eval.harness import EvalConfig, evaluate
from privacy_harm_heuristics.models.serialize import save_model_result
from privacy_harm_heuristics.models.trainers.decision_tree import train_decision_tree


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_compare_harness_decision_tree(tmp_path: Path):
    # Create a tiny labeled feature dataset
    data_path = tmp_path / "features.jsonl"
    rows = [
        {"kw_a": 1, "kw_b": 0, "harm_category": 1, "description": "camera"},
        {"kw_a": 0, "kw_b": 1, "harm_category": 0, "description": "public"},
        {"kw_a": 1, "kw_b": 0, "harm_category": 1, "description": "gps"},
        {"kw_a": 0, "kw_b": 1, "harm_category": 0, "description": "leak"},
    ]
    _write_jsonl(data_path, rows)

    ds = build_dataset(data_path, target="harm_category", test_size=0.5, random_state=42)
    result = train_decision_tree(
        ds.X_train,
        ds.y_train,
        ds.X_test,
        ds.y_test,
        max_depth=3,
        min_samples_leaf=1,
        random_state=42,
    )
    model_dir = tmp_path / "decision_tree"
    save_model_result(
        result,
        model_dir,
        feature_names=ds.feature_names,
        training_config={"model": "decision_tree"},
        data_path=str(data_path),
    )

    cfg = EvalConfig(
        data_path=str(data_path),
        target="harm_category",
        model_dirs=[str(model_dir)],
        include_heuristic=True,
        include_llm=False,
        limit=None,
    )
    report = evaluate(cfg)
    assert "heuristic" in report
    # Model key equals the directory name
    assert "decision_tree" in report
    assert isinstance(report["heuristic"].get("accuracy", 0.0), float)
