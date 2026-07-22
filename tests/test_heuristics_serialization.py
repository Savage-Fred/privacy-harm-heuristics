import json
from pathlib import Path

import numpy as np

from privacy_harm_heuristics.models.heuristics import extract_heuristics, save_heuristics_json
from privacy_harm_heuristics.models.trainers.decision_tree import train_decision_tree


def test_heuristics_save_roundtrip(tmp_path: Path):
    # Simple dataset
    X = np.array([[0], [1], [1], [0], [1], [0]], dtype=float)
    y = np.array([0, 1, 1, 0, 1, 0])
    # Train tiny tree
    result = train_decision_tree(
        X[:4], y[:4], X[4:], y[4:], max_depth=2, min_samples_leaf=1, random_state=0
    )
    items = extract_heuristics(result, ["f0"], X_train=X[:4], y_train=y[:4])
    assert items, "Expected extracted heuristics"
    out_path = tmp_path / "heuristics.jsonl"
    save_heuristics_json(items, str(out_path))
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(items)
    first = json.loads(lines[0])
    assert "provenance_hash" in first["extra"]
    assert first["extra"].get("version") == 1
