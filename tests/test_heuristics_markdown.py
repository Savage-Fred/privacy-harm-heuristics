from pathlib import Path

import numpy as np

from privacy_harm_heuristics.models.heuristics import (
    extract_heuristics,
    save_heuristics_json,
    save_heuristics_markdown,
)
from privacy_harm_heuristics.models.trainers.decision_tree import train_decision_tree


def test_heuristics_markdown_written(tmp_path: Path):
    # Tiny dataset
    X = np.array([[0], [1], [1], [0], [1], [0]], dtype=float)
    y = np.array([0, 1, 1, 0, 1, 0])
    result = train_decision_tree(
        X[:4], y[:4], X[4:], y[4:], max_depth=2, min_samples_leaf=1, random_state=0
    )
    items = extract_heuristics(result, ["f0"], X_train=X[:4], y_train=y[:4], top_n=5)

    # JSONL + Markdown write
    out_json = tmp_path / "heuristics.jsonl"
    out_md = tmp_path / "HEURISTICS.md"
    save_heuristics_json(items, str(out_json))
    save_heuristics_markdown(
        items,
        str(out_md),
        title="Unit Test Summary",
        context={"model": "decision_tree", "notes": "test"},
        top_n=5,
    )

    assert out_json.exists()
    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "# Unit Test Summary" in content
    assert "## Top Heuristics" in content
    # Ensure at least one heuristic text is present
    assert any("IF" in line for line in content.splitlines())
