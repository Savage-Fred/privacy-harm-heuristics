import json
import os
import random
from pathlib import Path

import pytest

from privacy_harm_heuristics.models.data import build_dataset


def _write_synthetic(tmp_path: Path, n: int = 40) -> Path:
    path = tmp_path / "synthetic.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for _ in range(n):
            kw_privacy = random.randint(0, 1)
            kw_security = random.randint(0, 1)
            label = 1 if (kw_privacy == 1 and kw_security == 1) else 0
            rec = {
                "id": f"r{random.random():.6f}",
                "source": "synthetic",
                "type": "test",
                "created_date": "2024-01-01",
                "kw_privacy": kw_privacy,
                "kw_security": kw_security,
                "label": label,
            }
            fh.write(json.dumps(rec) + "\n")
    return path


@pytest.mark.slow
def test_train_brl(tmp_path):
    if os.getenv("FAST_TESTS"):
        pytest.skip("Skipping slow BRL test under FAST_TESTS mode")
    # brl.py imports imodels lazily and raises ImportError only at *call* time, so
    # guard on the actual dependency here (not around the module import) — otherwise
    # `make test` hard-fails instead of skipping when the `models` extra is absent.
    pytest.importorskip("imodels", reason="imodels not installed (pip install '.[models]')")
    data_path = _write_synthetic(tmp_path)
    ds = build_dataset(data_path, target="label")
    from privacy_harm_heuristics.models.trainers.brl import train_brl

    result = train_brl(
        ds.X_train, ds.y_train, ds.X_test, ds.y_test, n_steps=50
    )  # fewer steps for speed
    assert result.model_type == "brl"
    assert "accuracy" in result.metrics
    # Sanity: rules were extracted
    n_rules = result.extra.get("n_rules")
    assert n_rules is None or n_rules >= 1
