"""Shared metric averaging helpers.

Centralizes the binary vs macro averaging decision to avoid sklearn
ValueError when a test split drops one of the training classes.

Call `choose_average(y_train, y_test)` to obtain (average, pos_label).

Rules:
  - Consider union of train/test labels; if union size != 2 -> multiclass -> macro.
  - If union size == 2 but one label missing in y_test -> macro.
  - Otherwise binary; if labels contain {0,1} use pos_label=1 else pos_label="positive".
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple


def _to_list(y: Any) -> list:
    """Best-effort conversion to list without forcing materialization for generators."""
    if hasattr(y, "tolist"):
        try:
            return list(y.tolist())  # type: ignore[arg-type]
        except Exception:
            pass
    try:
        return list(y)
    except Exception:
        return [y]


def choose_average(y_train: Iterable, y_test: Iterable) -> Tuple[str, str | int]:
    """Return (average, pos_label) for sklearn metrics.

    Ensures we only request binary averaging if the test set actually
    contains both classes; otherwise fall back to macro.
    """
    train_list = _to_list(y_train)
    test_list = _to_list(y_test)
    union_labels = set(train_list) | set(test_list)
    if len(union_labels) != 2:
        return "macro", "positive"
    # Two labels overall; ensure both appear in test split
    if not (set(test_list) >= union_labels):
        return "macro", "positive"
    # True binary; determine pos_label
    if union_labels.issuperset({0, 1}):
        return "binary", 1
    return "binary", "positive"


__all__ = ["choose_average"]
