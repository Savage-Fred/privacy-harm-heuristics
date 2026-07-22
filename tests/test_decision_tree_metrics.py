import numpy as np
from privacy_harm_heuristics.models.trainers.decision_tree import train_decision_tree
from privacy_harm_heuristics.models import ModelResult


def test_decision_tree_macro_when_test_drops_class():
    # Create a dataset with 3 classes; craft split where test lacks one class
    X = np.random.RandomState(0).randn(15, 4)
    y = np.array(
        [
            "aggregation",
            "insecurity",
            "surveillance",
            "aggregation",
            "insecurity",
            "surveillance",
            "aggregation",
            "insecurity",
            "surveillance",
            "aggregation",
            "insecurity",
            "surveillance",
            "aggregation",
            "insecurity",
            "surveillance",
        ]
    )
    # Manual split: y_test lacks 'insecurity'
    X_train, y_train = X[:10], y[:10]
    X_test, y_test = X[10:], y[10:]
    # Force y_test to not contain 'insecurity'
    mask = y_test != "insecurity"
    X_test = X_test[mask]
    y_test = y_test[mask]

    res: ModelResult = train_decision_tree(X_train, y_train, X_test, y_test, max_depth=3)
    # If metrics were computed with average='binary', sklearn would raise; here we just assert keys exist.
    assert "f1" in res.metrics and "precision" in res.metrics and "recall" in res.metrics
    # Ensure values are finite floats
    for k in ("f1", "precision", "recall", "accuracy"):
        v = res.metrics[k]
        assert isinstance(v, float)
        assert np.isfinite(v)
