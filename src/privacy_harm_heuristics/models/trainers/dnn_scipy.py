"""DNN trainer using SciPy/Scikit-learn (MLPClassifier).

This serves as the "DNN built using SciPy" requested by the user.
It uses sklearn.neural_network.MLPClassifier which relies on scipy.
"""

from __future__ import annotations


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ..metrics_utils import choose_average

from .. import ModelResult


def train_dnn_scipy(
    X_train,
    y_train,
    X_test,
    y_test,
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size="auto",
    learning_rate="constant",
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
) -> ModelResult:
    """Train a Multi-Layer Perceptron classifier."""

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    avg, pos_label = choose_average(y_train, y_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average=avg, pos_label=pos_label)),
        "precision": float(precision_score(y_test, preds, average=avg, pos_label=pos_label)),
        "recall": float(recall_score(y_test, preds, average=avg, pos_label=pos_label)),
    }

    return ModelResult(
        model_type="dnn_scipy",
        metrics=metrics,
        artifacts={"model": clf},
        extra={"params": clf.get_params(), "n_layers": clf.n_layers_, "n_outputs": clf.n_outputs_},
    )
