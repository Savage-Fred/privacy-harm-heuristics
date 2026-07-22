import pytest
import numpy as np
from privacy_harm_heuristics.evals.metrics import (
    calculate_mlc_metrics,
    calculate_ranking_metrics,
    calculate_ordinal_metrics,
    calculate_metrics,
    bootstrap_confidence_interval,
)

# Test data for Multi-Label Classification
y_true_mlc = [
    {"cat", "dog"},
    {"cat"},
    {"dog", "fish"},
    {"fish"},
    {"cat", "dog", "fish"},
]
y_pred_mlc = [
    {"cat", "dog"},
    {"dog"},
    {"fish"},
    {"cat"},
    {"cat", "dog", "fish"},
]
all_classes = ["cat", "dog", "fish", "bird"]


def test_calculate_mlc_metrics_perfect_match():
    y_true = [{"a", "b"}, {"c"}]
    y_pred = [{"a", "b"}, {"c"}]
    metrics = calculate_mlc_metrics(y_true, y_pred)
    assert metrics["instance_jaccard"] == 1.0
    assert metrics["instance_f1"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["hamming_loss"] == 0.0
    assert metrics["exact_match_ratio"] == 1.0


def test_calculate_mlc_metrics_no_match():
    y_true = [{"a"}, {"b"}]
    y_pred = [{"c"}, {"d"}]
    classes = ["a", "b", "c", "d"]
    metrics = calculate_mlc_metrics(y_true, y_pred, classes=classes)
    assert metrics["instance_jaccard"] == 0.0
    assert metrics["instance_f1"] == 0.0
    assert metrics["micro_f1"] == 0.0
    assert metrics["macro_f1"] == 0.0
    assert metrics["hamming_loss"] == 0.5  # (1+1)/4 for each sample
    assert metrics["exact_match_ratio"] == 0.0


def test_calculate_mlc_metrics_partial_match():
    y_true = [{"a", "b"}, {"c"}]
    y_pred = [{"a"}, {"c", "d"}]
    classes = ["a", "b", "c", "d"]
    metrics = calculate_mlc_metrics(y_true, y_pred, classes=classes)
    assert metrics["exact_match_ratio"] == 0.0
    assert pytest.approx(metrics["instance_jaccard"]) == (0.5 + 0.5) / 2
    assert pytest.approx(metrics["hamming_loss"]) == (1 + 1) / (2 * 4)


def test_calculate_mlc_metrics_with_classes():
    y_true = [{"a"}, {"b"}]
    y_pred = [{"a"}, {"c"}]
    classes = ["a", "b", "c", "d"]
    metrics = calculate_mlc_metrics(y_true, y_pred, classes=classes)
    # 2 correct, 1 false positive, 1 false negative out of 8 possible labels
    assert metrics["hamming_loss"] == 2 / (2 * 4)
    assert metrics["exact_match_ratio"] == 0.5


def test_calculate_mlc_metrics_empty_input():
    metrics = calculate_mlc_metrics([], [])
    for value in metrics.values():
        assert value == 0.0 or value == 1.0  # accuracy can be 1.0 for no samples


# Test data for Ranking
y_true_rank = [["a", "b"], ["c"], ["d", "e", "f"]]
y_pred_rank = [["a", "c", "b"], ["d"], ["e", "f", "g", "h", "d"]]


def test_calculate_ranking_metrics_basic():
    y_true = [["a", "b"]]
    y_pred = [["a", "c", "b"]]
    k = 3
    metrics = calculate_ranking_metrics(y_true, y_pred, k=k)

    # nDCG@3: relevance=[1,0,1], ideal=[1,1,0]
    # dcg = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    # idcg = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    # ndcg = 1.5 / 1.6309 = 0.92
    assert pytest.approx(metrics[f"ndcg@{k}"], 0.01) == 0.919

    # MAP@3: precisions = [1/1, 2/3], min(len(true), k) = 2
    # ap = (1 + 0.666) / 2 = 0.833
    assert pytest.approx(metrics[f"map@{k}"], 0.01) == 0.833

    # MRR: first correct is at pos 1
    assert metrics["mrr"] == 1.0


def test_calculate_ranking_metrics_no_relevant_pred():
    y_true = [["a", "b"]]
    y_pred = [["c", "d", "e"]]
    metrics = calculate_ranking_metrics(y_true, y_pred, k=3)
    assert metrics["ndcg@3"] == 0.0
    assert metrics["map@3"] == 0.0
    assert metrics["mrr"] == 0.0


def test_calculate_ranking_metrics_empty_true():
    y_true = [[]]
    y_pred = [["a", "b"]]
    metrics = calculate_ranking_metrics(y_true, y_pred, k=2)
    assert metrics["ndcg@2"] == 0.0
    assert metrics["map@2"] == 0.0
    assert metrics["mrr"] == 0.0


# Test data for Ordinal
y_true_ord = [1, 2, 3, 4, 5]
y_pred_ord = [1, 1, 3, 5, 4]


def test_calculate_ordinal_metrics_perfect_agreement():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    metrics = calculate_ordinal_metrics(y_true, y_pred)
    assert metrics["weighted_kappa"] == 1.0
    assert metrics["spearman_rho"] == 1.0
    assert metrics["kendall_tau"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_calculate_ordinal_metrics_no_agreement():
    y_true = [1, 2, 3]
    y_pred = [3, 2, 1]
    metrics = calculate_ordinal_metrics(y_true, y_pred)
    assert metrics["spearman_rho"] == -1.0
    assert metrics["kendall_tau"] == -1.0
    assert metrics["accuracy"] < 1.0


def test_calculate_ordinal_metrics_with_nan():
    # spearmanr returns nan if input is constant
    y_true = [1, 1, 1]
    y_pred = [1, 1, 1]
    metrics = calculate_ordinal_metrics(y_true, y_pred)
    assert metrics["spearman_rho"] == 0.0  # Should handle nan and return 0
    assert metrics["kendall_tau"] == 0.0  # Also can be nan
    assert metrics["weighted_kappa"] == 1.0
    assert metrics["accuracy"] == 1.0


# Test data for simple metrics
y_true_simple = [[1, 1, 0], [1, 0, 0], [0, 1, 1]]
y_pred_simple = [[1, 1, 0], [0, 1, 0], [0, 1, 0]]


def test_calculate_metrics_basic():
    metrics = calculate_metrics(y_true_simple, y_pred_simple)
    # hamming: (0+2+1)/ (3*3) = 3/9 = 0.333
    assert pytest.approx(metrics["hamming_loss"]) == 3 / 9
    # f1:
    # class 0: p=1/1, r=1/2, f1=2/3
    # class 1: p=3/3, r=3/3, f1=1
    # class 2: p=0/0, r=0/1, f1=0
    # macro_f1 = (f1_class0 + f1_class1 + f1_class2) / 3
    # f1_class0 = 2/3; f1_class1 = 0.8; f1_class2 = 0.0
    # macro_f1 = (0.666... + 0.8 + 0) / 3 = 0.4888...
    assert pytest.approx(metrics["macro_f1"]) == ((2 / 3) + 0.8) / 3


def test_calculate_metrics_perfect():
    y_true = [[1, 0], [0, 1]]
    y_pred = [[1, 0], [0, 1]]
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["hamming_loss"] == 0.0
    assert metrics["macro_f1"] == 1.0


def test_bootstrap_confidence_interval():
    data = np.random.randn(100)
    lower, upper = bootstrap_confidence_interval(data, n_bootstraps=100)
    assert lower <= upper
    assert isinstance(lower, float)
    assert isinstance(upper, float)


def test_bootstrap_confidence_interval_empty_data():
    lower, upper = bootstrap_confidence_interval([])
    assert lower == 0.0
    assert upper == 0.0


def test_bootstrap_confidence_interval_constant_data():
    data = [5.0] * 20
    lower, upper = bootstrap_confidence_interval(data)
    assert lower == 5.0
    assert upper == 5.0
