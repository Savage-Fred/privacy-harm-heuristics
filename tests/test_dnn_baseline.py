"""Tests for DNN baseline trainer module."""

import numpy as np
import pytest

# Skip if torch not available
pytest.importorskip("torch")

from privacy_harm_heuristics.models.trainers.dnn_baseline import PrivacyDNN, train_dnn_baseline


def test_privacy_dnn_initialization():
    """Test that PrivacyDNN can be initialized with various configurations."""
    # Default hidden sizes
    model = PrivacyDNN(n_features=50)
    assert model is not None

    # Custom hidden sizes
    model = PrivacyDNN(n_features=50, hidden_sizes=[64, 32], dropout=0.2)
    assert model is not None

    # Single hidden layer
    model = PrivacyDNN(n_features=50, hidden_sizes=[32], dropout=0.0)
    assert model is not None


def test_privacy_dnn_forward_pass():
    """Test forward pass through the network."""
    import torch

    model = PrivacyDNN(n_features=10, hidden_sizes=[16, 8])
    x = torch.randn(5, 10)  # batch of 5 samples with 10 features
    output = model(x)

    assert output.shape == (5, 1)  # Binary classification output
    assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output in [0, 1]


def test_train_dnn_baseline_structure():
    """Test that train_dnn_baseline creates correct structure without full training."""
    # Generate minimal synthetic data
    np.random.seed(42)
    n_samples = 50
    n_features = 10

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    sentiment_scores = np.random.randn(n_samples) * 2  # Range around -2 to +2
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Train with minimal epochs for speed
    result = train_dnn_baseline(
        X=X,
        feature_names=feature_names,
        sentiment_scores=sentiment_scores,
        hidden_sizes=[16, 8],
        epochs=2,  # Very short training for test speed
        batch_size=16,
        early_stopping_patience=5,
        random_state=42,
    )

    # Check ModelResult structure
    assert result.model_type == "dnn_baseline"
    assert "accuracy" in result.metrics
    assert "precision_neg" in result.metrics
    assert "recall_neg" in result.metrics
    assert "f1_weighted" in result.metrics
    assert "best_val_loss" in result.metrics

    # Check artifacts
    assert "model" in result.artifacts
    assert "train_losses" in result.artifacts
    assert "val_losses" in result.artifacts
    assert "val_f1_scores" in result.artifacts

    # Check extra metadata
    assert result.extra["n_samples"] == n_samples
    assert result.extra["n_features"] == n_features
    assert "negative_rate" in result.extra
    assert result.extra["hidden_sizes"] == [16, 8]
    assert result.extra["n_epochs_trained"] >= 1


def test_train_dnn_baseline_binary_labels():
    """Test that sentiment scores are correctly binarized."""
    np.random.seed(42)
    n_samples = 30
    n_features = 5

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # All positive sentiment
    sentiment_scores = np.ones(n_samples) * 0.5
    feature_names = [f"feature_{i}" for i in range(n_features)]

    result = train_dnn_baseline(
        X=X,
        feature_names=feature_names,
        sentiment_scores=sentiment_scores,
        epochs=1,
        batch_size=10,
        random_state=42,
    )

    # With all positive sentiment, negative_rate should be 0
    assert result.extra["negative_rate"] == 0.0

    # All negative sentiment
    sentiment_scores = np.ones(n_samples) * -0.5
    result = train_dnn_baseline(
        X=X,
        feature_names=feature_names,
        sentiment_scores=sentiment_scores,
        epochs=1,
        batch_size=10,
        random_state=42,
    )

    # With all negative sentiment, negative_rate should be 1
    assert result.extra["negative_rate"] == 1.0


def test_train_dnn_baseline_train_val_split():
    """Test that train/validation split is working correctly."""
    np.random.seed(42)
    n_samples = 100
    n_features = 8

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    sentiment_scores = np.random.randn(n_samples)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    result = train_dnn_baseline(
        X=X,
        feature_names=feature_names,
        sentiment_scores=sentiment_scores,
        test_size=0.2,
        epochs=2,
        random_state=42,
    )

    # Check split sizes
    assert result.extra["n_train"] == 80
    assert result.extra["n_val"] == 20
    assert result.extra["n_train"] + result.extra["n_val"] == n_samples


def test_train_dnn_baseline_checkpoint_dir(tmp_path):
    """Test that checkpoints are saved when checkpoint_dir is provided."""
    np.random.seed(42)
    n_samples = 40
    n_features = 6

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    sentiment_scores = np.random.randn(n_samples)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    checkpoint_dir = tmp_path / "dnn_checkpoints"

    result = train_dnn_baseline(
        X=X,
        feature_names=feature_names,
        sentiment_scores=sentiment_scores,
        epochs=3,
        checkpoint_dir=str(checkpoint_dir),
        random_state=42,
    )

    # Check that checkpoint directory was created
    assert checkpoint_dir.exists()
    assert result.extra["checkpoint_dir"] == str(checkpoint_dir)

    # Check that best model checkpoint exists
    best_model_path = checkpoint_dir / "best_model.pt"
    assert best_model_path.exists()


def test_train_dnn_baseline_early_stopping():
    """Test that early stopping works when validation loss doesn't improve."""
    np.random.seed(42)
    n_samples = 50
    n_features = 10

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    sentiment_scores = np.random.randn(n_samples)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    result = train_dnn_baseline(
        X=X,
        feature_names=feature_names,
        sentiment_scores=sentiment_scores,
        epochs=100,  # Many epochs
        early_stopping_patience=3,  # But stop early
        random_state=42,
    )

    # Should stop before 100 epochs
    assert result.extra["n_epochs_trained"] < 100
    assert "early_stopped" in result.extra
    assert "best_epoch" in result.extra


def test_train_dnn_baseline_device_selection():
    """Test device selection logic."""
    np.random.seed(42)
    n_samples = 30
    n_features = 5

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    sentiment_scores = np.random.randn(n_samples)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Force CPU device
    result = train_dnn_baseline(
        X=X,
        feature_names=feature_names,
        sentiment_scores=sentiment_scores,
        epochs=1,
        device="cpu",
        random_state=42,
    )

    assert result.extra["device"] == "cpu"
