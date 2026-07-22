"""Tests for root cause analysis and autoencoder trainers.

Validates supervised (root cause) and unsupervised (autoencoder) model training
on synthetic privacy feature datasets.
"""

import numpy as np
import pytest

from privacy_harm_heuristics.models.trainers.root_cause import train_root_cause

# autoencoder.py does an unguarded `import torch` (unlike ebm.py/brl.py's optional
# try/except pattern) and torch isn't a base dependency of this package (see
# pyproject.toml). Import it lazily inside test_train_autoencoder (guarded by
# importorskip, matching test_dnn_baseline.py) so this module still collects,
# and test_train_root_cause (torch-independent) still runs, when torch is absent.


def dummy_dataset(n=40):
    # Two features strongly correlated with negative sentiment
    rng = np.random.default_rng(42)
    pf_always_on = rng.integers(0, 2, size=n)
    pf_session_replay = rng.integers(0, 2, size=n)
    noise = rng.integers(0, 2, size=(n, 3))
    X = np.column_stack([pf_always_on, pf_session_replay, noise])
    # Negative sentiment if (always_on or session_replay) with probability
    sentiment = []
    for a, s in zip(pf_always_on, pf_session_replay):
        if a or s:
            sentiment.append(-0.5)
        else:
            sentiment.append(0.3)
    return (
        X,
        ["pf_always_on_listening", "pf_session_replay", "pf_noise1", "pf_noise2", "pf_noise3"],
        sentiment,
    )


def test_train_root_cause():
    X, feature_names, sentiment = dummy_dataset()
    result = train_root_cause(X, feature_names, sentiment)
    assert result.model_type == "root_cause"
    # Coefficients should include our main features
    important = {c["feature"] for c in result.extra["coefficients"][:3]}
    assert "pf_always_on_listening" in important or "pf_session_replay" in important


def test_train_autoencoder(tmp_path):
    pytest.importorskip("torch")
    from privacy_harm_heuristics.models.trainers.autoencoder import train_autoencoder

    X, feature_names, _ = dummy_dataset()
    checkpoint_dir = tmp_path / "autoencoder_test"
    result = train_autoencoder(
        X, feature_names, epochs=2, latent_dim=4, batch_size=8, checkpoint_dir=checkpoint_dir
    )
    assert result.model_type == "autoencoder"
    assert "reconstruction_loss" in result.metrics
    assert result.extra["checkpoint_dir"]
    # Verify checkpoint was created
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "checkpoint_2.pt").exists()
