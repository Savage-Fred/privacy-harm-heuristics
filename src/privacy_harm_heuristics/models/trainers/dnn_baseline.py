"""Deep Neural Network baseline trainer for privacy sentiment classification.

A multi-layer feedforward network as a baseline for comparing against interpretable models.
This provides a standard DNN architecture for performance comparison, though it lacks
the explainability of interpretable alternatives (BRL, decision trees, sparse linear).

Architecture: input -> hidden layers with dropout -> binary classification
Training: Adam optimizer with early stopping capability
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .. import ModelResult


class PrivacyDNN(nn.Module):
    """Deep neural network for binary privacy sentiment classification.

    Args:
        n_features: Number of input features
        hidden_sizes: List of hidden layer sizes (default: [128, 64, 32])
        dropout: Dropout probability for regularization (default: 0.3)
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        layers: List[nn.Module] = []
        prev_size = n_features

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer for binary classification
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


def train_dnn_baseline(
    X,
    feature_names: Sequence[str],
    sentiment_scores,
    hidden_sizes: list[int] | None = None,
    dropout: float = 0.3,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
    checkpoint_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
    early_stopping_patience: int = 10,
    min_delta: float = 1e-4,
) -> ModelResult:
    """Train a deep neural network baseline for privacy sentiment classification.

    Args:
        X: Feature matrix (numpy array or similar)
        feature_names: List of feature names
        sentiment_scores: Sentiment scores (will be binarized: <0 = negative)
        hidden_sizes: List of hidden layer sizes (default: [128, 64, 32])
        dropout: Dropout probability (default: 0.3)
        epochs: Maximum number of training epochs (default: 50)
        batch_size: Batch size for training (default: 64)
        lr: Learning rate (default: 1e-3)
        test_size: Fraction of data for validation (default: 0.2)
        random_state: Random seed for reproducibility
        checkpoint_dir: Directory to save model checkpoints
        device: Device to train on ('cpu', 'cuda', 'mps', or None for auto)
        early_stopping_patience: Number of epochs without improvement before stopping

    Returns:
        ModelResult with trained model, metrics, and metadata
    """
    # Binarize sentiment: negative if score < 0
    y = np.array([1 if score < 0 else 0 for score in sentiment_scores])

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Setup device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Initialize model
    model = PrivacyDNN(
        n_features=X.shape[1],
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)

    # Use a simple manual gradient descent loop to avoid optimizer import issues in some
    # PyTorch builds/environments.
    criterion = nn.BCELoss()

    # Training loop with early stopping
    train_losses = []
    val_losses = []
    val_f1_scores = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    checkpoint_path = None
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Zero gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            # Manual SGD update
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-lr)

            epoch_train_loss += loss.item() * batch_X.size(0)

        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor.to(device))
            val_loss = criterion(val_predictions, y_val_tensor.to(device))
            val_losses.append(val_loss.item())

            # Calculate F1 for validation
            val_pred_labels = (val_predictions.cpu().numpy() > 0.5).astype(int).flatten()
            val_f1 = f1_score(y_val, val_pred_labels, average="weighted")
            val_f1_scores.append(val_f1)

        # Early stopping check
        # Consider it an improvement only if it decreases by at least min_delta
        if (best_val_loss - val_loss.item()) > min_delta:
            best_val_loss = val_loss.item()
            best_epoch = epoch
            patience_counter = 0

            # Save best model checkpoint
            if checkpoint_path:
                best_model_path = checkpoint_path / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": None,
                        "train_loss": epoch_train_loss,
                        "val_loss": val_loss.item(),
                        "val_f1": val_f1,
                    },
                    best_model_path,
                )
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    # Final evaluation on validation set
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor.to(device)).cpu().numpy()

    val_pred_labels = (val_predictions > 0.5).astype(int).flatten()

    # Compute metrics
    accuracy = accuracy_score(y_val, val_pred_labels)
    precision = precision_score(y_val, val_pred_labels, zero_division=0)
    recall = recall_score(y_val, val_pred_labels, zero_division=0)
    f1 = f1_score(y_val, val_pred_labels, average="weighted")

    metrics = {
        "accuracy": float(accuracy),
        "precision_neg": float(precision),
        "recall_neg": float(recall),
        "f1_weighted": float(f1),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(train_losses[-1]) if train_losses else 0.0,
    }

    artifacts: Dict[str, Any] = {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_f1_scores": val_f1_scores,
    }

    extra: Dict[str, Any] = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "negative_rate": float(y.mean()),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "hidden_sizes": hidden_sizes or [128, 64, 32],
        "dropout": dropout,
        "n_epochs_trained": len(train_losses),
        "best_epoch": best_epoch,
        "early_stopped": patience_counter >= early_stopping_patience,
        "device": device,
        "checkpoint_dir": str(checkpoint_path) if checkpoint_path else None,
    }

    return ModelResult(
        model_type="dnn_baseline",
        metrics=metrics,
        artifacts=artifacts,
        extra=extra,
    )
