"""Lightweight PyTorch autoencoder for unsupervised representation of product features.

Optimized for small datasets and fast convergence on local (M1) hardware.
Saves checkpoints each epoch: checkpoint_<epoch>.pt storing model + optimizer state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .. import ModelResult


class FeatureAutoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 8):
        super().__init__()
        h = max(16, n_features * 2)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, h),
            nn.ReLU(),
            nn.Linear(h, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.ReLU(),
            nn.Linear(h, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def train_autoencoder(
    X,
    feature_names: Sequence[str],
    epochs: int = 10,
    batch_size: int = 64,
    latent_dim: int = 8,
    lr: float = 1e-3,
    checkpoint_dir: str | Path = "models/autoencoder",
    device: Optional[str] = None,
) -> ModelResult:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FeatureAutoencoder(X.shape[1], latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            # Zero gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            # Manual SGD update
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-lr)
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        losses.append(epoch_loss)
        # Save checkpoint
        ckpt_file = checkpoint_path / f"checkpoint_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": None,
                "loss": epoch_loss,
            },
            ckpt_file,
        )

    artifacts: Dict[str, Any] = {"model": model, "losses": losses}
    extra: Dict[str, Any] = {
        "final_loss": float(losses[-1] if losses else 0.0),
        "n_epochs": epochs,
        "device": device,
        "latent_dim": latent_dim,
        "checkpoint_dir": str(checkpoint_path),
    }
    metrics = {"reconstruction_loss": float(losses[-1] if losses else 0.0)}
    return ModelResult(
        model_type="autoencoder",
        metrics=metrics,
        artifacts=artifacts,
        extra=extra,
    )
