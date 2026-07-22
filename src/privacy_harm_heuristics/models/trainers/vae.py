"""Lightweight Variational Autoencoder (VAE) for unsupervised feature representation.

Optimized for small tabular binary indicator feature sets (product features) and
M1/M2 Mac constraints. Uses a simple Gaussian latent prior and reparameterization.

Implementation note: We avoid torch.optim Adam on certain macOS/Python combos that
can implicitly pull torch._dynamo/functorch modules not present in minimal installs.
Instead, we perform a manual SGD parameter update under ``torch.no_grad()``.

Artifacts saved each epoch: checkpoint_<epoch>.pt with model + losses (no optimizer).
Final ModelResult contains latent dimension and final reconstruction+KL metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .. import ModelResult


class VAE(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 8, hidden_mult: int = 2):
        super().__init__()
        h = max(16, n_features * hidden_mult)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, h),
            nn.ReLU(),
        )
        self.mu = nn.Linear(h, latent_dim)
        self.logvar = nn.Linear(h, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.ReLU(),
            nn.Linear(h, n_features),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def _vae_loss(recon, x, mu, logvar):
    bce = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
    # KL divergence component
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0), bce / x.size(0), kld / x.size(0)


def train_vae(
    X,
    feature_names: Sequence[str],
    epochs: int = 15,
    batch_size: int = 64,
    latent_dim: int = 8,
    lr: float = 1e-3,
    checkpoint_dir: str | Path = "models/vae",
    device: Optional[str] = None,
) -> ModelResult:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    # BCE loss expects inputs in [0, 1]; clip feature matrix so newer torch releases
    # do not raise runtime errors when risk counters dip below zero.
    X_clipped = np.clip(X, 0.0, 1.0)
    X_tensor = torch.tensor(X_clipped, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VAE(X.shape[1], latent_dim=latent_dim).to(device)
    ckpt_dir_path = Path(checkpoint_dir)
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            # Zero grads, forward, loss
            model.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            loss, bce, kld = _vae_loss(recon, batch, mu, logvar)
            loss.backward()
            # Manual SGD step to avoid torch._dynamo/functorch optimizer imports
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-lr)
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        losses.append(epoch_loss)
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "loss": epoch_loss,
            },
            ckpt_dir_path / f"checkpoint_{epoch}.pt",
        )

    artifacts: Dict[str, Any] = {"model": model, "losses": losses}
    extra: Dict[str, Any] = {
        "final_loss": float(losses[-1] if losses else 0.0),
        "n_epochs": epochs,
        "latent_dim": latent_dim,
        "device": device,
        "checkpoint_dir": str(ckpt_dir_path),
    }
    metrics = {"loss": extra["final_loss"]}
    return ModelResult(
        model_type="vae",
        metrics=metrics,
        artifacts=artifacts,
        extra=extra,
    )
