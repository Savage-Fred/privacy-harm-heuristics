"""Unsupervised neural analysis of privacy harm indicators.

This module trains a light-weight autoencoder over the feature matrix
derived from ``with_features.jsonl``. It does **not** use harm labels
during training (fully unsupervised), but after reconstruction it
correlates error patterns with recorded harm scores to highlight which
features are most associated with privacy risk.

Usage (CLI):

    practicum unsupervised analyse --data data/with_features.jsonl

Outputs include:
  - reconstruction loss statistics
  - correlation between reconstruction error and harm score (if present)
  - feature rankings based on encoder weights and harm correlation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, cast

import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

NUMERIC_PREFIXES: tuple[str, ...] = (
    "kw_",
    "pf_",
    "f_",
    "rc_",
)


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_json(path, lines=True)


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols: List[str] = []
    for column in df.columns:
        if any(column.startswith(prefix) for prefix in NUMERIC_PREFIXES):
            if pd.api.types.is_numeric_dtype(df[column]):
                numeric_cols.append(column)
    return sorted(set(numeric_cols))


@dataclass
class AutoencoderConfig:
    hidden_dim: int = 64
    latent_dim: int = 16
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


@dataclass
class AutoencoderResult:
    loss_history: List[float]
    feature_importance: List[tuple[str, float]]
    harm_correlation: Optional[float]
    feature_harm_correlation: List[tuple[str, float]]
    reconstruction_stats: Dict[str, float]


def train_autoencoder(
    dataset_path: Path,
    config: Optional[AutoencoderConfig] = None,
    feature_columns: Optional[Iterable[str]] = None,
) -> AutoencoderResult:
    cfg = config or AutoencoderConfig()
    df = _load_dataset(dataset_path)

    columns = list(feature_columns) if feature_columns else _select_feature_columns(df)
    if not columns:
        raise ValueError("No numeric feature columns found for autoencoder training.")

    X = df[columns].fillna(0.0).astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    tensor_data = torch.tensor(X_scaled, dtype=torch.float32, device=cfg.device)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )

    model = Autoencoder(input_dim=X.shape[1], hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim)
    model = model.to(cfg.device)
    criterion = torch.nn.MSELoss()

    loss_history: List[float] = []
    model.train()

    # Note: Manual gradient descent is used instead of torch.optim.SGD for the following reasons:
    # 1. Minimal overhead: Avoids optimizer state tracking for simple SGD updates
    # 2. Deterministic behavior: Explicit parameter updates are more predictable for
    #    reproducibility in privacy analysis contexts where we may need to audit the
    #    exact update process
    # 3. Educational clarity: The explicit gradient application makes the learning
    #    process transparent for researchers reviewing the privacy harm detection logic
    # 4. Lightweight deployment: Reduces serialization complexity when deploying models
    #    to constrained environments (Cloud Run, edge devices)
    #
    # For more complex optimizers (Adam, RMSprop), use torch.optim instead.

    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        for (batch,) in dataloader:
            # Zero gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            # Manual SGD update: θ ← θ - lr * ∇θ
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-cfg.lr)
            epoch_loss += loss.item() * batch.size(0)
        loss_history.append(epoch_loss / len(dataset))

    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor_data).cpu().numpy()

    original = tensor_data.cpu().numpy()
    reconstruction_errors = ((original - reconstructed) ** 2).mean(axis=1)
    reconstruction_stats = {
        "mean_error": float(reconstruction_errors.mean()),
        "std_error": float(reconstruction_errors.std()),
        "max_error": float(reconstruction_errors.max()),
    }

    # Feature importance using first-layer weights (higher magnitude -> larger influence)
    first_linear = cast(torch.nn.Linear, model.encoder[0])
    encoder_weight = first_linear.weight.detach().cpu().abs().mean(dim=0).numpy()
    feature_importance = sorted(
        zip(columns, encoder_weight), key=lambda item: item[1], reverse=True
    )

    harm_series = None
    for candidate in ("harm_score", "harm", "risk_score"):
        if candidate in df.columns:
            harm_series = df[candidate].astype(float)
            break

    harm_correlation = None
    feature_harm_correlation: List[tuple[str, float]] = []
    if harm_series is not None:
        valid_mask = harm_series.notna()
        if valid_mask.any():
            harm_values = harm_series[valid_mask].to_numpy()
            recon_valid = reconstruction_errors[valid_mask.to_numpy()]
            harm_correlation = spearmanr(harm_values, recon_valid).correlation

            for col in columns:
                try:
                    corr = spearmanr(df.loc[valid_mask, col], harm_values).correlation
                except Exception:
                    corr = float("nan")
                feature_harm_correlation.append((col, corr))
            feature_harm_correlation.sort(
                key=lambda item: (abs(item[1]) if item[1] is not None else 0.0),
                reverse=True,
            )

    return AutoencoderResult(
        loss_history=loss_history,
        feature_importance=feature_importance,
        harm_correlation=harm_correlation,
        feature_harm_correlation=feature_harm_correlation,
        reconstruction_stats=reconstruction_stats,
    )


def summarize_result(result: AutoencoderResult, top_n: int = 15) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "reconstruction": result.reconstruction_stats,
        "feature_importance": [
            {"feature": feature, "weight": float(weight)}
            for feature, weight in result.feature_importance[:top_n]
        ],
    }
    if result.harm_correlation is not None:
        summary["harm_correlation"] = float(result.harm_correlation)
    if result.feature_harm_correlation:
        summary["top_harm_features"] = [
            {"feature": feature, "spearman": float(corr) if corr is not None else None}
            for feature, corr in result.feature_harm_correlation[:top_n]
        ]
    return summary


def run_privacy_autoencoder(dataset: Path, output: Optional[Path] = None) -> AutoencoderResult:
    result = train_autoencoder(dataset)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        summary = summarize_result(result)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result
