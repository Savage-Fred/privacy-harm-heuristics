"""Multi-dimensional composite risk scoring.

This module implements a weighted geometric mean approach to combine
multiple risk dimensions into a single composite score, while preventing
any single dimension (especially monetary sanctions) from dominating the
overall risk assessment.

Design principles:
- Use geometric mean to prevent dimension dominance
- Apply higher weights to non-monetary dimensions
- Ensure explainability: return both composite and per-dimension scores
- Calibrate probabilities before composition
"""

from __future__ import annotations

from math import prod
from typing import Dict, Optional

# Default weights for composite scoring
# Higher weights for non-monetary dimensions to counter historical bias
DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "monetary_sanction": 1.0,  # Baseline weight
    "user_distress": 1.3,  # Elevated: subjective/emotional harm
    "contextual_integrity": 1.2,  # Elevated: Nissenbaum CI violations
    "sensitive_data_exposure": 1.1,  # Elevated: special category data
}


def composite_score(
    dimensions: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    epsilon: float = 1e-6,
) -> float:
    """Calculate composite risk score from multiple dimensions using weighted geometric mean.

    The weighted geometric mean prevents any single high-scoring dimension from
    dominating the composite, which is critical for avoiding monetary bias.

    Formula: composite = (∏ᵢ (pᵢ + ε)^wᵢ)^(1/Σwᵢ)

    Args:
        dimensions: Dictionary mapping dimension name to risk score [0, 1]
        weights: Optional custom weights per dimension (defaults to DEFAULT_DIMENSION_WEIGHTS)
        epsilon: Small constant to avoid log(0) issues

    Returns:
        Composite risk score in [0, 1]

    Raises:
        ValueError: If dimension scores are not in [0, 1] or weights are negative

    Example:
        >>> dimensions = {
        ...     "monetary_sanction": 0.8,
        ...     "user_distress": 0.4,
        ...     "contextual_integrity": 0.5,
        ...     "sensitive_data_exposure": 0.6
        ... }
        >>> score = composite_score(dimensions)
        >>> 0.0 <= score <= 1.0
        True
    """
    if not dimensions:
        return 0.0

    # Validate inputs
    for dim, score in dimensions.items():
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Dimension '{dim}' has score {score} outside valid range [0, 1]")

    # Use default weights if not provided
    if weights is None:
        weights = DEFAULT_DIMENSION_WEIGHTS

    # Only use dimensions that have both score and weight
    active_dimensions = [dim for dim in dimensions.keys() if dim in weights]

    if not active_dimensions:
        # Fall back to simple mean if no weights match
        return sum(dimensions.values()) / len(dimensions)

    # Validate weights
    for dim in active_dimensions:
        if weights[dim] < 0:
            raise ValueError(f"Weight for dimension '{dim}' must be non-negative")

    # Calculate weighted geometric mean
    weighted_scores = [(dimensions[dim] + epsilon) ** weights[dim] for dim in active_dimensions]

    total_weight = sum(weights[dim] for dim in active_dimensions)

    if total_weight == 0:
        return 0.0

    geometric_mean = prod(weighted_scores) ** (1.0 / total_weight)

    # Remove epsilon contribution and clamp to [0, 1]
    result = geometric_mean - epsilon
    return max(0.0, min(1.0, result))


def dimension_scores_from_features(
    features: Dict[str, float],
    feature_to_dimension_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Aggregate feature-level scores into dimension-level scores.

    Args:
        features: Dictionary mapping feature names to their risk contributions
        feature_to_dimension_map: Optional mapping of features to dimensions

    Returns:
        Dictionary mapping dimension names to aggregated scores
    """
    if feature_to_dimension_map is None:
        # Default mapping based on feature name prefixes
        feature_to_dimension_map = _default_feature_to_dimension_map()

    dimension_contributions: Dict[str, list] = {
        "monetary_sanction": [],
        "user_distress": [],
        "contextual_integrity": [],
        "sensitive_data_exposure": [],
    }

    for feature, score in features.items():
        dimension = feature_to_dimension_map.get(feature)
        if dimension and dimension in dimension_contributions:
            dimension_contributions[dimension].append(score)

    # Aggregate: use max score per dimension to avoid double-counting
    dimension_scores = {}
    for dimension, scores in dimension_contributions.items():
        if scores:
            dimension_scores[dimension] = max(scores)
        else:
            dimension_scores[dimension] = 0.0

    return dimension_scores


def _default_feature_to_dimension_map() -> Dict[str, str]:
    """Create default mapping of feature names to risk dimensions."""
    mapping = {}

    # Monetary features
    monetary_features = [
        "penalty_amount",
        "fine",
        "settlement",
        "kw_monetary_penalty",
        "kw_reg_enforcement",
    ]
    for feat in monetary_features:
        mapping[feat] = "monetary_sanction"

    # User distress features
    distress_features = [
        "sentiment_score",
        "sentiment_label",
        "user_distress_indicators",
        "kw_harassment",
        "kw_anxiety",
    ]
    for feat in distress_features:
        mapping[feat] = "user_distress"

    # Contextual integrity features
    context_features = [
        "contextual_mismatch",
        "kw_secondary_use",
        "kw_unexpected",
        "kw_consent",
    ]
    for feat in context_features:
        mapping[feat] = "contextual_integrity"

    # Sensitive data features
    sensitivity_features = [
        "data_sensitivity_level",
        "kw_biometric",
        "kw_health",
        "kw_location",
        "kw_minors",
    ]
    for feat in sensitivity_features:
        mapping[feat] = "sensitive_data_exposure"

    return mapping


def explain_composite_score(
    composite: float,
    dimensions: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> str:
    """Generate human-readable explanation of composite score calculation.

    Args:
        composite: The calculated composite score
        dimensions: Individual dimension scores
        weights: Weights used in calculation

    Returns:
        Formatted explanation string
    """
    if weights is None:
        weights = DEFAULT_DIMENSION_WEIGHTS

    lines = [
        f"Composite Risk Score: {composite:.3f}",
        "",
        "Dimension Breakdown:",
    ]

    # Sort by contribution to composite (weighted score)
    dimension_items = [(dim, score, weights.get(dim, 1.0)) for dim, score in dimensions.items()]
    dimension_items.sort(key=lambda x: x[1] * x[2], reverse=True)

    for dim, score, weight in dimension_items:
        lines.append(f"  - {dim}: {score:.3f} (weight: {weight:.1f})")

    lines.append("")
    lines.append("Calculation: Weighted geometric mean prevents dimension dominance")

    return "\n".join(lines)
