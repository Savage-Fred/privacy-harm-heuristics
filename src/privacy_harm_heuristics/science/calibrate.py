import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from sklearn.linear_model import LinearRegression

from privacy_harm_heuristics.labeling.harm_labeler import label_harm_category, HARM_CATEGORIES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Anchored to the repo root (parents[3] from this file) rather than CWD;
# PHH_DATA_DIR overrides for callers that keep data elsewhere (e.g. Cloud Run).
ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("PHH_DATA_DIR") or ROOT_DIR / "data")
GOLDEN_FILE = DATA_DIR / "golden_cases.jsonl"
OUTPUT_FILE = DATA_DIR / "calibrated_weights.json"


def load_golden_cases(path: Path) -> List[Dict[str, Any]]:
    cases = []
    if not path.exists():
        logger.error(f"Golden file not found: {path}")
        return []

    with open(path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return cases


def featurize_case(case: Dict[str, Any]) -> Dict[str, float]:
    """Extract features (category scores) for a case."""
    # We use the same labeling logic as the app
    # Note: label_harm_category returns a dict of scores when return_scores=True
    scores = label_harm_category(case, return_scores=True, text_fields=["description"])

    # We also want to capture the 'keyword' count proxy if possible,
    # but for this V1 calibration, we'll focus on the semantic categories.
    return scores


def calibrate() -> None:
    """Run calibration routine."""
    logger.info("Starting calibration task...")

    cases = load_golden_cases(GOLDEN_FILE)
    if not cases:
        logger.error("No cases loaded. Aborting.")
        return

    logger.info(f"Loaded {len(cases)} golden cases.")

    # 1. Build Feature Matrix X and Target Y
    # Features = Score for each HARM_CATEGORY
    # Target = harm_score (0-10) -> scaled to 0-1 for stability if needed, but 0-10 is fine for linear

    feature_names = list(HARM_CATEGORIES.keys())
    X_rows = []
    y = []

    for case in cases:
        if "harm_score" not in case:
            continue

        feat_dict = featurize_case(case)
        row = [feat_dict.get(cat, 0.0) for cat in feature_names]
        X_rows.append(row)
        y.append(float(case["harm_score"]))

    X = np.array(X_rows)
    y_vec = np.array(y)

    # 2. Fit Linear Regression (positive coefficients only)
    # We want weights to be additive risks.
    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(X, y_vec)

    # 3. Extract Weights
    weights: Dict[str, float] = {}
    for name, coef in zip(feature_names, reg.coef_):
        # Round for cleanliness
        weights[name] = round(float(coef), 3)

    # Log results
    logger.info(f"Calibration complete. R2 Score: {reg.score(X, y_vec):.3f}")
    logger.info("Top detected risk drivers:")
    for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)[:5]:
        logger.info(f"  {k}: {v}")

    # 4. Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(weights, f, indent=2)

    logger.info(f"Saved weights to {OUTPUT_FILE}")


if __name__ == "__main__":
    calibrate()
