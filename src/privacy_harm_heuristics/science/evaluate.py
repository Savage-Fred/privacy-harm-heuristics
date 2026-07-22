import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from privacy_harm_heuristics.science.scoring import ScoringEngine, HybridScorer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Anchor to the repo root (this file lives at src/privacy_harm_heuristics/science/)
# instead of the caller's CWD, so `python -m ...evaluate` works from anywhere.
# PHH_DATA_DIR overrides for callers that keep data elsewhere (e.g. Cloud Run).
DATA_DIR = Path(os.environ.get("PHH_DATA_DIR") or Path(__file__).resolve().parents[3] / "data")
GOLDEN_DATA_PATH = DATA_DIR / "golden_cases.jsonl"


def load_gold_cases(path: Path) -> List[Dict[str, Any]]:
    cases = []
    if not path.exists():
        logger.error(f"Golden data not found at {path}")
        return []

    with open(path, "r") as f:
        for line in f:
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return cases


def calculate_metrics(true_positives: int, false_positives: int, false_negatives: int):
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_hybrid_scorer():
    logger.info("Initializing Hybrid Scorer...")
    engine = ScoringEngine(data_dir=DATA_DIR)

    # Attempt to load calibrated weights if available, otherwise default
    try:
        calibrated_path = DATA_DIR / "calibrated_weights.json"
        if calibrated_path.exists():
            engine.weights = json.loads(calibrated_path.read_text())
            logger.info("Loaded calibrated weights.")
        else:
            logger.warning("Calibrated weights not found, using defaults.")
    except Exception as e:
        logger.warning(f"Failed to load calibrated weights: {e}")

    scorer = HybridScorer(engine)

    cases = load_gold_cases(GOLDEN_DATA_PATH)
    logger.info(f"Loaded {len(cases)} golden cases.")

    tp_total = 0
    fp_total = 0
    fn_total = 0

    logger.info("\n--- Starting Evaluation ---")

    for i, case in enumerate(cases):
        description = case.get("description", "")
        # Ground truth: 'harms' list is the most reliable if present, else single 'harm_category'
        gold_harms = set(case.get("harms", []))
        if not gold_harms and "harm_category" in case:
            gold_harms.add(case["harm_category"])

        # Skip if no ground truth
        if not gold_harms:
            continue

        # Run Hybrid Scorer
        # Note: In a real eval for the LLM component, this would make live API calls.
        # For cost/speed, this script relies on the 'fallback' keyword provider inside classify_privacy_relevance
        # UNLESS the user has set up API keys.
        # Ideally, we should mock the LLM or use a cached provider for deterministic checks.
        # However, classify_privacy_relevance has a fallback mode.

        # Run Hybrid Scorer with fallback to avoid API timeouts
        result = scorer.hybrid_score(description, provider="fallback")
        predicted_harms = set(result["validated_harms"])

        # Calculate case stats
        tp = len(gold_harms.intersection(predicted_harms))
        fp = len(predicted_harms - gold_harms)
        fn = len(gold_harms - predicted_harms)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        if i % 10 == 0:
            print(f"Processed {i}/{len(cases)} cases...", end="\r")

    logger.info("\n\n--- Evaluation Results ---")
    precision, recall, f1 = calculate_metrics(tp_total, fp_total, fn_total)

    logger.info(f"Total Cases: {len(cases)}")
    logger.info(f"True Positives: {tp_total}")
    logger.info(f"False Positives: {fp_total}")
    logger.info(f"False Negatives: {fn_total}")
    logger.info("-" * 20)
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall:    {recall:.3f}")
    logger.info(f"F1 Score:  {f1:.3f}")
    logger.info("-" * 20)

    target_f1 = 0.531
    if f1 >= target_f1:
        logger.info(f"SUCCESS: F1 ({f1:.3f}) meets or exceeds paper baseline ({target_f1})")
    else:
        logger.warning(f"FAILURE: F1 ({f1:.3f}) is below paper baseline ({target_f1})")
        logger.info("Recommendation: Adjust VALIDATION_THRESHOLD or improve LLM prompt/heuristics.")


if __name__ == "__main__":
    evaluate_hybrid_scorer()
