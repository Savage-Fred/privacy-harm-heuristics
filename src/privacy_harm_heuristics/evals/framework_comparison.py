"""Compare empirical models against expert-consensus frameworks.

This module evaluates how well our data-driven models predict privacy risks
compared to established expert frameworks like NIST, Solove, ISO 29100, etc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from ..models.eval.metrics import (
    compute_comprehensive_metrics,
    format_metrics_for_output_contract,
)

logger = logging.getLogger(__name__)


@dataclass
class FrameworkPrediction:
    """Prediction made by an expert framework."""

    framework_name: str
    predicted_harm: str
    predicted_severity: str  # low, medium, high, critical
    predicted_likelihood: str  # unlikely, possible, likely, certain
    confidence: float
    reasoning: str


@dataclass
class ModelPrediction:
    """Prediction made by an empirical model."""

    model_name: str
    predicted_harm: str
    probability: float
    reasoning: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of comparing model vs framework on a test case."""

    case_id: str
    case_description: str
    actual_outcome: str

    framework_predictions: List[FrameworkPrediction]
    model_predictions: List[ModelPrediction]

    framework_correct: Dict[str, bool]
    model_correct: Dict[str, bool]

    best_framework: Optional[str]
    best_model: Optional[str]


class NISTFrameworkEvaluator:
    """Evaluate privacy risks using NIST Privacy Framework principles."""

    def __init__(self):
        self.name = "NIST Privacy Framework"

    def predict(self, case: Dict[str, Any]) -> FrameworkPrediction:
        """Predict using NIST framework principles."""
        description = case.get("description", "")

        # Simple keyword-based assessment (would be more sophisticated in practice)
        severity = "medium"
        likelihood = "possible"
        predicted_harm = "regulatory_violation"
        confidence = 0.6
        reasoning = "Based on NIST Privacy Framework assessment"

        # Check for high-severity indicators
        high_severity_keywords = ["breach", "exposure", "unauthorized", "leak"]
        if any(kw in description.lower() for kw in high_severity_keywords):
            severity = "high"
            likelihood = "likely"
            predicted_harm = "insecurity"
            confidence = 0.8

        # Check for data minimization issues
        if "excessive" in description.lower() or "unnecessary" in description.lower():
            predicted_harm = "aggregation"
            confidence = 0.7

        return FrameworkPrediction(
            framework_name=self.name,
            predicted_harm=predicted_harm,
            predicted_severity=severity,
            predicted_likelihood=likelihood,
            confidence=confidence,
            reasoning=reasoning,
        )


class SoloveFrameworkEvaluator:
    """Evaluate privacy risks using Solove's Taxonomy."""

    def __init__(self):
        self.name = "Solove's Taxonomy"

        # Load taxonomy structure
        self.harm_categories = {
            "surveillance": ["tracking", "monitoring", "watching"],
            "interrogation": ["questioning", "investigating"],
            "aggregation": ["combining", "profiling", "aggregating"],
            "identification": ["identifying", "de-anonymizing"],
            "insecurity": ["breach", "leak", "unauthorized access"],
            "secondary_use": ["repurposing", "selling", "sharing"],
            "exclusion": ["deny", "restrict", "exclude"],
            "disclosure": ["revealing", "exposing", "publishing"],
            "intrusion": ["spam", "harassment", "unwanted"],
            "decisional_interference": ["manipulation", "coercion", "dark pattern"],
        }

    def predict(self, case: Dict[str, Any]) -> FrameworkPrediction:
        """Predict using Solove's taxonomy."""
        description = case.get("description", "").lower()

        # Find best matching harm category
        best_harm = "regulatory_violation"
        best_score = 0.0

        for harm, keywords in self.harm_categories.items():
            score = sum(1 for kw in keywords if kw in description)
            if score > best_score:
                best_score = score
                best_harm = harm

        # Determine severity based on score
        if best_score >= 3:
            severity = "high"
            likelihood = "likely"
            confidence = 0.9
        elif best_score >= 2:
            severity = "medium"
            likelihood = "possible"
            confidence = 0.7
        else:
            severity = "medium"
            likelihood = "possible"
            confidence = 0.5

        reasoning = f"Matched {best_score} keywords in Solove's {best_harm} category"

        return FrameworkPrediction(
            framework_name=self.name,
            predicted_harm=best_harm,
            predicted_severity=severity,
            predicted_likelihood=likelihood,
            confidence=confidence,
            reasoning=reasoning,
        )


class ISO29100Evaluator:
    """Evaluate using ISO/IEC 29100 principles."""

    def __init__(self):
        self.name = "ISO/IEC 29100"

    def predict(self, case: Dict[str, Any]) -> FrameworkPrediction:
        """Predict using ISO 29100 principles."""
        description = case.get("description", "").lower()

        # Check against ISO principles
        violations = []

        if "consent" in description or "without permission" in description:
            violations.append("consent_choice")
        if "unnecessary" in description or "excessive" in description:
            violations.append("data_minimization")
        if "breach" in description or "unauthorized" in description:
            violations.append("information_security")
        if "opaque" in description or "hidden" in description:
            violations.append("openness_transparency")

        if len(violations) >= 3:
            severity = "critical"
            likelihood = "likely"
            confidence = 0.85
        elif len(violations) >= 2:
            severity = "high"
            likelihood = "possible"
            confidence = 0.75
        else:
            severity = "medium"
            likelihood = "possible"
            confidence = 0.6

        predicted_harm = "regulatory_violation" if violations else "data_handling"
        reasoning = f"ISO 29100: {len(violations)} principle violations detected"

        return FrameworkPrediction(
            framework_name=self.name,
            predicted_harm=predicted_harm,
            predicted_severity=severity,
            predicted_likelihood=likelihood,
            confidence=confidence,
            reasoning=reasoning,
        )


class FrameworkComparator:
    """Compare empirical models against expert frameworks."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        # No common base class between the evaluators below, so annotate explicitly
        # -- otherwise mypy infers `list[object]` and every `.name`/`.predict()` call
        # site on this list becomes an error.
        self.frameworks: List[Any] = [
            NISTFrameworkEvaluator(),
            SoloveFrameworkEvaluator(),
            ISO29100Evaluator(),
        ]
        self.loaded_models: Dict[str, Any] = {}

    def load_models(self):
        """Load all trained models."""
        for model_dir in self.models_dir.glob("*/"):
            if not model_dir.is_dir():
                continue

            model_file = model_dir / "model.joblib"
            if model_file.exists():
                try:
                    model = joblib.load(model_file)
                    self.loaded_models[model_dir.name] = model
                    logger.info(f"Loaded model: {model_dir.name}")
                except Exception as e:
                    logger.error(f"Failed to load {model_dir.name}: {e}")

    def predict_with_model(
        self,
        model_name: str,
        model: Any,
        features: pd.DataFrame,
    ) -> ModelPrediction:
        """Get prediction from an empirical model."""
        try:
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(features)[0]
                predicted_idx = probas.argmax()
                probability = probas[predicted_idx]
                predicted_harm = model.classes_[predicted_idx]
            else:
                predicted_harm = model.predict(features)[0]
                probability = 0.8  # Default confidence

            return ModelPrediction(
                model_name=model_name,
                predicted_harm=predicted_harm,
                probability=probability,
            )
        except Exception as e:
            logger.error(f"Model {model_name} prediction failed: {e}")
            return ModelPrediction(
                model_name=model_name,
                predicted_harm="unknown",
                probability=0.0,
            )

    def compare_on_cases(
        self,
        test_cases: List[Dict[str, Any]],
        features_df: pd.DataFrame,
    ) -> List[ComparisonResult]:
        """Compare models and frameworks on test cases.

        Args:
            test_cases: List of test cases with actual outcomes
            features_df: DataFrame with features for each case

        Returns:
            List of comparison results
        """
        if not self.loaded_models:
            self.load_models()

        results = []

        for i, case in enumerate(test_cases):
            case_id = case.get("id", f"case_{i}")
            description = case.get("description", "")
            actual_harm = case.get("actual_harm", "unknown")

            # Get framework predictions
            framework_preds = []
            framework_correct = {}

            for framework in self.frameworks:
                pred = framework.predict(case)
                framework_preds.append(pred)
                framework_correct[framework.name] = pred.predicted_harm == actual_harm

            # Get model predictions
            model_preds = []
            model_correct = {}

            if i < len(features_df):
                features = features_df.iloc[[i]]

                for model_name, model in self.loaded_models.items():
                    pred = self.predict_with_model(model_name, model, features)
                    model_preds.append(pred)
                    model_correct[model_name] = pred.predicted_harm == actual_harm

            # Determine best performers
            best_framework = None
            best_framework_score = 0.0
            for pred in framework_preds:
                if framework_correct.get(pred.framework_name, False):
                    if pred.confidence > best_framework_score:
                        best_framework_score = pred.confidence
                        best_framework = pred.framework_name

            best_model = None
            best_model_score = 0.0
            for pred in model_preds:
                if model_correct.get(pred.model_name, False):
                    if pred.probability > best_model_score:
                        best_model_score = pred.probability
                        best_model = pred.model_name

            results.append(
                ComparisonResult(
                    case_id=case_id,
                    case_description=description,
                    actual_outcome=actual_harm,
                    framework_predictions=framework_preds,
                    model_predictions=model_preds,
                    framework_correct=framework_correct,
                    model_correct=model_correct,
                    best_framework=best_framework,
                    best_model=best_model,
                )
            )

        return results

    def compute_comprehensive_metrics(
        self,
        results: List[ComparisonResult],
    ) -> Dict[str, Any]:
        """Compute comprehensive metrics for all models and frameworks.

        Args:
            results: List of comparison results

        Returns:
            Dictionary with metrics formatted per Output Contract
        """
        # Extract data for each approach
        models_metrics = {}
        frameworks_metrics = {}

        # Collect predictions and ground truth
        for model_name in self.loaded_models.keys():
            y_true_causes = []
            y_pred_causes = []
            y_true_outcomes = []
            y_pred_outcomes = []
            y_true_binary = []
            y_pred_proba = []

            for result in results:
                y_true_causes.append(result.actual_outcome)
                y_true_outcomes.append(result.actual_outcome)
                y_true_binary.append(
                    1 if result.actual_outcome not in ["unknown", None, "none"] else 0
                )

                # Find model prediction
                model_pred = next(
                    (p for p in result.model_predictions if p.model_name == model_name), None
                )
                if model_pred:
                    y_pred_causes.append(model_pred.predicted_harm)
                    # ModelPrediction has no separate "outcome" field; predicted_harm
                    # is the single prediction dimension it exposes (matches the
                    # y_true_outcomes population above, which reuses actual_outcome).
                    y_pred_outcomes.append(model_pred.predicted_harm)
                    y_pred_proba.append(model_pred.probability)
                else:
                    y_pred_causes.append("unknown")
                    y_pred_outcomes.append("unknown")
                    y_pred_proba.append(0.0)

            # Compute metrics
            metrics = compute_comprehensive_metrics(
                y_true_causes,
                y_pred_causes,
                y_true_outcomes,
                y_pred_outcomes,
                y_true_binary,
                y_pred_proba,
            )
            models_metrics[model_name] = metrics

        # Do the same for frameworks
        for framework in self.frameworks:
            y_true_causes = []
            y_pred_causes = []
            y_true_outcomes = []
            y_pred_outcomes = []
            y_true_binary = []
            y_pred_proba = []

            for result in results:
                y_true_causes.append(result.actual_outcome)
                # ComparisonResult has no separate "consequence" field; reuse
                # actual_outcome as the single ground-truth dimension (mirrors the
                # model-comparison block above).
                y_true_outcomes.append(result.actual_outcome)
                y_true_binary.append(
                    1 if result.actual_outcome not in ["unknown", None, "none"] else 0
                )

                # Find framework prediction
                fw_pred = next(
                    (p for p in result.framework_predictions if p.framework_name == framework.name),
                    None,
                )
                if fw_pred:
                    y_pred_causes.append(fw_pred.predicted_harm)
                    # FrameworkPrediction has no separate "outcome" field; predicted_harm
                    # is the single prediction dimension it exposes.
                    y_pred_outcomes.append(fw_pred.predicted_harm)
                    y_pred_proba.append(fw_pred.confidence)
                else:
                    y_pred_causes.append("unknown")
                    y_pred_outcomes.append("unknown")
                    y_pred_proba.append(0.0)

            # Compute metrics
            metrics = compute_comprehensive_metrics(
                y_true_causes,
                y_pred_causes,
                y_true_outcomes,
                y_pred_outcomes,
                y_true_binary,
                y_pred_proba,
            )
            frameworks_metrics[framework.name] = metrics

        # Format for output contract
        return format_metrics_for_output_contract(models_metrics, frameworks_metrics)

    def generate_report(
        self,
        results: List[ComparisonResult],
        output_file: Path,
    ):
        """Generate comparison report.

        Args:
            results: Comparison results
            output_file: Path to write report
        """
        report = []
        report.append("# Expert Framework vs Empirical Model Comparison\n\n")
        report.append(f"Generated: {datetime.utcnow().isoformat()}\n\n")
        report.append(f"Test Cases: {len(results)}\n\n")

        # Calculate aggregate accuracy
        framework_accuracy: Dict[str, List[bool]] = {}
        model_accuracy: Dict[str, List[bool]] = {}

        for result in results:
            for fw_name, correct in result.framework_correct.items():
                framework_accuracy.setdefault(fw_name, []).append(correct)

            for model_name, correct in result.model_correct.items():
                model_accuracy.setdefault(model_name, []).append(correct)

        report.append("## Accuracy Summary\n\n")
        report.append("### Expert Frameworks\n\n")
        report.append("| Framework | Accuracy | Correct/Total |\n")
        report.append("|-----------|----------|---------------|\n")
        for fw_name, correct_list in sorted(framework_accuracy.items()):
            accuracy = sum(correct_list) / len(correct_list) if correct_list else 0.0
            total = len(correct_list)
            n_correct = sum(correct_list)
            report.append(f"| {fw_name} | {accuracy:.1%} | {n_correct}/{total} |\n")

        report.append("\n### Empirical Models\n\n")
        report.append("| Model | Accuracy | Correct/Total |\n")
        report.append("|-------|----------|---------------|\n")
        for model_name, correct_list in sorted(model_accuracy.items()):
            accuracy = sum(correct_list) / len(correct_list) if correct_list else 0.0
            total = len(correct_list)
            n_correct = sum(correct_list)
            report.append(f"| {model_name} | {accuracy:.1%} | {n_correct}/{total} |\n")

        # Write report
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write("".join(report))

        logger.info(f"Comparison report written to {output_file}")

        # Also write JSON version
        json_file = output_file.with_suffix(".json")
        json_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "test_cases": len(results),
            "framework_accuracy": {
                name: sum(vals) / len(vals) if vals else 0.0
                for name, vals in framework_accuracy.items()
            },
            "model_accuracy": {
                name: sum(vals) / len(vals) if vals else 0.0
                for name, vals in model_accuracy.items()
            },
        }

        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"JSON report written to {json_file}")


def load_golden_test_cases(golden_file: Path) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Load golden test cases for comparison.

    Args:
        golden_file: Path to golden cases JSONL

    Returns:
        Tuple of (test cases, features DataFrame)
    """
    test_cases = []

    with open(golden_file) as f:
        for line in f:
            if line.strip():
                test_cases.append(json.loads(line))

    # Extract features into DataFrame
    # This would need to match your actual feature engineering
    features_data = []
    for case in test_cases:
        features_data.append(
            {
                "description_length": len(case.get("description", "")),
                # Add more features as needed
            }
        )

    features_df = pd.DataFrame(features_data)

    return test_cases, features_df


if __name__ == "__main__":
    # Example usage

    logging.basicConfig(level=logging.INFO)

    base_dir = Path(__file__).resolve().parents[3]
    models_dir = base_dir / "models"

    comparator = FrameworkComparator(models_dir)

    # Would load real test cases in practice
    logger.info("Framework comparison module ready")
