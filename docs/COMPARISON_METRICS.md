# Comparison Metrics for Privacy Harm Models

> Practicum artifact (Dec 2025), extracted from the privacy-heuristics repo 2026-07-21.

This document describes the metrics system for comparing privacy harm prediction models and expert frameworks.

## Overview

The comparison metrics module (`src/privacy_harm_heuristics/models/eval/metrics.py`) implements a comprehensive set of metrics as specified in the repository's Output Contract. These metrics enable systematic evaluation and comparison of:

1. **Interpretable Models**: Decision Trees, EBM, BRL, Sparse Linear, etc.
2. **Expert Frameworks**: NIST Privacy Framework, Solove's Taxonomy, ISO/IEC 29100, etc.
3. **LLM Baselines**: GPT, Gemini, etc.

## Core Metrics

### 1. Root Cause Accuracy

**Purpose**: Measures how well an approach identifies the root cause of privacy incidents.

**Definition**: Standard classification accuracy for root cause identification.

```python
root_cause_accuracy = (correct_predictions / total_predictions)
```

**Range**: 0.0 (never correct) to 1.0 (always correct)

**Example**: 0.85 means the approach correctly identified the root cause 85% of the time.

### 2. Outcome Accuracy

**Purpose**: Measures how well an approach predicts actual outcomes (regulatory fines, lawsuits, settlements, etc.).

**Definition**: Standard classification accuracy for outcome prediction.

```python
outcome_accuracy = (correct_outcome_predictions / total_predictions)
```

**Range**: 0.0 to 1.0

**Example**: 0.80 means the approach correctly predicted the outcome 80% of the time.

### 3. Risk Calibration (ECE)

**Purpose**: Measures how well-calibrated the risk predictions are.

**Definition**: Expected Calibration Error (ECE) using binned calibration curves.

The ECE measures the weighted absolute difference between predicted probabilities and actual frequencies across probability bins:

```python
ECE = Σ (n_bin / n_total) * |avg_predicted_prob - actual_frequency|
```

**Range**: 0.0 (perfect calibration) to 1.0 (worst calibration)

**Lower is better!**

**Example**: 0.10 means predicted probabilities are off by 10% on average.

**Interpretation**:
- ECE < 0.05: Excellent calibration
- ECE 0.05-0.15: Good calibration
- ECE 0.15-0.30: Moderate calibration
- ECE > 0.30: Poor calibration

### 4. Predicted Risk Percentage

**Purpose**: The average predicted risk across all cases.

**Definition**: Mean of predicted probabilities, expressed as percentage.

```python
predicted_risk_pct = mean(predicted_probabilities) * 100
```

**Range**: 0% to 100%

**Example**: 42% means the model predicts an average risk of 42% across all cases.

**Use**: Compare with actual base rates to detect over/under-prediction.

## Output Contract Format

The metrics are formatted according to the repository's JSON schema:

```json
{
  "models": [
    {
      "name": "decision_tree",
      "root_cause_accuracy": 0.85,
      "outcome_accuracy": 0.80,
      "risk_calibration": 0.10,
      "predicted_risk_pct": 42.0,
      "accuracy": 0.85,
      "n_samples": 100,
      "n_correct": 85
    }
  ],
  "expert_frameworks": [
    {
      "name": "NIST Privacy Framework",
      "root_cause_accuracy": 0.70,
      "outcome_accuracy": 0.68
    }
  ],
  "ranking": [
    "decision_tree",
    "NIST Privacy Framework",
    "..."
  ]
}
```

The `ranking` array orders all approaches by `root_cause_accuracy` (descending).

## Usage Examples

### Basic Usage

```python
from privacy_harm_heuristics.models.eval.metrics import compute_comprehensive_metrics

# Ground truth
y_true_causes = ["breach", "surveillance", "aggregation"]
y_true_outcomes = ["fine", "lawsuit", "settlement"]
y_true_binary = [1, 1, 0]  # 1 = harm occurred

# Predictions
y_pred_causes = ["breach", "surveillance", "breach"]
y_pred_outcomes = ["fine", "lawsuit", "settlement"]
y_pred_proba = [0.9, 0.8, 0.2]

# Compute metrics
metrics = compute_comprehensive_metrics(
    y_true_causes,
    y_pred_causes,
    y_true_outcomes,
    y_pred_outcomes,
    y_true_binary,
    y_pred_proba,
)

print(f"Root Cause Accuracy: {metrics.root_cause_accuracy:.2%}")
print(f"Risk Calibration: {metrics.risk_calibration:.4f}")
```

### Integration with Framework Comparator

```python
from privacy_harm_heuristics.evals.framework_comparison import FrameworkComparator

# Initialize comparator
comparator = FrameworkComparator(models_dir)

# Run comparison on test cases
results = comparator.compare_on_cases(test_cases, features_df)

# Compute comprehensive metrics
metrics = comparator.compute_comprehensive_metrics(results)

# Get Output Contract format
output = metrics  # Already formatted
```

### Comparison Trials

```python
from privacy_harm_heuristics.evals.comparison_trials import ComparisonTrialsConfig, ComparisonTrialsRunner

config = ComparisonTrialsConfig(
    golden_cases=Path("data/golden_cases.jsonl"),
    models_dir=Path("models/"),
    llm_input=Path("data/labeled.jsonl"),
    output_dir=Path("results/"),
    iterations=20,
)

runner = ComparisonTrialsRunner(config)
summary = runner.run()

# Each iteration record includes comprehensive metrics
for record in summary["records"]:
    print(record["metrics"])
```

## Testing

The metrics module includes comprehensive tests:

```bash
# Run all metrics tests
pytest tests/test_comparison_metrics.py -v

# Run integration tests
pytest tests/test_metrics_integration.py -v

# Run demo
python examples/comparison_metrics_demo.py
```

## Performance Considerations

- **Root Cause Accuracy**: O(n) where n = number of samples
- **Outcome Accuracy**: O(n)
- **Risk Calibration**: O(n * b) where b = number of bins (default 10)
- **Predicted Risk %**: O(n)

For large datasets (>100k samples), consider sampling for calibration curve computation.

## References

1. **Expected Calibration Error (ECE)**:
   - Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning"
   - Guo, C., et al. (2017). "On Calibration of Modern Neural Networks"

2. **Brier Score**:
   - Brier, G. W. (1950). "Verification of forecasts expressed in terms of probability"

3. **Privacy Frameworks**:
   - NIST Privacy Framework: https://www.nist.gov/privacy-framework
   - Solove, D. J. (2006). "A Taxonomy of Privacy"
   - ISO/IEC 29100:2011 Privacy Framework

## Future Enhancements

Planned improvements:

1. **Multi-dimensional calibration**: Calibration curves per harm category
2. **Confidence intervals**: Bootstrap confidence intervals for metrics
3. **Statistical significance tests**: Compare models with statistical rigor
4. **Interpretability metrics**: Measure how interpretable the predictions are
5. **Fairness metrics**: Evaluate fairness across demographic groups
