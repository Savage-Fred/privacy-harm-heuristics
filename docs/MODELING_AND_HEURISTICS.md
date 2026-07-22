# Modeling and Heuristics

> Practicum artifact (Dec 2025), extracted from the privacy-heuristics repo 2026-07-21.

This page covers the interpretable modeling stack that turns labeled incidents into actionable privacy heuristics.

## Goals

- Predict harm categories and risk scores with transparency.
- Surface human-readable rules and explanations that practitioners can trust.
- Support experimentation with multiple model families while keeping outputs comparable.

## Model Portfolio

| Model | Module | Best For |
| --- | --- | --- |
| **Explainable Boosting Machine (EBM)** | `src/privacy_harm_heuristics/models/trainers/ebm.py` | Highest accuracy with visual explanations. |
| **Bayesian Rule Lists (BRL)** | `src/privacy_harm_heuristics/models/trainers/brl.py` | Compact rule sets for policy teams. |
| **Sparse Linear Models** | `src/privacy_harm_heuristics/models/trainers/root_cause.py` | Interpretable coefficients linking features to harms. |
| **Transformer Classifier** | `src/privacy_harm_heuristics/models/harm_classifier_transformer.py` | Fast labeling of new incidents. |

All trainers share utilities in `src/privacy_harm_heuristics/models/trainers/` and rely on feature matrices produced by the data pipeline.

## Training Workflow

1. Ensure the pipeline has produced a labeled dataset with features (`data/large/with_features.jsonl`).
2. Trigger the desired trainer via `python -m privacy_harm_heuristics.train_cli <trainer>`.
3. Persist model artifacts under `models/` with metadata for reproducibility.
4. Register model version and configuration in project documentation (e.g., `IMPLEMENTATION.md`).

### Example: Train EBM

```bash
python -m privacy_harm_heuristics.train_cli train-ebm \
  --in data/large/with_features.jsonl \
  --target harm_category \
  --out models/ebm \
  --max-bins 256 \
  --learning-rate 0.01
```

### Example: Train Bayesian Rule Lists

```bash
python -m privacy_harm_heuristics.train_cli train-brl \
  --in data/large/with_features.jsonl \
  --target harm_category \
  --out models/brl \
  --max-rules 50 \
  --min-support 10
```

## Heuristic Extraction

- **Rule Summaries:** `src/privacy_harm_heuristics/models/explain.py` converts model outputs into readable heuristics.
- **Temperature Check:** `src/privacy_harm_heuristics/web/templates/temperature_check.html` renders 0-100 risk scores with mitigation advice.
- **Root Cause Insights:** `src/privacy_harm_heuristics/models/trainers/root_cause.py` links incidents to product features and design choices.

## Evaluation and Monitoring

- Unit tests live in `tests/test_transformer_classifier.py`, `tests/test_bayes_net_trainer.py`, and related files.
- Add custom evaluation scripts under `scripts/` or expand `generate_sources_matrix.py` for coverage analysis.
- Track precision/recall per harm category and monitor drift as new data sources are introduced.

## Next Steps

- Return to the [Home](Home.md) <!-- not extracted --> page or proceed to the [Web Experience](Web-Experience.md) <!-- not extracted --> to learn how models drive user-facing tools.
