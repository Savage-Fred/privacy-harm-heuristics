# Comparison Trials Runner

> Practicum artifact (Dec 2025), extracted from the privacy-heuristics repo 2026-07-21.

This document captures how to run the multi-iteration comparison between expert frameworks, interpretable models, and LLM baselines.

## CLI command

Run the trials from source:

```bash
PYTHONPATH=src python -m privacy_harm_heuristics.main evals comparison-trials \
  --golden data/golden_cases.jsonl \
  --models-dir models \
  --iterations 20 \
  --llm-input data/golden_cases.jsonl \
  --gemini-request-cap 400
```

The command:

- Runs up to 20 iterations (or less if the Gemini limit is hit) and logs every iteration under `evaluations/comparison_trials/<timestamp>`.
- Tracks confidence for frameworks, models, and each LLM provider based on Beta priors.
- Falls back to GPT automatically when Gemini usage reaches the configured cap.

### Docker helper

```bash
./scripts/run_comparison_trials.sh --iterations 20 --llm-input data/golden_cases.jsonl
```

## Scheduling

To start trials 30 minutes after labeling completes:

```bash
./scripts/start_comparison_trials_after_labeling.sh
```

The script monitors `data/labeling_status.log` for `LABELING_COMPLETE`, then uses `at` to run the comparison command 30 minutes later (default command can be overridden by `TRIAL_COMMAND`).

## Latest Results

For the latest experimental results and analysis, please refer to `data/experiments/final_results_summary.md` and the [Final Paper](../paper/draft.md) <!-- not extracted -->.
