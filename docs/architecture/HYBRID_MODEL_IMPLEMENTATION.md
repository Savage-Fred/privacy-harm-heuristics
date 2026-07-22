# Hybrid Model Implementation Summary

> Practicum artifact (Dec 2025), extracted from the privacy-heuristics repo 2026-07-21.

## Overview

We have successfully implemented the infrastructure for the "Hybrid Model" pipeline, enabling:

1. **Versioned Rulesets**: Management of expert privacy frameworks as versioned JSON artifacts.
2. **Automated Reporting**: A CLI tool to process experiment results and automatically update the website and paper artifacts.
3. **Hybrid Experiment Runner**: An updated runner that supports loading specific ruleset versions and logging metadata.

## Key Components

### 1. Ruleset Management

- **Location**: `data/rules/`
- **Format**: `name.vX.X.json`
- **CLI Command**: `python -m privacy_harm_heuristics.main rules list` / `view`
- **Current Ruleset**: `expert_framework.v1.0.json` (Initial set based on GDPR/NIST)

### 2. Experiment Runner

- **Command**: `python -m privacy_harm_heuristics.main evals hybrid run-hybrid`
- **New Flags**:
  - `--ruleset`: Name of the ruleset (e.g., `expert_framework`)
  - `--ruleset-version`: Version to use (e.g., `1.0`)
- **Output**: Saves JSON results to `data/experiments/rules/` with full config metadata.

### 3. Automated Reporting

- **Command**: `python -m privacy_harm_heuristics.main reporting process-results`
- **Function**:
  - Scans `data/experiments/` (recursively) for `results_*.json`.
  - Aggregates metrics.
  - Updates `src/privacy_harm_heuristics/web/static/data/results_summary.json` (for the Web UI).
  - Updates `src/privacy_harm_heuristics/web/static/data/leaderboard.json`.
  - Updates `paper/latest_metrics_table.md` (Markdown table for the paper).

## Verification

- **Test Run**: Successfully executed a "Rules Static" experiment.
- **Reporting**: Successfully processed the test result and generated all artifacts.
- **Artifacts**:
  - `paper/latest_metrics_table.md`: Contains the latest run metrics.
  - `src/privacy_harm_heuristics/web/static/data/results_summary.json`: Contains the full dataset for the UI.

## Next Steps

1. **Run Full Experiments**: Execute the runner with real models and the full dataset.
2. **Verify Web UI**: Launch the web server (`python -m privacy_harm_heuristics.main web serve`) and check the "Results" page to ensure the data renders correctly (collapsible rules, etc.).
3. **Expand Rulesets**: Add more sophisticated rules or alternative frameworks (e.g., Solove's taxonomy).
