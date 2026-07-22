"""Runner for hybrid model experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from tqdm import tqdm

from privacy_harm_heuristics.constants.privacy_taxonomy import TAXONOMY_SOLOVE

from ..llm.provider import resolve_model_name
from ..models.hybrid import HybridMode, HybridModel
from .framework_comparison import load_golden_test_cases
from .metrics import (
    calculate_mlc_metrics,
    calculate_ordinal_metrics,
    calculate_ranking_metrics,
)

app = typer.Typer(help="Hybrid model evaluation runner")


@app.command("run-hybrid")
def run_hybrid_experiment(
    golden_file: Path = typer.Option(
        Path("data/golden_cases.jsonl"), exists=True, help="Path to golden test cases"
    ),
    mode: Optional[HybridMode] = typer.Option(
        None, help="Hybrid mode to run (defaults to config or baseline)"
    ),
    ruleset: Optional[str] = typer.Option(
        None, help="Name of the ruleset (e.g., expert_framework)"
    ),
    ruleset_version: Optional[str] = typer.Option(None, help="Version of the ruleset (e.g., 1.0)"),
    rules_dir: Path = typer.Option(Path("data/rules"), help="Directory containing rulesets"),
    output_dir: Path = typer.Option(Path("data/experiments"), help="Output directory"),
    provider: str = typer.Option("gemini", help="LLM provider"),
    model_name: Optional[str] = typer.Option(None, help="Specific model name"),
    limit: Optional[int] = typer.Option(None, help="Limit number of cases to run"),
):
    """Run a hybrid model experiment and compute comprehensive metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Default Config
    config_path = Path("data/default_model_config.json")
    if config_path.exists():
        try:
            defaults = json.loads(config_path.read_text())
            if mode is None and "default_mode" in defaults:
                try:
                    mode = HybridMode(defaults["default_mode"])
                    print(f"Using default mode from config: {mode.value}")
                except ValueError:
                    print(f"Warning: Invalid mode in config: {defaults['default_mode']}")

            if ruleset is None:
                ruleset = defaults.get("default_ruleset")
                if ruleset:
                    print(f"Using default ruleset from config: {ruleset}")

            if ruleset_version is None:
                ruleset_version = defaults.get("default_ruleset_version")
                if ruleset_version:
                    print(f"Using default ruleset version from config: {ruleset_version}")
        except Exception as e:
            print(f"Warning: Failed to load default config: {e}")

    # Fallback defaults
    if mode is None:
        mode = HybridMode.BASELINE
        print("Using fallback default mode: baseline")

    # Load Data
    print(f"Loading cases from {golden_file}...")
    cases, _ = load_golden_test_cases(golden_file)

    if limit:
        cases = cases[:limit]

    # Load Rules if needed
    rules = []
    ruleset_meta = {}
    if ruleset and ruleset_version:
        filename = f"{ruleset}.v{ruleset_version}.json"
        rules_path = rules_dir / filename
        if rules_path.exists():
            try:
                data = json.loads(rules_path.read_text())
                rules = data.get("rules", [])
                ruleset_meta = {
                    "name": data.get("name", ruleset),
                    "version": data.get("version", ruleset_version),
                    "description": data.get("description", ""),
                    "file": str(rules_path),
                    "rules": rules,
                }
                print(f"Loaded {len(rules)} rules from {rules_path}")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in {rules_path}")
        else:
            print(f"Warning: Ruleset file {rules_path} not found. Running without rules.")

    # Initialize Model
    model = HybridModel(
        mode=mode,
        rules=rules,
        provider=provider,
        model_name=model_name,
    )

    # Pin the model: record the concrete model string this run will use, so the
    # results file is never ambiguous the way Dec 2025's unpinned runs were
    # (see data/experiments/rerun_20260721/README.md).
    resolved_model = resolve_model_name(provider, model_name)
    print(f"Provider={provider} resolved_model={resolved_model}")

    # Build Taxonomy Mapping (Subtype -> Parent)
    # This ensures granular golden labels (e.g. "surveillance") match
    # the model's high-level output (e.g. "information_collection").
    subtype_map = {}
    for parent, spec in TAXONOMY_SOLOVE.items():
        # Map parent to itself
        subtype_map[parent] = parent
        # Map subtypes to parent
        for sub in spec.get("subtypes", []):
            if isinstance(sub, dict) and "name" in sub:
                subtype_map[sub["name"]] = parent

    # Custom mappings for legacy/golden labels
    custom_map = {
        "data_breach": "information_processing",  # insecurity -> information_processing
        "data_breach_internal": "information_processing",
        "data_breach_phishing": "information_processing",
        "data_breach_ransomware": "information_processing",
        "data_breach_credential_reuse": "information_processing",
        "data_breach_web_skimming": "information_processing",
        "data_breach_state": "information_processing",
        "data_breach_scraping": "information_processing",
        "data_breach_reuse": "information_processing",
        "data_leak": "information_dissemination",  # disclosure -> information_dissemination
        "data_leak_insider": "information_dissemination",
        "whistleblower_leak": "information_dissemination",
        "privacy_misuse": "invasion",  # ?
        "personal_privacy_violation": "invasion",
    }
    subtype_map.update(custom_map)

    # Run Predictions
    y_true_causes = []
    y_pred_causes = []
    y_true_ranking = []
    y_pred_ranking = []
    y_true_harm = []
    y_pred_harm = []

    results = []

    print(f"Running {len(cases)} cases in mode {mode.value}...")
    for case in tqdm(cases):
        text = case.get("description", "") or case.get("text", "")

        # Ground Truth Extraction (Adapt keys to your golden schema)
        # Use 'harms' field for Solove taxonomy labels (not 'root_causes' which are technical causes)
        # Fallback for legacy golden cases with only 'harm_category'
        true_causes = set(case.get("harms", []))
        if not true_causes:
            # Try root_causes as fallback for older data
            true_causes = set(case.get("root_causes", []))
        if not true_causes and "harm_category" in case:
            true_causes = {case["harm_category"]}

        true_ranking = case.get("ranking", list(case.get("harms", [])))
        if not true_ranking and "harm_category" in case:
            true_ranking = [case["harm_category"]]

        true_score = int(case.get("harm_score", 0))

        # Normalize Ground Truth Labels to High-Level Taxonomy
        # If a label is not in the map, keep it as-is (fallback)
        true_causes = {subtype_map.get(c, c) for c in true_causes}
        true_ranking = [subtype_map.get(c, c) for c in true_ranking]
        # Deduplicate ranking while preserving order (if multiple subtypes map to same parent)
        seen = set()
        true_ranking_dedup = []
        for c in true_ranking:
            if c not in seen:
                true_ranking_dedup.append(c)
                seen.add(c)
        true_ranking = true_ranking_dedup

        # Predict
        pred = model.predict(text)

        # Normalize Predicted Labels
        pred_causes = {subtype_map.get(c, c) for c in pred["root_causes"]}
        pred_ranking = [subtype_map.get(c, c) for c in pred["ranking"]]
        # Deduplicate ranking
        seen_pred = set()
        pred_ranking_dedup = []
        for c in pred_ranking:
            if c not in seen_pred:
                pred_ranking_dedup.append(c)
                seen_pred.add(c)
        pred_ranking = pred_ranking_dedup

        # Store for Metrics
        y_true_causes.append(true_causes)
        y_pred_causes.append(pred_causes)

        y_true_ranking.append(true_ranking)
        y_pred_ranking.append(pred_ranking)

        y_true_harm.append(true_score)
        y_pred_harm.append(pred["harm_score"])

        # Log result
        results.append(
            {
                "case_id": case.get("id"),
                "text": text[:100] + "...",
                "true": {"causes": list(true_causes), "ranking": true_ranking, "score": true_score},
                "pred": pred,
            }
        )

    # Compute Metrics
    print("Computing metrics...")
    mlc_metrics = calculate_mlc_metrics(y_true_causes, y_pred_causes)
    ranking_metrics = calculate_ranking_metrics(y_true_ranking, y_pred_ranking)
    ordinal_metrics = calculate_ordinal_metrics(y_true_harm, y_pred_harm)

    final_metrics = {
        "mode": mode.value,
        "mlc": mlc_metrics,
        "ranking": ranking_metrics,
        "ordinal": ordinal_metrics,
    }

    # Save
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"results_{mode.value}_{timestamp_str}.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "id": f"results_{mode.value}_{timestamp_str}",
                "timestamp": pd.Timestamp.now().timestamp(),
                "config": {
                    "mode": mode.value,
                    "ruleset": ruleset_meta if ruleset_meta else None,
                    "provider": provider,
                    "model_name": model_name,
                    "resolved_model": resolved_model,
                },
                "metrics": final_metrics,
                "details": results,
            },
            f,
            indent=2,
        )

    print(f"Results saved to {out_file}")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    app()
