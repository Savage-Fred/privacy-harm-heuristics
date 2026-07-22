from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .framework_comparison import FrameworkComparator, load_golden_test_cases
from .llm_eval import LLMEvalConfig, run_llm_eval


@dataclass
class ComparisonTrialsConfig:
    """Configuration for comparison trial runs."""

    golden_cases: Path
    models_dir: Path
    llm_input: Path
    output_dir: Path
    iterations: int = 20
    llm_sample: Optional[int] = 25
    llm_text_field: str = "description"
    llm_target_field: str = "harm_category"
    llm_labels: Optional[List[str]] = None
    llm_max_tokens: int = 96
    gemini_request_cap: Optional[int] = 400
    primary_provider: str = "gemini"
    fallback_provider: str = "openai"
    gemini_model: Optional[str] = None
    gpt_model: Optional[str] = None


class ConfidenceTracker:
    """Beta-distribution confidence tracker with coverage weighting."""

    def __init__(self, *, prior: float = 1.0, coverage_scale: float = 10.0) -> None:
        self.alpha = prior
        self.beta = prior
        self.prior = prior
        self.coverage_scale = coverage_scale

    def update(self, successes: int, total: int) -> None:
        if total <= 0:
            return
        failures = total - successes
        if failures < 0:
            failures = 0
        self.alpha += max(0, successes)
        self.beta += failures

    def _observations(self) -> float:
        return max(0.0, (self.alpha + self.beta) - (2 * self.prior))

    def score(self) -> float:
        total = self._observations()
        if total <= 0:
            return 0.0
        accuracy = self.alpha / (self.alpha + self.beta)
        coverage = 1 - math.exp(-total / max(1.0, self.coverage_scale))
        return round(float(accuracy * coverage), 4)

    def snapshot(self) -> Dict[str, float]:
        denom = self.alpha + self.beta
        return {
            "score": self.score(),
            "mean_accuracy": round(float(self.alpha / denom), 4) if denom else 0.0,
            "observations": int(self._observations()),
        }


class ConfidenceRegistry:
    """Maintains per-approach confidence trackers."""

    def __init__(self) -> None:
        self._trackers: Dict[str, ConfidenceTracker] = {}

    def update(self, key: str, successes: int, total: int) -> None:
        if total <= 0:
            return
        tracker = self._trackers.setdefault(key, ConfidenceTracker())
        tracker.update(successes, total)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {name: tracker.snapshot() for name, tracker in self._trackers.items()}


class ComparisonTrialsRunner:
    """Runs repeated comparison trials with LLM fallbacks and confidence tracking."""

    def __init__(self, config: ComparisonTrialsConfig) -> None:
        self.config = config
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.run_dir = (config.output_dir / timestamp).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not config.golden_cases.exists():
            raise FileNotFoundError(f"Golden cases file not found: {config.golden_cases}")
        if not config.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {config.models_dir}")
        if not config.llm_input.exists():
            raise FileNotFoundError(f"LLM input file not found: {config.llm_input}")
        self.test_cases, self.features_df = load_golden_test_cases(config.golden_cases)
        self.comparator = FrameworkComparator(config.models_dir)
        self.confidence = ConfidenceRegistry()
        self.current_provider = config.primary_provider
        self.gemini_requests_used = 0
        self.provider_usage: Dict[str, int] = {}
        self.iteration_records: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        trials_log = self.run_dir / "trials.jsonl"
        for iteration in range(1, self.config.iterations + 1):
            provider = self._select_provider()
            record = self._run_iteration(iteration, provider)
            with trials_log.open("a", encoding="utf-8") as log_fh:
                log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.iteration_records.append(record)
            if provider == "gemini":
                self.gemini_requests_used += record.get("llm_metrics", {}).get("total", 0)
                if (
                    self.config.gemini_request_cap is not None
                    and self.gemini_requests_used >= self.config.gemini_request_cap
                ):
                    self.current_provider = self.config.fallback_provider
        summary = self._build_summary()
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        return summary

    def _select_provider(self) -> str:
        if self.current_provider != "gemini":
            return self.current_provider
        cap = self.config.gemini_request_cap
        if cap is None:
            return self.current_provider
        if self.gemini_requests_used >= cap:
            self.current_provider = self.config.fallback_provider
        return self.current_provider

    def _run_iteration(self, iteration: int, provider: str) -> Dict[str, Any]:
        iteration_dir = self.run_dir / f"iteration_{iteration:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        framework_metrics, model_metrics = self._run_framework_models()
        llm_metrics = self._run_llm(iteration_dir, provider)
        self._update_confidence(framework_metrics, model_metrics, llm_metrics)

        # Compute comprehensive metrics using new metrics module
        comprehensive_metrics = self._compute_comprehensive_metrics()

        record = {
            "iteration": iteration,
            "timestamp": datetime.utcnow().isoformat(),
            "provider": provider,
            "framework_metrics": framework_metrics,
            "model_metrics": model_metrics,
            "llm_metrics": llm_metrics,
            "confidence": self.confidence.snapshot(),
            "gemini_requests_used": self.gemini_requests_used,
            "metrics": comprehensive_metrics,  # Add comprehensive metrics
        }
        with (iteration_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        self.provider_usage[provider] = self.provider_usage.get(provider, 0) + 1
        return record

    def _run_framework_models(
        self,
    ) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        results = self.comparator.compare_on_cases(self.test_cases, self.features_df)
        framework_counts: Dict[str, Dict[str, int]] = {}
        model_counts: Dict[str, Dict[str, int]] = {}
        for result in results:
            for fw, correct in result.framework_correct.items():
                stats = framework_counts.setdefault(fw, {"correct": 0, "total": 0})
                stats["total"] += 1
                if correct:
                    stats["correct"] += 1
            for model, correct in result.model_correct.items():
                stats = model_counts.setdefault(model, {"correct": 0, "total": 0})
                stats["total"] += 1
                if correct:
                    stats["correct"] += 1
        framework_metrics = {
            name: {
                "accuracy": round(stats["correct"] / stats["total"], 4) if stats["total"] else 0.0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for name, stats in framework_counts.items()
        }
        model_metrics = {
            name: {
                "accuracy": round(stats["correct"] / stats["total"], 4) if stats["total"] else 0.0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for name, stats in model_counts.items()
        }
        return framework_metrics, model_metrics

    def _run_llm(self, iteration_dir: Path, provider: str) -> Dict[str, Any]:
        out_path = iteration_dir / f"llm_{provider}.jsonl"
        model_hint = None
        if provider == "gemini":
            model_hint = self.config.gemini_model
        elif provider == "openai":
            model_hint = self.config.gpt_model
        llm_config = LLMEvalConfig(
            in_path=self.config.llm_input,
            out_path=out_path,
            text_field=self.config.llm_text_field,
            target_field=self.config.llm_target_field,
            labels=self.config.llm_labels,
            provider=provider,
            model=model_hint,
            sample=self.config.llm_sample,
            max_tokens=self.config.llm_max_tokens,
        )
        metrics = run_llm_eval(llm_config)
        metrics["out_path"] = str(out_path)
        return metrics

    def _compute_comprehensive_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive metrics using the comparator."""
        # Run comparison on test cases to get results
        results = self.comparator.compare_on_cases(self.test_cases, self.features_df)

        # Compute comprehensive metrics
        return self.comparator.compute_comprehensive_metrics(results)

    def _update_confidence(
        self,
        framework_metrics: Dict[str, Dict[str, float]],
        model_metrics: Dict[str, Dict[str, float]],
        llm_metrics: Dict[str, Any],
    ) -> None:
        fw_success = sum(int(m["correct"]) for m in framework_metrics.values())
        fw_total = sum(int(m["total"]) for m in framework_metrics.values())
        self.confidence.update("frameworks", fw_success, fw_total)
        model_success = sum(int(m["correct"]) for m in model_metrics.values())
        model_total = sum(int(m["total"]) for m in model_metrics.values())
        self.confidence.update("interpretable_models", model_success, model_total)
        llm_success = int(llm_metrics.get("correct", 0))
        llm_total = int(llm_metrics.get("total", 0))
        llm_key = f"llm_{llm_metrics.get('provider', 'unknown')}".lower()
        self.confidence.update(llm_key, llm_success, llm_total)

    def _build_summary(self) -> Dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "iterations": len(self.iteration_records),
            "provider_usage": self.provider_usage,
            "gemini_requests_used": self.gemini_requests_used,
            "confidence": self.confidence.snapshot(),
            "records": self.iteration_records,
        }
