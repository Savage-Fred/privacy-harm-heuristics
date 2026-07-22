"""Generate a narrative, human-readable comparison report for models.

This module takes the JSON output from the evaluation harness
(`comparison_report.json`) and produces a Markdown report that is suitable
for docs or a paper appendix.

When an online LLM provider is configured (OpenAI, Gemini, Anthropic, or
LLaMA via llama-cpp), we use it via ``practicum.llm.provider`` to expand
the deterministic summary into a richer narrative. When no provider is
available, we fall back to a purely programmatic summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ...llm import provider as llm_provider


def _score_model(metrics: Dict[str, Any]) -> float:
    """Return a scalar score for ranking models (macro F1 preferred)."""
    macro = metrics.get("macro avg")
    if isinstance(macro, dict):
        f1 = macro.get("f1-score")
        if isinstance(f1, (int, float)):
            return float(f1)
    acc = metrics.get("accuracy")
    if isinstance(acc, (int, float)):
        return float(acc)
    return 0.0


def _rank_models(report: Dict[str, dict]) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    for name, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        score = _score_model(metrics)
        items.append((name, score))
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def build_narrative_markdown(
    report: Dict[str, dict],
    *,
    dataset: str | None = None,
    target: str | None = None,
) -> str:
    """Build a deterministic Markdown summary from comparison metrics."""
    lines: List[str] = []
    lines.append("# Model Comparison Narrative\n")

    # Context block
    if dataset or target:
        lines.append("## Evaluation context")
        if dataset:
            lines.append(f"- Dataset: `{dataset}`")
        if target:
            lines.append(f"- Target label: `{target}`")

    # Overall ranking
    ranked = _rank_models(report)
    if ranked:
        lines.append("\n## Overall ranking (by macro F1 / accuracy)")
        for idx, (name, score) in enumerate(ranked, start=1):
            metrics = report.get(name, {})
            acc = metrics.get("accuracy")
            macro = metrics.get("macro avg", {})
            f1 = macro.get("f1-score") if isinstance(macro, dict) else None
            if isinstance(acc, (int, float)) and isinstance(f1, (int, float)):
                lines.append(f"{idx}. **{name}** – accuracy={acc:.4f}, macro F1={f1:.4f}")
            elif isinstance(acc, (int, float)):
                lines.append(f"{idx}. **{name}** – accuracy={acc:.4f}")
            else:
                lines.append(f"{idx}. **{name}** – score={score:.4f}")

    # Per-model details
    for name, metrics in report.items():
        lines.append(f"\n## {name}")

        # Handle "note"-only entries (e.g., LLM baseline skipped)
        if "note" in metrics and "accuracy" not in metrics and "macro avg" not in metrics:
            lines.append(f"- {metrics['note']}")
            continue

        acc = metrics.get("accuracy")
        if isinstance(acc, (int, float)):
            lines.append(f"- Overall accuracy: {acc:.4f}")

        macro = metrics.get("macro avg", {})
        if isinstance(macro, dict):
            p = macro.get("precision") or 0.0
            r = macro.get("recall") or 0.0
            f1 = macro.get("f1-score") or 0.0
            lines.append(
                "- Macro average: precision={:.4f}, recall={:.4f}, F1={:.4f}".format(
                    float(p), float(r), float(f1)
                )
            )

        weighted = metrics.get("weighted avg", {})
        if isinstance(weighted, dict):
            wp = weighted.get("precision") or 0.0
            wr = weighted.get("recall") or 0.0
            wf1 = weighted.get("f1-score") or 0.0
            lines.append(
                "- Weighted average: precision={:.4f}, recall={:.4f}, F1={:.4f}".format(
                    float(wp), float(wr), float(wf1)
                )
            )

        # Per-class summaries: top-3 and bottom-3 by F1
        label_rows: List[Tuple[str, float]] = []
        for label, vals in metrics.items():
            if label in {
                "accuracy",
                "macro avg",
                "weighted avg",
                "micro avg",
                "samples avg",
            }:
                continue
            if not isinstance(vals, dict):
                continue
            f1 = vals.get("f1-score")
            if isinstance(f1, (int, float)):
                label_rows.append((label, float(f1)))

        if label_rows:
            label_rows.sort(key=lambda x: x[1], reverse=True)
            best = label_rows[:3]
            worst = label_rows[-3:]
            lines.append(
                "- Strongest classes (by F1): "
                + ", ".join(f"`{lbl}` ({score:.3f})" for lbl, score in best)
            )
            lines.append(
                "- Weakest classes (by F1): "
                + ", ".join(f"`{lbl}` ({score:.3f})" for lbl, score in worst)
            )

    return "\n".join(lines).strip() + "\n"


def generate_narrative_report(
    report_path: str | Path,
    out_path: str | Path,
    *,
    dataset: str | None = None,
    target: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
) -> Path:
    """Generate a narrative Markdown report, optionally using an LLM.

    Args:
        report_path: Path to comparison_report.json.
        out_path: Path to write Markdown report.
        dataset: Optional dataset path for context text.
        target: Optional target label name for context text.
        provider: Optional explicit provider name for LLM.
        model: Optional model hint/name for the provider.
        max_tokens: Max tokens for LLM narrative generation.
    """
    report_json = json.loads(Path(report_path).read_text(encoding="utf-8"))
    base_md = build_narrative_markdown(report_json, dataset=dataset, target=target)

    prov = llm_provider.select_provider(provider)
    final_md = base_md

    if prov != "fallback":
        # Ask the configured LLM to expand the deterministic summary into a richer narrative.
        prompt = (
            "You are a senior machine learning researcher focused on privacy risk modeling.\n"
            "Given the following structured evaluation metrics and an initial Markdown summary,\n"
            "write an expanded, well-structured Markdown report (roughly 2–4 pages) that:\n"
            "- Explains the evaluation setup in clear, non-jargony terms\n"
            "- Compares each model (including heuristic and LLM baselines) with pros and cons\n"
            "- Highlights where interpretable models agree or disagree with baselines\n"
            "- Calls out notable per-class performance patterns\n"
            "- Ends with concrete recommendations for which models to use and why.\n\n"
            "=== METRICS JSON ===\n"
            f"{json.dumps(report_json, indent=2)}\n\n"
            "=== BASE SUMMARY ===\n"
            f"{base_md}\n\n"
            "Now write the expanded Markdown report only.\n"
        )
        try:
            resp = llm_provider.complete(prompt, provider=prov, model=model, max_tokens=max_tokens)
            if isinstance(resp, str) and resp.strip():
                final_md = resp.strip()
        except Exception:
            # If LLM call fails for any reason, fall back to deterministic summary.
            final_md = base_md

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(final_md, encoding="utf-8")
    return out
