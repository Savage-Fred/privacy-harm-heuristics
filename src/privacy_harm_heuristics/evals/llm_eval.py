from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..llm.provider import complete, select_provider


@dataclass
class LLMEvalConfig:
    in_path: Path
    out_path: Path
    text_field: str = "description"
    target_field: str = "harm_category"
    labels: Optional[list[str]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    sample: Optional[int] = None
    seed: int = 42
    max_tokens: int = 64


def _infer_text(record: dict, text_field: str) -> str:
    txt = record.get(text_field)
    if isinstance(txt, str) and txt.strip():
        return txt
    # Common fallbacks
    parts = []
    for key in ("title", "description", "body", "selftext", "text"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return "\n\n".join(parts)[:8000]


def _infer_labels(records: Iterable[dict], target_field: str, limit: int = 500) -> list[str]:
    seen: dict[str, int] = {}
    count = 0
    for rec in records:
        if count >= limit:
            break
        label = rec.get(target_field)
        if isinstance(label, str) and label.strip():
            key = label.strip()
            seen[key] = seen.get(key, 0) + 1
        count += 1
    # Sort by frequency desc, then name
    return [k for k, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))]


def _build_prompt(text: str, labels: list[str]) -> str:
    labels_joined = ", ".join(labels)
    return (
        "You are an expert privacy analyst. Read the incident text and classify it into "
        "exactly one "
        f"harm category from this list: [{labels_joined}].\n\n"
        "Respond with ONLY the label, no punctuation, no quotes.\n\n"
        f"Incident text:\n{text}\n\nLabel:"
    )


def _parse_label(output: str, labels: list[str]) -> Optional[str]:
    if not output:
        return None
    line = output.strip().splitlines()[0]
    # normalize
    low = re.sub(r"[^a-z0-9_\- ]+", "", line.lower()).strip()
    # direct match
    for lab in labels:
        if low == lab.lower():
            return lab
    # prefix match / contains
    for lab in labels:
        llow = lab.lower()
        if low.startswith(llow) or llow in low:
            return lab
    return None


def _fallback_keyword_label(text: str, labels: list[str]) -> Optional[str]:
    t = text.lower()
    # prioritize some common categories if present in labels
    checks = [
        ("insecurity", ["breach", "leak", "expos", "ransom", "stolen", "hack"]),
        (
            "surveillance",
            ["track", "monitor", "camera", "microphone", "geofenc", "bluetooth", "spy"],
        ),
        ("location", ["gps", "location", "geoloc", "geofence"]),
        ("children", ["child", "minor", "kid", "coppa"]),
        ("ads", ["ad", "advert", "targeting", "adtech"]),
        ("finance", ["bank", "credit", "card", "loan", "account"]),
        ("health", ["hipaa", "medical", "patient", "hospital"]),
        ("biometrics", ["face", "fingerprint", "iris", "voice", "biometric"]),
        ("law_enforcement", ["police", "warrant", "subpoena", "sheriff", "dea", "ice"]),
    ]
    available = {lab.lower(): lab for lab in labels}
    for key, keywords in checks:
        if key in available:
            if any(k in t for k in keywords):
                return available[key]
    # fallback to most frequent (first label)
    return labels[0] if labels else None


def iter_llm_judgments(config: LLMEvalConfig) -> Iterable[dict]:
    provider = select_provider(config.provider)
    # Read input lazily to support sampling without loading entire file
    rng = None
    if config.sample is not None:
        import random

        rng = random.Random(config.seed)
    processed = 0
    written = 0
    # First pass to infer labels if not provided
    labels: list[str]
    if config.labels is None:
        with config.in_path.open("r", encoding="utf-8") as ftmp:
            labels = _infer_labels(
                (json.loads(line) for line in ftmp if line.strip()), config.target_field
            )
            if not labels:
                labels = []
    else:
        labels = list(config.labels)

    with config.in_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            processed += 1
            if config.sample is not None and rng is not None:
                # Bernoulli keep to approximate sample size without pre-counting
                # Keep probability = sample / processed_so_far (rough)
                # Simpler: keep when random() < (sample / (processed+1)) until hits
                if written >= config.sample:
                    break
                # Probabilistic keep to get to target roughly
                if rng.random() > max(0.1, min(0.9, config.sample / max(1, processed))):
                    continue
            text = _infer_text(rec, config.text_field)
            if not text:
                continue
            target = rec.get(config.target_field)
            prompt = _build_prompt(text, labels) if labels else _build_prompt(text, ["unknown"])
            t0 = time.perf_counter()
            out = complete(
                prompt,
                provider=config.provider,
                model=config.model,
                max_tokens=config.max_tokens,
            )
            ms = int((time.perf_counter() - t0) * 1000)
            pred = _parse_label(out, labels) if labels else None
            if pred is None:
                # Fallback keyword heuristic when the model fails or offline
                pred = _fallback_keyword_label(text, labels) if labels else "unknown"
            result = {
                "provider": provider,
                "model": config.model,
                "latency_ms": ms,
                "predicted": pred,
            }
            if target is not None:
                result["target"] = target
                result["correct"] = bool(str(pred) == str(target))
            yield result
            written += 1


def run_llm_eval(config: LLMEvalConfig) -> dict:
    """Run the LLM eval streaming and write JSONL judgments; return metrics summary."""
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    correct = 0
    with config.out_path.open("w", encoding="utf-8") as fout:
        for item in iter_llm_judgments(config):
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            total += 1
            if item.get("correct"):
                correct += 1
    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "provider": select_provider(config.provider),
        "model": config.model,
    }
    # Save a companion metrics file next to out_path
    metrics_path = config.out_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics
