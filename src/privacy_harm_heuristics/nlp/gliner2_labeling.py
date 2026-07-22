"""GLiNER2-based schema-driven information extraction for privacy heuristics.

This module standardizes span-level entity extraction and document-level
classification using the GLiNER2 model under a single, hierarchical schema.

Why GLiNER2
-----------
GLiNER2 provides a schema-driven, multi-task interface that can jointly handle
named entity recognition, text classification, and structured extraction in a
single pass over the text. This replaces earlier disjoint pipelines that used
separate models for NER (e.g., ``dslim/bert-base-NER``) and zero-shot
classification (e.g., ``facebook/bart-large-mnli``), while remaining efficient
enough for CPU-only local execution.

Reference
---------
Urchade Zaratiana, Gil Pasternak, Oliver Boyd, George Hurn-Maloney, and
Ash Lewis. *GLiNER2: An Efficient Multi-Task Information Extraction System
with Schema-Driven Interface.* arXiv:2507.18546 [cs.CL], 2025.
https://arxiv.org/abs/2507.18546

The GLiNER2 code and models are released under the Apache 2.0 license, which is
compatible with this project. See the GLiNER2 repository and model cards for
details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # Optional heavy dependency; imported lazily in helpers below.
    from gliner2 import GLiNER2  # type: ignore
    from gliner2.inference.engine import Schema  # type: ignore
except Exception:  # pragma: no cover - import guarded for environments without gliner2
    GLiNER2 = None  # type: ignore[assignment]
    Schema = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Canonical privacy schema for GLiNER2
# ---------------------------------------------------------------------------

privacy_schema: Dict[str, Dict[str, str]] = {
    # Span-level labels (NER-style entities)
    "span_labels": {
        # Product Features (pf_)
        "pf_biometric_collection": "The collection of biometric data like fingerprints, face scans, or voice prints.",
        "pf_dark_pattern": "A user interface element designed to trick or mislead the user.",
        "pf_location_tracking": "The tracking or collection of a user's precise physical location.",
        "pf_third_party_sharing": "The practice of sharing user data with third-party companies, advertisers, or partners.",
        # Data & Context
        "data_type_collected": "A specific type of personal data being collected (e.g., name, email, financial info, IP address).",
        "data_recipient": "The entity, organization, or third party that receives the data.",
        # Contextual Integrity Violations
        "ci_secondary_use": "The use of data for a purpose different from the one it was collected for.",
        "ci_unexpected_recipient": "Data is transferred to a party that the user would not expect.",
    },
    # Text-level labels (document classification)
    # These are grouped into harm_* and rc_* to align with the golden
    # dataset and root cause modeling docs.
    "text_labels": {
        # Solove-like harms / non-monetary categories
        "harm_surveillance": "This text discusses surveillance, monitoring, 'creepiness', or being watched.",
        "harm_insecurity": "This text discusses a data breach, security vulnerability, hack, or insecure data storage.",
        "harm_aggregation": "This text discusses combining user data from multiple different sources or services.",
        "harm_disclosure": "This text discusses the unwanted exposure or dissemination of personal information.",
        # Root Causes (rc_) / mitigations
        "rc_has_mitigation": "The text mentions a positive safeguard, a user control, or an ability to opt-out or consent.",
        "rc_no_mitigation": "The text complains about a lack of control, a missing opt-out, or forced consent.",
        # Additional root-cause style document labels inspired by ROOT_CAUSE_ANALYSIS.md
        "rc_forced_consent": "The text criticizes forced consent, pre-ticked boxes, or lack of genuine choice.",
        "rc_misleading_notice": "The text mentions misleading, confusing, or deceptive privacy interfaces or notices.",
        "rc_data_broker_usage": "The text describes use of data brokers or opaque third-party data markets.",
        "rc_inadequate_security": "The text highlights basic security failures (open databases, lack of encryption, misconfigurations).",
        "rc_tracking_profiling": "The text discusses cross-site tracking, profiling, fingerprinting, or opaque algorithms.",
    },
}


@dataclass
class GLiNER2Config:
    """Runtime configuration for GLiNER2 extraction.

    Attributes:
        model_name: HF Hub repo id or local directory for the GLiNER2 model.
        text_field: Record field to read input text from (JSON/JSONL).
        span_threshold: Global default span confidence threshold in [0,1].
        text_threshold: Threshold for multi-label document classification.
        batch_size: Number of texts to encode jointly in GLiNER2.
    """

    # Default to a public GLiNER2 checkpoint suitable for CPU use.
    # Users can override via config or CLI if they prefer a different variant.
    model_name: str = field(default_factory=lambda: "fastino/gliner2-base-v1")
    text_field: str = "description"
    span_threshold: float = 0.5
    text_threshold: float = 0.5
    batch_size: int = 8


def _load_gliner2_model(model_name: str) -> Optional["GLiNER2"]:
    """Return a GLiNER2 model instance if available, else None.

    This keeps GLiNER2 as an optional dependency. Callers should handle
    ``None`` by either skipping annotation or emitting a clear warning.
    """

    if GLiNER2 is None:
        return None
    try:
        # The HF repo id (or local directory) is provided via config.model_name.
        return GLiNER2.from_pretrained(model_name)  # type: ignore[arg-type]
    except Exception:
        return None


def build_schema(
    schema_spec: Mapping[str, Mapping[str, str]] | None = None,
    *,
    span_threshold: float = 0.5,
    text_threshold: float = 0.5,
) -> "Schema":
    """Build a GLiNER2 ``Schema`` from a project-specific schema spec.

    The input ``schema_spec`` is expected to look like :data:`privacy_schema`:

    .. code-block:: python

        {
            "span_labels": {"label": "description", ...},
            "text_labels": {"label": "description", ...},
        }

    Span labels are configured via ``Schema.entities`` and text labels via
    ``Schema.classification`` with a single multi-label task called
    ``\"text_labels\"``.
    """

    if Schema is None:  # pragma: no cover - import guarded
        raise RuntimeError("gliner2 is not available; cannot build GLiNER2 schema")

    spec = schema_spec or privacy_schema
    span_labels = spec.get("span_labels", {})
    text_labels = spec.get("text_labels", {})

    schema = Schema()

    if span_labels:
        # Use descriptions to guide the model for zero-shot/custom entity types.
        # dict(...) wrap: Schema wants a concrete dict, not just a Mapping.
        schema = schema.entities(dict(span_labels), dtype="list", threshold=span_threshold)

    if text_labels:
        # Multi-label classification over the whole document.
        schema = schema.classification(
            task="text_labels",
            labels=dict(text_labels),
            multi_label=True,
            cls_threshold=text_threshold,
        )

    return schema


def _convert_single_result(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a single GLiNER2 result into the project JSON shape.

    Output schema:

    .. code-block:: json

        {
          "text_labels": ["harm_surveillance", "rc_no_mitigation"],
          "text_label_scores": {"harm_surveillance": 0.94, ...},
          "span_labels": [
            {"span": "always listening", "label": "harm_surveillance", "score": null},
            {"span": "shares my voice commands with advertisers",
             "label": "pf_third_party_sharing", "score": null}
          ]
        }

    Notes:
        - GLiNER2's public API currently exposes confidences for classification
          tasks but not per-entity spans after formatting. We therefore set
          ``score`` for spans to ``None`` for now. Thresholds still control
          which spans appear.
    """

    # Text-level labels
    text_labels: List[str] = []
    text_label_scores: Dict[str, float] = {}

    text_task = raw.get("text_labels")
    if isinstance(text_task, list) and text_task:
        first = text_task[0]
        if isinstance(first, dict) and "label" in first and "confidence" in first:
            # include_confidence=True, multi-label case
            for item in text_task:
                label = str(item.get("label"))
                conf = float(item.get("confidence", 0.0))
                text_labels.append(label)
                text_label_scores[label] = conf
        else:
            # include_confidence=False or simple list of labels
            text_labels = [str(lbl) for lbl in text_task]

    # Span-level entities
    span_labels: List[Dict[str, Any]] = []
    entities = raw.get("entities")
    if isinstance(entities, dict):
        for label, spans in entities.items():
            if isinstance(spans, str):
                if spans:
                    span_labels.append({"span": spans, "label": str(label), "score": None})
            elif isinstance(spans, Iterable):
                for span in spans:
                    if not span:
                        continue
                    span_labels.append({"span": str(span), "label": str(label), "score": None})

    return {
        "text_labels": text_labels,
        "text_label_scores": text_label_scores,
        "span_labels": span_labels,
    }


def label_texts(
    texts: Sequence[str],
    schema_spec: Mapping[str, Mapping[str, str]] | None = None,
    *,
    config: Optional[GLiNER2Config] = None,
    model: Optional["GLiNER2"] = None,
) -> List[Dict[str, Any]]:
    """Run GLiNER2 over one or more texts with a project schema.

    Args:
        texts: Sequence of raw text strings to annotate.
        schema_spec: Optional schema mapping; defaults to :data:`privacy_schema`.
        config: Runtime configuration (model name, thresholds, batch size).
        model: Optional pre-loaded GLiNER2 instance to reuse across calls.

    Returns:
        A list of JSON-serializable dicts, one per input text, each containing
        ``text_labels``, ``text_label_scores``, and ``span_labels``.
    """

    if not texts:
        return []

    cfg = config or GLiNER2Config()

    mdl = model or _load_gliner2_model(cfg.model_name)
    if mdl is None:
        # Graceful fallback: return empty labels so callers can continue.
        return [{"text_labels": [], "text_label_scores": {}, "span_labels": []} for _ in texts]

    schema = build_schema(
        schema_spec, span_threshold=cfg.span_threshold, text_threshold=cfg.text_threshold
    )

    raw_results = mdl.batch_extract(
        list(texts),
        schema,
        batch_size=cfg.batch_size,
        threshold=cfg.span_threshold,
        format_results=True,
        include_confidence=True,
    )

    return [_convert_single_result(res or {}) for res in raw_results]


def annotate_jsonl_with_gliner2(
    in_path: str,
    out_path: str,
    *,
    config: Optional[GLiNER2Config] = None,
    schema_spec: Mapping[str, Mapping[str, str]] | None = None,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """Stream JSONL, annotate with GLiNER2, and write augmented JSONL.

    The input JSONL is expected to contain a text field (default: ``description``)
    on each line. For each record, this function adds a ``gliner2_labels`` field
    with the unified output structure and writes the augmented record to
    ``out_path``.

    Args:
        in_path: Input JSONL path (each line is a JSON object with a text field).
        out_path: Output JSONL path (augmented with ``gliner2_labels``).
        config: GLiNER2Config controlling model name, text field, thresholds, etc.
        schema_spec: Optional schema spec; defaults to :data:`privacy_schema`.
        overwrite: If False, raises if ``out_path`` already exists.

    Returns:
        Tuple ``(total_records, annotated_records)``.
    """

    import json
    from pathlib import Path

    cfg = config or GLiNER2Config()
    in_p = Path(in_path)
    out_p = Path(out_path)

    if not in_p.exists():
        raise FileNotFoundError(in_path)
    if out_p.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass overwrite=True to replace: {out_path}")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    mdl = _load_gliner2_model(cfg.model_name)
    if mdl is None:
        # If the model cannot be loaded, do a no-op pass-through.
        total = 0
        annotated = 0
        with (
            open(in_path, "r", encoding="utf-8") as fin,
            open(out_path, "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return total, annotated

    schema = build_schema(
        schema_spec, span_threshold=cfg.span_threshold, text_threshold=cfg.text_threshold
    )

    total = 0
    annotated = 0
    with (
        open(in_path, "r", encoding="utf-8") as fin,
        open(out_path, "w", encoding="utf-8") as fout,
    ):
        buffer: List[MutableMapping[str, Any]] = []
        texts: List[str] = []

        def flush_batch() -> None:
            nonlocal annotated, buffer, texts
            if not buffer:
                return
            results = mdl.batch_extract(
                texts,
                schema,
                batch_size=cfg.batch_size,
                threshold=cfg.span_threshold,
                format_results=True,
                include_confidence=True,
            )
            for obj, res in zip(buffer, results):
                obj["gliner2_labels"] = _convert_single_result(res or {})
                annotated += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            buffer.clear()
            texts.clear()

        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            text = str(obj.get(cfg.text_field) or "").strip()
            buffer.append(obj)
            texts.append(text)

            if len(buffer) >= cfg.batch_size:
                flush_batch()

        # Final flush
        if buffer:
            flush_batch()

    return total, annotated
