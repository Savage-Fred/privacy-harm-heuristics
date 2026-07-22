"""Projection utilities: map GLiNER2 outputs into stable schema fields.

This module takes the unified ``gliner2_labels`` structure produced by
``gliner2_labeling`` and projects it into the canonical :class:`Record`
fields (root causes, product features, harms). This keeps the GLiNER2 schema
decoupled from downstream interpretable models, while remaining easy to
evolve.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping


def _extract_labels(gliner2_labels: Mapping[str, Any]) -> List[str]:
    labels = gliner2_labels.get("text_labels") or []
    return [str(label) for label in labels]


def project_gliner2_to_record_fields(record: Dict[str, Any]) -> None:
    """Project ``gliner2_labels`` into stable ``Record`` fields in-place.

    This function is intentionally conservative: it only writes fields
    when GLiNER2 signals are present and leaves any existing fields intact.

    Mappings (current hypothesis, aligned with research docs):
      - harm_* text labels → ``harm_categories`` (string tags)
      - rc_* text labels → ``root_cause_features`` (feature slugs)
      - span_labels with label starting ``pf_`` → extend ``root_cause_features``
        and ``product_features`` (if present)

    The mapping is deliberately simple and transparent so that future
    iterations can refine it as more golden cases accumulate.
    """

    gl = record.get("gliner2_labels")
    if not isinstance(gl, Mapping):
        return

    text_labels = _extract_labels(gl)
    if not text_labels:
        return

    # Harm categories (document-level)
    harms = [lab for lab in text_labels if lab.startswith("harm_")]
    if harms:
        existing = set(record.get("harm_categories") or [])
        for h in harms:
            existing.add(h)
        record["harm_categories"] = sorted(existing)

    # Root cause style document-level signals
    rc_labels = [lab for lab in text_labels if lab.startswith("rc_")]
    if rc_labels:
        existing_rc = set(record.get("root_cause_features") or [])
        for rc in rc_labels:
            existing_rc.add(rc)
        record["root_cause_features"] = sorted(existing_rc)

    # Span-level product features from entities (pf_*)
    span_labels = gl.get("span_labels") or []
    pf_spans = [s for s in span_labels if str(s.get("label", "")).startswith("pf_")]
    if pf_spans:
        existing_rc = set(record.get("root_cause_features") or [])
        existing_pf = set(record.get("product_features") or [])
        for s in pf_spans:
            lab = str(s.get("label"))
            existing_rc.add(lab)
            existing_pf.add(lab.replace("pf_", "", 1) if lab.startswith("pf_") else lab)
        record["root_cause_features"] = sorted(existing_rc)
        record["product_features"] = sorted(existing_pf)
