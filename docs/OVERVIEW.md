# Overview — Privacy Harm Heuristics

> Practicum artifact (Dec 2025), extracted from the privacy-heuristics repo 2026-07-21.

> **This is the standalone, frozen practicum artifact.** For the project
> introduction, headline results, quickstart, and repo map, start at the
> top-level [`README.md`](../README.md). This page is a deeper narrative of the
> harm-heuristics pipeline; it is **closed** (masters practicum, ended Dec 2025)
> and maintained for reference, not under active development.

This repository is the extracted, self-contained version of the 2025
masters-practicum work. It collects public privacy incidents, labels them against
Solove's harm taxonomy, engineers features, and trains interpretable models
(decision tree, EBM, BRL, sparse linear, Bayes net) that export human-readable
heuristics. In the original multi-track umbrella repo this work was called
**"Privacy Heuristics (v1) / Track A"**; that name is preserved here for lineage.

> **Note on forward references below.** The sections that follow were copied from
> the original umbrella repo's v1 landing page. Their pointers into an in-repo
> `wiki/`, a `docs/NAMING.md`, sibling Track B / Track C specs, and a
> `temperature/` web UI describe the **original** repo — that material was **not**
> extracted into this standalone artifact, and those links are tagged
> `<!-- not extracted -->`. What *is* in this repo is described in the top-level
> [`README.md`](../README.md) and [`data/README.md`](../data/README.md).

## Where the detailed v1 docs live

- **In-repo wiki:** [`wiki/`](../../wiki/) <!-- not extracted --> is the source of truth for v1 —
  start at [`wiki/Home.md`](../../wiki/Home.md) <!-- not extracted -->, then
  [`wiki/Data-Pipeline.md`](../../wiki/Data-Pipeline.md) <!-- not extracted -->,
  [`wiki/Modeling-and-Heuristics.md`](MODELING_AND_HEURISTICS.md),
  [`wiki/Web-Experience.md`](../../wiki/Web-Experience.md) <!-- not extracted -->, and
  [`wiki/Deployment-and-Ops.md`](../../wiki/Deployment-and-Ops.md) <!-- not extracted -->.
- **Practicum archive (Drive):** proposal, TDD/DD, and the final paper
  "Human vs Human vs Machine" (Dec 2025), as referenced in
  [`docs/NAMING.md`](../NAMING.md) <!-- not extracted -->. This archive keeps the "Privacy Heuristics"
  title permanently.
  <!-- TODO: no direct Drive archive URL is committed in this repo. Add the
  canonical link here once available; do not confuse it with the live-proposal
  Google Doc linked from the top-level README's Research Blog Portal section. -->

## v1 codebase map (legacy packages)

All under `src/privacy_harm_heuristics/` (import paths use the current package
name, `privacy_harm_heuristics`). These are the v1-only packages; each carries a
`Legacy (Privacy Heuristics v1 …)` docstring header (see W-20):

| Package | What it did (high level) |
| --- | --- |
| `connectors/` | Pull raw incidents from external feeds (HHS breaches, SEC/FTC, Hacker News, Reddit, state portals, GDPR, Wikipedia) into normalized JSONL. |
| `collectors/` | Orchestrate/schedule connector runs and accumulate the raw corpus (append-only, dedup-aware). |
| `labeling/` | Assign Solove harm categories via GLiNER2 + weak-supervision rules and optional LLM signals. |
| `features/` | Derive engineered features (penalty buckets, keyword flags, product-feature / root-cause signals) for modeling. |
| `rules/` | Weak-supervision rule definitions and rule-based enrichment used during labeling. |
| `temperature/` | 0–100 privacy "temperature" risk scoring and mitigation logic behind the temperature-check UI. |
| `discovery/` | Source/candidate discovery — expanding the set of incidents and feeds to ingest. |
| `processing/` | Merge, validate, deduplicate, and transform records (JSONL ↔ Parquet, DuckDB) between pipeline stages. |

These packages are internal to Track A; Track B (`contextual_integrity/`) and
Track C (`agentic_dp/`) do **not** import them directly (verified 2026-07). See
[`docs/briefs/REFACTOR-BRIEF-2026-07.md`](../briefs/REFACTOR-BRIEF-2026-07.md) <!-- not extracted -->
(W-20) for the per-package legacy annotations.

## Where ACTIVE work lives instead

New work is **not** in v1. It lives under `docs/` as Track B and Track C:

- **Track B — Privacy Agent** (🔄 active, M1): contextual-integrity classifier.
  Spec: [`docs/specs/SPEC-01-ci-classifier.md`](../specs/SPEC-01-ci-classifier.md) <!-- not extracted -->;
  master design [`docs/DESIGN-SPEC.md`](../DESIGN-SPEC.md) <!-- not extracted -->;
  RFC [`docs/RFC-001-contextual-integrity.md`](../RFC-001-contextual-integrity.md) <!-- not extracted -->.
  Code: `src/privacy_harm_heuristics/contextual_integrity/`.
- **Track C — Contextual Differential Privacy** (🔄 active PoC): CI-aware DP
  budget engine. Specs:
  [`docs/specs/SPEC-06-contextual-dp-math.md`](../specs/SPEC-06-contextual-dp-math.md) <!-- not extracted -->
  and [`docs/specs/SPEC-02-inference-dp-budget.md`](../specs/SPEC-02-inference-dp-budget.md) <!-- not extracted -->.
  Code: `src/privacy_harm_heuristics/agentic_dp/`.

For the current roadmap and task tracker see
[`docs/WORKPLAN.md`](../WORKPLAN.md) <!-- not extracted --> and
[`docs/SPRINT-2026-07.md`](../SPRINT-2026-07.md) <!-- not extracted -->.
