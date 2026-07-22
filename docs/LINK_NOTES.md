# Link Notes — P1-C practicum docs extraction

Inventory of every relative link found in the files copied under `docs/` by the
P1-C extraction pass (old repo `privacy-heuristics` → new repo `privacy-harm-heuristics`,
branch `extract/docs`). For wave-P3: these are candidates to either extract the
target file too, or to leave as historical references into the old repo.

Legend: **FIXED** = link rewritten to point at its new in-repo location.
**MARKED** = link left with its original text/href, tagged `<!-- not extracted -->`
because the target was not part of this copy set.

## docs/OVERVIEW.md
(from `docs/v1-privacy-heuristics/README.md`)

| Original link | Status | Note |
| --- | --- | --- |
| `../../README.md` | **FIXED** (P3-E) | Root README now exists in-repo; the opening was rewritten and the link corrected to `../README.md` (the old `../../` escaped the repo root). Marker removed. |
| `../NAMING.md` (×3, lines 8/17/33) | MARKED | `docs/NAMING.md` not in the copy list. |
| `../../wiki/` | MARKED | Whole wiki dir, explicitly excluded except the files enumerated in the task. |
| `../../wiki/Home.md` | MARKED | Not in the copy list. |
| `../../wiki/Data-Pipeline.md` | MARKED | Not in the copy list. |
| `../../wiki/Modeling-and-Heuristics.md` | **FIXED** | Was copied to `docs/MODELING_AND_HEURISTICS.md` — rewrote link to `MODELING_AND_HEURISTICS.md`. |
| `../../wiki/Web-Experience.md` | MARKED | Not in the copy list. |
| `../../wiki/Deployment-and-Ops.md` | MARKED | Not in the copy list. |
| `../briefs/REFACTOR-BRIEF-2026-07.md` | MARKED | Not in the copy list. |
| `../specs/SPEC-01-ci-classifier.md` | MARKED | Not in the copy list. |
| `../DESIGN-SPEC.md` | MARKED | Not in the copy list. |
| `../RFC-001-contextual-integrity.md` | MARKED | Not in the copy list. |
| `../specs/SPEC-06-contextual-dp-math.md` | MARKED | Not in the copy list. |
| `../specs/SPEC-02-inference-dp-budget.md` | MARKED | Not in the copy list. |
| `../WORKPLAN.md` | MARKED | Not in the copy list. |
| `../SPRINT-2026-07.md` | MARKED | Not in the copy list. |

## docs/COMPARISON_METRICS.md
(from `wiki/reports/COMPARISON_METRICS.md`) — no relative links.

## docs/COMPARISON_TRIALS.md
(from `wiki/reports/COMPARISON_TRIALS.md`)

| Original link | Status | Note |
| --- | --- | --- |
| `../paper/draft.md` | MARKED | Resolves to `wiki/paper/draft.md` in the old repo, which does not exist there either (already-dangling link at source). Left as-is per "do not rewrite prose", tagged not-extracted. |

## docs/MODELING_AND_HEURISTICS.md
(from `wiki/Modeling-and-Heuristics.md`)

| Original link | Status | Note |
| --- | --- | --- |
| `Home.md` | MARKED | `wiki/Home.md` not in the copy list. |
| `Web-Experience.md` | MARKED | `wiki/Web-Experience.md` not in the copy list. |

## docs/architecture/INTERPRETABILITY_GUIDE.md
(from `wiki/architecture/INTERPRETABILITY_GUIDE.md`) — no relative links.

## docs/architecture/HYBRID_MODEL_IMPLEMENTATION.md
(from `wiki/architecture/HYBRID_MODEL_IMPLEMENTATION.md`) — no relative links.
Note: it references `[Final Paper](../paper/draft.md)`-style artifacts only via
plain prose paths (`paper/latest_metrics_table.md`, `data/experiments/...`), not
as markdown links, so nothing to fix/mark there.

## docs/reference/PROPOSAL_AND_STATUS_UPDATES.md
(from `wiki/reference/Privacy Heuristics Proposal and Status Updates.md`)
No relative links (all links in this file are absolute `https://` URLs to
external sources, plus one footnote marker `[^1]` which is self-contained).

## docs/reference/LIT_REVIEW.md
(from `wiki/reference/Privacy Heuristics –\xa0Lit Review.md`) — no relative links
(external `https://` citations only).

## docs/reference/INTEGRATED_SYNTHESIS.md
(from `wiki/reference/Integrated_Privacy_Synthesis_and_Updated_Practicum.docx.md`)
— no relative links found.

## docs/reference/GLOBAL_PRIVACY_FRAMEWORKS.md
(from `wiki/reference/Global Privacy Frameworks and Privacy Theory_ A Comprehensive Overview.md`)
— no relative links found (external `https://` citations only).

## docs/reference/DATA_SOURCES.csv
(from `wiki/reference/Practicum Data Sources - Data Sources.csv`) — CSV data
file, not prose; no markdown links applicable. The practicum-artifact banner
line was **deliberately not injected** into this file (it would corrupt the
CSV's row structure / column count). Flagging here instead for wave-P3
awareness. First column contains an in-repo-context reference
("Context: Proposal,Privacy Heuristics Proposal and Status Updates") — that's
prose data, not a link, left untouched.

## Not copied (per task scope, not omissions)
- `wiki/site/` — excluded explicitly (static build output).
- `wiki/reference/collections/`, `wiki/reference/internal/`, `wiki/reference/research/`,
  `wiki/reference/templates/` — the task's `wiki/reference/**` line names exactly
  five deliverables (practicum proposal, lit review, integrated synthesis,
  frameworks overview, data-sources CSV); these subdirectories hold other
  material (internal resume-point notes, AI-usage notes, portal design doc,
  BibTeX collections, IEEE LaTeX template) not named in that list, so they were
  left behind. Flagging in case wave-P3 wants them too.
