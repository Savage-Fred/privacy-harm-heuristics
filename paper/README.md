# Close-out paper: "When Constants Win"

`lessons.tex` / `lessons.pdf` is a short (5-page) negative-results paper. It
re-analyses this repository's committed evidence offline and records what the
practicum does and does not support. It was revised after an adversarial
peer-style review (see PR #10). It is anonymised for double-blind review;
remove the repository URL in §"Artifact availability" before submitting to a
venue that requires it.

## What we learned

The headline, that a static rule block in the prompt beats LLM baselines, does
not survive constants fixed in advance. On the same 50 gold cases:

- Predicting no harm matches or beats every arm on exact-match (0.700).
- Predicting all four Solove groups beats every arm on Jaccard (0.175) and
  nDCG@5 (0.237), with paired bootstrap intervals that exclude zero.
- On micro-F1, the recorded arm leads, but no per-case data survive. The 2026
  rerun ties the best constant, and beats it only after a post-hoc filter
  removes out-of-taxonomy labels.
- Five of the six arms sent an identical prompt.
- The first research question (can heuristics estimate harm?) is still open,
  because every training label came from a keyword fallback.

Evaluation lessons:

1. Pre-specify constant baselines (empty, all labels in a fixed order, modal severity), score them with the same scorer, and report oracle constants separately.
2. Disclose empty-set conventions and the ceiling they imply. Here a perfect predictor scores Jaccard and nDCG 0.30.
3. Field names lie, so pin the prompt-scorer contract. The prompt invited root causes into the field scored as harms.
4. Diff what each arm actually sends. Five "different" arms sent one prompt.
5. Make placeholders unable to pass for results. Every stored multi-arm run was a dry run, and API errors echo the prompt.
6. Keep the gold set independent of the models under test, and keep its review auditable.
7. Measure the label pipeline you ran. A swallowed GLiNER2 load error left 100% keyword labels.
8. Record the model version, scorer revision, prompt hash, trial count and per-case outputs in every result file.

Data-ethics lesson: scrub at collection, not at release. Before the audit, the
corpus reproduced the harm it studied.

## Rebuild every number

No network and no API calls. From the repo root:

```bash
make setup                                     # once
make fetch-data                                # optional: full-corpus label sources
PYTHONPATH=src .venv/bin/python paper/reanalysis.py   # -> paper/reanalysis.json
cd paper && latexmk -pdf lessons.tex           # -> paper/lessons.pdf
```

`reanalysis.py` asserts that its scoring path reproduces the recorded 2026
rerun (set, ranking and ordinal metrics) and the repository's offline arm
exactly before it reports anything. Without `make fetch-data`, the
`release_asset_v1_0_0` fields in the JSON are `null`. The recorded 2025 arms
come from `data/experiments/final_results_summary.md`, and the offline arm
comes from the top-level README.

## Status

Drafted 2026-09-24 as the artifact's close-out (#9). Candidate venues are
negative-results and evaluation-methodology workshops; check current calls for
papers before choosing one.
