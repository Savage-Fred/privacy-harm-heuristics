# Close-out paper: "A Constant Beats Every Arm"

`lessons.tex` / `lessons.pdf` is a short (4-page) negative-results paper that
re-analyses this repository's committed evidence offline and records what the
practicum does and does not support. It is anonymised for double-blind review;
remove the repository URL in §"Artifact availability" before submitting to a
venue that requires it.

## What we learned

The headline, that a static rule block in the prompt beats LLM baselines, does
not survive a constant predictor: on the same 50 gold cases, predicting no harm
gives the best exact-match (0.700), and predicting all four Solove groups gives
the best Jaccard (0.175) and nDCG@5 (0.282). The first research question
(can heuristics estimate harm?) is still open, because every training label in
the released corpus came from the keyword fallback.

1. Report constant baselines (empty, modal, all-labels, modal severity) with the same scorer.
2. On an empty-majority gold set, exact-match mostly measures emptiness. Report the empty share and score the non-empty subset separately.
3. Keep the gold set independent of the systems under test, and review it before the headline run.
4. Make scorer fallbacks loud. A missing `harms` field silently switched the target to `root_causes`.
5. Make placeholder runs unable to pass for results. Every per-arm metric in the 14 "comprehensive" runs came from a dry run.
6. Measure the label pipeline you ran, not the one you designed. 100% of labels came from the keyword fallback.
7. Pin model versions and keep per-case outputs.
8. Name arms by mechanism ("rules static" still called an LLM), and include a model-free arm.
9. Scrub at collection, not at release. Before the audit, the corpus reproduced the harm it studied.

## Rebuild every number

No network and no API calls. From the repo root:

```bash
make setup                                     # once
make fetch-data                                # optional: full-corpus label sources
PYTHONPATH=src .venv/bin/python paper/reanalysis.py   # -> paper/reanalysis.json
cd paper && latexmk -pdf lessons.tex           # -> paper/lessons.pdf
```

`reanalysis.py` asserts that its scoring path reproduces the recorded 2026
rerun exactly before it computes anything else. Without `make fetch-data`, the
`release_asset_v1_0_0` fields in the JSON are `null`. The recorded 2025 arms
come from `data/experiments/final_results_summary.md`, and the offline arm
comes from the top-level README.

## Status

Drafted 2026-09-24 as the artifact's close-out (#9). Candidate venues are
negative-results and evaluation-methodology workshops; check current calls for
papers before choosing one.
