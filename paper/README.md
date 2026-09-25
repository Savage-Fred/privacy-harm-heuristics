# Close-out paper: "When Constants Win"

`lessons.tex` / `lessons.pdf` is a short (5-page) negative-results paper. It
re-analyses this repository's committed evidence offline and records what the
practicum does and does not support. It was revised after an adversarial
peer-style review (see PR #10). It is anonymised for double-blind review;
remove the repository URL in §"Artifact availability" before submitting to a
venue that requires it.

## What we learned

The headline, that a static rule block in the prompt beats LLM baselines, does
not survive label-free constants. On the same 50 gold cases, using the
recovered per-case outputs of all 18 recorded trials:

- Predicting all four Solove groups beats the headline arm on Jaccard and
  nDCG@5. The paired bootstrap intervals exclude zero even after correcting
  for 15 comparisons.
- Predicting no harm ties the arm on exact match.
- On micro-F1, neither the headline arm nor the 2026 rerun is significantly
  better than the best constant, even with out-of-taxonomy labels filtered.
- Five of the six arms sent an identical prompt. The rule block did change
  outputs (p = 0.007), but not enough to beat a constant.
- The first research question (can heuristics estimate harm?) is still open,
  because every training label came from a keyword fallback.

Evaluation lessons:

1. Pre-specify constant baselines (empty, all labels in a fixed order, modal severity), score them with the same scorer, and report oracle constants separately.
2. Disclose empty-set conventions and the ceiling they imply. Here a perfect predictor scores Jaccard and nDCG 0.30.
3. Field names lie, so pin the prompt-scorer contract. The prompt invited root causes into the field scored as harms.
4. Diff what each arm actually sends. Five "different" arms sent one prompt.
5. Make placeholders unable to pass for results. The multi-arm runs shipped with the artifact were all dry runs, and API errors echo the prompt.
6. Keep the gold set independent of the models under test, and keep its review auditable.
7. Measure the label pipeline you ran. A swallowed GLiNER2 load error left 100% keyword labels.
8. Ship the per-case outputs behind every reported number. The extraction kept the dry runs and dropped the live trials. Also make the analysis deterministic.

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

Before it reports anything, `reanalysis.py` asserts that it exactly reproduces:

- all 18 recorded 2025 trial files, and their means against the recorded table;
- the 2026 rerun (set, ranking and ordinal metrics);
- the repository's offline arm.

It also asserts that its fast bootstrap metrics equal the repository scorer.

Its output does not depend on the hash seed. To check, run it twice with
different `PYTHONHASHSEED` values and `cmp` the JSON. Without
`make fetch-data`, the `release_asset_v1_0_0` fields are `null`.

`trials_2025/` holds the trial summary and the 18 per-case result files behind
the recorded 2025 table. They were harvested read-only from the private
predecessor repository at commit `d1d38594` (2025-11-24) and are unchanged.

For double-blind submission, set `\anontrue` near the top of `lessons.tex`.

## Status

Drafted 2026-09-24 as the artifact's close-out (#9). Candidate venues are
negative-results and evaluation-methodology workshops; check current calls for
papers before choosing one.
