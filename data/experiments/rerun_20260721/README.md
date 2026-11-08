# Live rerun — 2026-07-21

`rules_static`, n=50 `golden_cases_v3.jsonl`, provider gemini, run from the
extracted repo at the P2 state. The run config records `model_name: null`; the
provider default in `llm/provider.py` was `gemini-2.5-flash`, the same default
as in 2025.

| metric | recorded (Dec 2025, model unrecorded) | rerun 2026-07-21 (model unrecorded) |
|---|---|---|
| instance_jaccard | 0.0678 | 0.1052 |
| exact_match_ratio | 0.6667 | 0.3600 |
| micro_f1 | 0.3908 | 0.3356 |
| ndcg@5 | 0.0896 | 0.1760 |

The numbers differ, but this is **not evidence of model drift**. Between the two
runs the prompt was revised (2025-11-29: a root-cause allowlist, and permission
to put root causes in the scored field), the scorer changed (it now maps
predictions to groups as well as gold), and one trial is compared with three.
Neither run records its model. See `paper/lessons.tex`, "Scorer and prompt
changed between runs". The lesson is to record the model string, prompt and
scorer with every run; the recorded table is a dated historical artifact.
