# Live rerun — 2026-07-21 (provider drift reference)

`rules_static`, n=50 `golden_cases_v3.jsonl`, provider gemini (`gemini-2.5-flash`,
the 2026-07 default in `llm/provider.py`), run from the extracted repo at the P2 state.

| metric | recorded (Dec 2025, model unpinned) | rerun 2026-07-21 |
|---|---|---|
| instance_jaccard | 0.0678 | 0.1052 |
| exact_match_ratio | 0.6667 | 0.3600 |
| micro_f1 | 0.3908 | 0.3356 |
| ndcg@5 | 0.0896 | 0.1760 |

Same regime, materially drifted: today's model predicts non-empty harm sets far
more often than the Dec-2025 model (EMR was dominated by empty-set agreement,
so it halves while jaccard/NDCG roughly double). This is provider-model drift —
the original runs recorded only `provider: gemini`, no model name. Lesson
applied: this rerun pins the model string, and the README documents that the
recorded table is a historical artifact of an unpinned cloud model.
