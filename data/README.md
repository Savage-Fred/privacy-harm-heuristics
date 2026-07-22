# Data

All records originate from the practicum's public breach-reports/incident
collection pipeline (HHS breaches, SEC filings, Hacker News, Reddit,
Mastodon, Wikipedia, state portals, ransomware-leak trackers, Kaggle
datasets) — see the parent repo's connector code for source detail. Labels
are Solove-taxonomy harm categories assigned by a mix of weak supervision,
trained interpretable models, and LLM judges during the practicum.

Checksums for every file below (plus the staged release asset) are in
[`CHECKSUMS.txt`](CHECKSUMS.txt). Regenerate `*.sample.jsonl` deterministically
with `python scripts/make_samples.py` (fixed seed — re-running against the
same source produces byte-identical output).

**PII / content scrub.** Every published row — both samples and the full
release asset — is run through `scripts/make_samples.py::scrub_row`, which
removes identity handles, raw source records, verbatim free text, and
re-identifying URLs, keeping only the engineered features + labels the models
consume. See that script's docstring for the exact kept/dropped field
inventory and `docs/RELEASE-AUDIT.md` for the rationale. `tests/test_scrub.py`
enforces forbidden-field absence over every committed sample row.

| File | Size | Status | Notes |
|---|---|---|---|
| `golden_cases.jsonl` | 224 KB | committed | Hand-curated gold-label eval set (base). |
| `golden_cases_v3.jsonl` | 2.7 MB | committed | Gold-label eval set, v3 revision. |
| `golden_with_features.jsonl` | 508 KB | committed | Golden cases joined with engineered features. |
| `calibrated_weights.json` | 4 KB | committed | Per-harm-category calibration weights. |
| `default_model_config.json` | <1 KB | committed | Default mode/ruleset config. |
| `rules/*.json` | 56 KB total | committed | Rule sets: `solove`, `nist`, `gdpr`, `expert_framework(_enhanced)`, `learned_dt` (v1.0/v1.1). |
| `experiments/final_results_summary.md` | 3 KB | committed | Headline experiment writeup. |
| `experiments/comprehensive_*/` | ~380 KB total | committed | Trimmed to `summary_by_{model,mode,ruleset,config}.json`, `pairwise_tests.json`, `final_ranking.json`, `metadata.json` per run — bulky per-trial logs/raw dumps dropped. |
| `with_features.sample.jsonl` | 10.9 MB | committed | Stratified-by-`harm_category` sample of the full feature-engineered corpus (2,549 of 33,472 rows), PII-scrubbed. |
| `labeled_gliner2.sample.jsonl` | 0.7 MB | committed | First 150 rows + stratified-by-`harm_category` sample of the GLiNER2-labeled Mastodon corpus (987 of 30,316 rows; rows >5 KB excluded from the sampling pool as outliers), PII-scrubbed. |
| `with_features.jsonl` | 143 MB | **release asset** (not committed) | Full feature-engineered corpus, PII-scrubbed. Ships as a `v1.0.0` GitHub release asset — `make fetch-data` (placeholder until P4 tag). Verify against `CHECKSUMS.txt`. |
| `labeled_gliner2_full.jsonl` | 159 MB | **omitted** | Full GLiNER2-labeled corpus. Not released; use the `.sample.jsonl` for reproduction. |

`heuristics/` (15 files, 132 KB) and `trained_models/` (6 checkpoints, 364 KB
total: `decision_tree_v4`, `ebm_v4`, `sparse_linear_v3`, `bayes_net_v4`,
`brl_v3`, `brl_cascade_v3`) live as sibling top-level directories, not under
`data/` — each trained-model directory holds `model.joblib` (where
applicable) plus `metrics.json`/`provenance.json`.
