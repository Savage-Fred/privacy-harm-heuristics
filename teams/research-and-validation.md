# Operational Playbook: Research and Validation Autonomous Team

## 1. Mission and Scientific Foundation

The **Research and Validation Autonomous Team** is tasked with autonomous research into privacy harm heuristics, interpretable model development, weak-supervision calibration, and multi-metric benchmark replication within the `privacy-harm-heuristics` repository.

### Core Scientific Questions
1. *Heuristic Efficacy*: Can small, interpretable heuristic pipelines (Decision Trees, Sparse Linear Models, Explainable Boosting Machines, Bayesian Rule Lists, Bayesian Networks, and expert rule sets) match or outperform large black-box language models when estimating privacy harm from factual incident reports?
2. *Framework Grounding*: How well do empirical heuristics ground in formal privacy taxonomy frameworks (Solove's taxonomy, NIST Privacy Framework, GDPR Article 83 criteria, ISO/IEC 29100)?
3. *Interpretability vs. Performance*: What is the empirical trade-off between rule sparsity/depth and multi-label classification accuracy, ranking fidelity, and ordinal risk calibration?

### Core Operating Principles
- **Evidence-First & Constant-Baseline Verification**: Never assert a heuristic or model works without empirical proof from the benchmark suite (`tests/`, `data/golden_cases_v3.jsonl`, `data/experiments/`, `paper/`) and explicit comparison against pre-specified label-free constant baselines (empty set, all four Solove groups, modal severity — see `paper/README.md`).
- **Frozen-Artifact Preservation**: Preserve historical recorded results, committed samples, checksums, and frozen experiment tables (`data/experiments/final_results_summary.md`, `paper/trials_2025/`) unchanged; write new runs under `results/` or new dated experiment directories.
- **Offline Determinism**: Core pipelines must operate 100% deterministically without network calls or cloud API keys. The `offline_deterministic` arm provides byte-stable verification.
- **Provider Drift Awareness**: Cloud LLM outputs drift over time. Experiments comparing against LLMs must record the exact pinned provider model configured via `src/privacy_harm_heuristics/llm/provider.py` (`GEMINI_MODEL`, `OPENAI_MODEL`) and validate against reference baselines with documented tolerance bands rather than bitwise equality.
- **Strict Data Hygiene**: Absolute prohibition of personal identifiable information (PII) in committed files. All datasets must pass `tests/test_scrub.py`.

---

## 2. Team Structure and Tier Allocations

The team operates using provider-agnostic multi-agent tier aliases:
- **`pro` Tier**: Reserved for high-level theoretical synthesis, taxonomy curation, hypothesis formulation, and adversarial validation/auditing.
- **`flash` Tier**: Assigned to bounded, deterministic implementation, model training, feature pipeline engineering, and benchmark execution.

```
+-------------------------------------------------------------------------+
|                  Lead Research & Taxonomy Architect                     |
|                      [research_lead] (Tier: pro)                        |
+-------------------------------------------------------------------------+
       |                                              |
       v                                              v
+-------------------------------+      +-------------------------------+
|   Interpretable Model Eng.    |      |  Benchmark & Eval Specialist  |
|  [model_engineer] (Tier: flash)|      |  [eval_specialist] (Tier: flash)|
+-------------------------------+      +-------------------------------+
       |                                              |
       v                                              v
+-------------------------------+      +-------------------------------+
|  Data Provenance & Ethics Aud.|      |   Hybrid & LLM Systems Eval   |
| [provenance_auditor] (Tier: pro)     | [hybrid_evaluator] (Tier: flash)|
+-------------------------------+      +-------------------------------+
```

---

## 3. Role Specifications

### 3.1 Lead Research & Taxonomy Architect (`research_lead`)
- **Tier**: `pro`
- **Primary Objective**: Maintain taxonomy integrity, align technical root causes with legal/ethical frameworks, design research trials, and synthesize findings.
- **Key Responsibilities**:
  - Curate Solove privacy taxonomy categories, subtypes, and keywords in `src/privacy_harm_heuristics/constants/privacy_taxonomy.py`.
  - Maintain rulesets in `data/rules/*.json` (`solove.v1.0.json`, `nist.v1.0.json`, `gdpr.v1.0.json`, `expert_framework.v1.0.json`).
  - Synthesize cross-framework mappings between legal articles (GDPR Art. 83, NIST PF) and the 16 standard technical root causes (`COMMON_ROOT_CAUSES`).
  - Formulate empirical hypotheses and review all experimental results before publication in `docs/`.
- **Allowed Scope**: `docs/**`, `heuristics/**`, `data/rules/**`, `src/privacy_harm_heuristics/constants/**`, `data/experiments/**`.
- **Validation Resource**: `tests/test_harm_taxonomy.py`.

### 3.2 Interpretable Model Engineer (`model_engineer`)
- **Tier**: `flash`
- **Primary Objective**: Train and tune interpretable models, extract human-readable heuristics, and manage model checkpoints.
- **Key Responsibilities**:
  - Maintain and train the 5 interpretable trainers in `src/privacy_harm_heuristics/models/trainers/`:
    1. `decision_tree.py`: Scikit-learn multi-output decision tree.
    2. `sparse_linear.py`: L1-penalized sparse logistic regression with odds ratio extraction.
    3. `ebm.py`: Explainable Boosting Machine using InterpretML.
    4. `brl.py`: Bayesian Rule Lists using imodels.
    5. `bayes_net.py`: Chow-Liu Bayesian Network using pomegranate.
  - Execute heuristics extraction (`src/privacy_harm_heuristics/models/heuristics.py`) to emit:
    - `heuristics.jsonl`: Parsed branch rules with support, precision, and lift.
    - `HEURISTICS.md`: Markdown tables of human-readable decision rules.
    - `heuristics_tree.json`: Nested AST representation for visual rendering.
  - Serialize checkpoints under `trained_models/<model_name>/` containing `model.joblib`, `metrics.json`, `provenance.json`, and `feature_names.json`.
- **Allowed Scope**: `src/privacy_harm_heuristics/models/**`, `src/privacy_harm_heuristics/features/**`, `heuristics/**`, `trained_models/**`, `results/heuristics/**`.
- **Validation Resources**: `tests/test_decision_tree_trainer.py`, `tests/test_sparse_linear_trainer.py`, `tests/test_ebm_trainer.py`, `tests/test_heuristics_serialization.py`.

### 3.3 Benchmark & Evaluation Specialist (`eval_specialist`)
- **Tier**: `flash`
- **Primary Objective**: Curate golden evaluation datasets, execute benchmark suites, compute standardized metrics, and gate regressions.
- **Key Responsibilities**:
  - Maintain the gold-standard evaluation corpus: `data/golden_cases_v3.jsonl` (50 curated incidents) and `data/golden_with_features.jsonl`.
  - Calculate multi-label classification (MLC) metrics via `src/privacy_harm_heuristics/evals/metrics.py`: instance Jaccard, Exact Match Ratio (EMR), Micro/Macro F1, Hamming loss.
  - Calculate ranking metrics: NDCG@5, MAP@5, MRR@5.
  - Calculate ordinal severity metrics: Cohen's quadratic-weighted kappa.
  - Verify byte-stability of the offline keyword baseline (`offline_deterministic` arm) via `cli reproduce` without `--check` (note: `cli reproduce` always routes hybrid arms through `provider="fallback"` offline, so `--check` intentionally exits `code 1` against live references as documented in `src/privacy_harm_heuristics/cli.py`).
- **Allowed Scope**: `src/privacy_harm_heuristics/evals/**`, `data/experiments/**`, `results/comparison/**`, `tests/evals/**`.
- **Validation Resources**: `tests/evals/test_metrics.py`, `tests/test_cli.py`, `data/experiments/final_results_summary.md`, `paper/README.md`.

### 3.4 Hybrid & LLM Systems Evaluator (`hybrid_evaluator`)
- **Tier**: `flash`
- **Primary Objective**: Evaluate hybrid reasoning architectures that combine deterministic heuristics with LLMs, manage provider configurations, and track model drift.
- **Key Responsibilities**:
  - Maintain and evaluate the 6 hybrid modes in `src/privacy_harm_heuristics/models/hybrid.py`:
    1. `baseline`: Direct LLM evaluation without heuristics.
    2. `rules_static`: Fixed prompt injection of normative expert rulesets.
    3. `rules_dynamic`: Incident-specific keyword-matched rule selection.
    4. `rag`: Vector retrieval of top-k related past incidents.
    5. `hybrid_deterministic_first`: Deterministic rules evaluate first; LLM adjudicates unresolved edge cases.
    6. `hybrid_llm_first`: LLM proposes harm categories; deterministic rules validate or prune.
  - Manage provider selection and model pinning in `src/privacy_harm_heuristics/llm/provider.py` (`GEMINI_MODEL`, `GEMINI_HEAVY_MODEL`, `OPENAI_MODEL`, `LLM_PROVIDER="fallback"`).
  - Execute multi-iteration comparison trials (`src/privacy_harm_heuristics/evals/comparison_trials.py`) with Beta-distribution confidence tracking.
  - Document provider drift across model iterations and maintain comparison logs in `data/experiments/`.
- **Allowed Scope**: `src/privacy_harm_heuristics/models/hybrid.py`, `src/privacy_harm_heuristics/llm/**`, `src/privacy_harm_heuristics/evals/hybrid_runner.py`, `src/privacy_harm_heuristics/evals/comparison_trials.py`, `data/experiments/**`.
- **Validation Resources**: `tests/test_provider_pinning.py`, `tests/test_llm_eval.py`, `tests/labeling/test_llm_fallback.py`.

### 3.5 Data Provenance & Ethics Auditor (`provenance_auditor`)
- **Tier**: `pro`
- **Primary Objective**: Enforce zero PII leakage, verify cryptographic checksums, audit data splits, and ensure research ethics compliance.
- **Key Responsibilities**:
  - Enforce automated PII scrubbing across all committed datasets via `tests/test_scrub.py`, dropping 20+ forbidden fields (`raw`, `author`, `username`, `account_display_name`, `url`, `permalink`, etc.).
  - Verify deterministic sample dataset regeneration with fixed random seeds (`scripts/make_samples.py`).
  - Verify cryptographic SHA-256 integrity digests of committed files in `data/CHECKSUMS.txt` (`grep -v 'release asset' data/CHECKSUMS.txt | sha256sum -c -`).
  - Audit train/validation/test splits to prevent evaluation contamination between `with_features.sample.jsonl` and `golden_with_features.jsonl`.
- **Allowed Scope**: `scripts/**`, `data/**`, `tests/test_scrub.py`, `docs/RELEASE-AUDIT.md`.
- **Validation Resources**: `tests/test_scrub.py`, `data/CHECKSUMS.txt`, `docs/RELEASE-AUDIT.md`.

---

## 4. Authoritative Execution Workflows

### Workflow 1: Taxonomy & Rules Curation Workflow (W1)
- **Trigger**: New legal frameworks, newly identified incident patterns, or taxonomy refinements.
- **Lead**: `research_lead` | **Supporting**: `provenance_auditor`
- **Execution Protocol**:
  1. `research_lead` edits `src/privacy_harm_heuristics/constants/privacy_taxonomy.py` or `data/rules/<framework>.v1.0.json`.
  2. Ensure all rule conditions use canonical regex or feature identifiers (`kw_*`, `pf_*`, `rc_*`).
  3. `provenance_auditor` inspects rule examples and text snippets for PII.
  4. Run validation: `.venv/bin/pytest tests/test_harm_taxonomy.py`.
  5. Verify zero taxonomy parse errors.

### Workflow 2: Model Training & Heuristics Extraction (W2)
- **Trigger**: Updated feature engineering, new sample datasets, or hyperparameter tuning.
- **Lead**: `model_engineer` | **Supporting**: `eval_specialist`
- **Execution Protocol**:
  1. Verify sample feature file exists: `test -f data/with_features.sample.jsonl`.
  2. Execute unified training CLI:
     ```bash
     .venv/bin/python -m privacy_harm_heuristics.cli train-all \
       --data data/with_features.sample.jsonl \
       --golden data/golden_with_features.jsonl
     ```
  3. Confirm artifacts emitted under `results/heuristics/` and `trained_models/`:
     - `heuristics.jsonl`
     - `HEURISTICS.md`
     - `heuristics_tree.json`
     - `model.joblib`
     - `metrics.json`
     - `provenance.json`
  4. Run trainer test suite:
     ```bash
     .venv/bin/pytest tests/test_decision_tree_trainer.py \
       tests/test_sparse_linear_trainer.py \
       tests/test_ebm_trainer.py \
       tests/test_heuristics_serialization.py
     ```

### Workflow 3: Offline Deterministic Benchmark & Regression Gating (W3)
- **Trigger**: Prior to creating any commit, PR, or releasing model artifacts.
- **Lead**: `eval_specialist` | **Supporting**: `model_engineer`, `hybrid_evaluator`
- **Execution Protocol**:
  1. Execute offline reproduction without `--check` (unset API keys so `provider='fallback'` runs deterministically; note `--check` evaluates the LLM-routed `rules_static` arm and intentionally exits non-zero offline as documented in `src/privacy_harm_heuristics/cli.py`):
     ```bash
     env -u OPENAI_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY \
       .venv/bin/python -m privacy_harm_heuristics.cli reproduce
     ```
  2. Verify in `results/comparison/comparison.json` and `results/comparison/comparison_table.md` that `Offline Deterministic (keyword rules)` yields target baselines:
     - Instance Jaccard: `0.0808` (±0.0001)
     - Exact Match Ratio: `0.0800` (±0.0001)
     - Micro-F1: `0.2222` (±0.0001)
     - NDCG@5: `0.1623` (±0.0001)
  3. Compare candidate models against pre-specified constant baselines per `paper/README.md`.
  4. Run full test suite: `.venv/bin/pytest tests/`.

### Workflow 4: Hybrid & LLM Systems Evaluation (W4)
- **Trigger**: Evaluating new LLM versions or testing prompt modification hypotheses.
- **Lead**: `hybrid_evaluator` | **Supporting**: `eval_specialist`
- **Execution Protocol**:
  1. Check for available API keys (`GEMINI_API_KEY` or `OPENAI_API_KEY`). If absent, verify graceful fallback:
     ```bash
     .venv/bin/pytest tests/labeling/test_llm_fallback.py
     ```
  2. Rely on the pinned default model in `src/privacy_harm_heuristics/llm/provider.py` (or override explicitly via `GEMINI_MODEL` / `OPENAI_MODEL` and record the exact string in provenance metadata).
  3. Execute live-provider hybrid comparison into an untracked output directory (do not use `cli reproduce`, which hardcodes `provider="fallback"`):
     ```bash
     .venv/bin/python -m privacy_harm_heuristics.evals.hybrid_runner \
       --golden-file data/golden_cases_v3.jsonl \
       --mode rules_static \
       --ruleset expert_framework \
       --ruleset-version 1.0 \
       --output-dir results/live_drift
     ```
  4. Compute delta against `data/experiments/rerun_20260721/README.md` and constant baselines (`paper/README.md`).
  5. If drift exceeds 10% on EMR or F1, flag to `research_lead` and log a new dated record under `data/experiments/`.

### Workflow 5: Data Hygiene & Provenance Audit (W5)
- **Trigger**: Any data sample regeneration or ingestion of external breach records.
- **Lead**: `provenance_auditor` | **Supporting**: `research_lead`
- **Execution Protocol**:
  1. If regenerating sample corpora from raw sources:
     ```bash
     .venv/bin/python scripts/make_samples.py
     ```
  2. Execute PII scrub audit:
     ```bash
     .venv/bin/pytest tests/test_scrub.py
     ```
  3. Validate cryptographic checksums of committed data files (excluding the external `v1.0.0` release-asset entry):
     ```bash
     grep -v 'release asset' data/CHECKSUMS.txt | sha256sum -c -
     ```
  4. Verify that no data from `data/golden_cases_v3.jsonl` appears in training samples.

---

## 5. Repository Validation Resources Map

| Category | Path | Description & Usage |
|---|---|---|
| **Test Suite** | `tests/` | 22 test files covering trainers, evals, CLI, taxonomy, and serialization. |
| **Golden Datasets** | `data/golden_cases_v3.jsonl` | 50 curated incident cases for standard benchmarking. |
| **Feature Samples** | `data/with_features.sample.jsonl` | 2,549 stratified sample rows with engineered features (`f_*`, `kw_*`, `pf_*`, `rc_*`). |
| **GLiNER2 Labels** | `data/labeled_gliner2.sample.jsonl` | 987 GLiNER2-labeled incident snippets. |
| **Integrity Hashes** | `data/CHECKSUMS.txt` | SHA-256 checksums for committed datasets and release archives. |
| **Calibrated Weights** | `data/calibrated_weights.json` | Empirical regression coefficients mapping harm categories to 1–5 severity scores. |
| **Rule Sets** | `data/rules/` | Versioned rule definitions: `solove.v1.0.json`, `nist.v1.0.json`, `gdpr.v1.0.json`, `expert_framework.v1.0.json`, `learned_dt.v1.1.json`. |
| **Heuristics Catalog** | `heuristics/` | 14 human-readable heuristic files (`augmented_solove.json`, `decision_tree_v3.jsonl`, `sparse_linear_v3.jsonl`, `ebm_v3.jsonl`, `bayes_net_v3.jsonl`, `brl_cascade_v3.jsonl`). |
| **Experimental Baselines** | `data/experiments/final_results_summary.md` | Frozen headline table recording the 6 evaluation modes across n=50 golden cases. |
| **Provider Drift Logs** | `data/experiments/rerun_20260721/README.md` | Reference live rerun tracking provider drift against recorded 2025 baselines. |
| **Re-Analysis & Constant Baselines** | `paper/` | `README.md`, `reanalysis.py`, `reanalysis.json`, and `trials_2025/` documenting offline trial verification and constant-predictor baselines. |
| **Core Documentation** | `docs/` | `OVERVIEW.md`, `MODELING_AND_HEURISTICS.md`, `COMPARISON_METRICS.md`, `COMPARISON_TRIALS.md`, `RELEASE-AUDIT.md`, `architecture/INTERPRETABILITY_GUIDE.md`, `reference/GLOBAL_PRIVACY_FRAMEWORKS.md`, `ops/MAINTENANCE.md`. |

---

## 6. Gating and Regression Policy

Every pull request, model release, or heuristic revision must pass through the four quality gates:

1. **Gate 1: Unit & Integration Health**
   - Command: `.venv/bin/pytest tests/`
   - Requirement: 70 passed, 0 failed, 6 skipped.
2. **Gate 2: Code Quality & Typing**
   - Command: `make lint` (`ruff check`, `black --check`, `mypy src/`)
   - Requirement: Zero errors or warnings.
3. **Gate 3: Deterministic Baseline Invariance**
   - Command:
     ```bash
     env -u OPENAI_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY .venv/bin/python -m privacy_harm_heuristics.cli reproduce && \
       .venv/bin/python -c "import json; m = json.load(open('results/comparison/comparison.json'))['modes']['offline_deterministic']; assert abs(m['instance_jaccard'] - 0.0808) < 1e-4 and abs(m['exact_match_ratio'] - 0.0800) < 1e-4 and abs(m['micro_f1'] - 0.2222) < 1e-4 and abs(m['ndcg@5'] - 0.1623) < 1e-4, m"
     ```
   - Requirement: Exits `0` offline and asserts `offline_deterministic` keyword arm metrics match within `0.0001`.
4. **Gate 4: Privacy & Ethics Invariance**
   - Command: `.venv/bin/pytest tests/test_scrub.py && (grep -v 'release asset' data/CHECKSUMS.txt | sha256sum -c -)`
   - Requirement: Zero PII detected; all committed checksums match.
