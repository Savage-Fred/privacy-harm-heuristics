# Autonomous Teams Framework: privacy-harm-heuristics

## 1. Overview

This directory contains the authoritative configuration files and operational playbooks for autonomous multi-agent teams operating on the `privacy-harm-heuristics` repository.

The system is organized around two specialized teams:
1. **Research and Validation Autonomous Team** (`teams/research-and-validation.*`): Focuses on empirical privacy harm research, interpretable model training (Decision Trees, Sparse Linear, EBM, BRL, Bayes Net), taxonomy grounding (Solove, NIST, GDPR, ISO 29100), weak-supervision calibration, and multi-metric benchmark replication.
2. **UI Development and Overhaul Autonomous Team** (`teams/ui-development-and-overhaul.*`): Focuses on greenfield exposition and overhaul of the user interface, building an offline-first **Privacy Harm Heuristics Studio**, exposing REST API endpoints (`/api/root-cause-explanations`, `/api/score`), visualizing decision trees, and integrating CLI serving commands.

---

## 2. Directory Layout & Dual-Specification Standard

Each team is defined using a **Dual-Specification Standard**:
- **Declarative Machine Schema (`.yaml`)**: Structured, machine-parseable definitions of team metadata, model tiers, role rosters, tool capabilities, strict file scopes, and execution workflows. Designed for programmatic consumption by multi-agent runners, Antigravity SDK orchestrators, and CI pipelines.
- **Authoritative Operational Playbook (`.md`)**: Human-readable and agent-executable operational manuals detailing scientific mandates, role personas, step-by-step protocols, boundary conditions, and quality gates.

```
teams/
├── README.md                           # This framework index and operational guide
├── research-and-validation.yaml        # Declarative machine schema (Research & Validation)
├── research-and-validation.md          # Authoritative operational playbook (Research & Validation)
├── ui-development-and-overhaul.yaml    # Declarative machine schema (UI Development & Overhaul)
└── ui-development-and-overhaul.md      # Authoritative operational playbook (UI Development & Overhaul)
```

---

## 3. Team Summary & Role Matrix

### 3.1 Research and Validation Team
- **Playbook**: [`teams/research-and-validation.md`](./research-and-validation.md)
- **Machine Schema**: [`teams/research-and-validation.yaml`](./research-and-validation.yaml)
- **Default Orchestrator**: `research_lead`

| Role ID | Title | Tier | Primary Scope | Key Tools |
|---|---|---|---|---|
| `research_lead` | Lead Research & Taxonomy Architect | `pro` | Taxonomy curation, Solove alignment, ruleset authoring | Python, JSON validator, Git |
| `model_engineer` | Interpretable Model Engineer | `flash` | Training 5 interpretable models, heuristic rule extraction | Scikit-learn, InterpretML, Joblib |
| `eval_specialist` | Benchmark & Evaluation Specialist | `flash` | Golden dataset evaluation, MLC & ranking metrics, regression gating | Pytest, CLI reproduce runner |
| `hybrid_evaluator` | Hybrid & LLM Systems Evaluator | `flash` | Hybrid mode evaluation, provider pinning, drift monitoring | Hybrid runner, Provider API |
| `provenance_auditor` | Data Provenance & Ethics Auditor | `pro` | Zero-PII enforcement, checksum validation, split hygiene | Make samples, sha256sum, Scrub tests |

#### Core Workflows
- **W1 (Taxonomy & Rules Curation)**: Evolving taxonomy in `src/privacy_harm_heuristics/constants/privacy_taxonomy.py` and rules in `data/rules/`.
- **W2 (Model Training & Heuristics Extraction)**: Training models on `data/with_features.sample.jsonl` and exporting heuristics to `heuristics/` and `trained_models/`.
- **W3 (Benchmark & Evaluation)**: Running `make reproduce` and validating offline deterministic keyword baseline (Jaccard: 0.0808, EMR: 0.0800, F1: 0.2222).
- **W4 (Hybrid & LLM Systems Evaluation)**: Multi-iteration comparison trials and provider drift tracking.
- **W5 (Data Hygiene & Provenance Audit)**: Enforcing 20+ forbidden PII fields drop via `pytest tests/test_scrub.py`.

---

### 3.2 UI Development and Overhaul Team
- **Playbook**: [`teams/ui-development-and-overhaul.md`](./ui-development-and-overhaul.md)
- **Machine Schema**: [`teams/ui-development-and-overhaul.yaml`](./ui-development-and-overhaul.yaml)
- **Default Orchestrator**: `ui_architect`

| Role ID | Title | Tier | Primary Scope | Key Tools |
|---|---|---|---|---|
| `ui_architect` | UI Architect & Product Lead | `pro` | System architecture, API contracts, CLI integration | Typer, Pyproject packager |
| `backend_api_engineer` | Backend Service & API Engineer | `flash` | REST API routes, scoring engine adapters, caching | Starlette/Flask/FastAPI, Pydantic |
| `frontend_viz_engineer` | Frontend & Visualization Engineer | `flash` | 4 Studio modules, decision tree graph, dark mode | Canvas/SVG, CSS tokens, Cytoscape |
| `ui_qa_engineer` | UI Verification & QA Engineer | `flash` | API unit tests, CLI smoke tests, a11y (WCAG 2.1 AA) | Pytest, TestClient, CliRunner |

#### Core Studio Modules
1. **Incident Harm Assessor ("Temperature Check")**: Interactive 0–100 privacy risk scoring, validated harms, and framework provenance radar (% Solove, % NIST, % GDPR, % Zuboff).
2. **Interactive Decision Tree & Rule Catalog Visualizer**: Inspectable tree graph parsing `trained_models/decision_tree_v4/heuristics_tree.json` with branch support and leaf predictions.
3. **Root Cause & Interpretability Explorer**: Implementation of `GET /api/root-cause-explanations` displaying red risk-increasing badges (odds ratio > 1.0) and green protective badges.
4. **Experiment Leaderboard & Benchmark Matrix**: Visual comparison across hybrid and deterministic baseline models.

---

## 4. Multi-Agent Tier Strategy

The autonomous teams follow a two-tier provider-agnostic specialization architecture using role/tier aliases rather than hard-coded model IDs:
- **`pro` Tier**: Orchestrators, architects, and auditors. Responsible for system design, theoretical grounding, ethical gating, and adversarial falsification.
- **`flash` Tier**: Specialized execution workers. Responsible for deterministic implementation, component rendering, metric calculation, and test execution.

---

## 5. Verification Commands

To verify the team configurations, repository health, and underlying resources:

```bash
# 1. Full regression test suite (expects 70 passed)
.venv/bin/pytest tests/

# 2. Run offline deterministic benchmark reproduction and assert target scores
env -u OPENAI_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY .venv/bin/python -m privacy_harm_heuristics.cli reproduce && \
  .venv/bin/python -c "import json; m = json.load(open('results/comparison/comparison.json'))['modes']['offline_deterministic']; assert abs(m['instance_jaccard'] - 0.0808) < 1e-4 and abs(m['exact_match_ratio'] - 0.0800) < 1e-4 and abs(m['micro_f1'] - 0.2222) < 1e-4 and abs(m['ndcg@5'] - 0.1623) < 1e-4, m"

# 3. Verify PII scrubbing and data hygiene
.venv/bin/pytest tests/test_scrub.py

# 4. Verify cryptographic checksums of committed data files (excluding external v1.0.0 release asset line)
grep -v 'release asset' data/CHECKSUMS.txt | sha256sum -c -

# 5. Verify YAML schema validity (requires PyYAML, e.g. system python3-yaml)
python3 -c "import yaml; yaml.safe_load(open('teams/research-and-validation.yaml')); yaml.safe_load(open('teams/ui-development-and-overhaul.yaml')); print('All team YAML schemas valid')"
```
