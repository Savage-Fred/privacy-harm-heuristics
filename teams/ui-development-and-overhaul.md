# Operational Playbook: UI Development and Overhaul Autonomous Team

## 1. Mission and UI Overhaul Mandate

The **UI Development and Overhaul Autonomous Team** is tasked with autonomously designing, implementing, serving, and testing a modern user interface for the `privacy-harm-heuristics` repository.

### Context of the Overhaul
Historical documentation (`docs/OVERVIEW.md`, `docs/MODELING_AND_HEURISTICS.md`, `docs/architecture/INTERPRETABILITY_GUIDE.md`) references several user-facing tools:
1. A 0–100 privacy "Temperature Check" UI (`temperature_check.html`).
2. An interactive root-cause interpretability panel (`http://localhost:8000/interpretability`).
3. Web-based experiment results and leaderboards.

When the standalone research artifact was frozen from the masters practicum ("Human vs Human vs Machine", Dec 2025), frontend web code was **not extracted** into the repository. Consequently, there are currently **zero** web templates or frontend assets in `src/`.

The UI team's mission is **not** to patch legacy spaghetti code, but to execute a **greenfield architectural exposition and overhaul**: building a lightweight, high-performance, offline-first **Privacy Harm Heuristics Studio** that exposes the rich predictive and interpretable models directly to practitioners and researchers.

### Core Operating Principles
- **Offline-First Guarantee**: All features must function completely offline using local artifacts (`trained_models/`, `heuristics/`, `data/rules/`). When calling `HybridScorer`, the backend must explicitly use `provider="fallback"` to prevent missing-API-key crashes.
- **Ponytail Discipline**: Write the smallest, cleanest implementation that meets requirements. Use Python standard library and pre-existing packages in `.venv` before introducing external dependencies.
- **Contract Conformance**: All API endpoints and response payloads must strictly match the schemas documented in `docs/architecture/INTERPRETABILITY_GUIDE.md` and repository Pydantic models.
- **Zero Regression**: UI additions must not break or alter the core research test suite (70 passing unit/integration tests).

---

## 2. Team Structure and Tier Allocations

The team operates using Antigravity multi-agent tiers:
- **`pro` Tier (Gemini 4 Argon High)**: Reserved for `ui_architect` to design systems, specify API contracts, direct component hierarchies, and resolve cross-functional ambiguities.
- **`flash` Tier (Gemini 3.8 Flash High)**: Assigned to `backend_api_engineer`, `frontend_viz_engineer`, and `ui_qa_engineer` for bounded implementation, component rendering, and test automation.

```
+-------------------------------------------------------------------------+
|                        UI Architect & Product Lead                      |
|                       [ui_architect] (Tier: pro)                        |
+-------------------------------------------------------------------------+
       |                                              |
       v                                              v
+-------------------------------+      +-------------------------------+
|   Backend Service & API Eng.  |      |  Frontend & Viz Engineer      |
| [backend_api_engineer] (flash)|      | [frontend_viz_engineer] (flash)|
+-------------------------------+      +-------------------------------+
       |                                              |
       +-----------------------+----------------------+
                               |
                               v
               +-------------------------------+
               |     UI Verification & QA      |
               |     [ui_qa_engineer] (flash)  |
               +-------------------------------+
```

---

## 3. Privacy Harm Heuristics Studio: Core Feature Specifications

The team will deliver a unified web interface structured into four core interactive modules:

```
+------------------------------------------------------------------------------------+
|                         Privacy Harm Heuristics Studio                             |
+------------------------------------------------------------------------------------+
| [1. Incident Risk Assessor] | [2. Tree Visualizer] | [3. Interpretability] | [4. Benchmarks] |
+------------------------------------------------------------------------------------+
```

### Module 1: Incident Harm Assessor ("Temperature Check")
- **Purpose**: Interactive privacy risk scoring and mitigation guidance for incident reports.
- **Input**: User-provided incident text, breach description, or system architecture summary.
- **Engine**: Connects to `ScoringEngine` and `HybridScorer` in `src/privacy_harm_heuristics/science/scoring.py` with `provider="fallback"`.
- **Outputs**:
  - **0–100 Privacy Risk Gauge**: Instantaneous risk score mapped from empirical harm weights.
  - **Harm Breakdown**: Validated harms vs. proposed/rejected harms.
  - **Framework Provenance**: Radar/bar chart showing grounding percentage (% Solove, % NIST, % GDPR, % Zuboff).
  - **Actionable Mitigations**: Protective feature recommendations (e.g., granular user consent, localized processing, automated retention limits).

### Module 2: Interactive Decision Tree & Rule Catalog Visualizer
- **Purpose**: Visual, inspectable exploration of trained heuristic decision logic.
- **Data Source**: `trained_models/decision_tree_v4/heuristics_tree.json` and `data/rules/*.json`.
- **Capabilities**:
  - Interactive AST rendering: expand/collapse tree nodes, view decision conditions (e.g. `kw_privacy <= 0.5000 AND pf_third_party_sharing > 0.5000`).
  - Branch diagnostics: support sample counts, branch weights, leaf prediction distributions.
  - Filterable rule catalog: browse and search rules across Solove, GDPR, NIST, and learned decision trees.

### Module 3: Root Cause & Interpretability Explorer
- **Purpose**: Fulfill the interactive interpretability panel specification from `docs/architecture/INTERPRETABILITY_GUIDE.md`.
- **API Contract**: `GET /api/root-cause-explanations`
- **Capabilities**:
  - **Top Risk-Increasing Features**: Red badges displaying features that strongly correlate with privacy violations, accompanied by odds ratios (> 1.0) and sample support.
  - **Top Risk-Mitigating Features**: Green badges displaying protective features that reduce harm probability, accompanied by odds ratios (< 1.0).
  - **Coefficient Browser**: Paginated, filterable table of logistic and EBM feature weights.

### Module 4: Experiment Leaderboard & Comparison Matrix
- **Purpose**: Visual presentation of empirical research benchmarks.
- **Data Source**: `data/experiments/final_results_summary.md` and `results/comparison/comparison.json`.
- **Capabilities**:
  - Multi-mode benchmark table: compare Rules Static, Baseline LLM, Rules Dynamic, RAG, Hybrid Deterministic First, and Offline Deterministic.
  - Metric filters: sort by Exact Match Ratio (EMR), Instance Jaccard, Micro-F1, and NDCG@5.

---

## 4. Serving and Interaction Architecture

### 4.1 CLI Entry Point Integration
The UI application must be launchable directly via the repository's standard Typer CLI:
```bash
# Launch Studio UI and open browser automatically
privacy-harm-heuristics ui --port 8000 --open-browser

# Launch background API service only
privacy-harm-heuristics serve --host 127.0.0.1 --port 8000
```

### 4.2 API Contract Schemas
In accordance with `docs/architecture/INTERPRETABILITY_GUIDE.md`, the primary endpoints are:

1. `GET /api/root-cause-explanations`:
   ```json
   {
     "status": "success",
     "metadata": { "model": "decision_tree_v4", "timestamp": "...", "total_features": 81 },
     "summary": { "risk_increasing_count": 45, "risk_mitigating_count": 36 },
     "risk_increasing": [
       { "feature": "kw_surveillance", "odds_ratio": 3.42, "coefficient": 1.23, "description": "Surveillance or monitoring keyword" }
     ],
     "risk_mitigating": [
       { "feature": "pf_user_consent_granular", "odds_ratio": 0.35, "coefficient": -1.05, "description": "Granular user consent controls" }
     ]
   }
   ```
2. `POST /api/score`:
   - Request: `{"text": "Company collects precise GPS data from minors without parental notice."}`
   - Response: `{"risk_score_0_100": 78.5, "score_1_to_5": 3.9, "validated_harms": ["surveillance", "exclusion"], "frameworks": {"solove": 0.45, "nist": 0.25, "gdpr": 0.30}}`
3. `GET /api/tree`:
   - Returns full AST from `trained_models/decision_tree_v4/heuristics_tree.json`.
4. `GET /api/rules`:
   - Returns versioned rules catalog from `data/rules/`.

### 4.3 Asset & Styling Conventions
- **Offline Bundling**: All JavaScript and CSS assets must be self-contained in `src/privacy_harm_heuristics/web/static/`. Absolutely no external CDN `<script>` or `<link>` tags (`cdnjs`, `unpkg`, `google fonts`).
- **Design Tokens**: Standardized CSS custom properties for light and dark themes:
  - `--bg-primary`, `--bg-secondary`, `--text-primary`, `--text-muted`
  - `--risk-high` (crimson), `--risk-med` (amber), `--risk-low` (emerald)
  - System font stack (`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`).
- **Accessibility (a11y)**: WCAG 2.1 AA compliant. Explicit ARIA landmarks (`role="main"`, `aria-label`), visible `:focus-visible` outlines, minimum 4.5:1 text contrast.

---

## 5. Role Protocols & Detailed Responsibilities

### 5.1 UI Architect & Product Lead (`ui_architect`)
- **Tier**: `pro`
- **Scope**: `pyproject.toml`, `Makefile`, `docs/**`, `src/privacy_harm_heuristics/cli.py`, `src/privacy_harm_heuristics/web/**`.
- **Protocols**:
  1. Define packaging extras in `pyproject.toml` under `[project.optional-dependencies] ui = [...]` so fresh clones can install UI tools deterministically.
  2. Review and freeze API endpoint schemas before backend implementation.
  3. Register Typer CLI commands in `src/privacy_harm_heuristics/cli.py`.
  4. Perform architectural review and sign-off on all PRs.

### 5.2 Backend Service & API Engineer (`backend_api_engineer`)
- **Tier**: `flash`
- **Scope**: `src/privacy_harm_heuristics/web/**`, `src/privacy_harm_heuristics/cli.py`, `tests/test_web_api.py`.
- **Protocols**:
  1. Implement lightweight service runner using standard, battle-tested Python web frameworks (FastAPI/Starlette or Flask).
  2. Connect `/api/score` directly to `ScoringEngine` and `HybridScorer` with `provider="fallback"`.
  3. Implement caching with `os.path.getmtime` checking to ensure zero latency overhead on repeated calls.
  4. Write unit tests in `tests/test_web_api.py` verifying status codes, schema validation, and edge-case handling.

### 5.3 Frontend & Visualization Engineer (`frontend_viz_engineer`)
- **Tier**: `flash`
- **Scope**: `src/privacy_harm_heuristics/web/static/**`, `src/privacy_harm_heuristics/web/templates/**`, `src/privacy_harm_heuristics/web/ui/**`.
- **Protocols**:
  1. Build modular HTML/CSS/JS components for each of the 4 Studio modules.
  2. Implement interactive tree graph rendering using SVG canvas or lightweight layout algorithms.
  3. Bind incident submission form to `/api/score` with immediate UI updates and loading states.
  4. Ensure complete offline operation by auditing network tabs for any outbound HTTP requests.

### 5.4 UI Verification & QA Engineer (`ui_qa_engineer`)
- **Tier**: `flash`
- **Scope**: `tests/test_web_api.py`, `tests/test_ui_cli.py`, `tests/ui/**`, `tests/conftest.py`.
- **Protocols**:
  1. Test API endpoints using HTTP test client without starting network sockets.
  2. Test CLI commands using Typer `CliRunner` verifying `--help`, option parsing, and clean termination.
  3. Audit DOM structure for WCAG 2.1 AA compliance (labels, roles, contrast).
  4. Benchmark cached endpoint latency ensuring responses complete in under 15ms.

---

## 6. Authoritative Execution Workflows

### Workflow 1: UI Architecture & Packaging Workflow (W1)
- **Lead**: `ui_architect` | **Supporting**: `backend_api_engineer`
- **Protocol**:
  1. Inspect `pyproject.toml` and declare optional dependencies for UI serving.
  2. Document API request/response structures in `docs/UI_GUIDE.md`.
  3. Confirm package builds cleanly: `.venv/bin/pip install -e .`.

### Workflow 2: Backend API & CLI Integration (W2)
- **Lead**: `backend_api_engineer` | **Supporting**: `ui_architect`, `ui_qa_engineer`
- **Protocol**:
  1. Implement `src/privacy_harm_heuristics/web/server.py` and routers.
  2. Verify offline fallback in scoring router:
     ```python
     scorer = HybridScorer(ScoringEngine(Path("data")))
     result = scorer.hybrid_score(text, provider="fallback")
     ```
  3. Add `ui` command group to `src/privacy_harm_heuristics/cli.py`.
  4. Run API tests: `.venv/bin/pytest tests/test_web_api.py`.

### Workflow 3: Frontend Visualization Build (W3)
- **Lead**: `frontend_viz_engineer` | **Supporting**: `ui_architect`
- **Protocol**:
  1. Author HTML/CSS/JS assets under `src/privacy_harm_heuristics/web/`.
  2. Test interactive rendering of `trained_models/decision_tree_v4/heuristics_tree.json`.
  3. Validate dark mode color tokens and responsive viewport breakpoints.
  4. Audit network egress to ensure zero external calls.

### Workflow 4: Comprehensive UI Verification & Accessibility Audit (W4)
- **Lead**: `ui_qa_engineer` | **Supporting**: `backend_api_engineer`, `frontend_viz_engineer`
- **Protocol**:
  1. Run full project test suite: `.venv/bin/pytest tests/`.
  2. Run code style and type checkers: `make lint`.
  3. Verify CLI help output:
     ```bash
     .venv/bin/python -m privacy_harm_heuristics.cli ui --help
     ```
  4. Confirm working tree remains completely clean.

---

## 7. Gating and Regression Policy

Every UI change must pass the three UI quality gates before approval:
1. **Gate 1: Zero Core Regressions**: Full pytest suite must pass with zero failures (70 passing core tests).
2. **Gate 2: Offline Isolation**: Zero outbound network requests during serving or test execution.
3. **Gate 3: Latency & A11y Standards**: Cached API response latency < 15ms; zero critical WCAG 2.1 AA accessibility violations.
