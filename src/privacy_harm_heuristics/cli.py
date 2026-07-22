"""privacy-harm-heuristics CLI.

Two commands make up the "runnable story" for this practicum artifact:

- ``train-all``: train the five interpretable models (decision tree, sparse
  linear, EBM, BRL, Bayes net) on the committed feature sample and export
  heuristics. Models whose optional dependency is not installed (``imodels``
  for BRL, ``pomegranate`` for Bayes net) are skipped with a message rather
  than failing the run.
- ``reproduce``: run the hybrid-model arms (baseline, rules_static, and
  friends -- see ``models/hybrid.py::HybridMode``), the genuinely-offline
  ``offline_deterministic`` keyword arm, and the expert-framework evaluators
  (NIST/Solove/ISO 29100) against ``data/golden_cases_v3.jsonl``, print +
  persist a comparison table, and optionally (``--check``) diff the
  ``rules_static`` row against a reference (``--reference``):
    * ``recorded-2025`` (default): the recorded headline numbers in
      ``data/experiments/final_results_summary.md``. Offline this MISMATCHES
      by construction (see the IMPORTANT note below); the honest root-cause /
      provider-drift explanation is printed.
    * ``rerun-2026``: the live 2026-07-21 rerun in
      ``data/experiments/rerun_20260721/`` with a looser tolerance (0.05, for
      LLM stochasticity). This is the regression guard for *live* runs (real
      ``GEMINI_API_KEY``); offline it also mismatches (near-zero fallback).

Both commands are offline-safe by construction: ``reproduce`` pins the LLM
provider to ``"fallback"`` (see ``llm/provider.py::complete`` -- "Always safe
to call offline") so no network call is attempted regardless of the caller's
environment. The ``offline_deterministic`` arm calls no LLM at all -- it is a
pure keyword/pattern scorer (``labeling/harm_labeler.py``) and produces
byte-identical output on every run. It is NOT part of the recorded 2025
comparison and does not attempt to reproduce those numbers.

IMPORTANT, read before "fixing" a --check mismatch: ``HybridMode.RULES_STATIC``
is NOT a pure deterministic classifier. ``HybridModel.predict()`` always calls
the LLM (``models/hybrid.py`` step 3) -- "rules_static" only means the prompt
has a *static* rules block injected into it; the actual label extraction is
still done by the LLM's JSON response. The offline fallback provider returns a
truncated echo of the prompt (not JSON), so `_parse_response` cannot recover
real labels and the offline metrics come out near-zero. The recorded numbers
in final_results_summary.md were produced by a live Gemini call. This is a
structural property of the extracted code, not a bug in this CLI -- do not
"fix" it by faking a match; see the P2 report for the honest writeup.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

app = typer.Typer(
    help="Train interpretable models and reproduce the practicum's headline comparison.",
    no_args_is_help=True,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_DATA = REPO_ROOT / "data" / "with_features.sample.jsonl"
DEFAULT_GOLDEN_FEATURES = REPO_ROOT / "data" / "golden_with_features.jsonl"
DEFAULT_HEURISTICS_OUT = REPO_ROOT / "results" / "heuristics"
DEFAULT_GOLDEN_CASES = REPO_ROOT / "data" / "golden_cases_v3.jsonl"
DEFAULT_RULES_DIR = REPO_ROOT / "data" / "rules"
DEFAULT_COMPARISON_OUT = REPO_ROOT / "results" / "comparison"
DEFAULT_SUMMARY_MD = REPO_ROOT / "data" / "experiments" / "final_results_summary.md"
DEFAULT_RERUN_DIR = REPO_ROOT / "data" / "experiments" / "rerun_20260721"

ALL_MODELS = "decision_tree,sparse_linear,ebm,brl,bayes_net"

# The genuinely-offline deterministic arm (offline keyword scorer). Not a HybridMode:
# it is a pure keyword/pattern scorer, handled specially in `reproduce`.
OFFLINE_DETERMINISTIC = "offline_deterministic"
# Per-category keyword score (labeling/harm_labeler.py) required to count a
# category as present. Fixed constant -> identical output on every run.
OFFLINE_DET_THRESHOLD = 1.0

# ---------------------------------------------------------------------------
# train-all
# ---------------------------------------------------------------------------


def _train_one(name: str, ds: Any, *, random_state: int) -> Any:
    """Dispatch to the trainer for ``name``. Raises ImportError if the
    model's optional dependency is not installed (caller decides to skip)."""
    if name == "decision_tree":
        from .models.trainers.decision_tree import train_decision_tree

        return train_decision_tree(
            ds.X_train, ds.y_train, ds.X_test, ds.y_test, random_state=random_state
        )
    if name == "sparse_linear":
        from .models.trainers.sparse_linear import train_sparse_linear

        return train_sparse_linear(
            ds.X_train, ds.y_train, ds.X_test, ds.y_test, random_state=random_state
        )
    if name == "ebm":
        from .models.trainers.ebm import train_ebm

        return train_ebm(ds.X_train, ds.y_train, ds.X_test, ds.y_test, random_state=random_state)
    if name == "brl":
        from .models.trainers.brl import train_brl

        return train_brl(ds.X_train, ds.y_train, ds.X_test, ds.y_test, random_state=random_state)
    if name == "bayes_net":
        from .models.trainers.bayes_net import train_bayes_net

        return train_bayes_net(
            ds.X_train,
            ds.y_train,
            ds.X_test,
            ds.y_test,
            ds.feature_names,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model: {name}")


@app.command("train-all")
def train_all(
    data: Path = typer.Option(DEFAULT_TRAIN_DATA, "--data", help="Feature JSONL to train on."),
    target: str = typer.Option("harm_category", "--target", help="Target column name."),
    golden: Optional[Path] = typer.Option(
        DEFAULT_GOLDEN_FEATURES,
        "--golden",
        help="Golden feature JSONL used as the held-out test split (overrides --test-size).",
    ),
    out_dir: Path = typer.Option(
        DEFAULT_HEURISTICS_OUT, "--out-dir", help="Base output directory for model artifacts."
    ),
    models: str = typer.Option(
        ALL_MODELS, "--models", help="Comma-separated list of models to train."
    ),
    test_size: float = typer.Option(0.2, "--test-size"),
    random_state: int = typer.Option(42, "--random-state"),
) -> None:
    """Train decision-tree, sparse-linear, EBM, BRL, and Bayes-net models.

    Exports heuristics (JSON/Markdown/tree JSON) for the decision tree, which
    is the only model whose structure heuristics extraction currently
    targets. Models whose optional dependency (``imodels`` for BRL,
    ``pomegranate`` for Bayes net) is not installed are skipped with a
    message rather than failing the whole run.
    """
    from .models.data import build_dataset
    from .models.heuristics import (
        extract_heuristics,
        save_heuristics_json,
        save_heuristics_markdown,
        save_heuristics_tree_json,
    )
    from .models.serialize import save_model_result

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    golden_arg = str(golden) if golden and Path(golden).exists() else None
    if golden and not golden_arg:
        typer.echo(
            f"[train-all] golden set not found at {golden}; falling back to --test-size split."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    trained = 0
    skipped = 0
    for name in model_list:
        model_out = out_dir / name
        typer.echo(f"[train-all] {name}: training...")
        try:
            ds = build_dataset(
                data,
                target=target,
                test_size=test_size,
                random_state=random_state,
                golden_path=golden_arg,
            )
        except Exception as exc:
            typer.echo(f"[train-all] {name}: SKIP (failed to build dataset: {exc})")
            skipped += 1
            continue

        try:
            result = _train_one(name, ds, random_state=random_state)
        except ImportError as exc:
            typer.echo(f"[train-all] {name}: SKIP (optional dependency missing: {exc})")
            skipped += 1
            continue
        except Exception as exc:
            typer.echo(f"[train-all] {name}: FAILED ({exc})")
            skipped += 1
            continue

        out_dir_path = save_model_result(
            result,
            model_out,
            feature_names=ds.feature_names,
            training_config={"model": name, "target": target, "test_size": test_size},
            data_path=str(data),
        )
        typer.echo(f"[train-all] {name}: saved artifacts to {out_dir_path}")
        trained += 1

        if name == "decision_tree":
            try:
                items = extract_heuristics(
                    result, ds.feature_names, X_train=ds.X_train, y_train=ds.y_train, top_n=50
                )
                save_heuristics_json(items, str(out_dir_path / "heuristics.jsonl"))
                save_heuristics_markdown(
                    items, str(out_dir_path / "HEURISTICS.md"), context={"model": name}
                )
                save_heuristics_tree_json(
                    result, ds.feature_names, str(out_dir_path / "heuristics_tree.json")
                )
            except Exception as exc:
                typer.echo(f"[train-all] {name}: heuristics export failed ({exc})")

    typer.echo(f"[train-all] done. trained={trained} skipped={skipped}")


# ---------------------------------------------------------------------------
# reproduce
# ---------------------------------------------------------------------------

# HybridMode value -> the row label used in data/experiments/final_results_summary.md
_MODE_TO_SUMMARY_LABEL = {
    "baseline": "Baseline",
    "rules_static": "Rules Static",
    "rules_dynamic": "Rules Dynamic",
    "rag": "RAG",
    "hybrid_deterministic_first": "Hybrid (Det. First)",
    "hybrid_llm_first": "Hybrid (LLM First)",
    OFFLINE_DETERMINISTIC: "Offline Deterministic (keyword rules)",
}
_METRIC_COLUMNS = ["instance_jaccard", "exact_match_ratio", "micro_f1", "ndcg@5"]
_METRIC_HEADERS = ["Instance Jaccard", "Exact Match Ratio", "Micro F1", "Ranking NDCG@5"]


def _load_golden_cases_and_taxonomy(golden_file: Path) -> tuple[list[dict], dict]:
    from .constants.privacy_taxonomy import TAXONOMY_SOLOVE
    from .evals.framework_comparison import load_golden_test_cases

    cases, _ = load_golden_test_cases(golden_file)

    subtype_map: dict[str, str] = {}
    for parent, spec in TAXONOMY_SOLOVE.items():
        subtype_map[parent] = parent
        for sub in spec.get("subtypes", []):
            if isinstance(sub, dict) and "name" in sub:
                subtype_map[sub["name"]] = parent
    custom_map = {
        "data_breach": "information_processing",
        "data_breach_internal": "information_processing",
        "data_breach_phishing": "information_processing",
        "data_breach_ransomware": "information_processing",
        "data_breach_credential_reuse": "information_processing",
        "data_breach_web_skimming": "information_processing",
        "data_breach_state": "information_processing",
        "data_breach_scraping": "information_processing",
        "data_breach_reuse": "information_processing",
        "data_leak": "information_dissemination",
        "data_leak_insider": "information_dissemination",
        "whistleblower_leak": "information_dissemination",
        "privacy_misuse": "invasion",
        "personal_privacy_violation": "invasion",
    }
    subtype_map.update(custom_map)
    return cases, subtype_map


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _run_hybrid_mode(
    mode_value: str,
    cases: list[dict],
    subtype_map: dict,
    rules: list,
) -> Dict[str, float]:
    from .evals.metrics import (
        calculate_mlc_metrics,
        calculate_ordinal_metrics,
        calculate_ranking_metrics,
    )
    from .models.hybrid import HybridMode, HybridModel

    # provider="fallback" pins the deterministic offline path in
    # llm/provider.py::complete -- explicit rather than relying on
    # select_provider() auto-detection, which would otherwise default to
    # hybrid_runner's hardcoded provider="gemini" and log a scary (but
    # harmless) "package not installed" error per case.
    model = HybridModel(mode=HybridMode(mode_value), rules=rules, provider="fallback")

    y_true_causes, y_pred_causes = [], []
    y_true_ranking, y_pred_ranking = [], []
    y_true_harm, y_pred_harm = [], []

    for case in cases:
        text = case.get("description", "") or case.get("text", "")

        true_causes = set(case.get("harms", [])) or set(case.get("root_causes", []))
        if not true_causes and "harm_category" in case:
            true_causes = {case["harm_category"]}
        true_ranking = case.get("ranking", list(case.get("harms", [])))
        if not true_ranking and "harm_category" in case:
            true_ranking = [case["harm_category"]]
        true_score = int(case.get("harm_score", 0))

        true_causes = {subtype_map.get(c, c) for c in true_causes}
        true_ranking = [subtype_map.get(c, c) for c in true_ranking]
        true_ranking = _dedupe_preserve_order(true_ranking)

        pred = model.predict(text)
        pred_causes = {subtype_map.get(c, c) for c in pred["root_causes"]}
        pred_ranking = [subtype_map.get(c, c) for c in pred["ranking"]]
        pred_ranking = _dedupe_preserve_order(pred_ranking)

        y_true_causes.append(true_causes)
        y_pred_causes.append(pred_causes)
        y_true_ranking.append(true_ranking)
        y_pred_ranking.append(pred_ranking)
        y_true_harm.append(true_score)
        y_pred_harm.append(pred["harm_score"])

    mlc = calculate_mlc_metrics(y_true_causes, y_pred_causes)
    ranking = calculate_ranking_metrics(y_true_ranking, y_pred_ranking)
    ordinal = calculate_ordinal_metrics(y_true_harm, y_pred_harm)
    return {
        "instance_jaccard": mlc["instance_jaccard"],
        "exact_match_ratio": mlc["exact_match_ratio"],
        "micro_f1": mlc["micro_f1"],
        "ndcg@5": ranking["ndcg@5"],
        "weighted_kappa": ordinal["weighted_kappa"],
    }


def _run_deterministic_offline(
    cases: list[dict],
    subtype_map: dict,
) -> Dict[str, float]:
    """Score each case with the pure keyword/pattern labeler -- no LLM.

    Unlike every ``HybridMode`` arm (which routes through ``llm/provider.py``
    and therefore collapses to near-zero offline), this arm calls
    ``labeling/harm_labeler.py::label_harm_category`` directly, so it produces
    real, byte-identical predictions on every run with no network and no keys.

    It is deliberately NOT a reproduction of the recorded 2025 numbers -- it is
    a different (deterministic) classifier, reported only to give the offline
    story a genuinely reproducible arm. Determinism comes from: a fixed
    threshold (``OFFLINE_DET_THRESHOLD``), stable ``(-score, category)`` sort,
    and the labeler being a pure function of the input text.
    """
    from .evals.metrics import (
        calculate_mlc_metrics,
        calculate_ordinal_metrics,
        calculate_ranking_metrics,
    )
    from .labeling.harm_labeler import label_harm_category

    y_true_causes, y_pred_causes = [], []
    y_true_ranking, y_pred_ranking = [], []
    y_true_harm, y_pred_harm = [], []

    for case in cases:
        text = case.get("description", "") or case.get("text", "")

        true_causes = set(case.get("harms", [])) or set(case.get("root_causes", []))
        if not true_causes and "harm_category" in case:
            true_causes = {case["harm_category"]}
        true_ranking = case.get("ranking", list(case.get("harms", [])))
        if not true_ranking and "harm_category" in case:
            true_ranking = [case["harm_category"]]
        true_score = int(case.get("harm_score", 0))

        true_causes = {subtype_map.get(c, c) for c in true_causes}
        true_ranking = _dedupe_preserve_order([subtype_map.get(c, c) for c in true_ranking])

        scores = label_harm_category(
            {"description": text}, return_scores=True, text_fields=["description"]
        )
        # Aggregate subtype scores up to Solove parents (max), threshold, rank.
        parent_scores: Dict[str, float] = {}
        for category, score in scores.items():
            if score < OFFLINE_DET_THRESHOLD:
                continue
            parent = subtype_map.get(category, category)
            parent_scores[parent] = max(parent_scores.get(parent, 0.0), score)
        ranked = [p for p, _ in sorted(parent_scores.items(), key=lambda kv: (-kv[1], kv[0]))]
        pred_causes = set(ranked)
        pred_ranking = _dedupe_preserve_order(ranked)
        top = max(parent_scores.values()) if parent_scores else 0.0
        harm_score = 1
        for thr, val in ((6, 5), (4, 4), (2, 3), (1, 2)):
            if top >= thr:
                harm_score = val
                break

        y_true_causes.append(true_causes)
        y_pred_causes.append(pred_causes)
        y_true_ranking.append(true_ranking)
        y_pred_ranking.append(pred_ranking)
        y_true_harm.append(true_score)
        y_pred_harm.append(harm_score)

    mlc = calculate_mlc_metrics(y_true_causes, y_pred_causes)
    ranking = calculate_ranking_metrics(y_true_ranking, y_pred_ranking)
    ordinal = calculate_ordinal_metrics(y_true_harm, y_pred_harm)
    return {
        "instance_jaccard": mlc["instance_jaccard"],
        "exact_match_ratio": mlc["exact_match_ratio"],
        "micro_f1": mlc["micro_f1"],
        "ndcg@5": ranking["ndcg@5"],
        "weighted_kappa": ordinal["weighted_kappa"],
    }


def _run_framework_evaluators(cases: list[dict], subtype_map: dict) -> Dict[str, Dict[str, Any]]:
    """Run the NIST/Solove/ISO29100 keyword evaluators for context.

    These are informational only -- final_results_summary.md's headline
    table is scoped to the hybrid-mode arms, not these framework evaluators,
    so there is nothing recorded to `--check` them against. We report a
    simple "true-set hit rate" (does the single predicted_harm fall in the
    normalized ground-truth harm set) since these evaluators are single-label,
    unlike the multi-label hybrid arms.
    """
    from .evals.framework_comparison import (
        ISO29100Evaluator,
        NISTFrameworkEvaluator,
        SoloveFrameworkEvaluator,
    )

    # No common base class between the evaluators (mirrors framework_comparison.py's
    # own FrameworkComparator.frameworks annotation) -- without this, mypy infers
    # `list[object]` and every `.predict()`/`.name` call site below becomes an error.
    evaluators: List[Any] = [
        NISTFrameworkEvaluator(),
        SoloveFrameworkEvaluator(),
        ISO29100Evaluator(),
    ]
    results: Dict[str, Dict[str, Any]] = {}
    for ev in evaluators:
        hits = 0
        total = 0
        for case in cases:
            true_causes = set(case.get("harms", [])) or set(case.get("root_causes", []))
            if not true_causes and "harm_category" in case:
                true_causes = {case["harm_category"]}
            true_causes = {subtype_map.get(c, c) for c in true_causes}
            if not true_causes:
                continue
            pred = ev.predict(case)
            total += 1
            if subtype_map.get(pred.predicted_harm, pred.predicted_harm) in true_causes:
                hits += 1
        results[ev.name] = {
            "hit_rate": round(hits / total, 4) if total else 0.0,
            "n": total,
        }
    return results


def _render_table(rows: List[tuple[str, Dict[str, float]]]) -> str:
    header = "| Mode | " + " | ".join(_METRIC_HEADERS) + " |"
    sep = "| :--- | " + " | ".join([":---"] * len(_METRIC_HEADERS)) + " |"
    lines = [header, sep]
    for label, metrics in rows:
        cells = [f"{metrics[col]:.4f}" for col in _METRIC_COLUMNS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


_SUMMARY_ROW_RE = re.compile(
    r"^\|\s*\*{0,2}(?P<label>[^|*]+?)\*{0,2}\s*\|\s*\*{0,2}(?P<jaccard>[-\d.]+)\*{0,2}\s*\|"
    r"\s*\*{0,2}(?P<emr>[-\d.]+)\*{0,2}\s*\|\s*\*{0,2}(?P<f1>[-\d.]+)\*{0,2}\s*\|"
    r"\s*\*{0,2}(?P<ndcg>[-\d.]+)\*{0,2}\s*\|\s*$"
)


def parse_recorded_summary(summary_md: Path) -> Dict[str, Dict[str, float]]:
    """Parse the "## 3. Key Metrics" table from final_results_summary.md.

    Returns a dict keyed by the summary's row label (e.g. "Rules Static")
    with instance_jaccard/exact_match_ratio/micro_f1/ndcg@5 values.
    """
    text = summary_md.read_text(encoding="utf-8")
    recorded: Dict[str, Dict[str, float]] = {}
    for line in text.splitlines():
        m = _SUMMARY_ROW_RE.match(line.strip())
        if not m:
            continue
        label = m.group("label").strip()
        if label.lower() in {"mode", ":---"}:
            continue
        try:
            recorded[label] = {
                "instance_jaccard": float(m.group("jaccard")),
                "exact_match_ratio": float(m.group("emr")),
                "micro_f1": float(m.group("f1")),
                "ndcg@5": float(m.group("ndcg")),
            }
        except ValueError:
            continue
    return recorded


def load_rerun_reference(rerun_dir: Path) -> Optional[Dict[str, float]]:
    """Load the rules_static metrics from the 2026-07-21 live rerun.

    Reads the newest ``results_rules_static_*.json`` in ``rerun_dir`` (written
    by ``evals/hybrid_runner.py``) and returns the four headline columns. This
    is the *live* regression reference; see ``rerun_20260721/README.md`` for the
    provider-drift story that motivated pinning the model.
    """
    candidates = sorted(rerun_dir.glob("results_rules_static_*.json"))
    if not candidates:
        return None
    data = json.loads(candidates[-1].read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    mlc = metrics.get("mlc", {})
    ranking = metrics.get("ranking", {})
    try:
        return {
            "instance_jaccard": float(mlc["instance_jaccard"]),
            "exact_match_ratio": float(mlc["exact_match_ratio"]),
            "micro_f1": float(mlc["micro_f1"]),
            "ndcg@5": float(ranking["ndcg@5"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


@app.command("reproduce")
def reproduce(
    golden_file: Path = typer.Option(DEFAULT_GOLDEN_CASES, "--golden-file"),
    ruleset: str = typer.Option("expert_framework", "--ruleset"),
    ruleset_version: str = typer.Option("1.0", "--ruleset-version"),
    rules_dir: Path = typer.Option(DEFAULT_RULES_DIR, "--rules-dir"),
    out_dir: Path = typer.Option(DEFAULT_COMPARISON_OUT, "--out-dir"),
    modes: str = typer.Option(
        "baseline,rules_static,rules_dynamic,rag,hybrid_deterministic_first,"
        "hybrid_llm_first,offline_deterministic",
        "--modes",
        help=(
            "Comma-separated arms to run: HybridMode values plus the pure-keyword "
            "'offline_deterministic' arm."
        ),
    ),
    check: bool = typer.Option(
        False,
        "--check/--no-check",
        help="Compare the rules_static row against the --reference table.",
    ),
    reference: str = typer.Option(
        "recorded-2025",
        "--reference",
        help=(
            "Which reference --check diffs rules_static against: 'recorded-2025' "
            "(the historical final_results_summary.md; mismatches offline by design) "
            "or 'rerun-2026' (the live 2026-07-21 rerun; regression guard for live runs)."
        ),
    ),
    summary_md: Path = typer.Option(DEFAULT_SUMMARY_MD, "--summary-md"),
    rerun_dir: Path = typer.Option(DEFAULT_RERUN_DIR, "--rerun-dir"),
    tolerance: Optional[float] = typer.Option(
        None,
        "--tolerance",
        help=(
            "Abs-diff tolerance for --check. Default depends on --reference: "
            "1e-3 for recorded-2025, 0.05 for rerun-2026 (LLM stochasticity)."
        ),
    ),
) -> None:
    """Reproduce the practicum's headline hybrid-mode comparison, offline.

    Runs the NIST/Solove/ISO29100 framework evaluators (informational), the
    requested hybrid-mode arms, and the pure-keyword ``offline_deterministic``
    arm against ``golden_file``, writes the table to ``out_dir``, prints it,
    and (with ``--check``) diffs the emitted ``rules_static`` row against the
    ``--reference`` table.
    """
    if reference not in {"recorded-2025", "rerun-2026"}:
        typer.echo(f"[reproduce] Unknown --reference {reference!r}.")
        raise typer.Exit(code=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    cases, subtype_map = _load_golden_cases_and_taxonomy(golden_file)
    typer.echo(f"[reproduce] Loaded {len(cases)} cases from {golden_file}")

    rules: list = []
    rules_path = rules_dir / f"{ruleset}.v{ruleset_version}.json"
    if rules_path.exists():
        rules = json.loads(rules_path.read_text()).get("rules", [])
        typer.echo(f"[reproduce] Loaded {len(rules)} rules from {rules_path}")
    else:
        typer.echo(f"[reproduce] WARNING: ruleset {rules_path} not found; running without rules.")

    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    rows: List[tuple[str, Dict[str, float]]] = []
    emitted: Dict[str, Dict[str, float]] = {}
    for mode_value in mode_list:
        if mode_value == OFFLINE_DETERMINISTIC:
            typer.echo(
                f"[reproduce] Running {OFFLINE_DETERMINISTIC} "
                f"({len(cases)} cases, pure keyword rules, no LLM)..."
            )
            metrics = _run_deterministic_offline(cases, subtype_map)
        else:
            typer.echo(
                f"[reproduce] Running mode={mode_value} ({len(cases)} cases, offline fallback)..."
            )
            metrics = _run_hybrid_mode(mode_value, cases, subtype_map, rules)
        label = _MODE_TO_SUMMARY_LABEL.get(mode_value, mode_value)
        rows.append((label, metrics))
        emitted[mode_value] = metrics

    table_md = _render_table(rows)
    typer.echo("\n" + table_md + "\n")

    if OFFLINE_DETERMINISTIC in emitted:
        typer.echo(
            "[reproduce] NOTE: 'Offline Deterministic (keyword rules)' is deterministic, "
            "offline-exact\n"
            "  (pure keyword/pattern scoring in labeling/harm_labeler.py; byte-identical every "
            "run).\n"
            "  It is NOT part of the recorded 2025 comparison and does not reproduce those "
            "numbers --\n"
            "  it is a different, genuinely-reproducible classifier reported alongside the LLM "
            "arms."
        )

    framework_results = _run_framework_evaluators(cases, subtype_map)
    typer.echo("[reproduce] Framework evaluators (informational; not part of the headline table):")
    for name, stats in framework_results.items():
        typer.echo(f"  {name}: hit_rate={stats['hit_rate']} (n={stats['n']})")

    # provider="fallback" here (see _run_hybrid_mode); record the resolved model
    # so the persisted comparison is never ambiguous about what produced it.
    from .llm.provider import resolve_model_name

    payload = {
        "golden_file": str(golden_file),
        "ruleset": ruleset,
        "ruleset_version": ruleset_version,
        "resolved_model": resolve_model_name("fallback"),
        "modes": emitted,
        "framework_evaluators": framework_results,
    }
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "comparison_table.md").write_text(table_md + "\n", encoding="utf-8")
    typer.echo(
        f"[reproduce] Wrote {out_dir / 'comparison.json'} and {out_dir / 'comparison_table.md'}"
    )

    if not check:
        return

    if "rules_static" not in emitted:
        typer.echo("[reproduce --check] rules_static was not in --modes; nothing to check.")
        raise typer.Exit(code=1)

    if reference == "recorded-2025":
        recorded = parse_recorded_summary(summary_md)
        reference_row = recorded.get("Rules Static")
        reference_label = str(summary_md)
        resolved_tolerance = 1e-3 if tolerance is None else tolerance
    else:  # rerun-2026
        reference_row = load_rerun_reference(rerun_dir)
        reference_label = str(rerun_dir)
        resolved_tolerance = 0.05 if tolerance is None else tolerance

    if reference_row is None:
        typer.echo(
            f"[reproduce --check] Could not load a 'Rules Static' reference "
            f"({reference}) from {reference_label}."
        )
        raise typer.Exit(code=1)

    emitted_row = emitted["rules_static"]
    mismatches = []
    for col in _METRIC_COLUMNS:
        got = emitted_row[col]
        want = reference_row[col]
        if abs(got - want) > resolved_tolerance:
            mismatches.append((col, got, want, abs(got - want)))

    if not mismatches:
        typer.echo(
            f"[reproduce --check] MATCH: rules_static metrics match the {reference} reference "
            f"within tolerance ({resolved_tolerance})."
        )
        return

    typer.echo(
        f"[reproduce --check] MISMATCH: rules_static metrics do not match the {reference} "
        f"reference (tolerance {resolved_tolerance})."
    )
    typer.echo(f"{'metric':<20}{'emitted':>12}{'reference':>12}{'abs diff':>12}")
    for col, got, want, diff in mismatches:
        typer.echo(f"{col:<20}{got:>12.4f}{want:>12.4f}{diff:>12.4f}")
    _print_check_root_cause(reference)
    raise typer.Exit(code=1)


def _print_check_root_cause(reference: str) -> None:
    """Explain why an offline --check mismatches, tuned to the reference used."""
    typer.echo(
        "\n[reproduce --check] ROOT CAUSE: HybridMode.RULES_STATIC still routes through the LLM\n"
        "(models/hybrid.py::HybridModel.predict -> llm/provider.py::complete); the static rules\n"
        "only change the prompt, not the extraction mechanism. Offline (no API keys), `complete()`\n"
        "falls back to a truncated echo of the prompt (not JSON), so `_parse_response` cannot\n"
        "recover real root_causes/ranking/harm_score and the offline metrics come out near-zero."
    )
    if reference == "recorded-2025":
        typer.echo(
            "The recorded 2025 numbers (final_results_summary.md) were produced by a live Gemini\n"
            "call whose model was never pinned -- recorded only as 'provider: gemini'. That model\n"
            "has since drifted: the 2026-07-21 rerun (data/experiments/rerun_20260721/README.md)\n"
            "shows EMR halving while Jaccard/NDCG roughly double under the current default\n"
            "gemini-2.5-flash. So even a live run today will not reproduce the 2025 table exactly.\n"
            "This is a structural property of the extracted code plus provider drift, not a bug in\n"
            "offline wiring -- see the P2 report. For a live regression guard use\n"
            "`--reference rerun-2026`; for a genuinely reproducible offline arm see\n"
            "'offline_deterministic' in the table above."
        )
    else:
        typer.echo(
            "The rerun-2026 reference (data/experiments/rerun_20260721/) was produced by a live\n"
            "Gemini call against the now-pinned gemini-2.5-flash. It is the regression guard for\n"
            "*live* runs (real GEMINI_API_KEY); offline it cannot match for the same reason above.\n"
            "See rerun_20260721/README.md for the provider-drift story that motivated pinning the\n"
            "model. For a genuinely reproducible offline arm see 'offline_deterministic'."
        )


def main() -> None:  # pragma: no cover - thin CLI wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
