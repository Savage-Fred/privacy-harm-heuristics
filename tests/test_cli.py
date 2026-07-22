"""Smoke tests for the P2 runnable story: `privacy-harm-heuristics` CLI + Makefile.

Keep these fast: `reproduce` runs on a 5-case slice (not the full 50-case
golden_cases_v3.jsonl) and `train-all` trains only decision_tree (the other
four models are exercised by their own dedicated trainer tests).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from privacy_harm_heuristics.cli import app, parse_recorded_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _write_golden_cases(path: Path, n: int = 5) -> None:
    # At least two distinct non-empty labels across cases: with only one label
    # value present, sklearn's MultiLabelBinarizer produces a single-column
    # matrix that `type_of_target` reads as "binary" rather than
    # "multilabel-indicator", and jaccard_score(average="samples") raises.
    # The real 50-case golden_cases_v3.jsonl has many distinct root_causes so
    # this never bites in practice; a 5-case synthetic slice must mimic that.
    labels = ["surveillance", "disclosure"]
    cases = []
    for i in range(n):
        label = labels[i % len(labels)] if i % 3 != 0 else None
        cases.append(
            {
                "id": f"g{i}",
                "source": "synthetic",
                "description": f"Synthetic case {i} about tracking and location data.",
                "root_causes": [label] if label else [],
                "ranking": [label] if label else [],
                "harm_score": 3 if label else 1,
            }
        )
    with path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")


def _write_feature_jsonl(path: Path, n: int = 40) -> None:
    # Three label classes, not two: models/metrics_utils.py::choose_average
    # only takes the "macro" (safe) path when the train/test label union size
    # != 2, or falls into a `pos_label="positive"` binary path that assumes
    # numeric {0,1} or a literal "positive" string label -- which crashes on
    # arbitrary string binary labels (e.g. "surveillance"/"none"). The real
    # committed data/with_features.sample.jsonl has many harm_category
    # classes so this pre-existing trainer quirk never bites in practice;
    # mirror that here rather than exercising the binary-string-label bug.
    categories = ["surveillance", "disclosure", "none"]
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            rec = {
                "id": f"r{i}",
                "source": "synthetic",
                "type": "record",
                "created_date": "2024-01-01T00:00:00Z",
                "raw": {},
                "kw_privacy": 1 if i % 2 == 0 else 0,
                "kw_security": 1 if i % 3 == 0 else 0,
                "penalty_amount": float(i % 5),
                "incident_date": "2024-01-01",
                "harm_category": categories[i % len(categories)],
            }
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# reproduce
# ---------------------------------------------------------------------------


def test_reproduce_offline_smoke(tmp_path: Path):
    """`reproduce` runs fully offline on a small slice and emits a table."""
    golden = tmp_path / "golden.jsonl"
    _write_golden_cases(golden)
    out_dir = tmp_path / "comparison"

    result = runner.invoke(
        app,
        [
            "reproduce",
            "--golden-file",
            str(golden),
            "--rules-dir",
            str(tmp_path),  # empty; no ruleset file present -> runs without rules
            "--out-dir",
            str(out_dir),
            "--modes",
            "baseline,rules_static",
            "--no-check",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Baseline" in result.output
    assert "Rules Static" in result.output
    assert (out_dir / "comparison.json").exists()
    assert (out_dir / "comparison_table.md").exists()

    payload = json.loads((out_dir / "comparison.json").read_text())
    assert set(payload["modes"].keys()) == {"baseline", "rules_static"}
    for metrics in payload["modes"].values():
        for col in ("instance_jaccard", "exact_match_ratio", "micro_f1", "ndcg@5"):
            assert col in metrics


def test_reproduce_check_mismatch_reports_honestly(tmp_path: Path):
    """--check against the tiny synthetic slice must not silently pass or crash.

    The offline LLM fallback can't recover real predictions (see cli.py's
    reproduce docstring), so a `--check` run is expected to report MISMATCH
    with an exit code of 1, and must not exit 0 by accident.
    """
    golden = tmp_path / "golden.jsonl"
    _write_golden_cases(golden)
    out_dir = tmp_path / "comparison"

    result = runner.invoke(
        app,
        [
            "reproduce",
            "--golden-file",
            str(golden),
            "--rules-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--modes",
            "rules_static",
            "--check",
        ],
    )

    assert result.exit_code == 1
    assert "MISMATCH" in result.output
    assert "ROOT CAUSE" in result.output


def _run_reproduce_modes(tmp_path: Path, modes: str, sub: str):
    golden = tmp_path / "golden.jsonl"
    _write_golden_cases(golden)
    out_dir = tmp_path / sub
    result = runner.invoke(
        app,
        [
            "reproduce",
            "--golden-file",
            str(golden),
            "--rules-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--modes",
            modes,
            "--no-check",
        ],
    )
    assert result.exit_code == 0, result.output
    return result, json.loads((out_dir / "comparison.json").read_text())


def test_offline_deterministic_arm_appears_and_is_marked(tmp_path: Path):
    """The offline_deterministic arm shows in the table and is clearly labeled
    as deterministic/offline-exact and NOT part of the recorded 2025 numbers."""
    result, payload = _run_reproduce_modes(tmp_path, "baseline,offline_deterministic", "cmp")
    assert "Offline Deterministic (keyword rules)" in result.output
    assert "deterministic, offline-exact" in result.output
    assert "NOT part of the recorded 2025 comparison" in result.output
    assert "offline_deterministic" in payload["modes"]
    for col in ("instance_jaccard", "exact_match_ratio", "micro_f1", "ndcg@5"):
        assert col in payload["modes"]["offline_deterministic"]


def test_offline_deterministic_arm_is_reproducible(tmp_path: Path):
    """Run-twice equality: the pure-keyword arm must emit identical metrics.

    Unlike the LLM-backed arms (which route through llm/provider.py), this arm
    is a pure function of the input text, so two independent runs must produce
    byte-identical metrics.
    """
    _, payload_a = _run_reproduce_modes(tmp_path, "offline_deterministic", "run_a")
    _, payload_b = _run_reproduce_modes(tmp_path, "offline_deterministic", "run_b")
    assert (
        payload_a["modes"]["offline_deterministic"] == payload_b["modes"]["offline_deterministic"]
    )


def test_reproduce_check_rerun_reference_mismatch(tmp_path: Path):
    """--reference rerun-2026 uses the 0.05 tolerance and still mismatches
    offline (near-zero fallback), citing the provider-drift story."""
    golden = tmp_path / "golden.jsonl"
    _write_golden_cases(golden)
    result = runner.invoke(
        app,
        [
            "reproduce",
            "--golden-file",
            str(golden),
            "--rules-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "cmp"),
            "--modes",
            "rules_static",
            "--check",
            "--reference",
            "rerun-2026",
        ],
    )
    assert result.exit_code == 1
    assert "rerun-2026" in result.output
    assert "0.05" in result.output
    assert "provider-drift" in result.output.replace("\n", " ")


def test_reproduce_rejects_unknown_reference(tmp_path: Path):
    golden = tmp_path / "golden.jsonl"
    _write_golden_cases(golden)
    result = runner.invoke(
        app,
        [
            "reproduce",
            "--golden-file",
            str(golden),
            "--rules-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "cmp"),
            "--check",
            "--reference",
            "made-up",
        ],
    )
    assert result.exit_code == 2


def test_parse_recorded_summary_against_real_file():
    """Unit test the --check parser against the actual recorded results file."""
    summary_md = REPO_ROOT / "data" / "experiments" / "final_results_summary.md"
    recorded = parse_recorded_summary(summary_md)

    assert recorded["Rules Static"] == {
        "instance_jaccard": 0.0678,
        "exact_match_ratio": 0.6667,
        "micro_f1": 0.3908,
        "ndcg@5": 0.0896,
    }
    assert recorded["Baseline"]["instance_jaccard"] == 0.0556
    assert recorded["RAG"]["micro_f1"] == 0.2424
    # All six modes from final_results_summary.md's methodology section should parse.
    assert len(recorded) == 6


# ---------------------------------------------------------------------------
# train-all
# ---------------------------------------------------------------------------


def test_train_all_smoke_decision_tree_only(tmp_path: Path):
    data = tmp_path / "features.jsonl"
    _write_feature_jsonl(data)
    out_dir = tmp_path / "heuristics"

    result = runner.invoke(
        app,
        [
            "train-all",
            "--data",
            str(data),
            "--target",
            "harm_category",
            "--golden",
            str(tmp_path / "does-not-exist.jsonl"),  # forces the --test-size split path
            "--out-dir",
            str(out_dir),
            "--models",
            "decision_tree",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "trained=1 skipped=0" in result.output
    dt_dir = out_dir / "decision_tree"
    assert (dt_dir / "model.joblib").exists()
    assert (dt_dir / "metrics.json").exists()
    assert (dt_dir / "heuristics.jsonl").exists()
    assert (dt_dir / "HEURISTICS.md").exists()


def test_train_all_skips_models_with_missing_optional_deps(tmp_path: Path):
    """brl/bayes_net need imodels/pomegranate; this venv only has interpret (EBM)."""
    data = tmp_path / "features.jsonl"
    _write_feature_jsonl(data)
    out_dir = tmp_path / "heuristics"

    result = runner.invoke(
        app,
        [
            "train-all",
            "--data",
            str(data),
            "--target",
            "harm_category",
            "--golden",
            str(tmp_path / "does-not-exist.jsonl"),
            "--out-dir",
            str(out_dir),
            "--models",
            "decision_tree,brl,bayes_net",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "trained=1 skipped=2" in result.output
    assert "brl: SKIP" in result.output
    assert "bayes_net: SKIP" in result.output


# ---------------------------------------------------------------------------
# Makefile
# ---------------------------------------------------------------------------


def test_makefile_targets_exist():
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("setup", "train", "reproduce", "reproduce-full", "fetch-data", "test", "lint"):
        assert f"\n{target}:" in makefile or makefile.startswith(
            f"{target}:"
        ), f"Makefile missing target: {target}"
