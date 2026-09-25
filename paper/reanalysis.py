"""Offline re-analysis behind ``paper/lessons.tex``. No network, no API calls.

Recomputes every number the paper states from committed artifacts and writes
them to ``paper/reanalysis.json``. Full-corpus fields need the v1.0.0 release
asset (``make fetch-data`` -> ``data/with_features.jsonl``); without it they
are ``null``. ``paper/trials_2025/`` holds the per-case outputs behind the
recorded 2025 table (harvested read-only from the predecessor repository).

Self-checks (fail loudly): the scoring path must reproduce the recorded 2026
rerun and all 18 recorded 2025 trial files exactly, the 2025 trial means must
reproduce the recorded table, the per-case offline arm must reproduce
``cli._run_deterministic_offline``, and the fast bootstrap metrics must equal
the repository scorer on the full sample. Output must not depend on
``PYTHONHASHSEED`` (run twice with different seeds and diff the JSON).

Run from the repo root:  PYTHONPATH=src python paper/reanalysis.py [corpus.jsonl]
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path

from privacy_harm_heuristics.cli import OFFLINE_DET_THRESHOLD, _run_deterministic_offline
from privacy_harm_heuristics.constants.privacy_taxonomy import TAXONOMY_SOLOVE
from privacy_harm_heuristics.evals.metrics import (
    calculate_mlc_metrics,
    calculate_ordinal_metrics,
    calculate_ranking_metrics,
)
from privacy_harm_heuristics.labeling.harm_labeler import label_harm_category
from privacy_harm_heuristics.models.hybrid import HybridMode, HybridModel

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRIALS = ROOT / "paper/trials_2025"
RERUN = DATA / "experiments/rerun_20260721/results_rules_static_20260721_212152.json"
SUMMARY = DATA / "experiments/final_results_summary.md"
GROUPS = list(TAXONOMY_SOLOVE)  # taxonomy order: collection, processing, dissemination, invasion
B, SEED, ALPHA = 4000, 0, 0.05

# Copied from evals/hybrid_runner.py `custom_map`; the rerun self-check fails
# if this copy stops matching the scorer.
CUSTOM_MAP = {
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


def solove_subtypes() -> set[str]:
    return {
        s["name"]
        for spec in TAXONOMY_SOLOVE.values()
        for s in spec.get("subtypes", [])
        if isinstance(s, dict) and "name" in s
    }


def label_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for parent, spec in TAXONOMY_SOLOVE.items():
        m[parent] = parent
        for sub in spec.get("subtypes", []):
            if isinstance(sub, dict) and "name" in sub:
                m[sub["name"]] = parent
    m.update(CUSTOM_MAP)
    return m


def to_parents(labels: list[str], m: dict[str, str]) -> list[str]:
    out: list[str] = []
    for label in labels:
        parent = m.get(label, label)
        if parent not in out:
            out.append(parent)
    return out


def case_title(case: dict) -> str:
    return case.get("title") or (case.get("description") or "")[:60]


def fast(T, P, idx) -> dict[str, float]:
    """Set metrics with the repo scorer's conventions, in plain Python (for the bootstrap)."""
    jac = j1 = emr = inter_sum = size_sum = 0.0
    for i in idx:
        t, p = T[i], P[i]
        inter, union = len(t & p), len(t | p)
        jac += inter / union if union else 0.0
        j1 += inter / union if union else 1.0
        emr += t == p
        inter_sum += inter
        size_sum += len(t) + len(p)
    n = len(idx)
    return {
        "instance_jaccard": jac / n,
        "jaccard_empty_as_1": j1 / n,
        "exact_match_ratio": emr / n,
        "micro_f1": 2 * inter_sum / size_sum if size_sum else 0.0,
    }


def ndcg(TR, PR, idx) -> float:
    return calculate_ranking_metrics([TR[i] for i in idx], [PR[i] for i in idx])["ndcg@5"]


def score(T, TR, P, PR) -> dict[str, float]:
    """Repository scorer (sklearn) plus the empty-as-1 Jaccard convention."""
    mlc = calculate_mlc_metrics(T, P)
    idx = range(len(T))
    return {
        "instance_jaccard": mlc["instance_jaccard"],
        "jaccard_empty_as_1": fast(T, P, idx)["jaccard_empty_as_1"],
        "exact_match_ratio": mlc["exact_match_ratio"],
        "micro_f1": mlc["micro_f1"],
        "ndcg@5": ndcg(TR, PR, idx),
    }


def check_fast(T, P) -> None:
    ref, got = calculate_mlc_metrics(T, P), fast(T, P, range(len(T)))
    for key in ("instance_jaccard", "exact_match_ratio", "micro_f1"):
        assert abs(ref[key] - got[key]) < 1e-12, f"fast metric drift on {key}"


def constant_arms() -> list[tuple[str, set[str], list[str]]]:
    """Every constant (label set, ranking order) over the four groups: 65 arms."""
    arms = []
    for k in range(len(GROUPS) + 1):
        for order in permutations(GROUPS, k):
            name = "+".join(g.split("_")[-1] for g in order) or "empty"
            arms.append((name, set(order), list(order)))
    return arms


def offline_arm(cases: list[dict], m: dict[str, str]):
    """Per-case predictions of the repo's pure-keyword arm (cli._run_deterministic_offline)."""
    P, PR, S = [], [], []
    for case in cases:
        text = case.get("description", "") or case.get("text", "")
        scores = label_harm_category(
            {"description": text}, return_scores=True, text_fields=["description"]
        )
        parents: dict[str, float] = {}
        for category, s in scores.items():
            if s >= OFFLINE_DET_THRESHOLD:
                parent = m.get(category, category)
                parents[parent] = max(parents.get(parent, 0.0), s)
        ranked = [p for p, _ in sorted(parents.items(), key=lambda kv: (-kv[1], kv[0]))]
        top = max(parents.values()) if parents else 0.0
        sev = next((v for thr, v in ((6, 5), (4, 4), (2, 3), (1, 2)) if top >= thr), 1)
        P.append(set(ranked))
        PR.append(ranked)
        S.append(sev)
    return P, PR, S


def recorded_2025() -> dict:
    """The recorded Nov-2025 table, parsed from final_results_summary.md."""
    rows = {}
    for line in SUMMARY.read_text().splitlines():
        cells = [c.strip(" *") for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and re.fullmatch(r"0\.\d{4}", cells[1]):
            j, emr, f1, nd = (float(c) for c in cells[1:])
            rows[cells[0]] = {
                "instance_jaccard": j,
                "exact_match_ratio": emr,
                "micro_f1": f1,
                "ndcg@5": nd,
                "emr_x250": round(emr * 250, 2),  # the summary claims 5 trials x 50 cases
                "emr_x150": round(emr * 150, 2),
            }
    return rows


def load_trials(T_gold, ids) -> dict:
    """The 18 recorded 2025 trial files named in trials_summary.json, self-checked."""
    summary = json.loads((TRIALS / "trials_summary.json").read_text())
    trials: dict[str, list[dict]] = {}
    for mode, v in summary.items():
        assert v["trials"] == len(v["runs"]) == 3, mode
        for run in v["runs"]:
            d = json.loads((TRIALS / Path(run["file"]).name).read_text())
            det = d["details"]
            assert [str(r["case_id"]) for r in det] == ids, "trial case order"
            assert [set(r["true"]["causes"]) for r in det] == T_gold, "trial gold"
            P = [set(r["pred"]["root_causes"]) for r in det]
            PR = [list(r["pred"]["ranking"]) for r in det]
            S = [r["pred"]["harm_score"] for r in det]
            # 2025 scorer: gold mapped to groups, predictions scored raw.
            got = score(T_gold, [r["true"]["ranking"] for r in det], P, PR)
            rec = d["metrics"]
            for key, want in [
                ("instance_jaccard", rec["mlc"]["instance_jaccard"]),
                ("exact_match_ratio", rec["mlc"]["exact_match_ratio"]),
                ("micro_f1", rec["mlc"]["micro_f1"]),
                ("ndcg@5", rec["ranking"]["ndcg@5"]),
            ]:
                assert abs(got[key] - want) < 1e-12, f"2025 scorer drift {run['file']} {key}"
            trials.setdefault(mode, []).append(
                {
                    "file": Path(run["file"]).name,
                    "config_model_name": d["config"].get("model_name"),
                    "P": P,
                    "PR": PR,
                    "S": S,
                    "metrics": got,
                    "parse_failures": sum(
                        1
                        for r in det
                        if (r["pred"].get("rationale") or "").startswith("Failed to parse")
                    ),
                }
            )
    return trials


def headline() -> dict:
    run = json.loads(RERUN.read_text())
    det, m = run["details"], label_map()
    n = len(det)
    T = [set(r["true"]["causes"]) for r in det]
    TR = [r["true"]["ranking"] for r in det]
    ys = [r["true"]["score"] for r in det]
    raw_P = [r["pred"]["root_causes"] for r in det]
    raw_PR = [r["pred"]["ranking"] for r in det]
    ps = [r["pred"]["harm_score"] for r in det]
    ids = [str(r["case_id"]) for r in det]

    cases = [json.loads(line) for line in open(DATA / "golden_cases_v3.jsonl")]
    assert [str(c["id"]) for c in cases] == ids, "case order"
    off_P, off_PR, off_S = offline_arm(cases, m)

    arms = {
        "rerun_current_scorer": (
            [set(to_parents(p, m)) for p in raw_P],
            [to_parents(p, m) for p in raw_PR],
        ),
        # The 2025 scorer: gold mapped to groups, predictions scored raw.
        "rerun_2025_scorer": ([set(p) for p in raw_P], [list(p) for p in raw_PR]),
        # Post-hoc sensitivity: drop predicted labels outside the four groups.
        "rerun_filtered_to_groups": (
            [set(to_parents(p, m)) & set(GROUPS) for p in raw_P],
            [[x for x in to_parents(p, m) if x in GROUPS] for p in raw_PR],
        ),
        "offline_keyword_arm": (off_P, off_PR),
    }

    # Self-checks against recorded outputs.
    rec = run["metrics"]
    got = score(T, TR, *arms["rerun_current_scorer"])
    for key, want in [
        ("instance_jaccard", rec["mlc"]["instance_jaccard"]),
        ("exact_match_ratio", rec["mlc"]["exact_match_ratio"]),
        ("micro_f1", rec["mlc"]["micro_f1"]),
        ("ndcg@5", rec["ranking"]["ndcg@5"]),
    ]:
        assert abs(got[key] - want) < 1e-12, f"scorer drift on {key}"
    got_ord = calculate_ordinal_metrics(ys, ps)
    for key in ("weighted_kappa", "accuracy", "spearman_rho", "kendall_tau"):
        assert abs(got_ord[key] - rec["ordinal"][key]) < 1e-12, f"ordinal drift on {key}"
    ref = _run_deterministic_offline(cases, m)
    got_off = score(T, TR, off_P, off_PR)
    for key in ("instance_jaccard", "exact_match_ratio", "micro_f1", "ndcg@5"):
        assert abs(got_off[key] - ref[key]) < 1e-12, f"offline arm drift on {key}"

    trials = load_trials(T, ids)
    recorded = recorded_2025()
    label_of = {
        "baseline": "Baseline",
        "rules_static": "Rules Static",
        "rules_dynamic": "Rules Dynamic",
        "rag": "RAG",
        "hybrid_deterministic_first": "Hybrid (Det. First)",
        "hybrid_llm_first": "Hybrid (LLM First)",
    }
    arms_2025 = {}
    for mode, runs in trials.items():
        mean = {k: statistics.fmean(r["metrics"][k] for r in runs) for k in runs[0]["metrics"]}
        for key in ("instance_jaccard", "exact_match_ratio", "micro_f1", "ndcg@5"):
            assert round(mean[key], 4) == recorded[label_of[mode]][key], f"table {mode} {key}"
        sev = [calculate_ordinal_metrics(ys, r["S"]) for r in runs]
        arms_2025[mode] = {
            "mean_over_3_trials": mean,
            "severity_mean": {
                k: statistics.fmean(s[k] for s in sev) for k in ("accuracy", "weighted_kappa")
            },
            "trial_micro_f1": [r["metrics"]["micro_f1"] for r in runs],
            "parse_failures": sum(r["parse_failures"] for r in runs),
            "config_model_names": sorted({str(r["config_model_name"]) for r in runs}),
        }
        for r in runs:
            check_fast(T, r["P"])
    rs_labels = Counter(
        "group" if x in GROUPS else "subtype" if x in solove_subtypes() else "other"
        for r in trials["rules_static"]
        for p in r["P"]
        for x in p
    )
    rs_mapped = [
        score(
            T,
            TR,
            [set(to_parents(list(p), m)) for p in r["P"]],
            [to_parents(p, m) for p in r["PR"]],
        )
        for r in trials["rules_static"]
    ]
    rs_mapped_mean = {k: statistics.fmean(s[k] for s in rs_mapped) for k in rs_mapped[0]}

    consts = constant_arms()
    rows = {name: score(T, TR, P, PR) for name, (P, PR) in arms.items()}
    rows["const_empty"] = score(T, TR, [set()] * n, [[]] * n)
    rows["const_all_four_taxonomy_order"] = score(T, TR, [set(GROUPS)] * n, [GROUPS] * n)
    rows["oracle_ceiling_perfect_predictor"] = score(T, TR, T, TR)
    for name, (P, _) in arms.items():
        check_fast(T, P)

    all_scores = {name: score(T, TR, [s] * n, [o] * n) for name, s, o in consts}
    best = {}
    for metric in (
        "instance_jaccard",
        "jaccard_empty_as_1",
        "exact_match_ratio",
        "micro_f1",
        "ndcg@5",
    ):
        name = max(all_scores, key=lambda a: (all_scores[a][metric], a))
        best[metric] = {"arm": name, "value": all_scores[name][metric]}
    four_orders = [all_scores[a]["ndcg@5"] for a, s, _ in consts if len(s) == 4]

    ne = [i for i, t in enumerate(T) if t]

    def subset(P, PR, idx):
        return {**fast(T, P, idx), "ndcg@5": ndcg(TR, PR, idx)}

    nonempty = {name: subset(P, PR, ne) for name, (P, PR) in arms.items()}
    nonempty["const_all_four_taxonomy_order"] = subset([set(GROUPS)] * n, [GROUPS] * n, ne)
    nonempty["oracle_ceiling_perfect_predictor"] = subset(T, TR, ne)
    for mode, runs in trials.items():
        nonempty[f"2025_{mode}_mean"] = {
            k: statistics.fmean(subset(r["P"], r["PR"], ne)[k] for r in runs)
            for k in ("instance_jaccard", "micro_f1", "ndcg@5")
        }

    # Paired case-resampling bootstrap. Every candidate constant is a frozenset,
    # so nothing below depends on set iteration order (PYTHONHASHSEED).
    label_sets = sorted({frozenset(s) for _, s, _ in consts}, key=lambda s: sorted(s))
    all4, empty = [set(GROUPS)] * n, [set()] * n
    rs25 = trials["rules_static"]
    same_prompt = [r for mode, runs in trials.items() if mode != "rules_static" for r in runs]
    rng = random.Random(SEED)
    draws: dict[str, list[float]] = {}

    def add(key, value):
        draws.setdefault(key, []).append(value)

    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        best_f1 = max(fast(T, [s] * n, idx)["micro_f1"] for s in label_sets)
        a, e = fast(T, all4, idx), fast(T, empty, idx)
        a_nd = ndcg(TR, [GROUPS] * n, idx)
        r = fast(T, arms["rerun_current_scorer"][0], idx)
        r_nd = ndcg(TR, arms["rerun_current_scorer"][1], idx)
        f = fast(T, arms["rerun_filtered_to_groups"][0], idx)
        o = fast(T, off_P, idx)
        o_nd = ndcg(TR, off_PR, idx)
        add("2026 emr: empty - rerun", e["exact_match_ratio"] - r["exact_match_ratio"])
        add("2026 jaccard: all_four - rerun", a["instance_jaccard"] - r["instance_jaccard"])
        add("2026 ndcg@5: all_four - rerun", a_nd - r_nd)
        add("2026 micro_f1: rerun - best_constant", r["micro_f1"] - best_f1)
        add("2026 micro_f1: filtered - best_constant", f["micro_f1"] - best_f1)
        add("offline jaccard: all_four - offline", a["instance_jaccard"] - o["instance_jaccard"])
        add("offline micro_f1: all_four - offline", a["micro_f1"] - o["micro_f1"])
        add("offline ndcg@5: all_four - offline", a_nd - o_nd)
        rs = [fast(T, t["P"], idx) for t in rs25]
        rs_m = {k: statistics.fmean(x[k] for x in rs) for k in rs[0]}
        rs_nd = statistics.fmean(ndcg(TR, t["PR"], idx) for t in rs25)
        sp = [fast(T, t["P"], idx) for t in same_prompt]
        sp_m = {k: statistics.fmean(x[k] for x in sp) for k in ("micro_f1", "exact_match_ratio")}
        add(
            "2025 jaccard: all_four - rules_static",
            a["instance_jaccard"] - rs_m["instance_jaccard"],
        )
        add("2025 ndcg@5: all_four - rules_static", a_nd - rs_nd)
        add("2025 emr: empty - rules_static", e["exact_match_ratio"] - rs_m["exact_match_ratio"])
        add(
            "2025 jaccard_empty_as_1: empty - rules_static",
            e["jaccard_empty_as_1"] - rs_m["jaccard_empty_as_1"],
        )
        add("2025 micro_f1: rules_static - best_constant", rs_m["micro_f1"] - best_f1)
        add("2025 micro_f1: rules_static - same_prompt_arms", rs_m["micro_f1"] - sp_m["micro_f1"])
        add(
            "2025 emr: rules_static - same_prompt_arms",
            rs_m["exact_match_ratio"] - sp_m["exact_match_ratio"],
        )
        add(
            "single: 2026 rerun weighted_kappa",
            calculate_ordinal_metrics([ys[i] for i in idx], [ps[i] for i in idx])["weighted_kappa"],
        )

    def interval(values, alpha):
        v = sorted(values)
        return [round(v[int(alpha / 2 * B)], 4), round(v[int((1 - alpha / 2) * B) - 1], 4)]

    family = [k for k in draws if not k.startswith("single")]
    ci95 = {k: interval(v, ALPHA) for k, v in draws.items()}
    bonf = {k: interval(draws[k], ALPHA / len(family)) for k in family}

    # Trial-level permutation test: are the 3 rules-static trials exchangeable
    # with the 15 identical-prompt trials on micro-F1?
    trial_f1 = [(mode, x) for mode, v in arms_2025.items() for x in v["trial_micro_f1"]]
    obs = statistics.fmean(x for mode, x in trial_f1 if mode == "rules_static")
    combos = list(combinations(range(len(trial_f1)), 3))

    def gap(c):  # mean of the chosen 3 trials minus mean of the other 15
        rest = [x for i, (_, x) in enumerate(trial_f1) if i not in c]
        return statistics.fmean(trial_f1[i][1] for i in c) - statistics.fmean(rest)

    rs_idx = tuple(i for i, (mode, _) in enumerate(trial_f1) if mode == "rules_static")
    obs_gap = gap(rs_idx)
    extreme = sum(1 for c in combos if gap(c) >= obs_gap - 1e-12)
    extreme_2 = sum(1 for c in combos if abs(gap(c)) >= abs(obs_gap) - 1e-12)

    empty_idx = [i for i, t in enumerate(T) if not t]
    P_cur = arms["rerun_current_scorer"][0]
    off_out = Counter(x for p in off_P for x in p if x not in GROUPS)
    return {
        "n": n,
        "rows": rows,
        "rows_2025": arms_2025,
        "rules_static_2025": {
            "label_kinds": dict(rs_labels),
            "mean_with_current_mapping": rs_mapped_mean,
        },
        "best_constant_selected_on_test_labels": best,
        "all_four_ndcg_range_over_24_orders": [min(four_orders), max(four_orders)],
        "nonempty_subset": {"n": len(ne), **nonempty},
        "paired_bootstrap": {
            "B": B,
            "seed": SEED,
            "family_size": len(family),
            "ci95": ci95,
            "bonferroni_ci": bonf,
        },
        "trial_permutation_rules_static_micro_f1": {
            "observed_mean": obs,
            "subsets": len(combos),
            "one_sided_at_least_as_extreme": extreme,
            "p_one_sided": extreme / len(combos),
            "two_sided_at_least_as_extreme": extreme_2,
            "p_two_sided": extreme_2 / len(combos),
        },
        "empty_set_decomposition": {
            "gold_empty": len(empty_idx),
            "gold_nonempty": len(ne),
            "rerun_pred_empty_on_gold_empty": sum(1 for i in empty_idx if not P_cur[i]),
            "rerun_exact_on_gold_nonempty": sum(1 for i in ne if P_cur[i] == T[i]),
            "gold_label_counts": dict(Counter(x for t in T for x in t)),
            "rerun_pred_labels_outside_groups": sorted({x for p in P_cur for x in p} - set(GROUPS)),
            "offline_pred_labels_outside_groups": dict(off_out),
        },
        "severity": {
            "rerun": run["metrics"]["ordinal"],
            "always_1": calculate_ordinal_metrics(ys, [1] * n),
            "offline_keyword_arm": calculate_ordinal_metrics(ys, off_S),
            "gold_severity_1": ys.count(1),
            "harm_labelled_cases_with_severity_1": [case_title(cases[i]) for i in ne if ys[i] == 1],
            "rerun_unparsed_predictions_scored_0": ps.count(0),
        },
        "rerun_config": {
            "model_name": run["config"].get("model_name"),
            "has_resolved_model": "resolved_model" in run["config"],
        },
    }


def superseded_runs() -> dict:
    """The 22 same-day trial files the 2025 summary did not use (paper/trials_2025/superseded)."""
    subtypes = solove_subtypes()
    rows = []
    for f in sorted((TRIALS / "superseded").glob("results_*.json")):
        d = json.loads(f.read_text())
        det, rec = d["details"], d["metrics"]["mlc"]
        rows.append(
            {
                "file": f.name,
                "n": len(det),
                "exact_match_ratio": rec["exact_match_ratio"],
                "micro_f1": rec["micro_f1"],
                "all_predictions_empty": all(not r["pred"]["root_causes"] for r in det),
                "prompt_echo_failures": sum(
                    1
                    for r in det
                    if (r["pred"].get("rationale") or "").startswith("Failed to parse")
                ),
                "gold_unmapped_subtypes": all(set(r["true"]["causes"]) <= subtypes for r in det)
                and any(r["true"]["causes"] for r in det),
            }
        )
    echo = [r for r in rows if r["all_predictions_empty"]]
    unmapped = [r for r in rows if r["gold_unmapped_subtypes"]]
    assert all(r["all_predictions_empty"] or r["gold_unmapped_subtypes"] for r in rows)
    return {
        "files": len(rows),
        "echo_runs_all_empty": len(echo),
        "echo_runs_emr": sorted({r["exact_match_ratio"] for r in echo}),
        "unmapped_gold_runs": len(unmapped),
        "unmapped_gold_runs_micro_f1_max": max(r["micro_f1"] for r in unmapped),
        "runs": rows,
    }


def prompt_identity() -> dict:
    """Which arms send the same prompt text (current code, runner's inputs)."""
    rules = json.loads((DATA / "rules/expert_framework.v1.0.json").read_text())["rules"]
    text = "A vendor exposed customer records through a misconfigured bucket."
    prompts = {
        mode.value: HybridModel(mode=mode, rules=rules, provider="gemini")._build_prompt(
            text, None, {}
        )
        for mode in HybridMode
    }
    groups: dict[str, list[str]] = {}
    for mode, p in prompts.items():
        groups.setdefault(p, []).append(mode)
    return {
        "distinct_prompts": len(groups),
        "modes_by_prompt": sorted(groups.values(), key=len, reverse=True),
        "expert_framework_rules": rules,
    }


def note_kind(note: str | None) -> str:
    if note is None:
        return "none"
    if note == "confirmed":
        return "confirmed"
    return "corrected" if note.startswith("Corrected") else "annotated"


def gold_sets() -> dict:
    v3 = [json.loads(line) for line in open(DATA / "golden_cases_v3.jsonl")]
    base = [json.loads(line) for line in open(DATA / "golden_cases.jsonl")]
    stamps = sorted(datetime.fromisoformat(r["reviewed_at"]) for r in base)
    gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
    rc = sorted({x for r in v3 for x in r["root_causes"]})
    return {
        "v3": {
            "n": len(v3),
            "sources": dict(Counter(r["source"] for r in v3)),
            "empty_root_causes": sum(1 for r in v3 if not r["root_causes"]),
            "has_harms_field": sum(1 for r in v3 if "harms" in r),
            "root_causes_values": rc,
            "root_causes_all_solove_subtypes": set(rc) <= solove_subtypes(),
            "harm_score_counts": dict(sorted(Counter(r["harm_score"] for r in v3).items())),
            "titles": [{"title": case_title(r), "labelled": bool(r["root_causes"])} for r in v3],
        },
        "reviewed_base": {
            "n": len(base),
            "empty_harms": sum(1 for r in base if not r.get("harms")),
            "reviewed": sum(1 for r in base if r.get("reviewed")),
            "review_notes": dict(Counter(note_kind(r.get("review_notes")) for r in base)),
            "label_providers": dict(Counter(str(r.get("llm_provider")) for r in base)),
            "reviewed_at_min": stamps[0].isoformat(),
            "reviewed_at_max": stamps[-1].isoformat(),
            "median_gap_seconds": statistics.median(gaps),
            "gaps_under_10s": sum(g < 10 for g in gaps),
            "gaps": len(gaps),
        },
    }


def comprehensive_runs() -> dict:
    rows, max_dry = [], 0.0
    for d in sorted((DATA / "experiments").glob("comprehensive_*")):
        meta = json.loads((d / "metadata.json").read_text())
        means = {}
        if (d / "summary_by_mode.json").exists():
            for mode, v in json.loads((d / "summary_by_mode.json").read_text()).items():
                means[mode] = v["metrics"]["mlc.micro_f1"]["mean"]
        pair = {"present": (d / "pairwise_tests.json").exists()}
        if pair["present"]:
            try:
                tests = json.loads((d / "pairwise_tests.json").read_text())
            except json.JSONDecodeError:
                pair["parses"] = False
            else:
                entries = [e for row in tests.values() for e in row.values()]
                pair.update(
                    parses=True,
                    entries=len(entries),
                    placeholders=sum(
                        1
                        for e in entries
                        if "insufficient" in json.dumps(e)
                        or (e.get("p_value") == 1.0 and not e.get("mean_diff"))
                    ),
                    bonferroni=any("bonferroni_alpha" in e for e in entries),
                    cliffs_delta=any("effect_size_cliffs_d" in e for e in entries),
                )
        if meta.get("dry_run") and means:
            max_dry = max(max_dry, *means.values())
        rows.append(
            {
                "run": d.name,
                "dry_run": meta.get("dry_run"),
                "golden_file": meta.get("golden_file"),
                "n_configs": meta.get("n_configs"),
                "n_trials": meta.get("n_trials"),
                "modes_with_metrics": len(means),
                "distinct_micro_f1_means": len({round(x, 12) for x in means.values()}),
                "pairwise_tests": pair,
            }
        )
    return {"runs": rows, "max_dry_run_micro_f1_mean": max_dry}


def interpretable_models(corpus: list[dict] | None) -> dict:
    out: dict = {}
    for d in sorted((ROOT / "trained_models").iterdir()):
        metrics = json.loads((d / "metrics.json").read_text())
        row = {k: metrics[k] for k in ("accuracy", "f1") if k in metrics}
        if "accuracy" in metrics:
            frac = Fraction(metrics["accuracy"]).limit_denominator(1000)
            row["accuracy_fraction"] = f"{frac.numerator}/{frac.denominator}"
            row["correct_of_984"] = round(metrics["accuracy"] * 984, 6)
        if "n_successful" in metrics:
            row.update(
                mean_accuracy=metrics["mean_accuracy"],
                mean_f1=metrics["mean_f1"],
                classifiers=metrics["n_classifiers"],
                successful=metrics["n_successful"],
            )
        if "status" in metrics:
            row["status"] = metrics["status"]
        prov = d / "provenance.json"
        if prov.exists():
            row["saved_at"] = json.loads(prov.read_text()).get("saved_at")
        extra = d / "extra.json"
        if extra.exists():
            preds = json.loads(extra.read_text()).get("predictions")
            if isinstance(preds, str):
                items = re.findall(r"'([^']+)'", preds)
                row["n_test_predictions"] = len(items)
                row["distinct_test_predictions"] = sorted(set(items))
        if (d / "feature_names.json").exists():
            row["features"] = json.loads((d / "feature_names.json").read_text())
        out[d.name] = row

    dt = ROOT / "trained_models/decision_tree_v4"
    rules = [json.loads(line) for line in open(dt / "heuristics.jsonl")]
    support = sorted({r["support"] for r in rules})
    node = json.loads((dt / "heuristics_tree.json").read_text())
    while node.get("left"):
        node = node["left"]  # leftmost leaf = rule 1: no keyword present
    train_rows = round(1 / support[0])
    exported = {
        "n_rules": len(rules),
        "distinct_support": support,
        "training_rows_from_1_over_support": train_rows,
        "leaf_samples": sorted({r["extra"]["leaf_samples"] for r in rules}),
        "rule_1_training_share": node["weight"],
        "rule_1_training_rows": round(node["weight"] * train_rows, 6),
        "labelled_rows_train_plus_984_test": train_rows + 984,
    }
    if corpus is not None:
        kws = ["kw_privacy", "kw_monetary_penalty", "kw_biometric", "kw_reg_enforcement"]
        kws += ["kw_video_surveillance", "kw_location"]
        hits = sum(1 for r in corpus if all((r.get(k) or 0) <= 0.5 for k in kws))
        exported["rule_1_release_share"] = round(hits / len(corpus), 4)
        boc = sum(1 for r in corpus if r.get("harm_category") == "breach_of_confidentiality")
        out["bayes_net_v4"]["predicted_class_share_in_release"] = round(boc / len(corpus), 4)
    out["decision_tree_v4_exported_rules"] = exported
    return out


def label_sources(rows: list[dict] | None) -> dict | None:
    if rows is None:
        return None
    src = Counter(str(r.get("harm_category_source")) for r in rows)
    cat = Counter(r.get("harm_category") for r in rows)
    classes = sorted(c for c in cat if c != "unknown")
    return {
        "n": len(rows),
        "harm_category_source": dict(src),
        "unknown_share": round(cat["unknown"] / len(rows), 4),
        "labelled_classes": len(classes),
        "non_solove_classes": sorted(set(classes) - solove_subtypes()),
    }


def load(path: Path) -> list[dict] | None:
    return [json.loads(line) for line in open(path)] if path.exists() else None


def main() -> None:
    full = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "with_features.jsonl"
    corpus = load(full)
    result = {
        "recorded_2025": recorded_2025(),
        "headline": headline(),
        "superseded_2025_runs": superseded_runs(),
        "prompt_identity": prompt_identity(),
        "gold_sets": gold_sets(),
        "comprehensive_runs": comprehensive_runs(),
        "interpretable_models": interpretable_models(corpus),
        "label_sources": {
            "committed_sample": label_sources(load(DATA / "with_features.sample.jsonl")),
            "release_asset_v1_0_0": label_sources(corpus),
        },
    }
    out = ROOT / "paper/reanalysis.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
