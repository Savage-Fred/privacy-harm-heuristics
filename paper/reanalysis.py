"""Offline re-analysis behind ``paper/lessons.tex``. No network, no API calls.

Recomputes every number the paper states from committed artifacts and writes
them to ``paper/reanalysis.json``. Full-corpus fields need the v1.0.0 release
asset (``make fetch-data`` -> ``data/with_features.jsonl``); without it they
are ``null``.

Self-checks (fail loudly): the scoring path must reproduce the recorded 2026
rerun exactly (set, ranking and ordinal metrics), and the per-case offline arm
must reproduce ``cli._run_deterministic_offline`` exactly.

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
from itertools import permutations
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
RERUN = DATA / "experiments/rerun_20260721/results_rules_static_20260721_212152.json"
SUMMARY = DATA / "experiments/final_results_summary.md"
GROUPS = list(TAXONOMY_SOLOVE)  # taxonomy order: collection, processing, dissemination, invasion
B, SEED = 2000, 0

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


def jaccard_empty_as_one(T, P) -> float:
    """Instance Jaccard with the other common convention: empty vs empty = 1."""
    return sum(1.0 if not t and not p else len(t & p) / len(t | p) for t, p in zip(T, P)) / len(T)


def score(T, TR, P, PR) -> dict[str, float]:
    mlc = calculate_mlc_metrics(T, P)
    return {
        "instance_jaccard": mlc["instance_jaccard"],
        "jaccard_empty_as_1": jaccard_empty_as_one(T, P),
        "exact_match_ratio": mlc["exact_match_ratio"],
        "micro_f1": mlc["micro_f1"],
        "ndcg@5": calculate_ranking_metrics(TR, PR)["ndcg@5"],
    }


def constant_arms() -> list[tuple[str, set[str], list[str]]]:
    """Every constant (label set, ranking order) over the four groups: 65 arms."""
    arms = []
    for k in range(len(GROUPS) + 1):
        for order in permutations(GROUPS, k):
            arms.append(
                ("+".join(g.split("_")[-1] for g in order) or "empty", set(order), list(order))
            )
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
            j, emr, f1, ndcg = (float(c) for c in cells[1:])
            rows[cells[0]] = {
                "instance_jaccard": j,
                "exact_match_ratio": emr,
                "micro_f1": f1,
                "ndcg@5": ndcg,
                # The summary says 5 trials x 50 cases (250 case-evaluations).
                "emr_x250": round(emr * 250, 2),
                "emr_x150": round(emr * 150, 2),
            }
    return rows


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

    cases = [json.loads(line) for line in open(DATA / "golden_cases_v3.jsonl")]
    assert [str(c["id"]) for c in cases] == [str(r["case_id"]) for r in det], "case order"
    off_P, off_PR, off_S = offline_arm(cases, m)

    arms = {
        "rerun_current_scorer": (
            [set(to_parents(p, m)) for p in raw_P],
            [to_parents(p, m) for p in raw_PR],
        ),
        # Earliest committed runner (2025-11-24) mapped gold labels to parents
        # but scored raw predicted labels and rankings.
        "rerun_nov2025_scorer": ([set(p) for p in raw_P], [list(p) for p in raw_PR]),
        # Sensitivity: drop predicted labels outside the four gold groups.
        "rerun_filtered_to_groups": (
            [set(to_parents(p, m)) & set(GROUPS) for p in raw_P],
            [[x for x in to_parents(p, m) if x in GROUPS] for p in raw_PR],
        ),
        "offline_keyword_arm": (off_P, off_PR),
    }

    # Self-checks.
    rec = run["metrics"]
    got = score(T, TR, *arms["rerun_current_scorer"])
    got_ord = calculate_ordinal_metrics(ys, ps)
    for key, want in [
        ("instance_jaccard", rec["mlc"]["instance_jaccard"]),
        ("exact_match_ratio", rec["mlc"]["exact_match_ratio"]),
        ("micro_f1", rec["mlc"]["micro_f1"]),
        ("ndcg@5", rec["ranking"]["ndcg@5"]),
    ]:
        assert abs(got[key] - want) < 1e-12, f"scorer drift on {key}"
    for key in ("weighted_kappa", "accuracy"):
        assert abs(got_ord[key] - rec["ordinal"][key]) < 1e-12, f"ordinal drift on {key}"
    ref = _run_deterministic_offline(cases, m)
    got_off = score(T, TR, off_P, off_PR)
    for key in ("instance_jaccard", "exact_match_ratio", "micro_f1", "ndcg@5"):
        assert abs(got_off[key] - ref[key]) < 1e-12, f"offline arm drift on {key}"

    consts = constant_arms()
    rows = {name: score(T, TR, P, PR) for name, (P, PR) in arms.items()}
    rows["const_empty"] = score(T, TR, [set()] * n, [[]] * n)
    rows["const_all_four_taxonomy_order"] = score(T, TR, [set(GROUPS)] * n, [GROUPS] * n)
    rows["oracle_ceiling_perfect_predictor"] = score(T, TR, T, TR)

    all_scores = {name: score(T, TR, [s] * n, [o] * n) for name, s, o in consts}
    best = {}
    for metric in ("instance_jaccard", "exact_match_ratio", "micro_f1", "ndcg@5"):
        name = max(all_scores, key=lambda a: all_scores[a][metric])
        best[metric] = {"arm": name, "value": all_scores[name][metric]}
    four_orders = [all_scores[a]["ndcg@5"] for a, s, _ in consts if len(s) == 4]

    ne = [i for i, t in enumerate(T) if t]

    def subset(P, PR, idx):
        return score(
            [T[i] for i in idx], [TR[i] for i in idx], [P[i] for i in idx], [PR[i] for i in idx]
        )

    nonempty = {name: subset(P, PR, ne) for name, (P, PR) in arms.items()}
    nonempty["const_all_four_taxonomy_order"] = subset([set(GROUPS)] * n, [GROUPS] * n, ne)

    # Paired case-resampling bootstrap.
    rng = random.Random(SEED)
    base = {
        **arms,
        "const_empty": ([set()] * n, [[]] * n),
        "const_all_four": ([set(GROUPS)] * n, [GROUPS] * n),
    }
    subsets = {name: s for name, s, _ in consts if list(s) == sorted(s, key=GROUPS.index)}
    diffs: dict[str, list[float]] = {
        k: []
        for k in (
            "emr: const_empty - rerun",
            "jaccard: const_all_four - rerun",
            "ndcg@5: const_all_four - rerun",
            "micro_f1: rerun - best_constant_reselected",
            "micro_f1: filtered - best_constant_reselected",
            "jaccard: const_all_four - offline",
            "ndcg@5: const_all_four - offline",
            "micro_f1: const_all_four - offline",
            "rerun micro_f1",
            "rerun weighted_kappa",
        )
    }
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]

        def s(name, _idx=idx):
            P, PR = base[name]
            return subset(P, PR, _idx)

        r, f, o = s("rerun_current_scorer"), s("rerun_filtered_to_groups"), s("offline_keyword_arm")
        e, a = s("const_empty"), s("const_all_four")
        t = [T[i] for i in idx]
        best_f1 = max(calculate_mlc_metrics(t, [st] * n)["micro_f1"] for st in subsets.values())
        diffs["emr: const_empty - rerun"].append(e["exact_match_ratio"] - r["exact_match_ratio"])
        diffs["jaccard: const_all_four - rerun"].append(
            a["instance_jaccard"] - r["instance_jaccard"]
        )
        diffs["ndcg@5: const_all_four - rerun"].append(a["ndcg@5"] - r["ndcg@5"])
        diffs["micro_f1: rerun - best_constant_reselected"].append(r["micro_f1"] - best_f1)
        diffs["micro_f1: filtered - best_constant_reselected"].append(f["micro_f1"] - best_f1)
        diffs["jaccard: const_all_four - offline"].append(
            a["instance_jaccard"] - o["instance_jaccard"]
        )
        diffs["ndcg@5: const_all_four - offline"].append(a["ndcg@5"] - o["ndcg@5"])
        diffs["micro_f1: const_all_four - offline"].append(a["micro_f1"] - o["micro_f1"])
        diffs["rerun micro_f1"].append(r["micro_f1"])
        diffs["rerun weighted_kappa"].append(
            calculate_ordinal_metrics([ys[i] for i in idx], [ps[i] for i in idx])["weighted_kappa"]
        )
    ci = {}
    for k, v in diffs.items():
        v.sort()
        ci[k] = [round(v[int(0.025 * B)], 4), round(v[int(0.975 * B) - 1], 4)]

    empty = [i for i, t in enumerate(T) if not t]
    P_cur = arms["rerun_current_scorer"][0]
    harm_sev1 = [case_title(cases[i]) for i in ne if ys[i] == 1]
    return {
        "n": n,
        "rows": rows,
        "best_constant_selected_on_test_labels": best,
        "all_four_ndcg_range_over_24_orders": [min(four_orders), max(four_orders)],
        "nonempty_subset": {"n": len(ne), **nonempty},
        "paired_bootstrap_95ci": {"B": B, "seed": SEED, **ci},
        "empty_set_decomposition": {
            "gold_empty": len(empty),
            "gold_nonempty": len(ne),
            "rerun_pred_empty_on_gold_empty": sum(1 for i in empty if not P_cur[i]),
            "rerun_exact_on_gold_nonempty": sum(1 for i in ne if P_cur[i] == T[i]),
            "gold_label_counts": dict(Counter(x for t in T for x in t).most_common()),
            "rerun_pred_labels_outside_groups": sorted({x for p in P_cur for x in p} - set(GROUPS)),
        },
        "severity": {
            "rerun": run["metrics"]["ordinal"],
            "always_1": calculate_ordinal_metrics(ys, [1] * n),
            "offline_keyword_arm": calculate_ordinal_metrics(ys, off_S),
            "gold_severity_1": ys.count(1),
            "harm_labelled_cases_with_severity_1": harm_sev1,
            "rerun_unparsed_predictions_scored_0": ps.count(0),
        },
        "rerun_config": {
            "model_name": run["config"].get("model_name"),
            "has_resolved_model": "resolved_model" in run["config"],
        },
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
            "sources": dict(Counter(r["source"] for r in v3).most_common()),
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
            text = (d / "pairwise_tests.json").read_text()
            try:
                tests = json.loads(text)
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
        if "n_successful" in metrics:
            row.update(
                mean_accuracy=metrics["mean_accuracy"],
                mean_f1=metrics["mean_f1"],
                classifiers=metrics["n_classifiers"],
                successful=metrics["n_successful"],
            )
        if "status" in metrics:
            row["status"] = metrics["status"]
        extra = d / "extra.json"
        if extra.exists():
            preds = json.loads(extra.read_text()).get("predictions")
            if isinstance(preds, str):
                row["distinct_test_predictions"] = sorted(set(re.findall(r"'([^']+)'", preds)))
        if (d / "feature_names.json").exists():
            row["features"] = json.loads((d / "feature_names.json").read_text())
        out[d.name] = row
    heur = (ROOT / "trained_models/decision_tree_v4/HEURISTICS.md").read_text()
    support = [float(s) for s in re.findall(r"support=([0-9.]+)", heur)]
    precision = [float(s) for s in re.findall(r"precision=([0-9.]+)", heur)]
    rule1 = {"n_rules": len(support), "max_support": max(support), "max_precision": max(precision)}
    if corpus is not None:
        kws = ["kw_privacy", "kw_monetary_penalty", "kw_biometric", "kw_reg_enforcement"]
        kws += ["kw_video_surveillance", "kw_location"]
        hits = sum(1 for r in corpus if all((r.get(k) or 0) <= 0.5 for k in kws))
        rule1["rule_1_release_coverage"] = round(hits / len(corpus), 4)
        boc = sum(1 for r in corpus if r.get("harm_category") == "breach_of_confidentiality")
        out["bayes_net_v4"]["predicted_class_share_in_release"] = round(boc / len(corpus), 4)
    out["decision_tree_v4_exported_rules"] = rule1
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
