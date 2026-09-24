"""Offline re-analysis behind ``paper/lessons.tex``. No network, no API calls.

Recomputes every number the paper states from committed artifacts and writes
them to ``paper/reanalysis.json``. The full-corpus label-source figures need
the v1.0.0 release asset (``make fetch-data`` -> ``data/with_features.jsonl``);
without it those fields are ``null``.

Run from the repo root:  PYTHONPATH=src python paper/reanalysis.py [corpus.jsonl]
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from privacy_harm_heuristics.constants.privacy_taxonomy import TAXONOMY_SOLOVE
from privacy_harm_heuristics.evals.metrics import (
    calculate_mlc_metrics,
    calculate_ordinal_metrics,
    calculate_ranking_metrics,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RERUN = DATA / "experiments/rerun_20260721/results_rules_static_20260721_212152.json"
GROUPS = [
    "information_processing",
    "invasion",
    "information_collection",
    "information_dissemination",
]

# Copied from evals/hybrid_runner.py `custom_map`. The self-check in
# rerun_and_baselines() fails if this copy stops matching the scorer.
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


def score(T, TR, P, PR) -> dict[str, float]:
    mlc = calculate_mlc_metrics(T, P)
    return {
        "instance_jaccard": mlc["instance_jaccard"],
        "exact_match_ratio": mlc["exact_match_ratio"],
        "micro_f1": mlc["micro_f1"],
        "ndcg@5": calculate_ranking_metrics(TR, PR)["ndcg@5"],
    }


def rerun_and_baselines() -> dict:
    run = json.loads(RERUN.read_text())
    det, m = run["details"], label_map()
    n = len(det)
    T = [set(r["true"]["causes"]) for r in det]
    TR = [r["true"]["ranking"] for r in det]
    P = [set(to_parents(r["pred"]["root_causes"], m)) for r in det]
    PR = [to_parents(r["pred"]["ranking"], m) for r in det]

    recomputed = score(T, TR, P, PR)
    recorded = run["metrics"]
    for key, rec in [
        ("instance_jaccard", recorded["mlc"]["instance_jaccard"]),
        ("exact_match_ratio", recorded["mlc"]["exact_match_ratio"]),
        ("micro_f1", recorded["mlc"]["micro_f1"]),
        ("ndcg@5", recorded["ranking"]["ndcg@5"]),
    ]:
        assert abs(recomputed[key] - rec) < 1e-12, f"scorer drift on {key}"

    const = {
        "always_empty": ([set() for _ in T], [[] for _ in T]),
        "always_information_processing": (
            [{GROUPS[0]} for _ in T],
            [[GROUPS[0]] for _ in T],
        ),
        "always_invasion": ([{GROUPS[1]} for _ in T], [[GROUPS[1]] for _ in T]),
        "always_all_four_groups": ([set(GROUPS) for _ in T], [GROUPS for _ in T]),
    }
    baselines = {name: score(T, TR, p, pr) for name, (p, pr) in const.items()}

    # Severity (1-5): the recorded run vs. always predicting the modal score 1.
    ys = [r["true"]["score"] for r in det]
    severity = {
        "rerun_rules_static": recorded["ordinal"],
        "always_1": {
            "accuracy": calculate_ordinal_metrics(ys, [1] * n)["accuracy"],
            "gold_score_1_count": ys.count(1),
        },
    }

    empty = [i for i, t in enumerate(T) if not t]
    nonempty = [i for i, t in enumerate(T) if t]
    decomposition = {
        "n": n,
        "gold_empty": len(empty),
        "gold_nonempty": len(nonempty),
        "pred_empty_on_gold_empty": sum(1 for i in empty if not P[i]),
        "exact_on_gold_nonempty": sum(1 for i in nonempty if P[i] == T[i]),
        "gold_label_counts": dict(Counter(x for t in T for x in t).most_common()),
        "pred_labels_outside_gold_space": sorted({x for p in P for x in p} - set(GROUPS)),
    }

    # Seeded case-resampling bootstrap; paired against the constant predictor.
    rng, B = random.Random(0), 2000
    cp = const["always_information_processing"][0]
    draws: dict[str, list[float]] = {
        "instance_jaccard": [],
        "exact_match_ratio": [],
        "micro_f1": [],
        "micro_f1_minus_constant": [],
    }
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        t = [T[i] for i in idx]
        a = calculate_mlc_metrics(t, [P[i] for i in idx])
        c = calculate_mlc_metrics(t, [cp[i] for i in idx])
        for k in ("instance_jaccard", "exact_match_ratio", "micro_f1"):
            draws[k].append(a[k])
        draws["micro_f1_minus_constant"].append(a["micro_f1"] - c["micro_f1"])
    ci = {}
    for k, v in draws.items():
        v.sort()
        ci[k] = [round(v[int(0.025 * B)], 4), round(v[int(0.975 * B) - 1], 4)]

    return {
        "rerun_rules_static_recomputed": recomputed,
        "trivial_baselines": baselines,
        "severity_baseline": severity,
        "empty_set_decomposition": decomposition,
        "bootstrap_95ci_rerun": {"B": B, "seed": 0, **ci},
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
    return {
        "v3": {
            "n": len(v3),
            "sources": dict(Counter(r["source"] for r in v3).most_common()),
            "empty_root_causes": sum(1 for r in v3 if not r["root_causes"]),
            "has_harms_field": sum(1 for r in v3 if "harms" in r),
            "harm_score_counts": dict(sorted(Counter(r["harm_score"] for r in v3).items())),
        },
        "reviewed_base": {
            "n": len(base),
            "reviewed": sum(1 for r in base if r.get("reviewed")),
            "review_notes": dict(Counter(note_kind(r.get("review_notes")) for r in base)),
            "label_providers": dict(Counter(str(r.get("llm_provider")) for r in base)),
            "reviewed_at_min": min(r["reviewed_at"] for r in base),
            "reviewed_at_max": max(r["reviewed_at"] for r in base),
        },
    }


def comprehensive_runs() -> list[dict]:
    rows = []
    for d in sorted((DATA / "experiments").glob("comprehensive_*")):
        meta = json.loads((d / "metadata.json").read_text())
        by_mode = d / "summary_by_mode.json"
        means = {}
        if by_mode.exists():
            for mode, v in json.loads(by_mode.read_text()).items():
                means[mode] = v["metrics"]["mlc.micro_f1"]["mean"]
        rows.append(
            {
                "run": d.name,
                "dry_run": meta.get("dry_run"),
                "golden_file": meta.get("golden_file"),
                "n_configs": meta.get("n_configs"),
                "modes_with_metrics": len(means),
                "distinct_micro_f1_means": len({round(x, 12) for x in means.values()}),
            }
        )
    return rows


def interpretable_models() -> dict:
    out = {}
    for d in sorted((ROOT / "trained_models").iterdir()):
        metrics = json.loads((d / "metrics.json").read_text())
        row = {k: metrics[k] for k in ("accuracy", "f1") if k in metrics}
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
        feats = d / "feature_names.json"
        if feats.exists():
            row["features"] = json.loads(feats.read_text())
        out[d.name] = row
    heur = (ROOT / "trained_models/decision_tree_v4/HEURISTICS.md").read_text()
    support = [float(s) for s in re.findall(r"support=([0-9.]+)", heur)]
    precision = [float(s) for s in re.findall(r"precision=([0-9.]+)", heur)]
    out["decision_tree_v4_exported_rules"] = {
        "n_rules": len(support),
        "max_support": max(support),
        "max_precision": max(precision),
    }
    return out


def label_sources(path: Path) -> dict | None:
    if not path.exists():
        return None
    src, cat, n = Counter(), Counter(), 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n += 1
            src[str(r.get("harm_category_source"))] += 1
            cat[r.get("harm_category")] += 1
    return {
        "n": n,
        "harm_category_source": dict(src),
        "unknown_share": round(cat["unknown"] / n, 4),
        "distinct_categories": len(cat),
    }


def main() -> None:
    full = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "with_features.jsonl"
    result = {
        "gold_sets": gold_sets(),
        **rerun_and_baselines(),
        "comprehensive_runs": comprehensive_runs(),
        "interpretable_models": interpretable_models(),
        "label_sources": {
            "committed_sample": label_sources(DATA / "with_features.sample.jsonl"),
            "release_asset_v1_0_0": label_sources(full),
        },
    }
    out = ROOT / "paper/reanalysis.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
