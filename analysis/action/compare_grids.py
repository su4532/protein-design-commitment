#!/usr/bin/env python
"""Compare published seven-step, coherent seven-step, and ten-step action timing."""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr

QUALITIES = ("interface_quality", "fold_identity", "structural_confidence", "designability")
ADDED = {150, 250, 350}
DEPLOYED = {100, 200, 300}


def ci(rows: list[float], seed: int, n_boot: int = 10000) -> dict:
    a = np.asarray([x for x in rows if np.isfinite(x)], dtype=float)
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(1)
    return {
        "mean": float(a.mean()),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "n": int(len(a)),
    }


def cluster_values(values: dict[str, float], clusters: dict[str, str]) -> tuple[list[str], list[float]]:
    grouped: dict[str, list[float]] = {}
    for target, value in values.items():
        if np.isfinite(value):
            grouped.setdefault(clusters[target], []).append(float(value))
    names = sorted(grouped)
    return names, [float(np.mean(grouped[name])) for name in names]


def count_steps(q: dict) -> dict:
    global_counts = collections.Counter(int(row["t_global"]) for row in q["rotation"]["chosen_fixed_steps"])
    adaptive_counts = collections.Counter()
    for counts in q["rotation"]["adaptive_step_counts"].values():
        adaptive_counts.update({int(step): int(n) for step, n in counts.items()})
    n_global = sum(global_counts.values())
    n_adaptive = sum(adaptive_counts.values())
    return {
        "panel_global": {
            "counts": dict(sorted(global_counts.items())),
            "n_total": n_global,
            "n_added": sum(global_counts[s] for s in ADDED),
            "fraction_added": sum(global_counts[s] for s in ADDED) / n_global,
            "n_deployed": sum(global_counts[s] for s in DEPLOYED),
            "fraction_deployed": sum(global_counts[s] for s in DEPLOYED) / n_global,
            "modal_step": global_counts.most_common(1)[0][0],
        },
        "per_target": {
            "counts": dict(sorted(adaptive_counts.items())),
            "n_total": n_adaptive,
            "n_added": sum(adaptive_counts[s] for s in ADDED),
            "fraction_added": sum(adaptive_counts[s] for s in ADDED) / n_adaptive,
        },
    }


def gain_row(gain: dict, quality: str) -> dict:
    q = gain["by_quality"][quality]
    primary = q["scored_conservative"]["gain_adaptive_vs_deployed"]
    return {
        "adaptive_vs_deployed": primary,
        "adaptive": q["scored_conservative"]["adaptive"],
        "deployed": q["scored_conservative"]["deployed"],
        "global": q["scored_conservative"]["global"],
        "step100": q["scored_conservative"]["step100"],
        "selected_steps": count_steps(q),
    }


def settling_relationship(gain: dict, commitment: dict, quality: str, seed: int) -> dict:
    q = gain["by_quality"][quality]
    steps = np.asarray(gain["capture_steps"], dtype=float)
    t_half = commitment["pooled"]["t_half_per_target"][quality]
    clusters = gain["clusters"]
    per_target = {}
    for target in q["targets"]:
        curve = np.asarray([q["curves_full"][target][str(int(s))] for s in steps], dtype=float)
        interp = PchipInterpolator(steps, curve, extrapolate=False)
        th = float(t_half[target])
        peak = float(q["regret"][target]["peak_step_interpolated"])
        peak_j = float(q["regret"][target]["peak_J"])
        j_at_th = float(interp(float(np.clip(th, steps.min(), steps.max()))))
        regret = (peak_j - j_at_th) / peak_j if peak_j > 0 else float("nan")
        per_target[target] = {
            "cluster": clusters[target],
            "t_half": th,
            "objective_peak": peak,
            "t_half_minus_peak": th - peak,
            "abs_offset": abs(th - peak),
            "J_at_t_half": j_at_th,
            "peak_J": peak_j,
            "relative_regret_at_t_half": regret,
        }

    th_names, th_cluster = cluster_values({t: d["t_half"] for t, d in per_target.items()}, clusters)
    pk_names, pk_cluster = cluster_values({t: d["objective_peak"] for t, d in per_target.items()}, clusters)
    if th_names != pk_names:
        raise AssertionError("cluster rows differ")
    x, y = np.asarray(th_cluster), np.asarray(pk_cluster)
    rho = float(spearmanr(x, y).correlation)
    rng = np.random.default_rng(seed)
    null = np.asarray([spearmanr(x, rng.permutation(y)).correlation for _ in range(20000)])
    _, regret_cluster = cluster_values(
        {t: d["relative_regret_at_t_half"] for t, d in per_target.items()}, clusters
    )
    return {
        "n_targets": len(per_target),
        "n_structural_clusters": len(th_names),
        "median_t_half": float(np.median([d["t_half"] for d in per_target.values()])),
        "median_objective_peak": float(np.median([d["objective_peak"] for d in per_target.values()])),
        "median_abs_offset": float(np.median([d["abs_offset"] for d in per_target.values()])),
        "spearman_by_cluster": rho,
        "spearman_permutation_p_two_sided": float((np.abs(null) >= abs(rho)).mean()),
        "mean_relative_regret_at_t_half": ci(regret_cluster, seed=seed + 1),
        "per_target": per_target,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--published-seven", required=True)
    p.add_argument("--recorded-seven", required=True)
    p.add_argument("--joint-seven", required=True)
    p.add_argument("--ten", required=True)
    p.add_argument("--commitment-ten", required=True)
    p.add_argument("--timing", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    sources = {
        "published_seven": json.load(open(args.published_seven)),
        "joint_label_seven": json.load(open(args.joint_seven)),
        "ten": json.load(open(args.ten)),
    }
    recorded_seven = json.load(open(args.recorded_seven))
    commitment = json.load(open(args.commitment_ten))
    timing = json.load(open(args.timing))

    out = {
        "capture_grids": {name: gain["capture_steps"] for name, gain in sources.items()},
        "comparison_scope": {
            "structural_bootstrap": (
                "all inferential seven- and ten-step comparisons reselect the global and deployed "
                "comparators within every structural-cluster draw; the originally recorded seven-step "
                "interval is retained as superseded provenance only"
            ),
            "interface_quality": "saved seven-step predictions and unchanged labels; seven-to-ten isolates added capture steps",
            "structural_confidence": "saved seven-step predictions and unchanged labels; seven-to-ten isolates added capture steps",
            "designability": "saved seven-step predictions and unchanged labels; seven-to-ten isolates added capture steps",
            "fold_identity": (
                "published_seven uses the original seven-step Foldseek clustering; joint_label_seven "
                "and ten refit every step against the ten-step joint clustering, so joint_label_seven-to-ten "
                "isolates the added capture steps"
            ),
        },
        "by_quality": {},
        "settling_vs_objective_by_quality": {},
        "recorded_seven_reference": {
            quality: {
                "adaptive_vs_deployed": recorded_seven["by_quality"][quality]["scored_conservative"][
                    "gain_adaptive_vs_deployed"
                ],
                "status": "superseded: comparator was not reselected inside cluster bootstrap draws",
            }
            for quality in QUALITIES
        },
        "compute_timing": timing,
    }
    for qi, quality in enumerate(QUALITIES):
        rows = {name: gain_row(gain, quality) for name, gain in sources.items()}
        p7 = rows["published_seven"]["adaptive_vs_deployed"]["mean"]
        j7 = rows["joint_label_seven"]["adaptive_vs_deployed"]["mean"]
        ten = rows["ten"]["adaptive_vs_deployed"]["mean"]
        rows["shifts"] = {
            "published_seven_to_ten": ten - p7,
            "published_seven_to_joint_label_seven": j7 - p7,
            "joint_label_seven_to_ten": ten - j7,
        }
        out["by_quality"][quality] = rows
        out["settling_vs_objective_by_quality"][quality] = settling_relationship(
            sources["ten"], commitment, quality, seed=100 + qi * 10
        )

    primary = out["by_quality"]["interface_quality"]["ten"]["adaptive_vs_deployed"]
    out["decision"] = {
        "rule": "positive 95% structural-cluster interval excludes zero",
        "criterion_met": bool(primary["lo"] > 0),
        "estimate": primary["mean"],
        "ci95": [primary["lo"], primary["hi"]],
        "assessment": (
            "per-target timing outperforms the best deployed point"
            if primary["lo"] > 0
            else "per-target timing was not shown to outperform the best deployed point"
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"decision": out["decision"], "compute_timing": timing}, indent=1))
    for quality in QUALITIES:
        row = out["by_quality"][quality]["ten"]
        g = row["adaptive_vs_deployed"]
        s = row["selected_steps"]
        print(
            f"{quality:24s} {g['mean']:+.5f} [{g['lo']:+.5f},{g['hi']:+.5f}] "
            f"global added {s['panel_global']['n_added']}/{s['panel_global']['n_total']} "
            f"adaptive added {s['per_target']['n_added']}/{s['per_target']['n_total']}"
        )


if __name__ == "__main__":
    main()
