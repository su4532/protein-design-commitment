#!/usr/bin/env python3
"""Regenerate the eight manuscript tables from anonymous frozen fixtures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

QUALITIES = ("fold_identity", "structural_confidence", "designability", "interface_quality")


def _estimate(block: dict[str, Any]) -> dict[str, float]:
    return {key: float(block[key]) for key in ("mean", "lo", "hi")}


def build_tables(fixtures: Path) -> dict[str, list[dict[str, Any]]]:
    fig = json.loads((fixtures / "figure_data.json").read_text())
    order = json.loads((fixtures / "order_power.json").read_text())
    opportunity = json.loads((fixtures / "adaptive_opportunity.json").read_text())
    joint = json.loads((fixtures / "joint_opportunity.json").read_text())

    clocks = []
    bb = fig["schedule"]["time_at_capture_steps"]["bb_ca"]
    latent = fig["schedule"]["time_at_capture_steps"]["local_latents"]
    for step, a, b in zip(fig["schedule"]["capture_steps"], bb, latent):
        clocks.append({"step": step, "backbone_time": a, "latent_time": b,
                       "uniform_time": step / fig["schedule"]["nsteps"]})

    order_null = []
    labels = {
        "common_value": "shared order, pooled magnitudes",
        "common_order": "shared order, per-target speed",
        "own_gaps": "shared order, each target's own gaps",
        "own_gaps_deconvolved": "shared order, own gaps, noise-corrected",
    }
    for key in labels:
        block = order["nulls"][key]
        order_null.append({
            "condition": labels[key],
            "kendall_w_median": block["kendall_w"]["median"],
            "kendall_w_p05": block["kendall_w"]["p05"],
            "kendall_w_p95": block["kendall_w"]["p95"],
            "distinct_orders_median": block["n_distinct_orders"]["median"],
            "distinct_orders_p05": block["n_distinct_orders"]["p05"],
            "distinct_orders_p95": block["n_distinct_orders"]["p95"],
            "p_w_le_observed": block["p_w_le_observed"],
        })
    order_null.append({"condition": "observed", "kendall_w_median": order["observed"]["kendall_w"],
                       "distinct_orders_median": order["observed"]["n_distinct_orders"]})

    families = order["gap_scale_sweep"]["families"]
    by_family = {name: {row["gap_scale"]: row for row in rows} for name, rows in families.items()}
    own_scale = order["gap_scale_sweep"]["own_separation_steps_median"]
    pooled_scale = order["gap_scale_sweep"]["pooled_separation_steps"]
    gap_scale = []
    for scale in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        row = {"gap_scale": scale, "pooled_separation_steps": scale * pooled_scale,
               "own_separation_steps": scale * own_scale}
        for family in ("gap_scale_uniform", "gap_scale_varying", "gap_scale_own"):
            src = by_family[family][scale]
            row[f"{family}_median_w"] = src["median_w"]
            row[f"{family}_p"] = src["p_w_le_observed"]
        gap_scale.append(row)

    cv = []
    for quality in QUALITIES:
        c = fig["spread_cluster_unit"]["per_metric"][quality]
        t = fig["spread_target_unit"]["per_metric"][quality]
        cv.append({"quality": quality, "cluster_cv": c["cv_corrected"],
                   "cluster_lo": c["cv_corrected_ci95"][0], "cluster_hi": c["cv_corrected_ci95"][1],
                   "target_cv": t["cv_corrected"], "target_lo": t["cv_corrected_ci95"][0],
                   "target_hi": t["cv_corrected_ci95"][1],
                   "estimation_error_steps": c["mean_within_se_steps"]})

    pairwise = []
    for key in sorted(fig["spread_cluster_unit"]["pairwise"]):
        c = fig["spread_cluster_unit"]["pairwise"][key]
        t = fig["spread_target_unit"]["pairwise"][key]
        pairwise.append({"difference": key, "cluster_point": c["observed"],
                         "cluster_lo": c["ci95"][0], "cluster_hi": c["ci95"][1],
                         "target_point": t["observed"], "target_lo": t["ci95"][0],
                         "target_hi": t["ci95"][1]})

    oa = []
    j = joint["measures"]["interface_quality"]
    for measure in ("O", "A"):
        for statistic in ("adaptive", "deployed", "gain_adaptive_vs_deployed"):
            oa.append({"measure": measure, "statistic": statistic,
                       "clusters_only": _estimate(j[measure]["cluster_only"][statistic]),
                       "clusters_and_design": _estimate(j[measure]["cluster_and_design"][statistic]),
                       "joint": _estimate(j[measure]["joint_cluster_rotation_design"][statistic])})

    thresholds = []
    for quality in QUALITIES:
        blocks = opportunity["commitment_threshold_sensitivity"]["by_quality"][quality]["thresholds"]
        for threshold in ("0.25", "0.5", "0.75", "0.9"):
            block = blocks[threshold]
            thresholds.append({"quality": quality, "threshold": float(threshold),
                               "median_step": block["resolved_dispersion"]["median"],
                               "n_resolved": block["censoring_counts"]["resolved"],
                               "interpret": block["interpret"]})

    boot = [
        {"family": "cell variance and commitment", "unit": "roots then children", "draws": 2000, "seed": "target-derived"},
        {"family": "pooled commitment", "unit": "structural clusters", "draws": 5000, "seed": 42},
        {"family": "settling-step spread", "unit": "clusters or targets plus saved design draws", "draws": 10000, "seed": 42},
        {"family": "clock remapping", "unit": "same saved spread replicates", "draws": 10000, "seed": 42},
        {"family": "action timing and O/A", "unit": "clusters, root rotations and saved design draws", "draws": 4000, "seed": 17},
        {"family": "scoring-noise sensitivity", "unit": "clusters or targets", "draws": 10000, "seed": 42},
        {"family": "shared-order simulations", "unit": "simulated panels, not a bootstrap", "draws": 20000, "seed": 42},
    ]

    return {"clocks": clocks, "order-null": order_null, "gap-scale": gap_scale,
            "cv": cv, "pairwise": pairwise, "oa": oa, "thresholds": thresholds, "boot": boot}


def write_tables(tables: dict[str, list[dict[str, Any]]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "manuscript_tables.json").write_text(json.dumps(tables, indent=2, sort_keys=True) + "\n")
    for name, rows in tables.items():
        keys = sorted({key for row in rows for key in row})
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in keys})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures"))
    parser.add_argument("--out", type=Path, default=Path("generated_tables"))
    args = parser.parse_args(argv)
    write_tables(build_tables(args.fixtures), args.out)


if __name__ == "__main__":
    main()
