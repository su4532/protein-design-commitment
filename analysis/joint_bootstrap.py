#!/usr/bin/env python3
"""Joint cluster x rotation x design bootstrap for available (O) and readable (A) opportunity.

The adaptive-selection source record reported two intervals per measure:

  * primary_cluster_bootstrap        - resamples the 12 Foldseek clusters only, averaging all
                                       60 cross-fitting rotations inside every draw;
  * nested_split_design_sensitivity  - additionally samples one of 200 saved design replicates
                                       per outer draw, still averaging all 60 rotations.

Neither resamples the root rotations themselves, so both understate uncertainty relative to the
accepted analysis plan, which asks for clusters, root rotations, and saved design replicates
resampled jointly. The maintained implementation defines exactly that as `conservative_bootstrap`, but only
executes it for the readable measure A (`if measure == "A"` in its `main()`); the available
measure O has no joint interval anywhere in its outputs.

This script rebuilds the frozen O and A rotation and design-replicate arrays with the source record's
maintained opportunity implementation and runs `conservative_bootstrap` for BOTH measures, so the manuscript can quote a joint
interval for O on the same footing as A.

Provenance checks, all of which must pass:
  1. every rebuilt O curve reproduces the saved per-target child_share from the frozen curves;
  2. the rebuilt primary cluster bootstrap reproduces the source record's published O and A intervals;
  3. the rebuilt nested split/design sensitivity reproduces its published O and A intervals;
  4. the joint bootstrap for A reproduces the source record's published legacy_one_rotation_mixture,
     which is the same joint calculation and the manuscript's existing pre-registered A bound.

CPU only. Output: <artifact-root>/<output-subdir>/joint_opportunity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent


def load_opportunity_analysis():
    """Load the maintained opportunity implementation and record its digest."""
    from . import opportunity

    path = Path(opportunity.__file__).resolve()
    return opportunity, hashlib.sha256(path.read_bytes()).hexdigest()


def close(observed: float, expected: float, tol: float) -> bool:
    return bool(np.isfinite(observed) and np.isfinite(expected) and abs(observed - expected) <= tol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE.parent / "configs" / "opportunity.yaml"))
    parser.add_argument("--published", required=True, help="adaptive-opportunity result JSON")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-subdir", default="joint_opportunity")
    parser.add_argument("--qualities", default="", help="comma list; default = config qualities")
    parser.add_argument("--max-targets", type=int, default=0, help="smoke only: cap target count")
    parser.add_argument("--n-boot", type=int, default=0, help="smoke only: override draw count")
    args = parser.parse_args()

    az, implementation_digest = load_opportunity_analysis()
    config = yaml.safe_load(open(args.config))
    published = json.load(open(args.published))

    steps = tuple(int(x) for x in config["capture_steps"])
    deployed_steps = tuple(int(x) for x in config["deployed_steps"])
    qualities = tuple(
        x for x in (args.qualities.split(",") if args.qualities else config["qualities"]) if x
    )
    primary_quality = config["primary_quality"]
    n_partitions = int(config["root_partitions"])
    n_design = int(config["design_bootstrap_replicates"])
    n_boot = int(args.n_boot or config["panel_bootstrap_draws"])
    bootstrap_seed = int(config["panel_bootstrap_seed"])
    sources = {key: Path(value) for key, value in config["sources"].items()}
    smoke = bool(args.max_targets or args.n_boot)

    output_dir = Path(args.artifact_root) / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== phase: frozen input audit ===", flush=True)
    curve_files = sorted(sources["readable_curves"].glob("*_curves.json"))
    targets = sorted(path.name.removesuffix("_curves.json") for path in curve_files)
    reference_gain = json.load(open(sources["published_readable_gain"]))
    clusters = reference_gain["clusters"]
    if not smoke and (len(targets) != 19 or len(set(clusters.values())) != 12):
        raise AssertionError("expected 19 targets in 12 structural clusters")
    if args.max_targets:
        targets = targets[: args.max_targets]
    clusters = {target: clusters[target] for target in targets}
    print(
        f"targets={len(targets)} clusters={len(set(clusters.values()))} steps={len(steps)} "
        f"qualities={list(qualities)} n_boot={n_boot} smoke={smoke}",
        flush=True,
    )

    measures: dict[str, dict[str, dict[str, Any]]] = {
        quality: {measure: {"full": {}, "rotation": {}, "design_boot": {}} for measure in ("O", "A")}
        for quality in qualities
    }
    max_child_share_deviation = 0.0
    started = time.time()

    for target_index, target in enumerate(targets, start=1):
        curve = json.load(open(sources["readable_curves"] / f"{target}_curves.json"))
        jgroup = np.load(sources["readable_curves"] / f"{target}_Jgroup.npz")
        if curve["target"] != target or set(map(int, curve["steps"])) != set(steps):
            raise AssertionError(f"readable curve identity mismatch for {target}")

        o_rotation = {
            q: np.full((n_partitions, 3, 3, len(steps)), np.nan, dtype=float) for q in qualities
        }
        o_boot = {
            q: np.full((n_partitions, 3, 3, len(steps), n_design), np.nan, dtype=float)
            for q in qualities
        }
        o_full = {q: np.empty(len(steps), dtype=float) for q in qualities}
        root_reference: np.ndarray | None = None
        seed_base = az.stable_seed(f"threeway|{target}")
        uniforms = np.random.default_rng(az.stable_seed(f"dboot|{target}")).random((n_design, 16, 9))

        for step_index, step in enumerate(steps):
            labels = np.load(sources["labels"] / f"{target}_step{step}.npz")
            roots_now = np.unique(labels["root_idx"])
            if root_reference is None:
                root_reference = roots_now
            if not np.array_equal(roots_now, root_reference) or len(roots_now) != 16:
                raise AssertionError(f"root identities changed for {target} step {step}")
            partitions = [
                az.three_way_roots(root_reference, rep, seed_base) for rep in range(n_partitions)
            ]
            for quality in qualities:
                if quality == "designability":
                    y, root, child = (
                        labels["y_designability"],
                        labels["des_root"],
                        labels["des_child"],
                    )
                else:
                    y, root, child = labels[f"y_{quality}"], labels["root_idx"], labels["child_idx"]
                full_available = float(az.available_opportunity(az.as_rck(y, root, child)))
                o_full[quality][step_index] = full_available
                max_child_share_deviation = max(
                    max_child_share_deviation,
                    abs(full_available - float(curve["steps"][str(step)][quality]["child_share"])),
                )
                for rep, groups in enumerate(partitions):
                    for score_group in range(3):
                        rows = np.where(np.isin(root, groups[score_group]))[0]
                        nested = az.as_rck(y[rows], root[rows], child[rows])
                        value = float(az.available_opportunity(nested))
                        boot_values = az.available_opportunity(az.design_resamples(nested, uniforms))
                        for fit_group in range(3):
                            if fit_group == score_group:
                                continue
                            o_rotation[quality][rep, fit_group, score_group, step_index] = value
                            o_boot[quality][rep, fit_group, score_group, step_index] = boot_values

        for quality in qualities:
            measures[quality]["O"]["full"][target] = o_full[quality]
            measures[quality]["O"]["rotation"][target] = o_rotation[quality]
            measures[quality]["O"]["design_boot"][target] = o_boot[quality]
            measures[quality]["A"]["full"][target] = np.asarray(
                [curve["steps"][str(step)][quality]["J"] for step in steps], dtype=float
            )
            rotation = np.full((n_partitions, 3, 3, len(steps)), np.nan, dtype=float)
            for step_index, step in enumerate(steps):
                for pair, values in curve["steps"][str(step)][quality]["group_pairs"].items():
                    fit_group, score_group = (int(x) for x in pair.split("->"))
                    if len(values) != n_partitions:
                        raise AssertionError(f"root partition mismatch {target}/{quality}/{step}")
                    rotation[:, fit_group, score_group, step_index] = values
            boot = np.stack(
                [jgroup[f"{quality}|{step}"] for step in steps], axis=3
            ).astype(np.float32)
            if boot.shape != (n_partitions, 3, 3, len(steps), n_design):
                raise AssertionError(f"unexpected readable bootstrap shape {boot.shape}")
            measures[quality]["A"]["rotation"][target] = rotation
            measures[quality]["A"]["design_boot"][target] = boot
        print(
            f"audited {target_index:02d}/{len(targets)} {target} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    print("=== phase: joint cluster x rotation x design bootstrap ===", flush=True)
    shared_design_indices = np.random.default_rng(bootstrap_seed + 1).integers(
        0, n_design, size=n_boot
    )
    results: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    draw_arrays: dict[str, np.ndarray] = {}

    for quality in qualities:
        results[quality] = {}
        for measure in ("O", "A"):
            payload = measures[quality][measure]
            primary, primary_draws = az.primary_cluster_bootstrap(
                payload["rotation"], clusters, steps, deployed_steps, n_boot, bootstrap_seed
            )
            nested, _ = az.nested_split_design_sensitivity(
                payload["design_boot"],
                clusters,
                steps,
                deployed_steps,
                primary_draws["cluster_draw_indices"],
                shared_design_indices,
            )
            joint, joint_draws = az.conservative_bootstrap(
                payload["rotation"],
                payload["design_boot"],
                clusters,
                steps,
                deployed_steps,
                n_boot,
                bootstrap_seed,
            )
            results[quality][measure] = {
                "name": "available opportunity" if measure == "O" else "readable opportunity",
                "cluster_only": {
                    key: primary[key]
                    for key in ("adaptive", "deployed", "gain_adaptive_vs_deployed")
                },
                "cluster_and_design": {
                    key: nested[key]
                    for key in ("adaptive", "deployed", "gain_adaptive_vs_deployed")
                },
                "joint_cluster_rotation_design": {
                    key: value
                    for key, value in joint.items()
                    if key
                    in (
                        "adaptive",
                        "deployed",
                        "gain_adaptive_vs_deployed",
                        "n_boot",
                        "n_design_boot",
                        "n_rotations",
                        "bootstrap_reselects_comparators",
                        "bootstrap_selected_deployed_step_counts",
                        "bootstrap_selected_global_step_counts",
                    )
                },
                "joint_assessment": az.assessment(joint["gain_adaptive_vs_deployed"]),
            }
            for key, values in joint_draws.items():
                draw_arrays[f"{quality}|{measure}|joint|{key}"] = np.asarray(values, dtype=float)

            if not smoke:
                pub = published["measures"][quality][measure]
                for label, rebuilt, reference in (
                    ("primary_cluster_bootstrap", primary, pub["primary_cluster_bootstrap"]),
                    (
                        "nested_split_design_sensitivity",
                        nested,
                        pub["nested_split_design_sensitivity"],
                    ),
                ):
                    for metric in ("adaptive", "deployed", "gain_adaptive_vs_deployed"):
                        for stat in ("mean", "lo", "hi", "p_gt_0"):
                            checks.append(
                                {
                                    "check": f"{quality}|{measure}|{label}|{metric}|{stat}",
                                    "observed": float(rebuilt[metric][stat]),
                                    "expected": float(reference[metric][stat]),
                                    "pass": close(
                                        float(rebuilt[metric][stat]),
                                        float(reference[metric][stat]),
                                        1e-9,
                                    ),
                                }
                            )
                if measure == "A" and pub.get("legacy_one_rotation_mixture"):
                    reference = pub["legacy_one_rotation_mixture"]
                    for metric in ("adaptive", "deployed", "gain_adaptive_vs_deployed"):
                        for stat in ("mean", "lo", "hi", "p_gt_0"):
                            checks.append(
                                {
                                    "check": f"{quality}|A|joint_reproduces_published|{metric}|{stat}",
                                    "observed": float(joint[metric][stat]),
                                    "expected": float(reference[metric][stat]),
                                    "pass": close(
                                        float(joint[metric][stat]),
                                        float(reference[metric][stat]),
                                        1e-12,
                                    ),
                                }
                            )
                if measure == "O":
                    for target in targets:
                        pub_curve = pub["rotation"]["per_target"][target]["full_curve"]
                        rebuilt_curve = payload["full"][target]
                        for step_index, step in enumerate(steps):
                            checks.append(
                                {
                                    "check": f"{quality}|O|full_curve|{target}|{step}",
                                    "observed": float(rebuilt_curve[step_index]),
                                    "expected": float(pub_curve[str(step)]),
                                    "pass": close(
                                        float(rebuilt_curve[step_index]),
                                        float(pub_curve[str(step)]),
                                        1e-12,
                                    ),
                                }
                            )

            gain = joint["gain_adaptive_vs_deployed"]
            print(
                f"{quality:24s} {measure} JOINT {gain['mean']:+.6f} "
                f"[{gain['lo']:+.6f}, {gain['hi']:+.6f}] p>0={gain['p_gt_0']:.4f} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

    n_fail = sum(1 for row in checks if not row["pass"])
    payload = {
        "schema_version": 1,
        "question": (
            "What are the available (O) and readable (A) target-adaptive timing gains when "
            "structural clusters, root rotations, and saved design replicates are resampled "
            "jointly, as the accepted analysis plan specifies?"
        ),
        "source_record": "adaptive-selection",
        "maintained_implementation": {
            "path": "analysis/opportunity.py",
            "sha256": implementation_digest,
            "function": "conservative_bootstrap",
            "note": (
                "byte-identical copy of the source record's implementation of record; that experiment "
                "runs conservative_bootstrap only for measure A, so O had no joint interval"
            ),
        },
        "method": {
            "resampling_units": [
                "12 Foldseek structural clusters (with replacement)",
                "1 of 60 three-way root rotations per draw",
                "1 of 200 saved design replicates per draw",
            ],
            "comparator": "deployed step reselected from the sampled cluster selection curves in every draw",
            "deployed_steps": list(deployed_steps),
            "capture_steps": list(steps),
            "n_boot": n_boot,
            "seed": bootstrap_seed,
            "paired_O_and_A": (
                "both measures consume the identical draw sequence (same seed, same draw counts)"
            ),
            "contrast": (
                "cluster_only and cluster_and_design average all 60 rotations inside each draw; "
                "joint_cluster_rotation_design samples one rotation per draw"
            ),
        },
        "scope": {
            "targets": targets,
            "n_targets": len(targets),
            "clusters": clusters,
            "n_structural_clusters": len(set(clusters.values())),
            "primary_quality": primary_quality,
            "smoke": smoke,
        },
        "measures": results,
        "input_audit": {
            "max_abs_full_O_vs_saved_child_share": float(max_child_share_deviation),
            "n_checks": len(checks),
            "n_failed": int(n_fail),
        },
        "checks": checks,
        "runtime_seconds": float(time.time() - started),
    }
    out_path = output_dir / ("joint_opportunity_smoke.json" if smoke else "joint_opportunity.json")
    json.dump(payload, open(out_path, "w"), indent=1)
    if not smoke:
        np.savez_compressed(output_dir / "joint_bootstrap_draws.npz", **draw_arrays)
    print(f"wrote {out_path}", flush=True)
    print(
        f"checks {len(checks) - n_fail}/{len(checks)} pass; "
        f"max |O - saved child_share| = {max_child_share_deviation:.3g}",
        flush=True,
    )
    if not smoke:
        pq = results[primary_quality]
        for measure in ("O", "A"):
            row = pq[measure]["joint_cluster_rotation_design"]["gain_adaptive_vs_deployed"]
            print(
                f"PRIMARY {measure} joint {row['mean']:+.6f} [{row['lo']:+.6f}, {row['hi']:+.6f}] "
                f"-> {pq[measure]['joint_assessment']}",
                flush=True,
            )
    if n_fail:
        print(f"FAIL: {n_fail} provenance checks failed", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
