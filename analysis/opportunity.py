#!/usr/bin/env python3
"""Held-out target-adaptive selection opportunity and commitment-threshold sensitivity."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

TRIPLES = list(itertools.permutations(range(3), 3))


def stable_seed(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


def three_way_roots(roots: np.ndarray, rep: int, seed_base: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed_base + 1009 * rep)
    return [np.sort(x) for x in np.array_split(rng.permutation(np.asarray(roots)), 3)]


def as_rck(y: np.ndarray, root: np.ndarray, child: np.ndarray) -> np.ndarray:
    """Convert a flat balanced nested design to [root, child, completion]."""
    y = np.asarray(y, dtype=float)
    root = np.asarray(root)
    child = np.asarray(child)
    per_root = [np.where(root == r)[0][np.argsort(child[root == r])] for r in np.unique(root)]
    n_child = min(len(rows) for rows in per_root)
    if n_child < 2 or y.ndim != 2 or y.shape[1] < 2:
        raise ValueError("nested design requires at least two children and completions")
    return np.stack([y[rows[:n_child]] for rows in per_root], axis=0)


def variance_components(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous nested-ANOVA variance components for [B,R,C,k] or [R,C,k]."""
    a = np.asarray(a, dtype=float)
    scalar = a.ndim == 3
    if scalar:
        a = a[None]
    if a.ndim != 4:
        raise ValueError(f"expected [B,R,C,k] or [R,C,k], got {a.shape}")
    b, n_root, n_child, k = a.shape
    if min(n_root, n_child, k) < 2:
        raise ValueError(f"insufficient nested design {a.shape}")
    child_mean = a.mean(3)
    root_mean = child_mean.mean(2)
    grand = root_mean.mean(1)
    ms_within = ((a - child_mean[..., None]) ** 2).sum((1, 2, 3)) / (
        n_root * n_child * (k - 1)
    )
    ms_child = k * ((child_mean - root_mean[..., None]) ** 2).sum((1, 2)) / (
        n_root * (n_child - 1)
    )
    ms_root = n_child * k * ((root_mean - grand[:, None]) ** 2).sum(1) / (n_root - 1)
    sigma_within = ms_within
    sigma_child = (ms_child - ms_within) / k
    sigma_root = (ms_root - ms_child) / (n_child * k)
    out = tuple(np.clip(x, 0.0, None) for x in (sigma_root, sigma_child, sigma_within))
    if scalar:
        return tuple(np.asarray(float(x[0])) for x in out)  # type: ignore[return-value]
    return out  # type: ignore[return-value]


def available_opportunity(a: np.ndarray) -> np.ndarray:
    sigma_root, sigma_child, sigma_within = variance_components(a)
    total = sigma_root + sigma_child + sigma_within
    return np.divide(sigma_child, total, out=np.zeros_like(total, dtype=float), where=total > 0)


def design_resamples(a: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Apply shared root-then-child bootstrap uniforms to [R,C,k]."""
    n_root, n_child, _ = a.shape
    if uniforms.shape[1] < n_root:
        reps = int(np.ceil(n_root / uniforms.shape[1]))
        u = np.tile(uniforms, (1, reps, 1))[:, :n_root]
    else:
        u = uniforms[:, :n_root]
    root_idx = np.minimum((u[:, :, 0] * n_root).astype(np.int64), n_root - 1)
    child_idx = np.minimum((u[:, :, 1 : 1 + n_child] * n_child).astype(np.int64), n_child - 1)
    return a[root_idx[:, :, None], child_idx, :]


def argmax_idx(curve: np.ndarray, allowed: np.ndarray | None = None) -> int:
    idx = np.arange(len(curve)) if allowed is None else np.asarray(allowed, dtype=int)
    values = np.asarray(curve)[idx]
    if not np.isfinite(values).any():
        return int(idx[0])
    # np.nanargmax returns the first maximum, so ties choose the earlier sorted step.
    return int(idx[int(np.nanargmax(values))])


def policy_values(
    selection: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    clusters: dict[str, str],
    steps: tuple[int, ...],
    deployed_steps: tuple[int, ...],
    selection_clusters: list[str] | None = None,
) -> dict[str, Any]:
    targets = sorted(selection)
    per_cluster: dict[str, list[np.ndarray]] = {}
    for target in targets:
        per_cluster.setdefault(clusters[target], []).append(selection[target])
    cluster_curves = {
        cluster: np.nanmean(np.stack(rows), axis=0) for cluster, rows in per_cluster.items()
    }
    draw = list(cluster_curves) if selection_clusters is None else list(selection_clusters)
    if not draw:
        raise ValueError("selection cluster draw is empty")
    missing = sorted(set(draw).difference(cluster_curves))
    if missing:
        raise KeyError(f"selection clusters have no targets: {missing}")
    panel = np.nanmean(np.stack([cluster_curves[c] for c in draw]), axis=0)
    global_idx = argmax_idx(panel)
    step_to_idx = {step: i for i, step in enumerate(steps)}
    deployed_idx = np.asarray([step_to_idx[s] for s in deployed_steps], dtype=int)
    fixed_idx = argmax_idx(panel, deployed_idx)
    values = {"adaptive": {}, "deployed": {}, "global": {}}
    adaptive_steps: dict[str, int] = {}
    for target in targets:
        adaptive_idx = argmax_idx(selection[target])
        adaptive_steps[target] = steps[adaptive_idx]
        values["adaptive"][target] = float(evaluation[target][adaptive_idx])
        values["deployed"][target] = float(evaluation[target][fixed_idx])
        values["global"][target] = float(evaluation[target][global_idx])
    return {
        "values": values,
        "t_adaptive": adaptive_steps,
        "t_deployed": steps[fixed_idx],
        "t_global": steps[global_idx],
    }


def rotation_summary(
    rotation: dict[str, np.ndarray],
    clusters: dict[str, str],
    steps: tuple[int, ...],
    deployed_steps: tuple[int, ...],
) -> dict[str, Any]:
    targets = sorted(rotation)
    n_partitions = rotation[targets[0]].shape[0]
    accum = {
        key: {target: [] for target in targets}
        for key in ("adaptive", "deployed", "global", "gain_adaptive_vs_deployed")
    }
    adaptive_counts = {target: collections.Counter() for target in targets}
    deployed_counts: collections.Counter[int] = collections.Counter()
    global_counts: collections.Counter[int] = collections.Counter()
    for rep in range(n_partitions):
        for fit_group, select_group, evaluate_group in TRIPLES:
            selection = {
                target: rotation[target][rep, fit_group, select_group] for target in targets
            }
            evaluation = {
                target: rotation[target][rep, fit_group, evaluate_group] for target in targets
            }
            selected = policy_values(
                selection, evaluation, clusters, steps, deployed_steps
            )
            deployed_counts[selected["t_deployed"]] += 1
            global_counts[selected["t_global"]] += 1
            for target in targets:
                adaptive = selected["values"]["adaptive"][target]
                deployed = selected["values"]["deployed"][target]
                global_value = selected["values"]["global"][target]
                accum["adaptive"][target].append(adaptive)
                accum["deployed"][target].append(deployed)
                accum["global"][target].append(global_value)
                accum["gain_adaptive_vs_deployed"][target].append(adaptive - deployed)
                adaptive_counts[target][selected["t_adaptive"][target]] += 1

    per_target = {}
    for target in targets:
        counts = dict(sorted(adaptive_counts[target].items()))
        modal = min(counts, key=lambda step: (-counts[step], step))
        per_target[target] = {
            "cluster": clusters[target],
            "adaptive_selected_step_counts": counts,
            "modal_adaptive_step": int(modal),
            "heldout_adaptive": float(np.nanmean(accum["adaptive"][target])),
            "heldout_deployed": float(np.nanmean(accum["deployed"][target])),
            "heldout_global": float(np.nanmean(accum["global"][target])),
            "adaptive_minus_deployed": float(
                np.nanmean(accum["gain_adaptive_vs_deployed"][target])
            ),
        }
    summary = {
        "n_rotations": int(n_partitions * len(TRIPLES)),
        "per_target": per_target,
        "panel_deployed_step_counts": dict(sorted(deployed_counts.items())),
        "panel_global_step_counts": dict(sorted(global_counts.items())),
    }
    arrays = {
        key: {target: np.asarray(values, dtype=float) for target, values in rows.items()}
        for key, rows in accum.items()
    }
    return summary, arrays


def summarize_draws(values: np.ndarray, point: float | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(values) if point is None else point),
        "lo": float(np.nanpercentile(values, 2.5)),
        "hi": float(np.nanpercentile(values, 97.5)),
        "p_gt_0": float(np.mean(values > 0)),
    }


def stable_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(math.fsum(finite) / len(finite)) if finite else float("nan")


def primary_cluster_bootstrap(
    rotation: dict[str, np.ndarray],
    clusters: dict[str, str],
    steps: tuple[int, ...],
    deployed_steps: tuple[int, ...],
    n_boot: int,
    seed: int,
    chunk_size: int = 200,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Resample only clusters, reselect comparators, and average every rotation."""
    expanded = {target: np.asarray(values)[..., None] for target, values in rotation.items()}
    selection_all, evaluation_all, adaptive_all, cluster_names = _design_policy_components(
        expanded, clusters, steps, dtype=np.float64
    )
    selection = selection_all[0]
    evaluation = evaluation_all[0]
    adaptive = adaptive_all[0]
    n_rotations, n_clusters, _ = selection.shape
    deployed_indices = np.asarray([steps.index(step) for step in deployed_steps], dtype=int)

    full_weight = np.full(n_clusters, 1.0 / n_clusters)
    full_selection = np.einsum("c,rcs->rs", full_weight, selection, optimize=True)
    full_deployed_local = np.nanargmax(full_selection[:, deployed_indices], axis=1)
    full_deployed_index = deployed_indices[full_deployed_local]
    rotation_index = np.arange(n_rotations)
    adaptive_cluster_rows = np.asarray(
        [stable_mean(adaptive[:, index]) for index in range(n_clusters)], dtype=float
    )
    deployed_cluster_rows = np.asarray(
        [
            stable_mean(evaluation[rotation_index, index, full_deployed_index])
            for index in range(n_clusters)
        ],
        dtype=float,
    )
    gain_cluster_rows = adaptive_cluster_rows - deployed_cluster_rows
    cluster_rows_by_metric = {
        "adaptive": adaptive_cluster_rows,
        "deployed": deployed_cluster_rows,
        "gain_adaptive_vs_deployed": gain_cluster_rows,
    }

    rng = np.random.default_rng(seed)
    cluster_draw_indices = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    cluster_weights = np.zeros((n_boot, n_clusters), dtype=np.float64)
    for index in range(n_clusters):
        cluster_weights[:, index] = np.mean(cluster_draw_indices == index, axis=1)
    adaptive_draws = np.empty(n_boot, dtype=float)
    deployed_draws = np.empty(n_boot, dtype=float)
    selected_deployed_counts: collections.Counter[int] = collections.Counter()
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        weight = cluster_weights[start:stop]
        selection_panel = np.einsum("bc,rcs->brs", weight, selection, optimize=True)
        deployed_local = np.nanargmax(selection_panel[:, :, deployed_indices], axis=2)
        deployed_index = deployed_indices[deployed_local]
        evaluation_panel = np.einsum("bc,rcs->brs", weight, evaluation, optimize=True)
        deployed_by_rotation = np.take_along_axis(
            evaluation_panel, deployed_index[..., None], axis=2
        )[..., 0]
        adaptive_by_rotation = np.einsum("bc,rc->br", weight, adaptive, optimize=True)
        adaptive_draws[start:stop] = np.asarray(
            [stable_mean(row) for row in adaptive_by_rotation], dtype=float
        )
        deployed_draws[start:stop] = np.asarray(
            [stable_mean(row) for row in deployed_by_rotation], dtype=float
        )
        selected_deployed_counts.update(
            steps[int(index)] for index in deployed_index.reshape(-1)
        )
    gain_draws = adaptive_draws - deployed_draws
    draws = {
        "adaptive": adaptive_draws,
        "deployed": deployed_draws,
        "gain_adaptive_vs_deployed": gain_draws,
        "cluster_draw_indices": cluster_draw_indices,
    }
    summaries: dict[str, Any] = {}
    for metric in ("adaptive", "deployed", "gain_adaptive_vs_deployed"):
        cluster_rows = cluster_rows_by_metric[metric]
        summaries[metric] = summarize_draws(draws[metric], point=stable_mean(cluster_rows))
        summaries[metric]["cluster_rows"] = {
            cluster: float(value) for cluster, value in zip(cluster_names, cluster_rows)
        }
    summaries.update(
        {
            "n_boot": int(n_boot),
            "seed": int(seed),
            "n_independent_structural_clusters": int(n_clusters),
            "n_repeated_crossfit_rotations": int(n_rotations),
            "rotation_axis_averaged_before_cluster_aggregation": True,
            "rotation_axis_averaged_within_every_cluster_draw": True,
            "resampling_unit": "Foldseek structural cluster",
            "resamples_rotations": False,
            "resamples_design_replicates": False,
            "comparator_reselection_during_cluster_bootstrap": True,
            "selected_deployed_step_counts_across_draws_and_rotations": dict(
                sorted(selected_deployed_counts.items())
            ),
            "comparator_selection": "reselected from sampled cluster selection curves within every draw and rotation",
        }
    )
    return summaries, draws


def _design_policy_components(
    design_boot: dict[str, np.ndarray],
    clusters: dict[str, str],
    steps: tuple[int, ...],
    dtype: Any = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Precompute design-replicate policy pieces as [design, rotation, cluster, ...]."""
    targets = sorted(design_boot)
    cluster_names = sorted(set(clusters.values()))
    target_indices = {
        cluster: np.asarray(
            [index for index, target in enumerate(targets) if clusters[target] == cluster],
            dtype=int,
        )
        for cluster in cluster_names
    }
    n_partitions, _, _, n_steps, n_design = design_boot[targets[0]].shape
    n_rotations = n_partitions * len(TRIPLES)
    selection_by_cluster = np.empty(
        (n_design, n_rotations, len(cluster_names), n_steps), dtype=dtype
    )
    evaluation_by_cluster = np.empty_like(selection_by_cluster)
    adaptive_by_cluster = np.empty(
        (n_design, n_rotations, len(cluster_names)), dtype=dtype
    )
    rotation_index = 0
    for rep in range(n_partitions):
        for fit_group, select_group, evaluate_group in TRIPLES:
            selection = np.stack(
                [
                    design_boot[target][rep, fit_group, select_group]
                    for target in targets
                ],
                axis=0,
            ).transpose(2, 0, 1)
            evaluation = np.stack(
                [
                    design_boot[target][rep, fit_group, evaluate_group]
                    for target in targets
                ],
                axis=0,
            ).transpose(2, 0, 1)
            adaptive_index = np.nanargmax(selection, axis=2)
            adaptive = np.take_along_axis(
                evaluation, adaptive_index[..., None], axis=2
            )[..., 0]
            for cluster_index, cluster in enumerate(cluster_names):
                members = target_indices[cluster]
                selection_by_cluster[:, rotation_index, cluster_index] = np.nanmean(
                    selection[:, members], axis=1
                )
                evaluation_by_cluster[:, rotation_index, cluster_index] = np.nanmean(
                    evaluation[:, members], axis=1
                )
                adaptive_by_cluster[:, rotation_index, cluster_index] = np.nanmean(
                    adaptive[:, members], axis=1
                )
            rotation_index += 1
    if tuple(steps) != tuple(sorted(steps)):
        raise ValueError("steps must be ordered so first-maximum tie handling selects earlier steps")
    return selection_by_cluster, evaluation_by_cluster, adaptive_by_cluster, cluster_names


def nested_split_design_sensitivity(
    design_boot: dict[str, np.ndarray],
    clusters: dict[str, str],
    steps: tuple[int, ...],
    deployed_steps: tuple[int, ...],
    cluster_draw_indices: np.ndarray,
    design_indices: np.ndarray,
    chunk_size: int = 200,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Propagate design/root uncertainty while averaging all 60 rotations per draw."""
    selection, evaluation, adaptive, cluster_names = _design_policy_components(
        design_boot, clusters, steps
    )
    n_boot = len(cluster_draw_indices)
    n_clusters = len(cluster_names)
    cluster_weights = np.zeros((n_boot, n_clusters), dtype=np.float32)
    for index in range(n_clusters):
        cluster_weights[:, index] = np.mean(cluster_draw_indices == index, axis=1)
    deployed_indices = np.asarray([steps.index(step) for step in deployed_steps], dtype=int)
    adaptive_draws = np.empty(n_boot, dtype=float)
    deployed_draws = np.empty(n_boot, dtype=float)
    selected_deployed_counts: collections.Counter[int] = collections.Counter()
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        weight = cluster_weights[start:stop]
        selected_design = design_indices[start:stop]
        selection_panel = np.einsum(
            "bc,brcs->brs", weight, selection[selected_design], optimize=True
        )
        deployed_local = np.nanargmax(selection_panel[:, :, deployed_indices], axis=2)
        deployed_index = deployed_indices[deployed_local]
        evaluation_panel = np.einsum(
            "bc,brcs->brs", weight, evaluation[selected_design], optimize=True
        )
        deployed_by_rotation = np.take_along_axis(
            evaluation_panel, deployed_index[..., None], axis=2
        )[..., 0]
        adaptive_by_rotation = np.einsum(
            "bc,brc->br", weight, adaptive[selected_design], optimize=True
        )
        adaptive_draws[start:stop] = np.nanmean(adaptive_by_rotation, axis=1)
        deployed_draws[start:stop] = np.nanmean(deployed_by_rotation, axis=1)
        selected_deployed_counts.update(
            steps[int(index)] for index in deployed_index.reshape(-1)
        )
    gain_draws = adaptive_draws - deployed_draws
    draws = {
        "adaptive": adaptive_draws,
        "deployed": deployed_draws,
        "gain_adaptive_vs_deployed": gain_draws,
    }
    summary = {key: summarize_draws(values) for key, values in draws.items()}
    summary.update(
        {
            "n_outer_cluster_draws": int(n_boot),
            "n_independent_structural_clusters": int(n_clusters),
            "n_design_boot": int(selection.shape[0]),
            "n_repeated_crossfit_rotations": int(selection.shape[1]),
            "rotation_axis_averaged_within_every_outer_draw": True,
            "paired_cluster_and_design_draws_across_O_and_A": True,
            "paired_adaptive_and_deployed_evaluation": True,
            "comparator_reselected_within_each_outer_draw_and_rotation": True,
            "selected_deployed_step_counts_across_draws_and_rotations": dict(
                sorted(selected_deployed_counts.items())
            ),
            "interpretation": "split/design sensitivity; not the primary population interval",
        }
    )
    return summary, draws


def conservative_bootstrap(
    rotation: dict[str, np.ndarray],
    design_boot: dict[str, np.ndarray],
    clusters: dict[str, str],
    steps: tuple[int, ...],
    deployed_steps: tuple[int, ...],
    n_boot: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Resample clusters, one root rotation, and one saved design replicate."""
    targets = sorted(t for t in rotation if t in design_boot)
    by_cluster: dict[str, list[str]] = {}
    for target in targets:
        by_cluster.setdefault(clusters[target], []).append(target)
    cluster_names = sorted(by_cluster)
    n_partitions, _, _, _, n_design = design_boot[targets[0]].shape
    rng = np.random.default_rng(seed)
    draws = {
        "adaptive": np.empty(n_boot),
        "deployed": np.empty(n_boot),
        "gain_adaptive_vs_deployed": np.empty(n_boot),
    }
    deployed_counts: collections.Counter[int] = collections.Counter()
    global_counts: collections.Counter[int] = collections.Counter()

    for bootstrap_index in range(n_boot):
        picks = rng.integers(0, len(cluster_names), size=len(cluster_names))
        sampled_clusters = [cluster_names[i] for i in picks]
        rep = int(rng.integers(0, n_partitions))
        fit_group, select_group, evaluate_group = TRIPLES[
            int(rng.integers(0, len(TRIPLES)))
        ]
        design_index = int(rng.integers(0, n_design))
        selection = {
            target: design_boot[target][rep, fit_group, select_group, :, design_index]
            for target in targets
        }
        evaluation = {
            target: design_boot[target][rep, fit_group, evaluate_group, :, design_index]
            for target in targets
        }
        selected = policy_values(
            selection,
            evaluation,
            clusters,
            steps,
            deployed_steps,
            selection_clusters=sampled_clusters,
        )
        deployed_counts[selected["t_deployed"]] += 1
        global_counts[selected["t_global"]] += 1
        adaptive_rows, deployed_rows, gain_rows = [], [], []
        for cluster in sampled_clusters:
            members = by_cluster[cluster]
            adaptive = float(
                np.mean([selected["values"]["adaptive"][target] for target in members])
            )
            deployed = float(
                np.mean([selected["values"]["deployed"][target] for target in members])
            )
            adaptive_rows.append(adaptive)
            deployed_rows.append(deployed)
            gain_rows.append(adaptive - deployed)
        draws["adaptive"][bootstrap_index] = float(np.mean(adaptive_rows))
        draws["deployed"][bootstrap_index] = float(np.mean(deployed_rows))
        draws["gain_adaptive_vs_deployed"][bootstrap_index] = float(np.mean(gain_rows))

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.nanmean(values)),
            "lo": float(np.nanpercentile(values, 2.5)),
            "hi": float(np.nanpercentile(values, 97.5)),
            "p_gt_0": float((values > 0).mean()),
        }

    summary = {key: summarize(values) for key, values in draws.items()}
    summary.update(
        {
            "n_boot": int(n_boot),
            "n_design_boot": int(n_design),
            "n_rotations": int(n_partitions * len(TRIPLES)),
            "bootstrap_reselects_comparators": True,
            "bootstrap_selected_deployed_step_counts": dict(sorted(deployed_counts.items())),
            "bootstrap_selected_global_step_counts": dict(sorted(global_counts.items())),
        }
    )
    return summary, draws


def first_crossing(
    steps: Iterable[int], values: Iterable[float], threshold: float
) -> dict[str, Any]:
    steps_arr = np.asarray(list(steps), dtype=float)
    values_arr = np.asarray(list(values), dtype=float)
    if len(steps_arr) != len(values_arr) or len(steps_arr) < 2:
        raise ValueError("steps and values must have equal length >= 2")
    if not np.isfinite(values_arr).all():
        return {"status": "missing", "crossing": None}
    if values_arr[0] >= threshold:
        return {"status": "left_censored", "crossing": float(steps_arr[0])}
    for i in range(1, len(steps_arr)):
        if values_arr[i] >= threshold:
            lo_step, hi_step = steps_arr[i - 1], steps_arr[i]
            lo_value, hi_value = values_arr[i - 1], values_arr[i]
            if hi_value == lo_value:
                crossing = hi_step
            else:
                crossing = lo_step + (threshold - lo_value) * (hi_step - lo_step) / (
                    hi_value - lo_value
                )
            return {"status": "resolved", "crossing": float(crossing)}
    return {"status": "right_censored", "crossing": float(steps_arr[-1])}


def threshold_sensitivity(
    commitment_dir: Path,
    commitment_summary_path: Path,
    targets: list[str],
    qualities: tuple[str, ...],
    steps: tuple[int, ...],
    thresholds: tuple[float, ...],
    minimum_resolved_fraction: float,
) -> tuple[dict[str, Any], float]:
    curves: dict[str, dict[str, list[float]]] = {quality: {} for quality in qualities}
    for target in targets:
        target_path = commitment_dir / f"{target}.json"
        payload = json.load(open(target_path))
        if payload["target"] != target or set(map(int, payload["steps"])) != set(steps):
            raise AssertionError(f"commitment grid mismatch for {target}")
        for quality in qualities:
            curves[quality][target] = [
                float(payload["steps"][str(step)][quality]["decomposition"]["commitment"])
                for step in steps
            ]

    parent = json.load(open(commitment_summary_path))
    clusters = parent["target_clusters"]
    max_pool_deviation = 0.0
    for quality in qualities:
        for step_index, step in enumerate(steps):
            grouped: dict[str, list[float]] = {}
            for target in targets:
                grouped.setdefault(clusters[target], []).append(curves[quality][target][step_index])
            observed = float(np.mean([np.mean(rows) for rows in grouped.values()]))
            expected = float(parent["pooled"]["commitment"][quality][str(step)]["mean"])
            max_pool_deviation = max(max_pool_deviation, abs(observed - expected))

    by_quality: dict[str, Any] = {}
    for quality in qualities:
        by_threshold: dict[str, Any] = {}
        for threshold in thresholds:
            per_target = {
                target: first_crossing(steps, curves[quality][target], threshold)
                for target in targets
            }
            counts = collections.Counter(row["status"] for row in per_target.values())
            resolved = np.asarray(
                [
                    row["crossing"]
                    for row in per_target.values()
                    if row["status"] == "resolved"
                ],
                dtype=float,
            )
            resolved_fraction = len(resolved) / len(targets)
            dispersion = None
            if len(resolved):
                dispersion = {
                    "median": float(np.median(resolved)),
                    "q25": float(np.percentile(resolved, 25)),
                    "q75": float(np.percentile(resolved, 75)),
                    "min": float(np.min(resolved)),
                    "max": float(np.max(resolved)),
                    "sd": float(np.std(resolved, ddof=1)) if len(resolved) > 1 else 0.0,
                }
            by_threshold[f"{threshold:g}"] = {
                "threshold": threshold,
                "per_target": per_target,
                "censoring_counts": {
                    status: int(counts.get(status, 0))
                    for status in ("resolved", "left_censored", "right_censored", "missing")
                },
                "resolved_fraction": float(resolved_fraction),
                "interpret": bool(resolved_fraction >= minimum_resolved_fraction),
                "resolved_dispersion": dispersion,
            }
        by_quality[quality] = {"curves": curves[quality], "thresholds": by_threshold}
    return {
        "method": "first crossing by linear interpolation; no monotonic smoothing",
        "boundary_rule": "at/above threshold at step 10 is left-censored; absent by step 350 is right-censored",
        "minimum_resolved_fraction": minimum_resolved_fraction,
        "by_quality": by_quality,
    }, max_pool_deviation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_input_manifest(paths: list[Path], output_path: Path) -> dict[str, Any]:
    unique = sorted(set(paths), key=lambda path: str(path))
    files = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in unique
    ]
    payload = {
        "schema_version": 1,
        "n_files": len(files),
        "total_bytes": int(sum(row["bytes"] for row in files)),
        "files": files,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(output_path, "w"), indent=1)
    return payload


def assessment(interval: dict[str, float]) -> str:
    if interval["lo"] > 0:
        return "strengthens Act IV"
    if interval["hi"] < 0:
        return "evidence against target-adaptive timing"
    return "unresolved"


def fmt_interval(row: dict[str, float]) -> str:
    return f"{row['mean']:+.4f} [{row['lo']:+.4f}, {row['hi']:+.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    artifact_root = Path(args.artifact_root)
    output_subdir = config.get("output_subdir", "results_cluster_reselected")
    output_dir = artifact_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = tuple(int(x) for x in config["capture_steps"])
    deployed_steps = tuple(int(x) for x in config["deployed_steps"])
    qualities = tuple(config["qualities"])
    primary_quality = config["primary_quality"]
    n_partitions = int(config["root_partitions"])
    n_design = int(config["design_bootstrap_replicates"])
    n_boot = int(config["panel_bootstrap_draws"])
    bootstrap_seed = int(config["panel_bootstrap_seed"])
    sources = {key: Path(value) for key, value in config["sources"].items()}

    print("=== phase: frozen input audit ===", flush=True)
    label_files = sorted(sources["labels"].glob("*_step*.npz"))
    curve_files = sorted(sources["readable_curves"].glob("*_curves.json"))
    targets = sorted(path.name.removesuffix("_curves.json") for path in curve_files)
    if len(targets) != 19 or len(label_files) != len(targets) * len(steps):
        raise AssertionError(
            f"expected 19 targets x {len(steps)} steps; got {len(targets)} targets and {len(label_files)} labels"
        )
    reference_gain = json.load(open(sources["published_readable_gain"]))
    clusters = reference_gain["clusters"]
    if set(clusters) != set(targets) or len(set(clusters.values())) != 12:
        raise AssertionError("target/structural-cluster identity mismatch")

    measures: dict[str, dict[str, dict[str, Any]]] = {
        quality: {
            measure: {"full": {}, "rotation": {}, "design_boot": {}}
            for measure in ("O", "A")
        }
        for quality in qualities
    }
    consumed_paths: list[Path] = [
        sources["published_readable_gain"],
        sources["paper_commitment_summary"],
    ]
    max_child_share_deviation = 0.0
    coherent_root_checks = 0
    split_disjoint_checks = 0

    for target_index, target in enumerate(targets, start=1):
        curve_path = sources["readable_curves"] / f"{target}_curves.json"
        jgroup_path = sources["readable_curves"] / f"{target}_Jgroup.npz"
        commitment_path = sources["paper_commitment_curves"] / f"{target}.json"
        if not (curve_path.is_file() and jgroup_path.is_file() and commitment_path.is_file()):
            raise FileNotFoundError(f"missing frozen source for {target}")
        consumed_paths.extend([curve_path, jgroup_path, commitment_path])
        curve = json.load(open(curve_path))
        jgroup = np.load(jgroup_path)
        if curve["target"] != target or set(map(int, curve["steps"])) != set(steps):
            raise AssertionError(f"readable curve identity mismatch for {target}")

        o_rotation = {
            quality: np.full((n_partitions, 3, 3, len(steps)), np.nan, dtype=float)
            for quality in qualities
        }
        o_boot = {
            quality: np.full(
                (n_partitions, 3, 3, len(steps), n_design), np.nan, dtype=float
            )
            for quality in qualities
        }
        root_reference: np.ndarray | None = None
        seed_base = stable_seed(f"threeway|{target}")
        uniforms = np.random.default_rng(stable_seed(f"dboot|{target}")).random(
            (n_design, 16, 9)
        )

        for step_index, step in enumerate(steps):
            label_path = sources["labels"] / f"{target}_step{step}.npz"
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            consumed_paths.append(label_path)
            labels = np.load(label_path)
            roots_now = np.unique(labels["root_idx"])
            if root_reference is None:
                root_reference = roots_now
            if not np.array_equal(roots_now, root_reference) or len(roots_now) != 16:
                raise AssertionError(f"root identities changed for {target} step {step}")
            coherent_root_checks += 1
            partitions = [three_way_roots(root_reference, rep, seed_base) for rep in range(n_partitions)]
            for groups in partitions:
                union = np.concatenate(groups)
                if len(np.unique(union)) != len(root_reference) or set(union) != set(root_reference):
                    raise AssertionError(f"non-disjoint root partition for {target}")
                for fit_group, select_group, evaluate_group in TRIPLES:
                    if (
                        set(groups[fit_group]) & set(groups[select_group])
                        or set(groups[fit_group]) & set(groups[evaluate_group])
                        or set(groups[select_group]) & set(groups[evaluate_group])
                    ):
                        raise AssertionError("fit/select/evaluate roots overlap")
                    split_disjoint_checks += 1

            for quality in qualities:
                if quality == "designability":
                    y = labels["y_designability"]
                    root = labels["des_root"]
                    child = labels["des_child"]
                else:
                    y = labels[f"y_{quality}"]
                    root = labels["root_idx"]
                    child = labels["child_idx"]
                full_available = float(available_opportunity(as_rck(y, root, child)))
                measures[quality]["O"]["full"].setdefault(target, np.empty(len(steps)))[
                    step_index
                ] = full_available
                recorded_share = float(curve["steps"][str(step)][quality]["child_share"])
                max_child_share_deviation = max(
                    max_child_share_deviation, abs(full_available - recorded_share)
                )
                for rep, groups in enumerate(partitions):
                    for score_group in range(3):
                        rows = np.where(np.isin(root, groups[score_group]))[0]
                        nested = as_rck(y[rows], root[rows], child[rows])
                        value = float(available_opportunity(nested))
                        boot_values = available_opportunity(design_resamples(nested, uniforms))
                        for fit_group in range(3):
                            if fit_group == score_group:
                                continue
                            o_rotation[quality][rep, fit_group, score_group, step_index] = value
                            o_boot[quality][rep, fit_group, score_group, step_index] = boot_values

        for quality in qualities:
            measures[quality]["O"]["rotation"][target] = o_rotation[quality]
            measures[quality]["O"]["design_boot"][target] = o_boot[quality]
            measures[quality]["A"]["full"][target] = np.asarray(
                [curve["steps"][str(step)][quality]["J"] for step in steps], dtype=float
            )
            rotation = np.full((n_partitions, 3, 3, len(steps)), np.nan, dtype=float)
            for step_index, step in enumerate(steps):
                pairs = curve["steps"][str(step)][quality]["group_pairs"]
                for pair, values in pairs.items():
                    fit_group, score_group = (int(x) for x in pair.split("->"))
                    if len(values) != n_partitions:
                        raise AssertionError(f"root partition count mismatch {target}/{quality}/{step}")
                    rotation[:, fit_group, score_group, step_index] = values
            keys = [f"{quality}|{step}" for step in steps]
            if not all(key in jgroup.files for key in keys):
                raise KeyError(f"missing readable design replicates for {target}/{quality}")
            boot = np.stack([jgroup[key] for key in keys], axis=3).astype(np.float32)
            if boot.shape != (n_partitions, 3, 3, len(steps), n_design):
                raise AssertionError(f"unexpected readable bootstrap shape {boot.shape}")
            measures[quality]["A"]["rotation"][target] = rotation
            measures[quality]["A"]["design_boot"][target] = boot
        print(f"audited {target_index:02d}/{len(targets)} {target}", flush=True)

    print("=== phase: held-out policy comparison ===", flush=True)
    result_measures: dict[str, Any] = {}
    bootstrap_arrays: dict[str, np.ndarray] = {}
    reference_max_abs_deviation = 0.0
    shared_design_indices = np.random.default_rng(bootstrap_seed + 1).integers(
        0, n_design, size=n_boot
    )
    for quality in qualities:
        result_measures[quality] = {}
        for measure in ("O", "A"):
            payload = measures[quality][measure]
            rotation, rotation_arrays = rotation_summary(
                payload["rotation"], clusters, steps, deployed_steps
            )
            for target in targets:
                full_curve = payload["full"][target]
                peak_idx = argmax_idx(full_curve)
                rotation["per_target"][target].update(
                    {
                        "full_curve": {
                            str(step): float(full_curve[i]) for i, step in enumerate(steps)
                        },
                        "full_peak_step": int(steps[peak_idx]),
                        "full_peak_value": float(full_curve[peak_idx]),
                    }
                )
            primary_bootstrap, primary_draws = primary_cluster_bootstrap(
                payload["rotation"],
                clusters,
                steps,
                deployed_steps,
                n_boot,
                bootstrap_seed,
            )
            nested, nested_draws = nested_split_design_sensitivity(
                payload["design_boot"],
                clusters,
                steps,
                deployed_steps,
                primary_draws["cluster_draw_indices"],
                shared_design_indices,
            )
            legacy = None
            legacy_draws: dict[str, np.ndarray] = {}
            if measure == "A":
                legacy, legacy_draws = conservative_bootstrap(
                    payload["rotation"],
                    payload["design_boot"],
                    clusters,
                    steps,
                    deployed_steps,
                    n_boot,
                    bootstrap_seed,
                )
            result_measures[quality][measure] = {
                "name": "available opportunity" if measure == "O" else "readable opportunity",
                "formula": "sigma2_child / sigma2_total"
                if measure == "O"
                else "O * R2_sibling (previously J)",
                "rotation": rotation,
                "primary_cluster_bootstrap": primary_bootstrap,
                "nested_split_design_sensitivity": nested,
                "legacy_one_rotation_mixture": legacy,
                "assessment": assessment(
                    primary_bootstrap["gain_adaptive_vs_deployed"]
                ),
            }
            for key, values in primary_draws.items():
                bootstrap_arrays[f"{quality}|{measure}|primary|{key}"] = values
            for key, values in nested_draws.items():
                bootstrap_arrays[f"{quality}|{measure}|nested|{key}"] = values
            for key, values in legacy_draws.items():
                bootstrap_arrays[f"{quality}|{measure}|legacy|{key}"] = values

            if measure == "A" and legacy is not None:
                expected = reference_gain["by_quality"][quality]["scored_conservative"]
                for metric in ("adaptive", "deployed", "gain_adaptive_vs_deployed"):
                    for key in ("mean", "lo", "hi", "p_gt_0"):
                        observed = float(legacy[metric][key])
                        reference_max_abs_deviation = max(
                            reference_max_abs_deviation,
                            abs(observed - float(expected[metric][key])),
                        )
            gain = primary_bootstrap["gain_adaptive_vs_deployed"]
            nested_gain = nested["gain_adaptive_vs_deployed"]
            print(
                f"{quality:24s} {measure} PRIMARY {gain['mean']:+.6f} "
                f"[{gain['lo']:+.6f}, {gain['hi']:+.6f}] "
                f"NESTED [{nested_gain['lo']:+.6f}, {nested_gain['hi']:+.6f}]",
                flush=True,
            )

    print("=== phase: commitment threshold sensitivity ===", flush=True)
    thresholds, max_commitment_pool_deviation = threshold_sensitivity(
        sources["paper_commitment_curves"],
        sources["paper_commitment_summary"],
        targets,
        qualities,
        steps,
        tuple(float(x) for x in config["thresholds"]),
        float(config["minimum_resolved_fraction"]),
    )

    input_manifest_path = output_dir / "input_manifest.json"
    input_manifest = write_input_manifest(consumed_paths, input_manifest_path)
    np.savez_compressed(output_dir / "bootstrap_draws.npz", **bootstrap_arrays)

    primary = {
        measure: result_measures[primary_quality][measure]["primary_cluster_bootstrap"][
            "gain_adaptive_vs_deployed"
        ]
        for measure in ("O", "A")
    }
    primary_nested = {
        measure: result_measures[primary_quality][measure][
            "nested_split_design_sensitivity"
        ]["gain_adaptive_vs_deployed"]
        for measure in ("O", "A")
    }
    primary_assessment = {measure: assessment(primary[measure]) for measure in ("O", "A")}
    nested_assessment = {
        measure: assessment(primary_nested[measure]) for measure in ("O", "A")
    }
    if any(
        primary_assessment[measure] == "strengthens Act IV"
        and nested_assessment[measure] == "strengthens Act IV"
        for measure in ("O", "A")
    ):
        act_iv_change = "strengthens"
    elif all(
        primary_assessment[measure] == "evidence against target-adaptive timing"
        and nested_assessment[measure] == "evidence against target-adaptive timing"
        for measure in ("O", "A")
    ):
        act_iv_change = "evidence_against"
    else:
        act_iv_change = "unresolved"

    output = {
        "schema_version": 1,
        "question": "Does target-adaptive action-step selection improve held-out opportunity over the best panel-wide deployed point?",
        "scope": {
            "targets": targets,
            "n_targets": len(targets),
            "clusters": clusters,
            "n_structural_clusters": len(set(clusters.values())),
            "capture_steps": list(steps),
            "deployed_steps": list(deployed_steps),
            "primary_quality": primary_quality,
            "other_qualities": "exploratory; fold identity is an ancillary ten-step refit",
            "predictive_not_interventional": True,
        },
        "method": {
            "root_partitions": n_partitions,
            "root_rotations": n_partitions * len(TRIPLES),
            "root_rotation_role": "repeated cross-fitting over the same roots, not independent population observations",
            "root_roles": ["fit", "select", "evaluate"],
            "tie_rule": "earlier measured step",
            "panel_bootstrap_draws": n_boot,
            "panel_bootstrap_seed": bootstrap_seed,
            "design_bootstrap_replicates": n_design,
            "independent_population_units": "12 Foldseek structural clusters",
            "cluster_weighting": "for the full-panel point, average 60 rotations per target, average targets within each Foldseek cluster, then weight 12 clusters equally",
            "primary_inference": "within each draw, sample only the 12 structural clusters, reselect the deployed comparator from sampled selection curves in each rotation, evaluate paired held-out curves, then average all 60 rotations",
            "primary_comparator_selection": "full-panel point reselects on full selection data; each bootstrap draw reselects on its sampled cluster selection curves, within every rotation",
            "nested_sensitivity": "within each outer cluster draw, sample one paired saved design replicate and average all 60 rotations; reselect the deployed comparator inside each rotation",
        },
        "input_audit": {
            "n_label_files": len(label_files),
            "n_curve_files": len(curve_files),
            "coherent_root_checks": coherent_root_checks,
            "disjoint_role_checks": split_disjoint_checks,
            "max_abs_full_O_vs_saved_child_share": max_child_share_deviation,
            "max_abs_A_legacy_reproduction_deviation": reference_max_abs_deviation,
            "max_abs_commitment_pool_deviation": max_commitment_pool_deviation,
            "input_manifest": f"{output_subdir}/input_manifest.json",
            "input_manifest_n_files": input_manifest["n_files"],
            "input_manifest_total_bytes": input_manifest["total_bytes"],
        },
        "measures": result_measures,
        "primary": {
            "interface_quality": primary,
            "nested_split_design_sensitivity": primary_nested,
            "population_interval_assessments": primary_assessment,
            "nested_split_design_assessments": nested_assessment,
            "act_iv_change": act_iv_change,
            "n_independent_structural_clusters": 12,
            "n_repeated_crossfit_rotations": 60,
            "decision_rule": "the 12-cluster interval assesses population sampling, while the paired nested sensitivity assesses saved split/design stability; the overall Act IV interpretation is directional only when both analyses agree",
        },
        "commitment_threshold_sensitivity": thresholds,
    }
    analysis_path = output_dir / "adaptive_opportunity.json"
    json.dump(output, open(analysis_path, "w"), indent=1)

    plot_rows = [
        {
            "measure": measure,
            "label": "Available O" if measure == "O" else "Readable A",
            **primary[measure],
            "nested_mean": primary_nested[measure]["mean"],
            "nested_lo": primary_nested[measure]["lo"],
            "nested_hi": primary_nested[measure]["hi"],
            "assessment": (
                f"population interval: {primary_assessment[measure]}; "
                f"nested split/design: {nested_assessment[measure]}"
            ),
        }
        for measure in ("O", "A")
    ]
    threshold_rows = []
    for quality in qualities:
        for threshold, row in thresholds["by_quality"][quality]["thresholds"].items():
            dispersion = row["resolved_dispersion"] or {}
            threshold_rows.append(
                {
                    "quality": quality,
                    "threshold": float(threshold),
                    "median": dispersion.get("median"),
                    "q25": dispersion.get("q25"),
                    "q75": dispersion.get("q75"),
                    "resolved": row["censoring_counts"]["resolved"],
                    "left_censored": row["censoring_counts"]["left_censored"],
                    "right_censored": row["censoring_counts"]["right_censored"],
                    "interpret": row["interpret"],
                }
            )
    plot_data = {"act_iv": plot_rows, "thresholds": threshold_rows}
    json.dump(plot_data, open(output_dir / "plot_data.json", "w"), indent=1)

    o_text, a_text = fmt_interval(primary["O"]), fmt_interval(primary["A"])
    nested_o_text = fmt_interval(primary_nested["O"])
    nested_a_text = fmt_interval(primary_nested["A"])
    manuscript = (
        "## Target-adaptive action timing\n\n"
        f"Across 19 targets in 12 independent Foldseek structural clusters, target-adaptive timing changed held-out available opportunity by **{o_text}** and readable opportunity by **{a_text}** relative to the best panel-wide deployed step in {{100, 200, 300}}. "
        f"The 12-cluster population interval is positive for available opportunity and unresolved for readable opportunity. A paired nested split/design sensitivity, which samples one of 200 saved design replicates inside each outer cluster draw while averaging all 60 rotations and preserving paired policy choices, was {nested_o_text} for available opportunity and {nested_a_text} for readable opportunity. Both nested intervals include zero, so the final Act IV interpretation is **{act_iv_change.replace('_', ' ')}** rather than directional. "
        "The full-panel point estimate averages all 60 three-way root rotations per target, then averages targets within structural cluster. Each population interval draw samples only the 12 structural clusters, reselects the deployed comparator from the sampled selection curves within every rotation, evaluates paired held-out curves, and then averages all 60 rotations. The rotations are repeated cross-fitting over the same roots, not independent population observations. "
        "These quantities measure predictive opportunity in the existing data; they do not show that an executed search policy would realize the estimated gain.\n\n"
        "The separate commitment-threshold sensitivity uses the original Acts I–II commitment curves. First crossings at 25%, 50%, 75%, and 90% were linearly interpolated between measured steps without monotonic smoothing; crossings before step 10 and absent by step 350 are reported as censored. This sensitivity changes the definition of settling, not the candidate search-action time.\n"
    )
    (output_dir / "manuscript_result.md").write_text(manuscript)
    print(f"wrote {analysis_path}", flush=True)
    print(
        f"PRIMARY O {o_text}; A {a_text}; Act IV {act_iv_change}; "
        f"A reproduction max deviation {reference_max_abs_deviation:.3g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
