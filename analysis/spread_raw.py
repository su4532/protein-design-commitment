#!/usr/bin/env python
"""How much does each quality's settling time vary from target to target?

POST-HOC. The pre-registered test is the metric *ordering* (ordering_watch.py).
That test failed: the ordering is target-dependent. This script quantifies the
replacement observation - that some qualities settle on a schedule that barely
depends on the target while others do not - and puts an honest interval on it.

Statistic per metric: the across-target spread of t_half (the capture step at
which commitment first crosses 0.5), as coefficient of variation (sd/mean) and
as interquartile range in denoiser steps. Uncertainty: a nonparametric
bootstrap that resamples TARGETS with replacement, which is the unit that
generalises. t_half is treated as a point estimate; the four metrics share the
same folds, the same nesting and the same estimator within each target, so
their estimation noise is comparable and does not favour one metric.

    python spread_stats.py <snapshot_dir> <out.json> [--boot 10000] [--seed 42]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

METRICS = ["fold_identity", "designability", "structural_confidence", "interface_quality"]
SHORT = {"fold_identity": "fold", "designability": "design",
         "structural_confidence": "conf", "interface_quality": "iface"}


def cv(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.mean(x))


def iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def compute_spread(t_half_by_target: dict, boot: int = 10000, seed: int = 42,
                   clusters: dict | None = None, floor_step: float = 10.0,
                   drop_floor: bool = False) -> dict:
    """Across-target spread of t_half per metric, with a bootstrap over targets.

    ``t_half_by_target`` maps target -> metric -> t_half (or None where the
    quality never crossed 0.5 by the last capture step). A target is used only
    if all four metrics are uncensored, so every metric is compared on the same
    set of targets.

    ``floor_step`` is the earliest capture step. A quality whose commitment is
    already above 0.5 there has a settling step somewhere below the first
    measurement, and ``t_half`` records the floor instead. Those values are
    LEFT-CENSORED: the recorded spread is narrower than the true spread, so a
    metric with floor hits has an understated coefficient of variation.
    ``drop_floor`` reruns the same statistic on targets with no floor hit at
    all, which is the sensitivity check for that.

    ``clusters`` (target -> structural cluster representative) switches the unit
    from the target to the structural cluster: members of one cluster are
    averaged into a single row before anything is computed. Seven of this
    benchmark's nineteen targets are one immunoglobulin-like cluster, so the
    target-level numbers can be read as optimistic and this collapsed version is
    the sensitivity check. It removes within-cluster variation from the spread
    as well, so it is a floor rather than a better estimate.
    """
    targets = sorted(t_half_by_target)
    usable = [t for t in targets
              if all(t_half_by_target[t].get(m) is not None for m in METRICS)]
    censored = [t for t in targets if t not in usable]

    # Left censoring: already past half commitment at the earliest capture step.
    left = {m: sorted(t for t in usable
                      if float(t_half_by_target[t][m]) <= floor_step) for m in METRICS}
    floor_targets = sorted({t for v in left.values() for t in v})
    if drop_floor:
        usable = [t for t in usable if t not in set(floor_targets)]

    unit_members = None
    if clusters:
        groups: dict[str, list[str]] = {}
        for t in usable:
            groups.setdefault(clusters.get(t, t), []).append(t)
        unit_members = {k: sorted(v) for k, v in sorted(groups.items())}
        collapsed = {k: {m: float(np.mean([t_half_by_target[t][m] for t in v]))
                         for m in METRICS} for k, v in unit_members.items()}
        t_half_by_target = collapsed
        usable = sorted(collapsed)

    if len(usable) < 3:
        return {"n_targets": len(usable), "censored_targets": censored,
                "left_censored": left, "insufficient": True}

    T = np.array([[t_half_by_target[t][m] for m in METRICS] for t in usable], float)
    n = T.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(boot, n))

    point = {m: {"median_t_half": float(np.median(T[:, i])),
                 "mean_t_half": float(np.mean(T[:, i])),
                 "min_t_half": float(np.min(T[:, i])),
                 "max_t_half": float(np.max(T[:, i])),
                 "q1_t_half": float(np.percentile(T[:, i], 25)),
                 "q3_t_half": float(np.percentile(T[:, i], 75)),
                 "cv": cv(T[:, i]), "iqr": iqr(T[:, i])}
             for i, m in enumerate(METRICS)}

    bs_cv = np.empty((boot, 4))
    bs_iqr = np.empty((boot, 4))
    for b in range(boot):
        S = T[idx[b]]
        for i in range(4):
            col = S[:, i]
            bs_cv[b, i] = np.std(col, ddof=1) / np.mean(col) if np.mean(col) > 0 else np.nan
            bs_iqr[b, i] = np.percentile(col, 75) - np.percentile(col, 25)

    for i, m in enumerate(METRICS):
        lo, hi = np.nanpercentile(bs_cv[:, i], [2.5, 97.5])
        point[m]["cv_ci95"] = [float(lo), float(hi)]
        lo, hi = np.nanpercentile(bs_iqr[:, i], [2.5, 97.5])
        point[m]["iqr_ci95"] = [float(lo), float(hi)]

    pairs = {}
    for i in range(4):
        for j in range(4):
            if i >= j:
                continue
            d = bs_cv[:, j] - bs_cv[:, i]
            obs = point[METRICS[j]]["cv"] - point[METRICS[i]]["cv"]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            key = f"cv({SHORT[METRICS[j]]}) - cv({SHORT[METRICS[i]]})"
            pairs[key] = {"observed": float(obs), "ci95": [float(lo), float(hi)],
                          "p_gt_0": float(np.nanmean(d > 0)),
                          "excludes_zero": bool(lo > 0 or hi < 0)}

    # grouped contrast: the two structural qualities against the two outcome
    # qualities. This grouping is post-hoc, chosen after seeing per-metric spreads.
    gs = [METRICS.index("fold_identity"), METRICS.index("structural_confidence")]
    go = [METRICS.index("designability"), METRICS.index("interface_quality")]
    obs_den = float(np.mean([point[METRICS[i]]["cv"] for i in gs]))
    obs_ratio = (float(np.mean([point[METRICS[i]]["cv"] for i in go])) / obs_den
                 if obs_den > 0 else float("nan"))
    # A resample can draw a degenerate set of clusters whose structural spread is
    # exactly zero; those replicates carry no ratio rather than an infinity.
    den = np.nanmean(bs_cv[:, gs], axis=1)
    d = np.where(den > 0, np.nanmean(bs_cv[:, go], axis=1) / np.where(den > 0, den, 1.0), np.nan)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    grouped = {"note": "post-hoc grouping chosen after seeing per-metric spreads",
               "structure_metrics": [METRICS[i] for i in gs],
               "outcome_metrics": [METRICS[i] for i in go],
               "cv_ratio_outcome_over_structure": obs_ratio,
               "ci95": [float(lo), float(hi)], "p_gt_1": float(np.nanmean(d > 1.0))}

    # Grouping-free summary: the rank order of the four coefficients of
    # variation, and which metrics separate from every other metric. This does
    # not depend on the post-hoc 2-versus-2 grouping below.
    cv_rank = sorted(METRICS, key=lambda m: point[m]["cv"])
    separates_from_all = [
        m for m in METRICS
        if all(pairs[k]["excludes_zero"] for k in pairs if SHORT[m] in k)
    ]

    return {"status": "post-hoc; the pre-registered test is the ordering test",
            "unit": "structural cluster" if clusters else "target",
            "floor_step": floor_step,
            "left_censored": left,
            "n_left_censored_values": int(sum(len(v) for v in left.values())),
            "dropped_floor_targets": bool(drop_floor),
            "cv_rank_order": cv_rank,
            "cv_separates_from_all_others": separates_from_all,
            "cluster_members": unit_members,
            "n_targets": n, "targets": usable, "censored_targets": censored,
            "bootstrap_draws": boot, "seed": seed,
            "t_half_by_target": {t: {m: t_half_by_target[t][m] for m in METRICS} for t in usable},
            "per_metric": point, "pairwise_cv_differences": pairs,
            "grouped_contrast": grouped}


def report(res: dict) -> None:
    if res.get("insufficient"):
        print(f"only {res['n_targets']} uncensored target(s) - not enough for a spread estimate")
        return
    print(f"n={res['n_targets']} {res.get('unit', 'target')}s (uncensored)   "
          f"bootstrap draws={res['bootstrap_draws']}")
    point, pairs, g = res["per_metric"], res["pairwise_cv_differences"], res["grouped_contrast"]
    print(f"{'metric':<24}{'median':>8}{'IQR':>8}{'IQR 95% CI':>22}{'CV':>8}{'CV 95% CI':>20}")
    for m in METRICS:
        pt = point[m]
        print(f"{m:<24}{pt['median_t_half']:>8.1f}{pt['iqr']:>8.1f}"
              f"{'[%.1f, %.1f]' % tuple(pt['iqr_ci95']):>22}"
              f"{pt['cv']:>8.4f}{'[%.4f, %.4f]' % tuple(pt['cv_ci95']):>20}")
    print("\npairwise CV differences (bootstrap over targets)")
    for k, v in pairs.items():
        star = "  *" if v["excludes_zero"] else ""
        print(f"  {k:<28} {v['observed']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
              f"  P(>0)={v['p_gt_0']:.4f}{star}")
    lc = res.get("left_censored") or {}
    if any(lc.values()):
        bits = "; ".join(f"{SHORT[m]}: {', '.join(v)}" for m, v in lc.items() if v)
        print(f"\nleft-censored at the first capture step (true settling step is below "
              f"{res.get('floor_step')}, so that metric's CV is understated) -> {bits}")
    print(f"\nCV rank order (tightest first): "
          f"{' < '.join(SHORT[m] for m in res.get('cv_rank_order', []))}")
    sep = res.get("cv_separates_from_all_others") or []
    print(f"separates from every other metric: "
          f"{', '.join(SHORT[m] for m in sep) if sep else 'none'}")
    print(f"\npost-hoc grouped contrast  CV(outcome)/CV(structure) = "
          f"{g['cv_ratio_outcome_over_structure']:.4f}"
          f"  95% CI [{g['ci95'][0]:.4f}, {g['ci95'][1]:.4f}]  P(>1)={g['p_gt_1']:.4f}")


def main() -> None:
    snap, out = sys.argv[1], sys.argv[2]
    boot = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 10000
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 42

    clusters_path = sys.argv[sys.argv.index("--target-clusters") + 1] \
        if "--target-clusters" in sys.argv else None
    clusters = json.load(open(clusters_path)) if clusters_path and os.path.exists(clusters_path) else None

    index = json.load(open(os.path.join(snap, "index.json")))
    th = {t: index[t]["t_half"] for t in index}
    res = compute_spread(th, boot=boot, seed=seed)
    report(res)
    if res.get("n_left_censored_values"):
        res["by_uncensored"] = compute_spread(th, boot=boot, seed=seed, drop_floor=True)
        print("\n--- sensitivity: targets with any left-censored settling step dropped ---")
        report(res["by_uncensored"])
    if clusters:
        res["by_structural_cluster"] = compute_spread(th, boot=boot, seed=seed, clusters=clusters)
        print("\n--- sensitivity: same analysis with homologous targets collapsed ---")
        report(res["by_structural_cluster"])
    json.dump(res, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
