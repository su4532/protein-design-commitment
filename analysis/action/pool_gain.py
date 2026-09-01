#!/usr/bin/env python
"""Stage C2: how much is there to gain from moving the action step?

Four policies, scored with the same objective on the same states:

``adaptive``   each target acts at its own best step
``global``     every target acts at the single best step, chosen across targets
``deployed``   every target acts at the best of Complexa's action points 100/200/300
``step100``    every target acts at step 100, what the search does today

Every number here comes from a three-way rotation of a target's roots. One group
fits the readout, a second selects the step, a third scores it, and the three are
disjoint, so the scoring group's outcomes influenced neither the readout's
coefficients nor the choice of step. Rotating over the six orderings of each
partition uses every group in every role.

The interval resamples the 12 structural clusters and, for each drawn cluster
row, draws one rotation and one design-bootstrap replicate of that rotation's
curves. Each target therefore contributes its own estimation error rather than
being treated as measured exactly. A variant that holds each target's gain fixed
and resamples clusters alone is reported beside it as the optimistic reading.

    python pool_gain.py <curves_dir> <out_json> [--quality ...] [--boot 10000]
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from collections import Counter

import numpy as np


from .objective import (  # noqa: E402
    CAPTURE_STEPS,
    QUALITIES,
    curve_interpolate,
    stable_seed,
)

DEPLOYED_STEPS = (100, 200, 300)
CURRENT_STEP = 100
GRID = np.arange(min(CAPTURE_STEPS), max(CAPTURE_STEPS) + 1, 1.0)
PLATEAU_MARGIN_FRAC = 0.10
N_GROUPS = 3
TRIPLES = [t for t in itertools.permutations(range(N_GROUPS), 3)]   # (fit, select, evaluate)
S_IDX = {s: i for i, s in enumerate(CAPTURE_STEPS)}
DEP_IDX = np.asarray([S_IDX[s] for s in DEPLOYED_STEPS])
CUR_IDX = S_IDX[CURRENT_STEP]


def cluster_average(values: dict[str, float], clusters: dict[str, str]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for t, v in values.items():
        if np.isfinite(v):
            acc.setdefault(clusters[t], []).append(float(v))
    return {c: float(np.mean(v)) for c, v in acc.items() if v}


def boot_mean_ci(rows: list[float], n_boot: int, seed: int) -> dict:
    a = np.asarray([r for r in rows if np.isfinite(r)], dtype=float)
    if len(a) < 2:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_rows": len(a)}
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return {"mean": float(a.mean()), "lo": float(np.percentile(m, 2.5)),
            "hi": float(np.percentile(m, 97.5)), "p_gt_0": float((m > 0).mean()),
            "n_rows": int(len(a)), "n_boot": int(n_boot)}


def argmax_idx(curve: np.ndarray, allowed: np.ndarray | None = None) -> int:
    idx = np.arange(len(curve)) if allowed is None else allowed
    vals = curve[idx]
    if not np.isfinite(vals).any():
        return int(idx[0])
    # ties break to the earliest step: acting sooner leaves more trajectory to influence
    return int(idx[int(np.nanargmax(vals))])


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load(curves_dir: str, quality: str):
    """Full curves, rotation curves ``[rep, fit, scored, step]`` and replicates."""
    full, rot, boot, diag = {}, {}, {}, {}
    for fn in sorted(os.listdir(curves_dir)):
        if not fn.endswith("_curves.json"):
            continue
        d = json.load(open(os.path.join(curves_dir, fn)))
        t = d["target"]
        s0 = str(CAPTURE_STEPS[0])
        if quality not in d["steps"][s0]:
            continue
        full[t] = np.asarray([d["steps"][str(s)][quality]["J"] for s in CAPTURE_STEPS])
        diag[t] = {k: {str(s): d["steps"][str(s)][quality][k] for s in CAPTURE_STEPS}
                   for k in ("child_share", "r2_sibling", "r2_observed", "floor", "J_raw",
                             "commitment", "n_states")}
        reps = len(next(iter(d["steps"][s0][quality]["group_pairs"].values())))
        M = np.full((reps, N_GROUPS, N_GROUPS, len(CAPTURE_STEPS)), np.nan)
        for si, s in enumerate(CAPTURE_STEPS):
            for pk, vals in d["steps"][str(s)][quality]["group_pairs"].items():
                gt, gs = (int(x) for x in pk.split("->"))
                M[:, gt, gs, si] = vals
        rot[t] = M
        zp = os.path.join(curves_dir, f"{t}_Jgroup.npz")
        if os.path.exists(zp):
            z = np.load(zp)
            keys = [f"{quality}|{s}" for s in CAPTURE_STEPS]
            if all(k in z.files for k in keys):
                # [rep, fit, scored, step, b]
                boot[t] = np.stack([z[k] for k in keys], axis=3).astype(np.float32)
    return full, rot, boot, diag


# --------------------------------------------------------------------------- #
# policy scoring on one rotation
# --------------------------------------------------------------------------- #


def policy_values(
    sel: dict[str, np.ndarray],
    ev: dict[str, np.ndarray],
    clusters: dict[str, str],
    selection_clusters: list[str] | None = None,
) -> dict:
    """Select policy steps on selection curves, then score on evaluation curves.

    ``selection_clusters`` is a structural-cluster draw with replacement. Passing
    the draw is essential inside the bootstrap: the global and deployed
    comparators must be reselected with the same cluster multiplicities used to
    score that bootstrap replicate. The adaptive policy remains target-specific.
    """
    targets = sorted(sel)
    stack = np.stack([sel[t] for t in targets])
    per_cluster: dict[str, list[np.ndarray]] = {}
    for t, row in zip(targets, stack):
        per_cluster.setdefault(clusters[t], []).append(row)
    cluster_curves = {
        cluster: np.nanmean(np.stack(rows), axis=0)
        for cluster, rows in per_cluster.items()
    }
    draw = list(cluster_curves) if selection_clusters is None else selection_clusters
    if not draw:
        raise ValueError("selection cluster draw is empty")
    missing = sorted(set(draw).difference(cluster_curves))
    if missing:
        raise KeyError(f"selection clusters have no targets: {missing}")
    panel = np.nanmean(np.stack([cluster_curves[cluster] for cluster in draw]), axis=0)
    g = argmax_idx(panel)
    d = argmax_idx(panel, DEP_IDX)
    out = {"adaptive": {}, "global": {}, "deployed": {}, "step100": {}}
    steps = {}
    for t in targets:
        i = argmax_idx(sel[t])
        steps[t] = CAPTURE_STEPS[i]
        out["adaptive"][t] = float(ev[t][i])
        out["global"][t] = float(ev[t][g])
        out["deployed"][t] = float(ev[t][d])
        out["step100"][t] = float(ev[t][CUR_IDX])
    return {"values": out, "t_global": CAPTURE_STEPS[g], "t_deployed": CAPTURE_STEPS[d],
            "t_adaptive": steps}


def gains(values: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    ts = list(values["adaptive"])
    return {
        "gain_adaptive_vs_deployed": {t: values["adaptive"][t] - values["deployed"][t] for t in ts},
        "gain_adaptive_vs_global": {t: values["adaptive"][t] - values["global"][t] for t in ts},
        "gain_adaptive_vs_step100": {t: values["adaptive"][t] - values["step100"][t] for t in ts},
        "gain_global_vs_deployed": {t: values["global"][t] - values["deployed"][t] for t in ts},
        "gain_deployed_vs_step100": {t: values["deployed"][t] - values["step100"][t] for t in ts},
    }


def rotation_average(rot: dict[str, np.ndarray], clusters: dict[str, str]) -> dict:
    """Every (partition, ordering) combination, averaged over combinations."""
    targets = sorted(rot)
    reps = rot[targets[0]].shape[0]
    acc: dict[str, dict[str, list[float]]] = {}
    acc_same: dict[str, dict[str, list[float]]] = {}
    chosen, adaptive_steps = [], {t: [] for t in targets}
    for rep in range(reps):
        for gt, gs, ge in TRIPLES:
            sel = {t: rot[t][rep, gt, gs] for t in targets}
            ev = {t: rot[t][rep, gt, ge] for t in targets}
            r = policy_values(sel, ev, clusters)
            all_v = {**r["values"], **gains(r["values"])}
            for k, per_t in all_v.items():
                for t in targets:
                    acc.setdefault(k, {}).setdefault(t, []).append(per_t[t])
            # Same choice, scored on the very curve it was chosen from. Identical fit
            # group, and a scoring group of the same size (the selection group rather
            # than the third group), so the only difference from `r` is whether the
            # roots that score the step also chose it. This isolates the winner's curse
            # from any change in training or scoring sample size.
            rs = policy_values(sel, sel, clusters)
            same_v = {**rs["values"], **gains(rs["values"])}
            for k, per_t in same_v.items():
                for t in targets:
                    acc_same.setdefault(k, {}).setdefault(t, []).append(per_t[t])
            chosen.append({"rep": rep, "fit": gt, "select": gs, "evaluate": ge,
                           "t_global": r["t_global"], "t_deployed": r["t_deployed"]})
            for t in targets:
                adaptive_steps[t].append(r["t_adaptive"][t])
    values = {k: {t: float(np.nanmean(v[t])) for t in targets} for k, v in acc.items()}
    values_same = {k: {t: float(np.nanmean(v[t])) for t in targets}
                   for k, v in acc_same.items()}
    return {"values": values, "values_same_group": values_same, "raw": acc,
            "chosen": chosen, "n_rotations": len(chosen),
            "adaptive_steps": adaptive_steps}


def conservative_interval(rot: dict[str, np.ndarray], boot: dict[str, np.ndarray],
                          clusters: dict[str, str], n_boot: int, seed: int) -> dict:
    """Resample clusters; per drawn row draw one rotation and one design replicate.

    This is the interval the decision is scored against. Each target contributes
    its own estimation error, so a small mean difference built on noisy per-target
    objectives cannot be made to look precise by averaging rotations first.
    """
    targets = sorted(t for t in rot if t in boot)
    if not targets:
        return {}
    by_cluster: dict[str, list[str]] = {}
    for t in targets:
        by_cluster.setdefault(clusters[t], []).append(t)
    cl = sorted(by_cluster)
    reps, _, _, _, B = boot[targets[0]].shape
    rng = np.random.default_rng(seed)
    keys = ("gain_adaptive_vs_deployed", "gain_adaptive_vs_global",
            "gain_adaptive_vs_step100", "gain_global_vs_deployed",
            "gain_deployed_vs_step100", "adaptive", "global", "deployed", "step100")
    out = {k: np.empty(n_boot) for k in keys}
    selected_global_steps: Counter[int] = Counter()
    selected_deployed_steps: Counter[int] = Counter()

    for b in range(n_boot):
        pick = rng.integers(0, len(cl), size=len(cl))
        drawn_clusters = [cl[ci] for ci in pick]
        rep = int(rng.integers(0, reps))
        gt, gs, ge = TRIPLES[int(rng.integers(0, len(TRIPLES)))]
        db = int(rng.integers(0, B))
        sel = {t: boot[t][rep, gt, gs, :, db] for t in targets}
        ev = {t: boot[t][rep, gt, ge, :, db] for t in targets}
        r = policy_values(sel, ev, clusters, selection_clusters=drawn_clusters)
        selected_global_steps[int(r["t_global"])] += 1
        selected_deployed_steps[int(r["t_deployed"])] += 1
        allv = {**r["values"], **gains(r["values"])}
        for k in keys:
            rows = [float(np.mean([allv[k][t] for t in by_cluster[cluster]]))
                    for cluster in drawn_clusters]
            out[k][b] = float(np.mean(rows))

    return {k: {"mean": float(np.nanmean(v)), "lo": float(np.nanpercentile(v, 2.5)),
                "hi": float(np.nanpercentile(v, 97.5)), "p_gt_0": float((v > 0).mean())}
            for k, v in out.items()} | {
                "n_boot": n_boot,
                "n_design_boot": int(B),
                "n_rotations": int(reps * len(TRIPLES)),
                "bootstrap_reselects_comparators": True,
                "bootstrap_selected_global_step_counts": dict(sorted(selected_global_steps.items())),
                "bootstrap_selected_deployed_step_counts": dict(sorted(selected_deployed_steps.items())),
            }


def same_data_two_stage(full: dict[str, np.ndarray], jboot_dir: str, quality: str,
                        clusters: dict[str, str], n_boot: int, seed: int) -> dict:
    """The optimistic branch: the step is the argmax of the curve it is scored on."""
    targets = sorted(full)
    jb = {}
    for t in targets:
        p = os.path.join(jboot_dir, f"{t}_Jboot.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        keys = [f"{quality}|{s}" for s in CAPTURE_STEPS]
        if all(k in z.files for k in keys):
            jb[t] = np.stack([z[k] for k in keys], axis=1).astype(float)
    if not jb:
        return {}
    by_cluster: dict[str, list[str]] = {}
    for t in jb:
        by_cluster.setdefault(clusters[t], []).append(t)
    cl = sorted(by_cluster)
    B = next(iter(jb.values())).shape[0]
    rng = np.random.default_rng(seed)
    keys = ("adaptive", "global", "deployed", "step100", "gain_adaptive_vs_deployed")
    out = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        pick = rng.integers(0, len(cl), size=len(cl))
        rows = []
        for ci in pick:
            r = int(rng.integers(0, B))
            rows.append(np.nanmean(np.stack([jb[t][r] for t in by_cluster[cl[ci]]]), axis=0))
        M = np.nan_to_num(np.stack(rows), nan=0.0)
        panel = M.mean(axis=0)
        g = argmax_idx(panel)
        d = argmax_idx(panel, DEP_IDX)
        out["adaptive"][b] = np.nanmax(M, axis=1).mean()
        out["global"][b] = M[:, g].mean()
        out["deployed"][b] = M[:, d].mean()
        out["step100"][b] = M[:, CUR_IDX].mean()
        out["gain_adaptive_vs_deployed"][b] = out["adaptive"][b] - out["deployed"][b]
    return {k: {"mean": float(np.nanmean(v)), "lo": float(np.nanpercentile(v, 2.5)),
                "hi": float(np.nanpercentile(v, 97.5)), "p_gt_0": float((v > 0).mean())}
            for k, v in out.items()}


def regret_and_plateau(full: dict[str, np.ndarray]) -> dict:
    out = {}
    for t, vals in full.items():
        dense = curve_interpolate(CAPTURE_STEPS, vals, GRID)
        if not np.isfinite(dense).any():
            continue
        i = int(np.nanargmax(dense))
        peak_v, peak_t = float(dense[i]), float(GRID[i])
        within = GRID[np.nan_to_num(dense, nan=-np.inf) >= peak_v - PLATEAU_MARGIN_FRAC * peak_v]
        out[t] = {
            "peak_step_discrete": int(CAPTURE_STEPS[int(np.nanargmax(vals))]),
            "peak_step_interpolated": peak_t, "peak_J": peak_v,
            "plateau_lo": float(within.min()) if len(within) else float("nan"),
            "plateau_hi": float(within.max()) if len(within) else float("nan"),
            "plateau_width_steps": float(within.max() - within.min()) if len(within) else float("nan"),
            "regret_at_step": {int(s): float(peak_v - vals[i2]) for i2, s in enumerate(CAPTURE_STEPS)},
            "regret_rel_at_step": {int(s): float((peak_v - vals[i2]) / peak_v) if peak_v > 0 else float("nan")
                                   for i2, s in enumerate(CAPTURE_STEPS)},
            "curve": {int(s): float(vals[i2]) for i2, s in enumerate(CAPTURE_STEPS)},
        }
    return out


def analyse(curves_dir: str, quality: str, clusters: dict[str, str], n_boot: int) -> dict:
    full, rot, boot, diag = load(curves_dir, quality)
    if not full:
        return {}
    targets = sorted(full)
    ra = rotation_average(rot, clusters)

    optimistic: dict = {"by_cluster": {}, "by_target": {}}
    for name, per_t in ra["values"].items():
        cl = cluster_average(per_t, clusters)
        optimistic["by_cluster"][name] = boot_mean_ci(
            list(cl.values()), n_boot, stable_seed(f"optimistic|{quality}|{name}|cluster")
        )
        optimistic["by_target"][name] = boot_mean_ci(
            [per_t[t] for t in targets],
            n_boot,
            stable_seed(f"optimistic|{quality}|{name}|target"),
        )

    # Winner's curse at matched sample sizes: both arms use the same readout (fitted on
    # the same third of the roots) and the same scoring group; they differ only in
    # whether the step was chosen from the curve it is scored on.
    key = "gain_adaptive_vs_deployed"
    same_t = ra["values_same_group"][key]
    held_t = ra["values"][key]
    diff_t = {t: same_t[t] - held_t[t] for t in targets}
    wc = {
        "same_group": boot_mean_ci(list(cluster_average(same_t, clusters).values()), n_boot, 91),
        "held_out": boot_mean_ci(list(cluster_average(held_t, clusters).values()), n_boot, 92),
        "difference": boot_mean_ci(list(cluster_average(diff_t, clusters).values()), n_boot, 93),
        "note": ("Both arms fit the readout on one third of the roots and score on another "
                 "third; they differ only in whether the step was chosen from the curve it "
                 "is scored on. optimistic_same_data instead uses the full-data "
                 "out-of-fold curves, so it also differs in readout training size."),
    }
    sg = wc["same_group"]["mean"]
    wc["drop_fraction"] = float(1 - wc["held_out"]["mean"] / sg) if sg else float("nan")

    return {
        "quality": quality, "targets": targets,
        "winners_curse_matched": wc,
        "curves_full": {t: {int(s): float(full[t][i]) for i, s in enumerate(CAPTURE_STEPS)}
                        for t in targets},
        "diagnostics": diag,
        "rotation": {"n_rotations": ra["n_rotations"],
                     "per_target": ra["values"],
                     "chosen_fixed_steps": ra["chosen"][:60],
                     "adaptive_step_counts": {
                         t: {int(s): int(sum(1 for x in ra["adaptive_steps"][t] if x == s))
                             for s in CAPTURE_STEPS} for t in targets}},
        "scored_conservative": conservative_interval(rot, boot, clusters, n_boot=min(n_boot, 4000),
                                                     seed=17),
        "optimistic_fixed_per_target": optimistic,
        "optimistic_same_data": same_data_two_stage(full, curves_dir, quality, clusters,
                                                    n_boot=min(n_boot, 4000), seed=7),
        "regret": regret_and_plateau(full),
    }


def main() -> None:
    curves_dir, out_json = sys.argv[1], sys.argv[2]
    n_boot = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 10000
    qualities = (sys.argv[sys.argv.index("--quality") + 1].split(",")
                 if "--quality" in sys.argv else list(QUALITIES))
    if "--clusters-json" not in sys.argv:
        raise ValueError("--clusters-json is required")
    cluster_path = sys.argv[sys.argv.index("--clusters-json") + 1]
    clusters = json.load(open(cluster_path))["target_clusters"]

    out = {"clusters": clusters, "n_structural_clusters": len(set(clusters.values())),
           "capture_steps": list(CAPTURE_STEPS), "deployed_steps": list(DEPLOYED_STEPS),
           "current_step": CURRENT_STEP, "n_boot": n_boot,
           "design": "three-way root rotation: fit / select / evaluate on disjoint groups",
           "structural_bootstrap": (
               "resample 12 structural clusters; reselect global and deployed comparators "
               "from each draw's selection curves; score on disjoint evaluation curves with "
               "the same cluster multiplicities"
           ),
           "by_quality": {}}
    for q in qualities:
        print(f"[pool] {q}", flush=True)
        r = analyse(curves_dir, q, clusters, n_boot)
        if not r:
            continue
        out["by_quality"][q] = r
        c = r["scored_conservative"].get("gain_adaptive_vs_deployed", {})
        o = r["optimistic_fixed_per_target"]["by_cluster"]["gain_adaptive_vs_deployed"]
        s = r["optimistic_same_data"].get("gain_adaptive_vs_deployed", {})
        if c:
            print(f"  SCORED  adaptive-deployed {c['mean']:+.5f} [{c['lo']:+.5f}, {c['hi']:+.5f}] "
                  f"P(>0)={c['p_gt_0']:.3f}", flush=True)
        print(f"  optimistic (fixed per target) {o['mean']:+.5f} [{o['lo']:+.5f}, {o['hi']:+.5f}]",
              flush=True)
        if s:
            print(f"  optimistic (same data)        {s['mean']:+.5f} [{s['lo']:+.5f}, {s['hi']:+.5f}]",
                  flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    json.dump(out, open(out_json, "w"), indent=1)
    print(f"[pool] -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
