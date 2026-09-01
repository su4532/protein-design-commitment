#!/usr/bin/env python
"""Across-target spread of settling time, with estimation error taken out of it.

POST-HOC. The pre-registered test was the ordering of the four qualities, and it
failed: the order changes from target to target. This is the replacement
observation - that some qualities settle on a schedule that varies less from
target to target than others - measured in a way that survives the obvious
objections to it.

First objection: a half-commitment step is estimated, not observed. Fold identity
is estimated with a categorical Gini decomposition and the other three with a
continuous nested ANOVA; the curves have different slopes where they cross 0.5,
and the within-completion noise differs by quality. Resampling only the 19
target-level point estimates, as the first version did, silently credits a
noisier estimator with genuine target-to-target variation.

So each target's whole commitment curve is recomputed under 2000 two-stage
resamples of its own roots and children (``commitment.boot``), the same draw
applied at every capture step, and each replicate curve is interpolated to a
replicate settling step. That gives, per target and quality, a sampling
distribution for the settling step rather than a single number. Two things
follow:

``cv_observed``   spread of the point estimates - what the first version
                  reported, kept for comparison. It is inflated by estimation
                  error, by an amount that differs per quality.
``cv_corrected``  the same spread after subtracting each row's own mean
                  estimation variance, so what is left is variation between
                  rows. This is the number to read.

Second objection, and the reason this module computes the whole analysis twice:
the 19 benchmark targets are not 19 independent draws. Foldseek puts them in 12
structural clusters and the largest holds seven of them. A point estimate over
19 targets paired with an interval that resamples 12 clusters describes two
different quantities. So the analysis is run at each unit with the point estimate
and the interval matched to that unit:

``by_cluster``  a row is a structural cluster (its members averaged), the
                bootstrap resamples clusters. 12 rows. The honest independent
                unit, and the one to lead with. Collapsing also removes genuine
                within-cluster variation, so it is conservative about spread.
``by_target``   a row is a target, the bootstrap resamples targets. 19 rows.
                An optimistic bound: it treats seven homologues as seven
                independent draws.

A third, hybrid version is kept for transparency because it was what the earlier
draft reported: ``by_cluster_block`` computes the point estimate over targets and
blocks the bootstrap on clusters. That is a defensible construction, but its
point estimate and its interval are not the same estimand, so it is not led with.

Within each version, metrics share the replicate index within a row, which keeps
paired comparisons paired.

    python spread_stats2.py <per_target_dir> <out.json> \
        [--clusters target_clusters.json] [--boot 10000] [--seed 42]
"""

from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np

from .commitment.grid import steps as _grid_steps  # noqa: E402

#: capture grid for this run; ``CAPTURE_STEPS`` in the environment overrides it
STEPS = _grid_steps()
METRICS = ["fold_identity", "designability", "structural_confidence", "interface_quality"]
SHORT = {"fold_identity": "fold", "designability": "design",
         "structural_confidence": "conf", "interface_quality": "iface"}
FLOOR = float(STEPS[0])

# mode -> (what a row is, what the bootstrap resamples, do they match)
MODES = {
    "cluster": ("structural cluster", "structural cluster", True),
    "target": ("target", "target", True),
    "target_block": ("target", "structural cluster", False),
}


# --------------------------------------------------------------------------- #
# settling step
# --------------------------------------------------------------------------- #


def t_half_curve(vals: np.ndarray, steps=STEPS, level: float = 0.5) -> float:
    """First crossing of ``level``, linearly interpolated; NaN if never crossed."""
    ok = np.isfinite(vals)
    if not ok.any():
        return np.nan
    s = np.asarray(steps, float)[ok]
    v = np.asarray(vals, float)[ok]
    if v[0] >= level:
        return float(s[0])
    hit = np.where(v >= level)[0]
    if hit.size == 0:
        return np.nan
    i = hit[0]
    v0, v1, s0, s1 = v[i - 1], v[i], s[i - 1], s[i]
    if v1 == v0:
        return float(s1)
    return float(s0 + (s1 - s0) * (level - v0) / (v1 - v0))


def t_half_replicates(boot: np.ndarray, steps=STEPS) -> np.ndarray:
    """``boot`` is ``[n_steps, B]`` of replicate commitments -> ``[B]`` settling steps."""
    B = boot.shape[1]
    return np.array([t_half_curve(boot[:, b], steps) for b in range(B)])


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_boot(per_target_dir: str, targets: list[str], steps=STEPS) -> dict:
    """target -> metric -> [n_steps, B] replicate commitments."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for t in targets:
        cols: dict[str, list] = {}
        for s in steps:
            p = os.path.join(per_target_dir, f"{t}_step{s}_boot.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            for m in METRICS:
                if m in z.files:
                    cols.setdefault(m, []).append((s, z[m]))
        keep = {}
        for m, rows in cols.items():
            if len(rows) != len(steps):
                continue
            keep[m] = np.stack([v for _, v in sorted(rows)], axis=0)
        if keep:
            out[t] = keep
    return out


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def _cv(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2 or np.mean(x) <= 0:
        return np.nan
    return float(np.std(x, ddof=1) / np.mean(x))


def _cv_corrected(point: np.ndarray, within_var: np.ndarray) -> float:
    """Between-row CV after removing each row's own mean estimation variance."""
    ok = np.isfinite(point) & np.isfinite(within_var)
    if ok.sum() < 3:
        return np.nan
    mu = float(np.mean(point[ok]))
    if mu <= 0:
        return np.nan
    v_obs = float(np.var(point[ok], ddof=1))
    v_est = float(np.mean(within_var[ok]))
    return float(np.sqrt(max(v_obs - v_est, 0.0)) / mu)


def spread(th_point: dict, th_rep: dict, clusters: dict, boot: int = 10000,
           seed: int = 42, drop_floor: bool = False, mode: str = "cluster") -> dict:
    """Per-quality spread of settling step, corrected and with propagated intervals."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    row_unit, boot_unit, matched = MODES[mode]

    targets = sorted(t for t in th_point
                     if all(np.isfinite(th_point[t].get(m, np.nan)) for m in METRICS))
    censored = sorted(set(th_point) - set(targets))
    left = {m: sorted(t for t in targets if th_point[t][m] <= FLOOR) for m in METRICS}
    floor_targets = sorted({t for v in left.values() for t in v})
    if drop_floor:
        targets = [t for t in targets if t not in set(floor_targets)]
    if len(targets) < 3:
        return {"insufficient": True, "n_targets": len(targets)}

    B = min(th_rep[t][m].shape[0] for t in targets for m in METRICS)
    P = np.array([[th_point[t][m] for m in METRICS] for t in targets], float)          # [T, 4]
    Rrep = np.stack([np.stack([th_rep[t][m][:B] for m in METRICS], 1) for t in targets])  # [T, B, 4]

    # within-target estimation variance of the settling step, per quality
    within = np.nanvar(Rrep, axis=1, ddof=1)                                            # [T, 4]

    # the per-target view is raw data and is reported whatever the analysis unit is
    se_target = np.sqrt(within)
    per_target_point = {t: {m: float(P[i, k]) for k, m in enumerate(METRICS)}
                        for i, t in enumerate(targets)}
    per_target_se = {t: {m: float(se_target[i, k]) for k, m in enumerate(METRICS)}
                     for i, t in enumerate(targets)}

    cluster_of = [clusters.get(t, t) for t in targets]
    unit_members = {k: sorted(t for t, u in zip(targets, cluster_of) if u == k)
                    for k in sorted(set(cluster_of))}

    if mode == "cluster":
        # A row becomes one structural cluster: its members' settling steps
        # averaged. The sampling variance of that average is NOT the average of
        # the members' variances - it is the variance of the average, which is
        # smaller by roughly the cluster size. Taking the members' mean variance
        # here would over-subtract in _cv_corrected and understate the spread of
        # every quality. It is read straight off the averaged replicate curves
        # instead, which carries the right scale by construction.
        keys = sorted(unit_members)
        idx_by_unit = {k: [targets.index(t) for t in unit_members[k]] for k in keys}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            P = np.stack([np.nanmean(P[idx_by_unit[k]], axis=0) for k in keys])
            Rrep = np.stack([np.nanmean(Rrep[idx_by_unit[k]], axis=0) for k in keys])
            within = np.nanvar(Rrep, axis=1, ddof=1)
        rows = keys
        row_of_unit = keys
    else:
        rows = targets
        row_of_unit = cluster_of if mode == "target_block" else list(targets)

    se = np.sqrt(within)
    censor_frac = np.mean(~np.isfinite(Rrep), axis=1)                                   # [rows, 4]

    point = {}
    for i, m in enumerate(METRICS):
        col = P[:, i]
        point[m] = {
            "median_t_half": float(np.median(col)), "mean_t_half": float(np.mean(col)),
            "min_t_half": float(np.min(col)), "max_t_half": float(np.max(col)),
            "q1_t_half": float(np.percentile(col, 25)), "q3_t_half": float(np.percentile(col, 75)),
            "iqr": float(np.percentile(col, 75) - np.percentile(col, 25)),
            "cv_observed": _cv(col),
            "cv_corrected": _cv_corrected(col, within[:, i]),
            "mean_within_se_steps": float(np.mean(se[:, i])),
            "median_within_se_steps": float(np.median(se[:, i])),
            "mean_replicate_censor_frac": float(np.mean(censor_frac[:, i])),
        }

    # ---- interval: resample the analysis unit, then a replicate per drawn row
    #
    # In the two matched modes each unit contributes exactly one row, so a draw
    # has the same size as the panel. In the hybrid mode a drawn cluster carries
    # every one of its members, which keeps the draw's expected composition equal
    # to the observed panel; taking one representative per cluster instead would
    # shrink the seven-target 05_CD45 cluster to a single row in every draw and
    # estimate a different quantity from the one reported.
    rng = np.random.default_rng(seed)
    uniq = sorted(set(row_of_unit))
    members = {u: [i for i, x in enumerate(row_of_unit) if x == u] for u in uniq}
    nu = len(uniq)
    bs_obs = np.full((boot, 4), np.nan)
    bs_cor = np.full((boot, 4), np.nan)
    bs_iqr = np.full((boot, 4), np.nan)
    draw_sizes = []
    for b in range(boot):
        drawn = [uniq[i] for i in rng.integers(0, nu, size=nu)]
        rows_b = [i for u in drawn for i in members[u]]
        draw_sizes.append(len(rows_b))
        reps = rng.integers(0, B, size=len(rows_b))
        draw = np.stack([Rrep[r, q] for r, q in zip(rows_b, reps)])   # [n, 4] one replicate each
        wv = within[rows_b]                                            # [n, 4]
        for i in range(4):
            bs_obs[b, i] = _cv(draw[:, i])
            bs_cor[b, i] = _cv_corrected(draw[:, i], wv[:, i])
            col = draw[:, i][np.isfinite(draw[:, i])]
            if col.size > 1:
                bs_iqr[b, i] = np.percentile(col, 75) - np.percentile(col, 25)
    for i, m in enumerate(METRICS):
        for key, arr in (("cv_observed", bs_obs), ("cv_corrected", bs_cor), ("iqr", bs_iqr)):
            lo, hi = np.nanpercentile(arr[:, i], [2.5, 97.5])
            point[m][f"{key}_ci95"] = [float(lo), float(hi)]
    draw_sizes = np.asarray(draw_sizes, float)

    pairs = {}
    for i in range(4):
        for j in range(4):
            if i >= j:
                continue
            d = bs_cor[:, j] - bs_cor[:, i]
            obs = point[METRICS[j]]["cv_corrected"] - point[METRICS[i]]["cv_corrected"]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            pairs[f"cv({SHORT[METRICS[j]]}) - cv({SHORT[METRICS[i]]})"] = {
                "observed": float(obs), "ci95": [float(lo), float(hi)],
                "p_gt_0": float(np.nanmean(d > 0)),
                "excludes_zero": bool(lo > 0 or hi < 0),
            }

    sep = []
    for m in METRICS:
        others = [o for o in METRICS if o != m]
        if all(pairs.get(f"cv({SHORT[o]}) - cv({SHORT[m]})",
                         pairs.get(f"cv({SHORT[m]}) - cv({SHORT[o]})", {})).get("excludes_zero")
               for o in others):
            sep.append(m)

    order = sorted(METRICS, key=lambda m: point[m]["cv_corrected"])

    # A quality can only be placed on the spread axis if its across-row spread
    # is resolved above the uncertainty in its own per-row settling step. That
    # is exactly the corrected CV having a 95% interval whose lower bound clears
    # zero. A quality that fails this is not "tight": it is unmeasured, and
    # calling it the tightest would invert its meaning.
    resolved = [m for m in order if point[m]["cv_corrected_ci95"][0] > 0]
    unresolved = [m for m in order if m not in resolved]

    tight, wide = order[0], order[-1]
    d_ext = bs_cor[:, METRICS.index(wide)] - bs_cor[:, METRICS.index(tight)]
    tcol = bs_cor[:, METRICS.index(tight)]
    ratio = np.where(tcol > 0, bs_cor[:, METRICS.index(wide)] / np.where(tcol > 0, tcol, 1.0), np.nan)
    lo, hi = np.nanpercentile(d_ext, [2.5, 97.5])
    rlo, rhi = (np.nanpercentile(ratio, [2.5, 97.5]) if np.isfinite(ratio).any() else (np.nan, np.nan))
    extreme = {
        "tightest": tight, "widest": wide,
        "difference": float(point[wide]["cv_corrected"] - point[tight]["cv_corrected"]),
        "difference_ci95": [float(lo), float(hi)],
        "p_gt_0": float(np.nanmean(d_ext > 0)),
        "ratio": float(point[wide]["cv_corrected"] / point[tight]["cv_corrected"])
        if point[tight]["cv_corrected"] > 0 else float("nan"),
        "ratio_ci95": [float(rlo), float(rhi)],
        "n_ratio_defined": int(np.isfinite(ratio).sum()),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }

    extreme_resolved = None
    if len(resolved) >= 2:
        rt, rw = resolved[0], resolved[-1]
        d_r = bs_cor[:, METRICS.index(rw)] - bs_cor[:, METRICS.index(rt)]
        rcol = bs_cor[:, METRICS.index(rt)]
        rat = np.where(rcol > 0, bs_cor[:, METRICS.index(rw)] / np.where(rcol > 0, rcol, 1.0),
                       np.nan)
        rlo2, rhi2 = np.nanpercentile(d_r, [2.5, 97.5])
        qlo, qhi = (np.nanpercentile(rat, [2.5, 97.5]) if np.isfinite(rat).any()
                    else (np.nan, np.nan))
        extreme_resolved = {
            "tightest": rt, "widest": rw,
            "difference": float(point[rw]["cv_corrected"] - point[rt]["cv_corrected"]),
            "difference_ci95": [float(rlo2), float(rhi2)],
            "p_gt_0": float(np.nanmean(d_r > 0)),
            "ratio": float(point[rw]["cv_corrected"] / point[rt]["cv_corrected"])
            if point[rt]["cv_corrected"] > 0 else float("nan"),
            "ratio_ci95": [float(qlo), float(qhi)],
            "excludes_zero": bool(rlo2 > 0 or rhi2 < 0),
        }

    # the post-hoc two-versus-two grouping, kept so the report can test whether
    # it still describes the panel rather than assuming it does
    gs = [METRICS.index("fold_identity"), METRICS.index("structural_confidence")]
    go = [METRICS.index("designability"), METRICS.index("interface_quality")]
    # A corrected CV can legitimately be exactly zero, when a quality's observed
    # spread is no larger than its own estimation error. Ratios against zero are
    # undefined rather than infinite, so they are dropped from the interval.
    den = np.nanmean(bs_cor[:, gs], axis=1)
    gr = np.where(den > 0, np.nanmean(bs_cor[:, go], axis=1) / np.where(den > 0, den, 1.0), np.nan)
    glo, ghi = (np.nanpercentile(gr, [2.5, 97.5]) if np.isfinite(gr).any() else (np.nan, np.nan))
    den_pt = float(np.mean([point[METRICS[i]]["cv_corrected"] for i in gs]))
    grouped = {
        "note": "post-hoc grouping chosen after seeing per-metric spreads, on corrected CVs",
        "structure_metrics": [METRICS[i] for i in gs],
        "outcome_metrics": [METRICS[i] for i in go],
        "cv_ratio_outcome_over_structure": float(
            np.mean([point[METRICS[i]]["cv_corrected"] for i in go]) / den_pt)
        if den_pt > 0 else float("nan"),
        "n_ratio_defined": int(np.isfinite(gr).sum()),
        "ci95": [float(glo), float(ghi)], "p_gt_1": float(np.nanmean(gr > 1.0)),
    }

    scheme = (
        "resample the %d %ss with replacement (one row each), then draw one independent "
        "settling-step replicate per drawn row" % (nu, boot_unit)
        if matched else
        "resample the %d %ss with replacement, carrying every target in each drawn cluster, "
        "then draw one independent settling-step replicate per drawn row" % (nu, boot_unit))

    return {
        "status": "post-hoc; the pre-registered test is the ordering test",
        "mode": mode,
        "unit": row_unit,
        "estimand_unit": row_unit,
        "inference_unit": boot_unit,
        "estimands_matched": bool(matched),
        "floor_step": FLOOR,
        "n_rows": len(rows),
        "n_targets": len(targets), "n_units": nu, "n_boot": boot, "bootstrap_draws": boot,
        "n_replicates": int(B), "seed": seed,
        "bootstrap_scheme": scheme,
        "bootstrap_draw_size": {
            "observed_rows": len(rows),
            "mean": float(np.mean(draw_sizes)),
            "min": float(np.min(draw_sizes)),
            "max": float(np.max(draw_sizes)),
        },
        "rows": list(rows),
        "targets": targets, "censored_targets": censored, "cluster_members": unit_members,
        "left_censored": left, "n_left_censored_values": int(sum(len(v) for v in left.values())),
        "dropped_floor_targets": bool(drop_floor),
        "cv_rank_order": order,
        "cv_resolved_metrics": resolved,
        "cv_unresolved_metrics": unresolved,
        "cv_resolved_rule": "corrected-CV 95% interval lower bound above zero",
        "per_metric": point, "pairwise_cv_differences": pairs,
        "grouped_contrast": grouped,
        "cv_separates_from_all_others": sep, "extreme_contrast": extreme,
        "extreme_contrast_resolved": extreme_resolved,
        # raw per-target values, reported at every unit so the figure and the
        # reader always see the 19 targets behind the 12 rows
        "t_half_by_target": per_target_point,
        "t_half_se_by_target": per_target_se,
        "t_half_by_row": {r: {m: float(P[i, k]) for k, m in enumerate(METRICS)}
                          for i, r in enumerate(rows)},
    }


def _arg(name: str, default: str | None = None) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _report(tag: str, p: dict) -> None:
    print(f"\n===== {tag}: rows are {p['unit']}s (n={p['n_rows']}), bootstrap resamples "
          f"{p['inference_unit']}s; estimands matched: {p['estimands_matched']} =====")
    print(f"{'quality':24s} {'median':>8s} {'CV obs':>8s} {'CV corr':>9s} {'CI':>22s} {'SE(steps)':>10s}")
    for m in METRICS:
        e = p["per_metric"][m]
        ci = e["cv_corrected_ci95"]
        print(f"{m:24s} {e['median_t_half']:8.2f} {e['cv_observed']:8.4f} {e['cv_corrected']:9.4f} "
              f"[{ci[0]:8.4f},{ci[1]:8.4f}] {e['mean_within_se_steps']:10.2f}")
    print(f"resolved: {p['cv_resolved_metrics']}   unresolved: {p['cv_unresolved_metrics']}")
    for k, v in sorted(p["pairwise_cv_differences"].items(), key=lambda kv: -kv[1]["observed"]):
        print(f"  {k:26s} {v['observed']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] "
              f"P(>0)={v['p_gt_0']:.4f} {'EXCLUDES 0' if v['excludes_zero'] else 'includes 0'}")
    g = p["grouped_contrast"]
    print(f"  grouped 2v2 ratio {g['cv_ratio_outcome_over_structure']:.4f} "
          f"[{g['ci95'][0]:.4f},{g['ci95'][1]:.4f}] P(>1)={g['p_gt_1']:.4f}")


def main() -> None:
    per_target_dir, out_path = sys.argv[1], sys.argv[2]
    boot = int(_arg("--boot", "10000"))
    seed = int(_arg("--seed", "42"))
    cpath = _arg("--clusters", "")
    clusters = json.load(open(cpath)) if cpath and os.path.exists(cpath) else {}

    targets = sorted({os.path.basename(p).split("_step")[0]
                      for p in glob.glob(os.path.join(per_target_dir, "*_step*_boot.npz"))})
    reps = load_boot(per_target_dir, targets)
    print(f"{len(reps)} targets with complete replicate curves", flush=True)

    th_point, th_rep = {}, {}
    for t, per_m in reps.items():
        pt, rp = {}, {}
        for m, arr in per_m.items():
            rp[m] = t_half_replicates(arr)
            pt[m] = np.nan
        th_point[t], th_rep[t] = pt, rp
    # point curves come from the merged per-target analysis JSONs
    for t in list(th_point):
        p = os.path.join(per_target_dir, f"{t}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        for m in METRICS:
            vals = [j["steps"].get(str(s), {}).get(m, {}).get("decomposition", {}).get("commitment", np.nan)
                    for s in STEPS]
            th_point[t][m] = t_half_curve(np.asarray(vals, float))

    out = {
        "headline_unit": "structural cluster",
        "units_note": (
            "The same analysis at two matched units. by_cluster is the lead: a row is a Foldseek "
            "structural cluster and the bootstrap resamples clusters, which is the honest "
            "independent unit because the largest cluster holds seven of the nineteen targets. "
            "by_target is an optimistic bound: it treats every target as independent at both the "
            "point estimate and the interval. by_cluster_block is the earlier hybrid, kept for "
            "transparency, whose point estimate is over targets while its interval resamples "
            "clusters."),
        "by_cluster": spread(th_point, th_rep, clusters, boot=boot, seed=seed, mode="cluster"),
        "by_target": spread(th_point, th_rep, clusters, boot=boot, seed=seed, mode="target"),
        "by_cluster_block": spread(th_point, th_rep, clusters, boot=boot, seed=seed,
                                   mode="target_block"),
        "by_uncensored": spread(th_point, th_rep, clusters, boot=boot, seed=seed,
                                drop_floor=True, mode="cluster"),
        "by_uncensored_target": spread(th_point, th_rep, clusters, boot=boot, seed=seed,
                                       drop_floor=True, mode="target"),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    for tag in ("by_cluster", "by_target", "by_cluster_block"):
        _report(tag, out[tag])

    # the comparison the whole section rests on, stated at both matched units
    print("\n===== headline pair at both matched units =====")
    for tag in ("by_cluster", "by_target"):
        p = out[tag]
        v = p["pairwise_cv_differences"]["cv(iface) - cv(fold)"]
        print(f"{tag:16s} (n={p['n_rows']} {p['unit']}s): cv(iface) - cv(fold) = "
              f"{v['observed']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
              f"P(>0)={v['p_gt_0']:.4f} -> "
              f"{'CLEARS ZERO' if v['excludes_zero'] else 'DOES NOT CLEAR ZERO'}")
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
