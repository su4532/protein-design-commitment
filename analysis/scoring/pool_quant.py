#!/usr/bin/env python
"""Pool the fold-identity quantisation check across targets.

``quantisation.py`` re-clusters every folded binder of a target with the parent
run's exact Foldseek call, then measures each structure's TM-score to each cluster
representative. That gives two rulers for the same quality on the same structures:

``categorical``
    the cluster label, decomposed with the parent's Gini estimator - the published
    ruler.

``continuous_pairwise``
    the same Gini estimator with the 0/1 same-label distance replaced by the
    structural distance ``1 - TM`` over every pair of designs in the cell. It refers
    to no cluster at all, so nothing about it can drift from step to step, and fed a
    0/1 distance it reproduces the categorical estimator exactly.

``continuous_tm_modal_rep``
    a sensitivity only: TM-score to the modal cluster's representative. The reference
    cluster is re-chosen in every cell, and for some targets it moves across steps, so
    that curve can compare different quantities over time. Targets where it moved are
    flagged in ``modal_reference_stability``.

If fold identity looked tight because a label discards sub-threshold structural
variation, the continuous ruler would settle later or vary more across targets.
This script puts the two settling-step distributions side by side and bootstraps
targets for an interval on the difference in coefficient of variation.

The margin summary answers the companion question: how many completions sit close
enough to a cluster boundary that a slightly different clustering would have
labelled them differently.

    pool_quant.py <quant_dir> <out_json> [--boot 20000]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


from .. import spread as ss  # noqa: E402

STEPS = list(ss.STEPS)
# the primary pair the headline contrast uses, plus the modal-representative variant
# carried only as a labelled sensitivity
PRIMARY = ("categorical", "continuous_pairwise")
RULERS = ("categorical", "continuous_pairwise", "continuous_tm_modal_rep")


def _cv(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2 or x.mean() <= 0:
        return float("nan")
    return float(x.std(ddof=1) / x.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("quant_dir")
    ap.add_argument("out_json")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--clusters", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "parent", "target_clusters.json"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.quant_dir, "*.json")))
    per_target: dict[str, dict] = {}
    for f in files:
        with open(f) as fh:
            j = json.load(fh)
        t = j.get("target")
        if not t or "steps" not in j:
            continue
        per_target[t] = j

    targets = sorted(per_target)
    if not targets:
        raise SystemExit(f"no quantisation results in {args.quant_dir}")

    curves: dict[str, dict[str, list[float]]] = {r: {} for r in RULERS}
    t_half: dict[str, dict[str, float]] = {r: {} for r in RULERS}
    for t in targets:
        st = per_target[t]["steps"]
        for r in RULERS:
            vals = []
            for s in STEPS:
                e = st.get(str(s), {})
                d = e.get(r) or {}
                vals.append(float(d.get("commitment", np.nan)))
            curves[r][t] = vals
            t_half[r][t] = ss.t_half_curve(np.asarray(vals, float), STEPS)

    # margins: how close is a typical completion to a different label?
    margin_rows = []
    for t in targets:
        mo = per_target[t].get("margin_overall", {})
        margin_rows.append([mo.get("mean", np.nan), mo.get("median", np.nan),
                            mo.get("frac_below_0p05", np.nan), mo.get("frac_below_0p10", np.nan)])
    M = np.asarray(margin_rows, float)

    agree = np.asarray([per_target[t].get("clustering_reproduction", {})
                        .get("pair_agreement_with_cached_labels", np.nan) for t in targets], float)

    # how much of the pairwise structural variation the label throws away, over all
    # labels rather than inside one arbitrary cluster
    rounded_rows, ratio_rows, frac_within = [], [], []
    for t in targets:
        st = per_target[t]["steps"]
        for s in STEPS:
            e = st.get(str(s), {})
            ra = e.get("rounded_away") or {}
            if np.isfinite(ra.get("share_rounded_away", np.nan)):
                rounded_rows.append(float(ra["share_rounded_away"]))
                ratio_rows.append(float(ra["mean_distance_ratio_same_over_all"]))
            v = e.get("frac_tm_variance_within_label")
            if v is not None and np.isfinite(v):
                frac_within.append(float(v))
    rounded_rows = np.asarray(rounded_rows, float)
    ratio_rows = np.asarray(ratio_rows, float)
    frac_within = np.asarray(frac_within, float)

    # does the modal reference move across steps? (it must not, for that sensitivity
    # curve to mean anything)
    modal_moved = {}
    for t in targets:
        st = per_target[t]["steps"]
        mods = [st[str(s)].get("modal_cluster") for s in STEPS if str(s) in st]
        mods = [m for m in mods if m is not None]
        modal_moved[t] = {"n_distinct_modal_clusters": len(set(mods)),
                          "stable": len(set(mods)) <= 1,
                          "modal_by_step": mods}

    P = np.array([[t_half[r][t] for r in PRIMARY] for t in targets], float)   # [T, 2]
    full = np.isfinite(P).all(axis=1)
    obs = {r: _cv(P[full, i]) for i, r in enumerate(PRIMARY)}

    rng = np.random.default_rng(42)
    T = int(full.sum())
    Pf = P[full]
    draws = np.empty((args.boot, 3), float)
    for b in range(args.boot):
        idx = rng.integers(0, T, T)
        c0, c1 = _cv(Pf[idx, 0]), _cv(Pf[idx, 1])
        draws[b] = (c0, c1, c1 - c0)
    ok = np.isfinite(draws).all(axis=1)
    D = draws[ok]

    def ci(col: np.ndarray) -> list[float]:
        return [float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))]

    out = {
        "n_targets": len(targets),
        "n_targets_both_rulers_cross": T,
        "targets": targets,
        "steps": STEPS,
        "commitment_curves": {r: curves[r] for r in RULERS},
        "t_half": {r: t_half[r] for r in RULERS},
        "primary_rulers": list(PRIMARY),
        "t_half_summary": {
            r: {
                "median": float(np.nanmedian(P[:, i])),
                "mean": float(np.nanmean(P[:, i])),
                # absolute scatter, reported alongside the CV so a CV move driven by a
                # shift in the mean is not read as a change in absolute spread
                "sd": float(np.nanstd(P[:, i], ddof=1)),
                "min": float(np.nanmin(P[:, i])) if np.isfinite(P[:, i]).any() else float("nan"),
                "max": float(np.nanmax(P[:, i])) if np.isfinite(P[:, i]).any() else float("nan"),
                "n_never_crossed": int((~np.isfinite(P[:, i])).sum()),
                # a curve already above 0.5 at the first capture step is left-censored:
                # its settling step is reported as the floor, but the true value is
                # somewhere below it, which compresses the observed spread
                "n_left_censored_at_floor": int((P[:, i] <= float(STEPS[0])).sum()),
                "floor_step": float(STEPS[0]),
            }
            for i, r in enumerate(PRIMARY)
        },
        "cv_observed": obs,
        "cv_ci": {r: ci(D[:, i]) for i, r in enumerate(PRIMARY)},
        "cv_difference_continuous_minus_categorical": {
            "continuous_ruler": "continuous_pairwise",
            "point": float(obs["continuous_pairwise"] - obs["categorical"]),
            "ci": ci(D[:, 2]),
            "p_two_sided_sign": float(2 * min((D[:, 2] > 0).mean(), (D[:, 2] < 0).mean())),
            "n_bootstrap": int(D.shape[0]),
        },
        "margin": {
            "mean_of_target_means": float(np.nanmean(M[:, 0])),
            "median_of_target_medians": float(np.nanmedian(M[:, 1])),
            "frac_below_0p05_mean": float(np.nanmean(M[:, 2])),
            "frac_below_0p10_mean": float(np.nanmean(M[:, 3])),
            "per_target_mean": {t: float(M[i, 0]) for i, t in enumerate(targets)},
        },
        "clustering_reproduction": {
            "pair_agreement_mean": float(np.nanmean(agree)),
            "pair_agreement_min": float(np.nanmin(agree)) if np.isfinite(agree).any() else float("nan"),
            "per_target": {t: float(agree[i]) for i, t in enumerate(targets)},
        },
        "rounded_away": {
            "n_cells": int(rounded_rows.size),
            "share_mean": float(rounded_rows.mean()) if rounded_rows.size else float("nan"),
            "share_median": float(np.median(rounded_rows)) if rounded_rows.size else float("nan"),
            "share_max": float(rounded_rows.max()) if rounded_rows.size else float("nan"),
            "mean_distance_ratio_same_over_all_mean": float(ratio_rows.mean())
            if ratio_rows.size else float("nan"),
            "note": "share of total pairwise structural distance that lies between designs "
                    "sharing a cluster label, which the 0/1 ruler sets to zero; computed "
                    "over every label, not inside the modal cluster only",
        },
        "modal_reference_stability": {
            "n_targets_stable": int(sum(v["stable"] for v in modal_moved.values())),
            "n_targets": len(modal_moved),
            "unstable_targets": sorted(t for t, v in modal_moved.items() if not v["stable"]),
            "per_target": modal_moved,
            "note": "the modal-representative sensitivity curve is only interpretable for "
                    "targets whose reference cluster is the same at every capture step",
        },
        "frac_tm_variance_inside_modal_label": {
            "n_cells": int(frac_within.size),
            "mean": float(frac_within.mean()) if frac_within.size else float("nan"),
            "median": float(np.median(frac_within)) if frac_within.size else float("nan"),
            "note": "superseded by rounded_away; kept because it is what the earlier "
                    "modal-cluster version reported",
        },
    }

    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"targets {len(targets)}  both rulers cross 0.5 in {T}")
    for i, r in enumerate(PRIMARY):
        print(f"  {r:20s} median t_half {np.nanmedian(P[:, i]):8.2f}   CV {obs[r]:.4f} "
              f"[{out['cv_ci'][r][0]:.4f}, {out['cv_ci'][r][1]:.4f}]")
    d = out["cv_difference_continuous_minus_categorical"]
    print(f"  CV(continuous) - CV(categorical) = {d['point']:+.4f} "
          f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
    ra = out["rounded_away"]
    print(f"  structural distance rounded away by the label: mean {ra['share_mean']:.4f} "
          f"over {ra['n_cells']} cells")
    ms = out["modal_reference_stability"]
    print(f"  modal reference stable in {ms['n_targets_stable']}/{ms['n_targets']} targets"
          + (f"; moved for {', '.join(ms['unstable_targets'])}" if ms['unstable_targets'] else ""))
    print(f"  margin mean {out['margin']['mean_of_target_means']:.4f}, "
          f"frac below 0.05 {out['margin']['frac_below_0p05_mean']:.4f}")
    print(f"  clustering pair agreement {out['clustering_reproduction']['pair_agreement_mean']:.4f}")
    print(f"-> {args.out_json}")


if __name__ == "__main__":
    main()
