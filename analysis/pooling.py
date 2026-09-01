#!/usr/bin/env python
"""Cross-target synthesis: the ordering claim, the ceilings, and the readability gap.

Per-target curves come from ``analyze.py``. This pools them.

The headline claim is an **ordering**, not a level: a commitment curve rises with
the capture step by construction (later states have less noise left to inject), so
"commitment goes up" is not a finding. What is a finding is *which quality settles
first*, tested **paired within each target** so the per-target difficulty that
dominates the levels cancels.

Two summaries of "when":

``t_half``  the step at which a metric's commitment curve first crosses 0.5,
            linearly interpolated between capture steps. Undefined if the curve
            never crosses; reported as censored rather than imputed.
``paired``  the difference in commitment between two metrics at the *same*
            capture step, averaged over steps. This is defined even when a curve
            never crosses 0.5, so it is the primary test and ``t_half`` is the
            readable summary.

Uncertainty is bootstrapped over **Foldseek clusters of the target structures**,
not over target names: several benchmark targets are the same or homologous
proteins, and resampling names would treat them as independent.

    python pool_targets.py <per_target_dir> <out.json> [--target-clusters clusters.json]
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

from .spread_raw import compute_spread  # noqa: E402  (same directory, not a package)

from .commitment.grid import steps as _grid_steps  # noqa: E402

#: capture grid for this run; ``CAPTURE_STEPS`` in the environment overrides it
STEPS = _grid_steps()
METRIC_ORDER = ("fold_identity", "designability", "structural_confidence", "interface_quality")
#: a per-target paired commitment difference smaller than this counts as no
#: evidence of an ordering for that target, not as a vote in either direction
MARGIN = 0.02
LABELS = {
    "fold_identity": "fold identity",
    "designability": "designability (redesign-and-refold scRMSD)",
    "structural_confidence": "structural confidence",
    "interface_quality": "interface quality",
}
#: readout families reported separately, so a linear-activation claim is
#: answered by linear activation readouts and not by the geometry baseline or
#: the non-linear capacity control
FAMILIES = ("linear_activation", "geometry", "mlp")


def t_half(steps: list[int], vals: list[float], level: float = 0.5) -> float | None:
    """First crossing of ``level``, linearly interpolated. ``None`` if never crossed."""
    pts = [(s, v) for s, v in zip(steps, vals) if np.isfinite(v)]
    if not pts:
        return None
    if pts[0][1] >= level:
        return float(pts[0][0])  # already above at the earliest capture step
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if v1 >= level:
            if v1 == v0:
                return float(s1)
            return float(s0 + (s1 - s0) * (level - v0) / (v1 - v0))
    return None


def load_targets(d: str) -> dict[str, dict]:
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(p))
        # the same directory also holds derived summaries (ordering_watch.json,
        # pooled.json, target_clusters.json); a per-target analysis is identified
        # by its shape, not by a filename blacklist
        if isinstance(j, dict) and "target" in j and "steps" in j:
            out[j["target"]] = j
    return out


def curves(per_target: dict[str, dict], field: str = "commitment") -> dict[str, dict[str, dict[int, float]]]:
    """``metric -> target -> {step: value}``."""
    out: dict[str, dict[str, dict[int, float]]] = {}
    for tgt, j in per_target.items():
        for s_str, cell in j["steps"].items():
            s = int(s_str)
            for m, v in cell.items():
                if m.startswith("_") or "decomposition" not in v:
                    continue
                out.setdefault(m, {}).setdefault(tgt, {})[s] = float(v["decomposition"][field])
    return out


def probe_ceiling_curves(per_target: dict[str, dict]) -> dict[str, dict[str, dict[int, float]]]:
    """The ceiling the readout is actually measured against: ``metric -> target -> {step: value}``.

    Equals the metric's commitment for the continuous qualities, and the modal
    fold-membership commitment for fold identity.
    """
    out: dict[str, dict[str, dict[int, float]]] = {}
    for tgt, j in per_target.items():
        for s_str, cell in j["steps"].items():
            s = int(s_str)
            for m, v in cell.items():
                if m.startswith("_") or "decomposition" not in v:
                    continue
                c = v.get("probe_target", {}).get("ceiling", v["decomposition"])["commitment"]
                out.setdefault(m, {}).setdefault(tgt, {})[s] = float(c)
    return out


def readability_curves(per_target: dict[str, dict]) -> dict:
    out: dict[str, dict[str, dict[int, dict]]] = {}
    for tgt, j in per_target.items():
        for s_str, cell in j["steps"].items():
            s = int(s_str)
            for m, v in cell.items():
                if m.startswith("_") or "best_readout" not in v:
                    continue
                out.setdefault(m, {}).setdefault(tgt, {})[s] = v["best_readout"]
    return out


def block_bootstrap(values_by_target: dict[str, float], clusters: dict[str, str], n_boot=5000, seed=42) -> dict:
    """Percentile CI resampling target *clusters*, then averaging within a cluster."""
    by_cluster: dict[str, list[float]] = {}
    for t, v in values_by_target.items():
        if np.isfinite(v):
            by_cluster.setdefault(clusters.get(t, t), []).append(v)
    keys = sorted(by_cluster)
    if not keys:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_clusters": 0,
                "n_targets": 0, "n_targets_offered": len(values_by_target)}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        means.append(float(np.mean([np.mean(by_cluster[keys[i]]) for i in pick])))
    point = float(np.mean([np.mean(by_cluster[k]) for k in keys]))
    return {
        "mean": point,
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n_clusters": len(keys),
        # targets that actually contributed a finite value. A quality with zero
        # split-half reliability in a cell has an undefined noise-corrected
        # readout, so that target drops out of the readability pool for that
        # step and the pooled n is smaller than the number of targets run.
        "n_targets": sum(len(v) for v in by_cluster.values()),
        "n_targets_offered": len(values_by_target),
    }


def main() -> None:
    src, out_path = sys.argv[1], sys.argv[2]
    clusters_path = None
    for i, a in enumerate(sys.argv):
        if a == "--target-clusters":
            clusters_path = sys.argv[i + 1]

    per_target = load_targets(src)
    if not per_target:
        raise SystemExit(f"no per-target JSONs under {src}")

    # Restrict every derivation to the selected capture grid. ``load_targets``
    # returns whatever steps the per-cell analyses produced, which is the full
    # ten-step grid; when ``CAPTURE_STEPS`` asks for the parent's seven steps,
    # the commitment curves, the readouts and above all the t_half
    # interpolation have to see only those seven, or the "seven-step"
    # derivation silently interpolates through all ten and the two grids are
    # not actually being compared.
    keep = {str(s) for s in STEPS}
    dropped = 0
    for j in per_target.values():
        extra = [k for k in j["steps"] if k not in keep]
        dropped += len(extra)
        for k in extra:
            del j["steps"][k]
        missing = sorted(keep - set(j["steps"]), key=int)
        if missing:
            raise SystemExit(f"{j['target']}: capture steps {missing} requested but absent")
    print(f"capture grid {sorted(STEPS)}: {dropped} cells outside the grid dropped", flush=True)

    targets = sorted(per_target)
    clusters = json.load(open(clusters_path)) if clusters_path and os.path.exists(clusters_path) else {}
    clusters = {t: clusters.get(t, t) for t in targets}
    print(f"{len(targets)} targets in {len(set(clusters.values()))} structural clusters", flush=True)

    commit = curves(per_target, "commitment")
    childsh = curves(per_target, "child_share")
    rootsh = curves(per_target, "root_share")
    read = readability_curves(per_target)

    # ---- instrument control ------------------------------------------------
    ctrl = commit.get("control_binder_length", {})
    ctrl_vals = [v for t in ctrl for v in ctrl[t].values()]
    ctrl_root = rootsh.get("control_binder_length", {})
    ctrl_root_vals = [v for t in ctrl_root for v in ctrl_root[t].values()]
    control = {
        "n_cells": len(ctrl_vals),
        "commitment_min": float(np.min(ctrl_vals)) if ctrl_vals else float("nan"),
        "commitment_max": float(np.max(ctrl_vals)) if ctrl_vals else float("nan"),
        "root_share_min": float(np.min(ctrl_root_vals)) if ctrl_root_vals else float("nan"),
        "passes": bool(ctrl_root_vals and np.min(ctrl_root_vals) > 1 - 1e-6),
    }
    print(f"instrument control: root share min {control['root_share_min']:.6f} over {control['n_cells']} cells "
          f"-> {'PASS' if control['passes'] else 'FAIL'}", flush=True)

    # ---- pooled curves -----------------------------------------------------
    pooled = {}
    for field, src_curves in (("commitment", commit), ("child_share", childsh), ("root_share", rootsh)):
        pooled[field] = {}
        for m in METRIC_ORDER:
            if m not in src_curves:
                continue
            pooled[field][m] = {}
            for s in STEPS:
                vals = {t: src_curves[m][t].get(s, np.nan) for t in src_curves[m]}
                e = block_bootstrap(vals, clusters)
                e["by_target"] = block_bootstrap(vals, {t: t for t in vals})
                pooled[field][m][str(s)] = e

    # Readability is reported against **the probe's own ceiling**, which is not
    # always the metric's commitment. For the three continuous qualities the two
    # coincide. For fold identity they do not: the decomposition is categorical
    # over Foldseek clusters, while a linear readout has to predict something
    # real-valued, so its target is membership of the modal cluster - a binary
    # reduction with its own, higher, commitment. Dividing fold-identity
    # readability by the categorical commitment compares two different
    # quantities and can exceed 1.
    pceil = probe_ceiling_curves(per_target)
    for key, src in (("readability", "readability"),
                     ("readability_frac_of_ceiling", "readability_frac_of_ceiling")):
        pooled[key] = {}
        for m in METRIC_ORDER:
            if m not in read:
                continue
            pooled[key][m] = {}
            for s in STEPS:
                vals = {t: read[m][t].get(s, {}).get(src, np.nan) for t in read[m]}
                pooled[key][m][str(s)] = block_bootstrap(vals, clusters)
    pooled["probe_ceiling"] = {}
    for m in METRIC_ORDER:
        if m not in pceil:
            continue
        pooled["probe_ceiling"][m] = {}
        for s in STEPS:
            vals = {t: pceil[m][t].get(s, np.nan) for t in pceil[m]}
            pooled["probe_ceiling"][m][str(s)] = block_bootstrap(vals, clusters)

    # Readability per readout family. The reported number is the linear
    # activation family; geometry is the baseline it has to beat and the MLP is
    # a capacity control, so all three are carried separately rather than
    # collapsed into one maximum.
    pooled["readability_by_family"] = {}
    for m in METRIC_ORDER:
        fam_out: dict[str, dict] = {}
        for fam in FAMILIES:
            per_s: dict[str, dict] = {}
            for s in STEPS:
                vals = {}
                for tgt, j in per_target.items():
                    e = j["steps"].get(str(s), {}).get(m, {})
                    v = (e.get("nested") or {}).get(fam, {}).get("readability")
                    if v is not None:
                        vals[tgt] = float(v)
                if vals:
                    per_s[str(s)] = block_bootstrap(vals, clusters)
            if per_s:
                fam_out[fam] = per_s
        if fam_out:
            pooled["readability_by_family"][m] = fam_out

    # Where along the trajectory the linear readout is at its best, and whether
    # that beats the predefined step-100 capture point. If readability is still climbing
    # past it, that choice left signal on the table. A pooled maximum alone
    # cannot answer this, because the steps are measured on the same targets and
    # the gap can be small - so the comparison is paired within target and then
    # resampled over clusters, exactly like every other interval here.
    REF_STEP = 100
    pooled["readability_step_contrast"] = {"reference_step": REF_STEP,
                                           "family": "linear_activation", "metrics": {}}
    for m in METRIC_ORDER:
        fam = (pooled["readability_by_family"].get(m) or {}).get("linear_activation") or {}
        if not fam:
            continue
        cand = [(fam[str(s)]["mean"], s) for s in STEPS
                if str(s) in fam and fam[str(s)].get("mean") is not None
                and np.isfinite(fam[str(s)]["mean"])]
        if not cand:
            continue
        peak_mean, peak_step = max(cand)
        per_step_vals: dict[int, dict[str, float]] = {}
        for s in STEPS:
            vals = {}
            for tgt, j in per_target.items():
                v = ((j["steps"].get(str(s), {}).get(m, {}) or {}).get("nested") or {}) \
                    .get("linear_activation", {}).get("readability")
                if v is not None and np.isfinite(float(v)):
                    vals[tgt] = float(v)
            per_step_vals[s] = vals
        vs_ref = {}
        ref_vals = per_step_vals.get(REF_STEP, {})
        for s in STEPS:
            if s == REF_STEP:
                continue
            paired = {t: per_step_vals[s][t] - ref_vals[t]
                      for t in per_step_vals[s] if t in ref_vals}
            if len(paired) >= 3:
                vs_ref[str(s)] = block_bootstrap(paired, clusters)
        pooled["readability_step_contrast"]["metrics"][m] = {
            "peak_step": int(peak_step), "peak_mean": float(peak_mean),
            "reference_mean": float(fam[str(REF_STEP)]["mean"]) if str(REF_STEP) in fam
            else float("nan"),
            "paired_vs_reference": vs_ref,
        }

    # Which configuration the nested selection actually picked, per family.
    chosen_acc: dict[str, dict[str, int]] = {}
    for tgt, j in per_target.items():
        for s_str, cell in j["steps"].items():
            for m, v in cell.items():
                lin = ((v or {}).get("nested") or {}).get("linear_activation", {})
                for c in lin.get("chosen", []):
                    ro = c.split("|")[0]
                    chosen_acc.setdefault(m, {})[ro] = chosen_acc.setdefault(m, {}).get(ro, 0) + 1
    pooled["nested_choice_counts"] = chosen_acc

    # leakage: the same nested protocol with the lineage block removed
    leak = {}
    for tgt, j in per_target.items():
        for s_str, cell in j["steps"].items():
            for m, v in cell.items():
                if m.startswith("_") or not isinstance(v, dict):
                    continue
                b = v.get("best_readout") or {}
                r = v.get("best_readout_random_split") or {}
                if "r2_observed" in b and "r2_observed" in r:
                    leak.setdefault(m, {}).setdefault(s_str, []).append(
                        (r["r2_observed"], b["r2_observed"])
                    )
    pooled["leakage"] = {
        m: {
            s: {
                "random_split_r2_mean": float(np.nanmean([a for a, _ in v])),
                "root_blocked_r2_mean": float(np.nanmean([b for _, b in v])),
                "n": len(v),
            }
            for s, v in per_s.items()
        }
        for m, per_s in leak.items()
    }

    # ---- t_half per target and metric -------------------------------------
    th = {}
    for m in METRIC_ORDER:
        if m not in commit:
            continue
        th[m] = {}
        for t in commit[m]:
            steps = sorted(commit[m][t])
            th[m][t] = t_half(steps, [commit[m][t][s] for s in steps])
    pooled["t_half_per_target"] = th
    pooled["t_half_pooled"] = {
        m: block_bootstrap({t: v for t, v in th[m].items() if v is not None}, clusters)
        for m in th
    }
    pooled["t_half_censored"] = {m: sorted([t for t, v in th[m].items() if v is None]) for m in th}

    # ---- how consistent is the ORDER itself, target by target? ------------
    # Kendall's W over t_half ranks: 1 means every target ranks the four
    # qualities identically, 0 means the rankings are unrelated. This is the
    # single number the pre-registered ordering claim lives or dies by, so the
    # report reads it from here rather than from a side script.
    rank_rows, rank_targets = [], []
    for t in sorted({t for m in th for t in th[m]}):
        vals = [th[m].get(t) for m in METRIC_ORDER if m in th]
        if len(vals) == len(METRIC_ORDER) and all(v is not None for v in vals):
            order_idx = np.argsort(np.argsort(np.asarray(vals, float))) + 1
            rank_rows.append([float(x) for x in order_idx])
            rank_targets.append(t)
    if len(rank_rows) >= 2:
        r = np.asarray(rank_rows, float)
        n_j, m_i = r.shape
        col = r.sum(axis=0)
        denom = n_j ** 2 * (m_i ** 3 - m_i) / 12.0
        w = float(((col - col.mean()) ** 2).sum() / denom) if denom > 0 else float("nan")
    else:
        w = float("nan")
    distinct = len({tuple(sorted(range(len(METRIC_ORDER)), key=lambda i: row[i]))
                    for row in rank_rows})
    pooled["order_consistency"] = {
        "kendall_w_t_half": w, "n_targets_full_rank": len(rank_rows),
        "targets_full_rank": rank_targets, "n_distinct_orders": distinct,
        "metrics": list(METRIC_ORDER),
    }

    # ---- POST-HOC: how much does t_half vary from target to target? --------
    # The pre-registered claim is the ordering, tested below. This measures the
    # across-target spread of each quality's settling time, which is what the
    # ordering result turned into once the ordering proved target-dependent.
    # Uncertainty here resamples TARGETS, not clusters, because the question is
    # about variability across targets and collapsing homologues into one value
    # would remove part of the very spread being estimated. The homologous pairs
    # are listed in the report so the reader can see how much of the spread
    # could be carried by them.
    by_target = {t: {m: th[m].get(t) for m in METRIC_ORDER if m in th}
                 for t in sorted({t for m in th for t in th[m]})}
    # Per-layer readability of interface quality, aggregated here rather than in
    # the figure script: figures are built on the pod, which cannot see the
    # per-target JSONs on the cluster, so everything a figure needs has to be in
    # this file.
    layer_acc: dict[str, list[float]] = {}
    n_layers = 0
    for tgt, j in per_target.items():
        for s_str, cell in j["steps"].items():
            v = cell.get("interface_quality") or {}
            for key, gg in (v.get("probe_grid") or {}).items():
                ro, layer = key.split("|")
                if ro != "binder" or "readability" not in gg:
                    continue
                val = gg.get("readability")
                if val is None or not np.isfinite(val):
                    continue
                n_layers = max(n_layers, int(layer) + 1)
                layer_acc.setdefault(f"{int(s_str)}|{int(layer)}", []).append(float(val))
    pooled["layer_readability"] = {
        "metric": "interface_quality", "n_layers": n_layers, "steps": list(STEPS),
        "mean": {k: float(np.mean(v)) for k, v in sorted(layer_acc.items())},
        "n": {k: len(v) for k, v in sorted(layer_acc.items())},
    }

    pooled["t_half_spread_uncorrected"] = compute_spread(by_target)
    # sensitivity: drop targets where a quality is already past half commitment
    # at the earliest capture step, so its settling step is recorded at the
    # floor rather than measured. Those are real early-settling targets, so this
    # is a check on which targets drive the middle of the ranking, not a fix.
    if pooled["t_half_spread_uncorrected"].get("n_left_censored_values"):
        pooled["t_half_spread_uncorrected"]["by_uncensored"] = compute_spread(
            by_target, drop_floor=True)
    # sensitivity: the same analysis with homologous targets collapsed into one
    # row each. Seven of the nineteen benchmark targets are a single
    # immunoglobulin-like Foldseek cluster, so the target-level intervals above
    # are optimistic. Collapsing also removes real within-cluster variation, so
    # this is a floor on the contrast rather than a better estimate of it.
    if len(set(clusters.values())) < len(clusters):
        pooled["t_half_spread_uncorrected_by_cluster"] = compute_spread(by_target, clusters=clusters)

    # The reported spread comes from spread_stats2, which recomputes each
    # target's whole curve under 2000 resamples of its own roots and children
    # and subtracts each quality's own estimation variance before comparing
    # across targets. The uncorrected version above is kept only so the
    # difference the correction makes is visible.
    spread_path = None
    for i, a in enumerate(sys.argv):
        if a == "--spread":
            spread_path = sys.argv[i + 1]
    if spread_path and os.path.exists(spread_path):
        sp = json.load(open(spread_path))
        # The lead analysis is the one whose point estimate and interval share a
        # unit: a row is a Foldseek structural cluster and the bootstrap
        # resamples clusters. The target-level version is carried beside it as
        # the optimistic bound, and the earlier hybrid (target-level point,
        # cluster-blocked interval) is carried so the change is visible.
        pooled["t_half_spread_posthoc"] = sp.get("by_cluster", {})
        pooled["t_half_spread_posthoc"]["by_uncensored"] = sp.get("by_uncensored", {})
        pooled["t_half_spread_posthoc_by_target"] = sp.get("by_target", {})
        pooled["t_half_spread_posthoc_by_target"]["by_uncensored"] = sp.get(
            "by_uncensored_target", {})
        pooled["t_half_spread_posthoc_cluster_block"] = sp.get("by_cluster_block", {})
        pooled["t_half_spread_units_note"] = sp.get("units_note", "")
    else:
        pooled["t_half_spread_posthoc"] = pooled["t_half_spread_uncorrected"]

    loco_path = None
    for i, a in enumerate(sys.argv):
        if a == "--loco":
            loco_path = sys.argv[i + 1]
    if loco_path and os.path.exists(loco_path):
        pooled["loco_transfer"] = json.load(open(loco_path))

    # designability provenance: the repeat-measure control and the best-of-8
    # versus best-of-17 calibration, both required by the plan
    rep_r, rep_mad, n_rep = [], [], 0
    for tgt, j in per_target.items():
        for s_str, cell in j["steps"].items():
            rm = (cell.get("designability") or {}).get("repeat_measure") or {}
            if rm.get("n"):
                n_rep += int(rm["n"])
                if np.isfinite(rm.get("r", np.nan)):
                    rep_r.append(float(rm["r"]))
                if np.isfinite(rm.get("mad", np.nan)):
                    rep_mad.append(float(rm["mad"]))
    calib = {}
    if "designability_bo8" in commit and "designability" in commit:
        pairs = {t: float(np.mean([commit["designability"][t][s] - commit["designability_bo8"][t][s]
                                   for s in STEPS
                                   if s in commit["designability"][t] and s in commit["designability_bo8"][t]]))
                 for t in set(commit["designability"]) & set(commit["designability_bo8"])}
        calib = {"commitment_diff_bo17_minus_bo8": block_bootstrap(pairs, clusters)}
    pooled["designability_provenance"] = {
        "repeat_measure": {"n_states": n_rep,
                           "mean_r": float(np.mean(rep_r)) if rep_r else float("nan"),
                           "mean_abs_diff_A": float(np.mean(rep_mad)) if rep_mad else float("nan"),
                           "n_cells": len(rep_r)},
        "calibration_bo8_vs_bo17": calib,
    }

    # How far the quantity the first pass reported sits from the planned one.
    # `selfconsistency_rmsd` is Complexa's own initial-guess RMSD for the
    # generated sequence; `designability` is the redesign-and-refold criterion.
    if "selfconsistency_rmsd" in commit and "designability" in commit:
        common = sorted(set(commit["designability"]) & set(commit["selfconsistency_rmsd"]))
        d_pairs = {t: float(np.mean([commit["designability"][t][s] - commit["selfconsistency_rmsd"][t][s]
                                     for s in STEPS
                                     if s in commit["designability"][t]
                                     and s in commit["selfconsistency_rmsd"][t]]))
                   for t in common}
        th_sub = {t: t_half(sorted(commit["selfconsistency_rmsd"][t]),
                            [commit["selfconsistency_rmsd"][t][s]
                             for s in sorted(commit["selfconsistency_rmsd"][t])])
                  for t in common}
        th_des = {t: th.get("designability", {}).get(t) for t in common}
        both = [t for t in common if th_sub.get(t) is not None and th_des.get(t) is not None]
        pooled["substitute_comparison"] = {
            "paired_commitment_diff": block_bootstrap(d_pairs, clusters),
            "n_targets": len(common),
            "t_half_spearman": float(spearmanr([th_des[t] for t in both],
                                               [th_sub[t] for t in both]).correlation)
            if len(both) > 3 else float("nan"),
            "t_half_median_planned": float(np.median([th_des[t] for t in both])) if both else float("nan"),
            "t_half_median_substitute": float(np.median([th_sub[t] for t in both])) if both else float("nan"),
        }

    # ---- the ordering claim, tested paired within target -------------------
    ordering = {}
    for a, b in itertools.combinations([m for m in METRIC_ORDER if m in commit], 2):
        common = sorted(set(commit[a]) & set(commit[b]))
        # paired difference in commitment at matched steps: positive means a is
        # further along (settles earlier) than b
        per_t = {}
        for t in common:
            ds = [commit[a][t][s] - commit[b][t][s] for s in STEPS
                  if s in commit[a][t] and s in commit[b][t]
                  and np.isfinite(commit[a][t][s]) and np.isfinite(commit[b][t][s])]
            if ds:
                per_t[t] = float(np.mean(ds))
        boot = block_bootstrap(per_t, clusters)
        boot_by_target = block_bootstrap(per_t, {t: t for t in per_t})
        n_pos = int(sum(1 for v in per_t.values() if v > 0))
        # Whether the ordering is the SAME target to target is a separate question
        # from whether it holds on average, and it is the one that decides whether
        # the headline is an ordering or a target-dependent contrast. A per-target
        # difference below MARGIN is treated as no evidence rather than as a vote.
        a_first = sorted(t for t, v in per_t.items() if v > MARGIN)
        b_first = sorted(t for t, v in per_t.items() if v < -MARGIN)
        decided = len(a_first) + len(b_first)
        if decided == 0:
            consistency = "no margin"
        elif not b_first or not a_first:
            consistency = "consistent"
        elif min(len(a_first), len(b_first)) / decided >= 0.25:
            consistency = "split"
        else:
            consistency = "majority"
        # the same comparison expressed in steps
        th_pairs = {t: th[b][t] - th[a][t] for t in common
                    if th.get(a, {}).get(t) is not None and th.get(b, {}).get(t) is not None}
        ordering[f"{a}__before__{b}"] = {
            "paired_commitment_diff": boot,
            "paired_commitment_diff_by_target": boot_by_target,
            "n_targets_positive": n_pos,
            "n_targets": len(per_t),
            "sign_test_frac_positive": float(n_pos / len(per_t)) if per_t else float("nan"),
            "paired_t_half_diff": block_bootstrap(th_pairs, clusters),
            "verdict": (
                "supported" if boot["lo"] > 0 else "refuted" if boot["hi"] < 0 else "not separated"
            ),
            "consistency": consistency,
            "margin": MARGIN,
            "targets_a_first": a_first,
            "targets_b_first": b_first,
            "n_within_margin": len(per_t) - decided,
            "per_target_diff": per_t,
        }
        print(
            f"{LABELS.get(a, a):24s} before {LABELS.get(b, b):24s}: "
            f"paired commitment diff {boot['mean']:+.4f} [{boot['lo']:+.4f},{boot['hi']:+.4f}] "
            f"({n_pos}/{len(per_t)} targets positive) -> {ordering[f'{a}__before__{b}']['verdict']}",
            flush=True,
        )

    # ---- refutation check: already settled before the search can act -------
    early = {}
    for m in METRIC_ORDER:
        if m not in commit:
            continue
        vals = {t: commit[m][t].get(10, np.nan) for t in commit[m]}
        early[m] = block_bootstrap(vals, clusters)
    pooled["commitment_at_step10"] = early

    out = {
        "targets": targets,
        "target_clusters": clusters,
        "n_structural_clusters": len(set(clusters.values())),
        "instrument_control": control,
        "pooled": pooled,
        "ordering": ordering,
        "capture_steps": list(STEPS),
        "metric_labels": LABELS,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
