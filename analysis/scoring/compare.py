#!/usr/bin/env python
"""Every number the correction touches, beside the one the paper published.

The corrected curves are written in the source analysis's own per-target layout, so the
pooling here is the parent's audited code (``spread_stats2.spread`` for the spread,
``pool_targets.block_bootstrap`` for the pooled settling step), not a
reimplementation. Running the same code over the *uncorrected* recomputation is the
check that the pipeline reproduces the published numbers before anything is claimed
about how the corrected ones differ.

What this file reports for readability is only the scale factor: readability on the
total-variance scale is ``r2_corrected`` times the commitment, so it moves by exactly
the commitment ratio. The harder half of the readability question, whether
``r2_corrected`` itself is attenuated by ruler error the parent's disattenuation
misses, is answered in ``readability_corrected.py``. No probe is re-run in either
place: removing noise from a target changes the scale a fit is scored on, not the fit.

    compare.py <corrected_dir> <out_json> [--raw-suffix _boot_raw]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

# Maintained estimators are imported from the package so every scoring path
# uses the same implementation. Reference pooled results and the cluster map
# are read from a caller-configurable local directory.
PARENT_RESULTS = os.environ.get(
    "PARENT_RESULTS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "parent"),
)

from .. import spread as ss  # noqa: E402
from ..pooling import block_bootstrap  # noqa: E402

STEPS = ss.STEPS
METRICS = ss.METRICS
EXTRA = ["iptm"]


def load_boot(directory: str, targets: list[str], suffix: str, metrics: list[str]) -> dict:
    out: dict[str, dict[str, np.ndarray]] = {}
    for t in targets:
        cols: dict[str, list] = {}
        for s in STEPS:
            p = os.path.join(directory, f"{t}_step{s}{suffix}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            for m in metrics:
                if m in z.files:
                    cols.setdefault(m, []).append((s, z[m]))
        keep = {m: np.stack([v for _, v in sorted(rows)], axis=0)
                for m, rows in cols.items() if len(rows) == len(STEPS)}
        if keep:
            out[t] = keep
    return out


def point_curves(directory: str, targets: list[str], field: str) -> dict:
    """target -> metric -> settling step, from the per-target commitment curves."""
    th: dict[str, dict[str, float]] = {}
    for t in targets:
        p = os.path.join(directory, f"{t}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        th[t] = {}
        for m in METRICS + EXTRA:
            vals = [j["steps"].get(str(s), {}).get(m, {}).get("decomposition", {}).get(field, np.nan)
                    for s in STEPS]
            th[t][m] = ss.t_half_curve(np.asarray(vals, float))
    return th


def _cv(vals: np.ndarray) -> float:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size < 2 or not np.isfinite(np.mean(v)) or np.mean(v) == 0:
        return float("nan")
    return float(np.std(v, ddof=1) / np.mean(v))


def combined_sensitivity(th_raw_point: dict, th_corr_point: dict, quant_pooled: dict,
                         clusters: dict, boot: int, seed: int) -> dict:
    """Both robustness checks on one footing, as a 2x2.

    The two checks touch different sides of the published contrast. Removing the
    scoring noise changes interface quality's settling step; replacing the discrete
    fold label with a continuous structural score changes fold identity's. Each was
    measured on its own scale above, so neither answers what happens when both apply.

    Everything here is the *observed* coefficient of variation of per-unit settling
    steps, not the parent's noise-corrected CV. The continuous-TM fold ruler has no
    within-target replicate curves, so its estimation variance cannot be subtracted;
    using the observed CV for all four cells keeps the 2x2 internally comparable at
    the cost of not matching the published statistic exactly. The published cell is
    reported here too, so the offset between the two statistics is visible.
    """
    # the label-free pairwise structural ruler is the primary continuous one; the
    # modal-representative variant is a flagged sensitivity and is not used here
    th = quant_pooled.get("t_half", {})
    cont = th.get("continuous_pairwise") or th.get("continuous_tm", {})
    cat = th.get("categorical", {})
    targets = sorted(t for t in th_raw_point
                     if t in cont and t in th_corr_point
                     and np.isfinite(cont[t])
                     and np.isfinite(th_raw_point[t].get("interface_quality", np.nan))
                     and np.isfinite(th_corr_point[t].get("interface_quality", np.nan))
                     and np.isfinite(th_raw_point[t].get("fold_identity", np.nan)))
    if len(targets) < 3:
        return {"insufficient": True, "n_targets": len(targets)}

    series = {
        "fold_discrete_label": np.array([th_raw_point[t]["fold_identity"] for t in targets]),
        "fold_continuous_structural": np.array([cont[t] for t in targets]),
        "interface_published": np.array([th_raw_point[t]["interface_quality"] for t in targets]),
        "interface_noise_corrected": np.array(
            [th_corr_point[t]["interface_quality"] for t in targets]),
    }
    # consistency: the categorical settling step derived inside the quantisation pass
    # should be the same number as the fold-identity settling step in the main pass
    agree = None
    if all(t in cat for t in targets):
        a = np.array([cat[t] for t in targets])
        agree = float(np.max(np.abs(a - series["fold_discrete_label"])))

    cluster_of = np.array([clusters.get(t, t) for t in targets])
    keys = sorted(set(cluster_of.tolist()))
    idx_by_cluster = {k: np.where(cluster_of == k)[0] for k in keys}

    def collapse(vec: np.ndarray, unit: str) -> np.ndarray:
        if unit == "by_target":
            return vec
        return np.array([np.mean(vec[idx_by_cluster[k]]) for k in keys])

    cells = [("published", "interface_published", "fold_discrete_label"),
             ("noise_corrected_only", "interface_noise_corrected", "fold_discrete_label"),
             ("continuous_ruler_only", "interface_published", "fold_continuous_structural"),
             ("both", "interface_noise_corrected", "fold_continuous_structural")]

    out = {"n_targets": len(targets), "n_clusters": len(keys),
           "targets": targets,
           "statistic": "observed CV of per-unit settling step (not noise-corrected)",
           "categorical_t_half_agreement_max_abs_diff": agree,
           "per_series_t_half": {k: {t: float(v[i]) for i, t in enumerate(targets)}
                                 for k, v in series.items()}}

    for unit in ("by_cluster", "by_target"):
        rng = np.random.default_rng(seed)
        n_units = len(keys) if unit == "by_cluster" else len(targets)
        # resample the analysis unit itself, matching the parent's convention
        draws = rng.integers(0, n_units, size=(boot, n_units))
        collapsed = {k: collapse(v, unit) for k, v in series.items()}
        rows = {}
        for name, i_key, f_key in cells:
            gap = _cv(collapsed[i_key]) - _cv(collapsed[f_key])
            reps = np.array([_cv(collapsed[i_key][d]) - _cv(collapsed[f_key][d]) for d in draws])
            reps = reps[np.isfinite(reps)]
            rows[name] = {
                "cv_interface": _cv(collapsed[i_key]),
                "cv_fold": _cv(collapsed[f_key]),
                "gap": gap,
                "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
                "frac_reps_above_zero": float(np.mean(reps > 0)),
                "n_reps": int(reps.size),
                "interface_series": i_key,
                "fold_series": f_key,
            }
        out[unit] = {"n_units": n_units, "cells": rows,
                     "gap_removed_by_noise_correction":
                         rows["published"]["gap"] - rows["noise_corrected_only"]["gap"],
                     "gap_removed_by_continuous_ruler":
                         rows["published"]["gap"] - rows["continuous_ruler_only"]["gap"],
                     "gap_removed_by_both":
                         rows["published"]["gap"] - rows["both"]["gap"]}
    return out


def run_spread(directory: str, targets: list[str], suffix: str, field: str, clusters: dict,
               boot: int, seed: int) -> dict:
    reps = load_boot(directory, targets, suffix, METRICS)
    th_point_all = point_curves(directory, targets, field)
    th_rep, th_point = {}, {}
    for t, per_m in reps.items():
        th_rep[t] = {m: ss.t_half_replicates(arr) for m, arr in per_m.items()}
        th_point[t] = {m: th_point_all.get(t, {}).get(m, np.nan) for m in per_m}
        # ipTM has no published counterpart and is not part of the parent's spread
        # estimator, so it never enters ss.spread; carry its point settling step
        # anyway so the second-interface-ruler contrast can be computed.
        for extra in EXTRA:
            v = th_point_all.get(t, {}).get(extra, np.nan)
            if np.isfinite(v):
                th_point[t][extra] = v
    th_point_for_spread = {t: {m: v[m] for m in METRICS if m in v} for t, v in th_point.items()}
    return {
        "by_cluster": ss.spread(th_point_for_spread, th_rep, clusters, boot=boot, seed=seed,
                                mode="cluster"),
        "by_target": ss.spread(th_point_for_spread, th_rep, clusters, boot=boot, seed=seed,
                               mode="target"),
        "_t_half_point": th_point,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corrected_dir")
    ap.add_argument("out_json")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reliability", default="results/reliability.json")
    ap.add_argument("--quantisation-dir", default="")
    ap.add_argument("--quantisation-pooled", default="")
    args = ap.parse_args()

    clusters = json.load(open(os.path.join(PARENT_RESULTS, "target_clusters.json")))
    published = json.load(open(os.path.join(PARENT_RESULTS, "pooled.json")))["pooled"]

    targets = sorted({os.path.basename(p).split("_step")[0]
                      for p in glob.glob(os.path.join(args.corrected_dir, "*_step*_boot.npz"))})
    print(f"{len(targets)} targets")

    corr = run_spread(args.corrected_dir, targets, "_boot", "commitment", clusters, args.boot, args.seed)
    raw = run_spread(args.corrected_dir, targets, "_boot_raw", "commitment_uncorrected",
                     clusters, args.boot, args.seed)
    # The naive one-sided correction: the whole marginal ruler variance charged to the
    # within-child term. It is not the estimate this experiment reports, but it is the
    # obvious thing to do and it is worth showing what it does.
    win = run_spread(args.corrected_dir, targets, "_boot_win", "commitment_within_only",
                     clusters, args.boot, args.seed)

    out: dict = {"n_targets": len(targets), "n_clusters": len(set(clusters.values()))}

    # ---- validation: does the uncorrected recomputation reproduce the paper? ----
    val = {}
    for m in METRICS:
        pub = published["t_half_spread_posthoc"]["per_metric"][m]
        mine = raw["by_cluster"]["per_metric"][m]
        val[m] = {
            "published_cv_corrected": pub["cv_corrected"],
            "recomputed_cv_corrected": mine["cv_corrected"],
            "abs_diff": abs(pub["cv_corrected"] - mine["cv_corrected"]),
            "published_median_t_half": pub["median_t_half"],
            "recomputed_median_t_half": mine["median_t_half"],
        }
    out["reproduction_check"] = val

    # ---- cell-by-cell reproduction of the parent's own commitment ----------
    # Stronger than the pooled check above: every (target, step, quality) cell's
    # uncorrected recomputation is compared against the number the parent wrote,
    # and every bootstrap replicate against the parent's replicate from the same draw.
    cell_diffs, boot_diffs, n_cells = [], [], 0
    for t in targets:
        p = os.path.join(args.corrected_dir, f"{t}_diagnostics.json")
        if not os.path.exists(p):
            continue
        for _s, cell in json.load(open(p))["steps"].items():
            for _m, d in cell.items():
                if "reproduction_abs_diff" in d:
                    cell_diffs.append(float(d["reproduction_abs_diff"]))
                    n_cells += 1
                if "reproduction_boot_max_abs_diff" in d:
                    boot_diffs.append(float(d["reproduction_boot_max_abs_diff"]))
    if cell_diffs:
        out["reproduction_cellwise"] = {
            "n_cells_checked": n_cells,
            "max_abs_diff_commitment": float(np.nanmax(cell_diffs)),
            "mean_abs_diff_commitment": float(np.nanmean(cell_diffs)),
            "n_exact": int(sum(1 for v in cell_diffs if v == 0.0)),
            "n_bootstrap_arrays_checked": len(boot_diffs),
            "max_abs_diff_bootstrap_replicate": float(np.nanmax(boot_diffs)) if boot_diffs else None,
        }

    # ---- pooled settling step, published vs corrected ----------------------
    pooled = {}
    for m in METRICS:
        pub = published["t_half_pooled"][m]
        th_corr = {t: v[m] for t, v in corr["_t_half_point"].items() if np.isfinite(v.get(m, np.nan))}
        th_raw = {t: v[m] for t, v in raw["_t_half_point"].items() if np.isfinite(v.get(m, np.nan))}
        bc = block_bootstrap(th_corr, clusters)
        br = block_bootstrap(th_raw, clusters)
        pooled[m] = {
            "published": {"mean": pub["mean"], "ci95": [pub["lo"], pub["hi"]]},
            "recomputed_uncorrected": {"mean": br["mean"], "ci95": [br["lo"], br["hi"]]},
            "corrected": {"mean": bc["mean"], "ci95": [bc["lo"], bc["hi"]]},
            "shift_corrected_minus_published": bc["mean"] - pub["mean"],
        }
    # ipTM has no published counterpart: it was recorded but never analysed
    th_iptm = {t: v["iptm"] for t, v in corr["_t_half_point"].items()
               if np.isfinite(v.get("iptm", np.nan))}
    if th_iptm:
        bi = block_bootstrap(th_iptm, clusters)
        pooled["iptm"] = {"published": None, "corrected": {"mean": bi["mean"], "ci95": [bi["lo"], bi["hi"]]},
                          "n_targets": len(th_iptm)}
    out["pooled_t_half"] = pooled

    # ---- ipTM as a second interface ruler ----------------------------------
    # ipTM and the interface-PAE score come out of the same AlphaFold forward pass,
    # so their reading errors are correlated and this is not an independent replicate.
    # It is still the one alternative interface measurement available without
    # re-folding anything, and a settling step that moves with the choice of score
    # would say the late interface number is a property of the score, not the design.
    pairs = []
    for t, v in corr["_t_half_point"].items():
        a, b = v.get("interface_quality", np.nan), v.get("iptm", np.nan)
        if np.isfinite(a) and np.isfinite(b):
            pairs.append((t, float(a), float(b)))
    if pairs:
        d = np.array([b - a for _, a, b in pairs])
        rng = np.random.default_rng(args.seed)
        cl = sorted({clusters.get(t, t) for t, _, _ in pairs})
        by_cl: dict = {}
        for i, (t, _, _) in enumerate(pairs):
            by_cl.setdefault(clusters.get(t, t), []).append(i)
        reps = []
        for _ in range(args.boot):
            pick = rng.integers(0, len(cl), len(cl))
            idx = [i for k in pick for i in by_cl[cl[k]]]
            reps.append(float(np.mean(d[idx])))
        reps_arr = np.asarray(reps)
        a_arr = np.array([a for _, a, _ in pairs])
        b_arr = np.array([b for _, _, b in pairs])
        out["iptm_vs_interface"] = {
            "n_targets": len(pairs),
            "n_clusters": len(cl),
            "mean_t_half_interface_quality": float(np.mean(a_arr)),
            "mean_t_half_iptm": float(np.mean(b_arr)),
            "paired_difference_iptm_minus_interface": float(np.mean(d)),
            "ci95_cluster_bootstrap": [float(np.percentile(reps_arr, 2.5)),
                                       float(np.percentile(reps_arr, 97.5))],
            "pearson_r_across_targets": float(np.corrcoef(a_arr, b_arr)[0, 1]) if len(pairs) > 2 else None,
            "spearman_r_across_targets": float(np.corrcoef(
                np.argsort(np.argsort(a_arr)), np.argsort(np.argsort(b_arr)))[0, 1])
            if len(pairs) > 2 else None,
            "per_target": {t: {"interface_quality": a, "iptm": b} for t, a, b in pairs},
            "caveat": "ipTM and interface PAE are read off the same AlphaFold forward pass; "
                      "their reading errors are correlated, so agreement here is weaker "
                      "evidence than an independent re-fold would be",
        }

    # ---- the spread comparison the headline rests on -----------------------
    spread_out = {}
    for unit in ("by_cluster", "by_target"):
        pub_key = "t_half_spread_posthoc" if unit == "by_cluster" else "t_half_spread_posthoc_by_target"
        pub = published[pub_key]
        pubm = pub["per_metric"]
        entry = {"per_quality": {}, "pairwise": {}}
        for m in METRICS:
            entry["per_quality"][m] = {
                "published_cv_corrected": pubm[m]["cv_corrected"],
                "published_ci95": pubm[m]["cv_corrected_ci95"],
                "corrected_cv_corrected": corr[unit]["per_metric"][m]["cv_corrected"],
                "corrected_ci95": corr[unit]["per_metric"][m]["cv_corrected_ci95"],
                "uncorrected_recomputed_cv_corrected": raw[unit]["per_metric"][m]["cv_corrected"],
                "published_median_t_half": pubm[m]["median_t_half"],
                "corrected_median_t_half": corr[unit]["per_metric"][m]["median_t_half"],
                "within_only_cv_corrected": win[unit]["per_metric"][m]["cv_corrected"],
                "within_only_median_t_half": win[unit]["per_metric"][m]["median_t_half"],
            }
        for key, v in corr[unit]["pairwise_cv_differences"].items():
            pv = pub.get("pairwise_cv_differences", {}).get(key, {})
            wv = win[unit]["pairwise_cv_differences"].get(key, {})
            entry["pairwise"][key] = {
                "published": {"observed": pv.get("observed"), "ci95": pv.get("ci95"),
                              "excludes_zero": pv.get("excludes_zero")},
                "corrected": {"observed": v["observed"], "ci95": v["ci95"],
                              "excludes_zero": v["excludes_zero"], "p_gt_0": v["p_gt_0"]},
                "within_only": {"observed": wv.get("observed"), "ci95": wv.get("ci95"),
                                "excludes_zero": wv.get("excludes_zero")},
            }
        entry["resolved_corrected"] = corr[unit].get("cv_resolved_metrics")
        entry["resolved_within_only"] = win[unit].get("cv_resolved_metrics")
        entry["resolved_published"] = pub.get("cv_resolved_metrics")
        entry["rank_order_corrected"] = corr[unit].get("cv_rank_order")
        entry["rank_order_published"] = pub.get("cv_rank_order")
        entry["grouped_contrast_corrected"] = corr[unit].get("grouped_contrast")
        entry["grouped_contrast_published"] = pub.get("grouped_contrast")
        entry["t_half_by_target_corrected"] = corr[unit].get("t_half_by_target")
        entry["t_half_se_by_target_corrected"] = corr[unit].get("t_half_se_by_target")
        entry["t_half_by_target_published"] = pub.get("t_half_by_target")
        spread_out[unit] = entry
    out["spread"] = spread_out

    # ---- readability on the total-variance scale ---------------------------
    read = {}
    pub_read = published.get("readability", {})
    for m in METRICS + EXTRA:
        rows = []
        for t in targets:
            p = os.path.join(args.corrected_dir, f"{t}.json")
            if not os.path.exists(p):
                continue
            j = json.load(open(p))
            for s in STEPS:
                c = j["steps"].get(str(s), {}).get(m, {}).get("decomposition", {})
                if not c:
                    continue
                rows.append((t, s, c.get("commitment_uncorrected"), c.get("commitment")))
        if rows:
            corr_vals = np.array([b if b is not None else np.nan for _, _, _, b in rows], float)
            ratio = np.array([b / a for _, _, a, b in rows
                              if a and b is not None and np.isfinite(a) and a > 0])
            finite = ratio[np.isfinite(ratio)]
            read[m] = {
                "n_cells": len(rows),
                # a cell where the estimated ruler variance meets or exceeds the
                # measured spread cannot be corrected: the corrected commitment
                # saturates at 1 or leaves the interval entirely. These counts are
                # how far the correction is being pushed for each quality.
                "n_cells_corrected_nonfinite": int((~np.isfinite(corr_vals)).sum()),
                "n_cells_corrected_at_or_above_one": int(np.nansum(corr_vals >= 1.0)),
                "mean_commitment_ratio_corrected_over_published": float(np.mean(finite))
                if finite.size else float("nan"),
                "max_commitment_ratio": float(np.max(finite)) if finite.size else float("nan"),
                "note": "readability on the total-variance scale is r2_corrected x commitment; "
                        "r2_corrected is unchanged, so readability scales by exactly this ratio",
                "published_readability_by_step": pub_read.get(m),
            }
    out["readability_scaling"] = read

    if os.path.exists(args.reliability):
        rel = json.load(open(args.reliability))
        # The ruler summary carries slightly different fields per quality (the
        # designability control is a repeat-measure pair statistic, not a member
        # decomposition), so copy whatever each one actually has.
        top_fields = ("interaction_sd_mean_over_sample", "interaction_sd_peak",
                      "interaction_peak_at_level", "interaction_sd_at_lowest_bin",
                      "interaction_sd_at_highest_bin",
                      "spread_across_sampled_completions_sd")
        glob_fields = ("n_structures", "n_pairs", "n_members", "member_offsets",
                       "member_offset_sd", "recorded_member_offset",
                       "interaction_sd_unbinned", "structure_sd", "r", "mean_abs_diff")
        out["reliability"] = {}
        for q, v in rel["per_quality"].items():
            row = {f: v[f] for f in top_fields if f in v}
            g = v.get("global", {})
            row.update({f: g[f] for f in glob_fields if f in g})
            row["reading_error_sd"] = row.get("interaction_sd_mean_over_sample")
            row["spread_across_completions_sd"] = row.get(
                "spread_across_sampled_completions_sd")
            se = row["reading_error_sd"]
            sp = row["spread_across_completions_sd"]
            row["reading_error_share_of_total_variance"] = (
                float(se ** 2 / (se ** 2 + sp ** 2))
                if se is not None and sp not in (None, 0) else None
            )
            out["reliability"][q] = row

    if args.quantisation_dir and os.path.isdir(args.quantisation_dir):
        q_rows = []
        for p in sorted(glob.glob(os.path.join(args.quantisation_dir, "*.json"))):
            q_rows.append(json.load(open(p)))
        if q_rows:
            out["quantisation"] = {
                "n_targets": len(q_rows),
                "margin_mean": float(np.mean([r["margin_overall"]["mean"] for r in q_rows])),
                "margin_median": float(np.median([r["margin_overall"]["median"] for r in q_rows])),
                "frac_below_0p05": float(np.mean([r["margin_overall"]["frac_below_0p05"] for r in q_rows])),
                "frac_below_0p10": float(np.mean([r["margin_overall"]["frac_below_0p10"] for r in q_rows])),
                "clustering_pair_agreement": float(
                    np.mean([r["clustering_reproduction"]["pair_agreement_with_cached_labels"]
                             for r in q_rows])),
                "per_target": {r["target"]: r["margin_overall"] for r in q_rows},
            }

    qp_path = args.quantisation_pooled
    if not qp_path and args.quantisation_dir:
        qp_path = os.path.join(os.path.dirname(os.path.abspath(args.quantisation_dir.rstrip("/"))),
                               "quantisation_pooled.json")
    if qp_path and os.path.exists(qp_path):
        out["combined_sensitivity"] = combined_sensitivity(
            raw["_t_half_point"], corr["_t_half_point"], json.load(open(qp_path)),
            clusters, args.boot, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    cs = out.get("combined_sensitivity")
    if cs and not cs.get("insufficient"):
        for unit in ("by_cluster", "by_target"):
            print(f"\n=== both checks together, {unit} (observed CV, n={cs[unit]['n_units']}) ===")
            for name, row in cs[unit]["cells"].items():
                lo, hi = row["ci95"]
                print(f"  {name:24s} cv(iface) {row['cv_interface']:.4f}  "
                      f"cv(fold) {row['cv_fold']:.4f}  gap {row['gap']:+.4f} "
                      f"[{lo:+.4f}, {hi:+.4f}]")

    rc = out.get("reproduction_cellwise")
    if rc:
        print(f"\n=== cellwise reproduction: {rc['n_exact']}/{rc['n_cells_checked']} cells exact, "
              f"max |diff| {rc['max_abs_diff_commitment']:.2e}, "
              f"max replicate |diff| {rc['max_abs_diff_bootstrap_replicate']}")
    print("\n=== reproduction of the published spread (uncorrected recomputation) ===")
    for m, v in out["reproduction_check"].items():
        print(f"  {m:24s} cv_corrected published {v['published_cv_corrected']:.4f} "
              f"recomputed {v['recomputed_cv_corrected']:.4f}  diff {v['abs_diff']:.4f}")
    print("\n=== pooled settling step ===")
    for m, v in out["pooled_t_half"].items():
        if v.get("published"):
            print(f"  {m:24s} published {v['published']['mean']:8.2f}  "
                  f"corrected {v['corrected']['mean']:8.2f}  "
                  f"shift {v['shift_corrected_minus_published']:+.2f}")
        else:
            print(f"  {m:24s} (no published value)  corrected {v['corrected']['mean']:8.2f}")
    iv = out.get("iptm_vs_interface")
    if iv:
        print("\n=== ipTM vs interface PAE settling step ===")
        print(f"  interface_quality {iv['mean_t_half_interface_quality']:.2f}   "
              f"iptm {iv['mean_t_half_iptm']:.2f}   "
              f"paired diff {iv['paired_difference_iptm_minus_interface']:+.2f} "
              f"{iv['ci95_cluster_bootstrap']}")
    print("\n=== headline spread pair, cluster unit ===")
    hp = out["spread"]["by_cluster"]["pairwise"].get("cv(iface) - cv(fold)")
    if hp:
        print(f"  published {hp['published']['observed']:+.4f} {hp['published']['ci95']}")
        print(f"  corrected {hp['corrected']['observed']:+.4f} {hp['corrected']['ci95']}")
    print(f"\n-> {args.out_json}")


if __name__ == "__main__":
    main()
