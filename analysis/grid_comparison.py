#!/usr/bin/env python
"""What the three extra capture steps changed.

Three derivations of the same quantities are compared, and the difference
between each consecutive pair means something specific:

``published``  the source analysis's seven-step values, exactly as reported.
``seven``      the same seven capture steps, re-derived from cells re-analysed
               under the ten-step joint Foldseek clustering. Published -> seven
               is the effect of re-clustering alone; no capture step moved.
``ten``        the ten-step derivation. Seven -> ten is the interpolation error
               the run is about: same cells, same labels, same estimators, more
               places where the commitment curve is actually observed.

Everything the plan asks to be reported beside its published counterpart is
here: per-target settling steps, corrected CVs at both matched units, the
interface-quality vs fold-identity spread gap, the pairwise CV differences, the
readability peaks, and the pre-registered ordering test.

    python compare_grids.py --published P.json --seven S.json --ten T.json --out O.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

METRICS = ["fold_identity", "designability", "structural_confidence", "interface_quality"]
SHORT = {"fold_identity": "fold", "designability": "design",
         "structural_confidence": "conf", "interface_quality": "iface"}
UNITS = {"cluster": "t_half_spread_posthoc", "target": "t_half_spread_posthoc_by_target"}
HEADLINE = "cv(iface) - cv(fold)"


def _arg(name: str, default: str | None = None) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _f(x) -> float:
    return float(x) if x is not None and np.isfinite(x) else float("nan")


def per_target_table(p: dict) -> dict:
    """target -> quality -> settling step, from a pooled.json."""
    out: dict[str, dict[str, float]] = {}
    src = p["pooled"]["t_half_per_target"]
    for m in METRICS:
        for t, v in src.get(m, {}).items():
            out.setdefault(t, {})[m] = _f(v)
    return out


def spread_block(p: dict, unit: str) -> dict:
    return p["pooled"][UNITS[unit]]


def readability_peaks(p: dict) -> dict:
    c = p["pooled"]["readability_step_contrast"]["metrics"]
    return {m: {"peak_step": c[m]["peak_step"], "peak_mean": _f(c[m]["peak_mean"]),
                "reference_step": p["pooled"]["readability_step_contrast"]["reference_step"],
                "reference_mean": _f(c[m]["reference_mean"])}
            for m in METRICS if m in c}


def main() -> None:
    paths = {"published": _arg("--published"), "seven": _arg("--seven"), "ten": _arg("--ten")}
    out_path = _arg("--out")
    P = {k: json.load(open(v)) for k, v in paths.items()}

    out: dict = {
        "sources": paths,
        "grids": {k: P[k]["capture_steps"] for k in P},
        "what_each_comparison_isolates": {
            "published_vs_seven": "re-analysis under the ten-step joint Foldseek clustering, at the same seven capture steps",
            "seven_vs_ten": "the capture grid alone: same cells, same labels, same estimators",
        },
    }

    # ---- per-target settling steps ----------------------------------------
    tabs = {k: per_target_table(P[k]) for k in P}
    targets = sorted(set(tabs["ten"]) & set(tabs["seven"]) & set(tabs["published"]))
    per_target: dict[str, dict] = {}
    deltas: dict[str, dict[str, list]] = {m: {"published_vs_seven": [], "seven_vs_ten": [],
                                              "published_vs_ten": []} for m in METRICS}
    for t in targets:
        per_target[t] = {}
        for m in METRICS:
            a = tabs["published"][t].get(m, float("nan"))
            b = tabs["seven"][t].get(m, float("nan"))
            c = tabs["ten"][t].get(m, float("nan"))
            per_target[t][m] = {"published": a, "seven": b, "ten": c,
                                "d_published_seven": b - a, "d_seven_ten": c - b,
                                "d_published_ten": c - a}
            for key, d in (("published_vs_seven", b - a), ("seven_vs_ten", c - b),
                           ("published_vs_ten", c - a)):
                if np.isfinite(d):
                    deltas[m][key].append(d)
    out["per_target_t_half"] = per_target

    summ = {}
    for m in METRICS:
        summ[m] = {}
        for key, arr in deltas[m].items():
            a = np.asarray(arr, float)
            summ[m][key] = {
                "n": int(a.size),
                "mean_signed": float(a.mean()) if a.size else float("nan"),
                "mean_abs": float(np.abs(a).mean()) if a.size else float("nan"),
                "median_abs": float(np.median(np.abs(a))) if a.size else float("nan"),
                "max_abs": float(np.abs(a).max()) if a.size else float("nan"),
                "n_moved_gt_5_steps": int((np.abs(a) > 5).sum()),
                "n_moved_gt_25_steps": int((np.abs(a) > 25).sum()),
            }
        # the interpolation error relative to the spread it feeds
        base = np.asarray([per_target[t][m]["ten"] for t in targets], float)
        d = np.asarray(deltas[m]["seven_vs_ten"], float)
        summ[m]["grid_error_vs_panel_spread"] = {
            "mean_abs_shift_steps": float(np.abs(d).mean()) if d.size else float("nan"),
            "panel_sd_of_t_half_steps": float(np.std(base, ddof=1)) if base.size > 1 else float("nan"),
            "ratio": float(np.abs(d).mean() / np.std(base, ddof=1))
            if base.size > 1 and np.std(base, ddof=1) > 0 else float("nan"),
        }
    out["t_half_shift_summary"] = summ

    # ---- corrected CVs and the spread gap, at both matched units ------------
    cvs: dict = {}
    for unit in UNITS:
        cvs[unit] = {"per_metric": {}, "pairwise": {}, "resolved": {}, "extreme_resolved": {}}
        for m in METRICS:
            cvs[unit]["per_metric"][m] = {}
            for k in P:
                e = spread_block(P[k], unit)["per_metric"][m]
                cvs[unit]["per_metric"][m][k] = {
                    "cv_corrected": _f(e["cv_corrected"]),
                    "cv_corrected_ci95": [_f(x) for x in e["cv_corrected_ci95"]],
                    "cv_observed": _f(e["cv_observed"]),
                    "median_t_half": _f(e["median_t_half"]),
                    "min_t_half": _f(e["min_t_half"]),
                    "max_t_half": _f(e["max_t_half"]),
                    "mean_within_se_steps": _f(e["mean_within_se_steps"]),
                }
        keys = set()
        for k in P:
            keys |= set(spread_block(P[k], unit)["pairwise_cv_differences"])
        for pair in sorted(keys):
            cvs[unit]["pairwise"][pair] = {}
            for k in P:
                v = spread_block(P[k], unit)["pairwise_cv_differences"].get(pair)
                if v is None:
                    continue
                cvs[unit]["pairwise"][pair][k] = {
                    "observed": _f(v["observed"]), "ci95": [_f(x) for x in v["ci95"]],
                    "p_gt_0": _f(v["p_gt_0"]), "excludes_zero": bool(v["excludes_zero"]),
                }
        for k in P:
            b = spread_block(P[k], unit)
            cvs[unit]["resolved"][k] = {"resolved": b["cv_resolved_metrics"],
                                        "unresolved": b["cv_unresolved_metrics"],
                                        "cv_rank_order": b["cv_rank_order"],
                                        "n_rows": b["n_rows"]}
            er = b.get("extreme_contrast_resolved")
            if er:
                cvs[unit]["extreme_resolved"][k] = {
                    "tightest": er["tightest"], "widest": er["widest"],
                    "difference": _f(er["difference"]), "ci95": [_f(x) for x in er["difference_ci95"]],
                    "p_gt_0": _f(er["p_gt_0"]), "ratio": _f(er["ratio"]),
                    "ratio_ci95": [_f(x) for x in er["ratio_ci95"]],
                    "excludes_zero": bool(er["excludes_zero"])}
    out["cv"] = cvs

    # ---- the pre-registered decision ---------------------------------------
    verdict = {}
    for unit in UNITS:
        h = cvs[unit]["pairwise"].get(HEADLINE, {})
        ten = h.get("ten", {})
        verdict[unit] = {
            "reported_value_is": "ten-step grid",
            "gap_ten": ten.get("observed"), "ci95_ten": ten.get("ci95"),
            "p_gt_0_ten": ten.get("p_gt_0"),
            "excludes_zero_ten": ten.get("excludes_zero"),
            "gap_published": h.get("published", {}).get("observed"),
            "ci95_published": h.get("published", {}).get("ci95"),
            "gap_seven": h.get("seven", {}).get("observed"),
            "ci95_seven": h.get("seven", {}).get("ci95"),
            "claim": ("the spread gap still excludes zero on the better-sampled grid"
                      if ten.get("excludes_zero") else
                      "the spread gap no longer excludes zero on the better-sampled grid"),
        }
    out["pre_registered_verdict"] = verdict

    # ---- readability peaks --------------------------------------------------
    out["readability_peaks"] = {k: readability_peaks(P[k]) for k in P}

    # ---- ordering test ------------------------------------------------------
    out["ordering"] = {}
    for k in P:
        oc = P[k]["pooled"]["order_consistency"]
        out["ordering"][k] = {"kendall_w_t_half": _f(oc["kendall_w_t_half"]),
                              "n_distinct_orders": oc["n_distinct_orders"],
                              "n_targets_full_rank": oc["n_targets_full_rank"]}

    # ---- pooled settling steps ---------------------------------------------
    out["t_half_pooled"] = {
        m: {k: {"mean": _f(P[k]["pooled"]["t_half_pooled"][m]["mean"]),
                "lo": _f(P[k]["pooled"]["t_half_pooled"][m]["lo"]),
                "hi": _f(P[k]["pooled"]["t_half_pooled"][m]["hi"])} for k in P}
        for m in METRICS}

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- print --------------------------------------------------------------
    print("capture grids:", {k: len(v) for k, v in out["grids"].items()})
    print("\n=== per-target settling step: how far the added steps moved it ===")
    print(f"{'quality':24s} {'mean|d| 7->10':>14s} {'median|d|':>10s} {'max|d|':>9s} "
          f"{'>25 steps':>10s} {'|d| / panel SD':>15s}")
    for m in METRICS:
        s = summ[m]["seven_vs_ten"]
        r = summ[m]["grid_error_vs_panel_spread"]["ratio"]
        print(f"{m:24s} {s['mean_abs']:14.3f} {s['median_abs']:10.3f} {s['max_abs']:9.3f} "
              f"{s['n_moved_gt_25_steps']:10d} {r:15.4f}")
    print("\n(published -> seven, i.e. re-clustering alone)")
    for m in METRICS:
        s = summ[m]["published_vs_seven"]
        print(f"{m:24s} mean|d| {s['mean_abs']:8.3f}  max|d| {s['max_abs']:8.3f}  "
              f"moved >5 steps: {s['n_moved_gt_5_steps']}/{s['n']}")

    for unit in UNITS:
        print(f"\n=== corrected CV, {unit} unit ===")
        print(f"{'quality':24s} {'published':>22s} {'seven':>22s} {'ten':>22s}")
        for m in METRICS:
            row = cvs[unit]["per_metric"][m]
            cells = []
            for k in ("published", "seven", "ten"):
                e = row[k]
                cells.append(f"{e['cv_corrected']:.4f} [{e['cv_corrected_ci95'][0]:.3f},{e['cv_corrected_ci95'][1]:.3f}]")
            print(f"{m:24s} {cells[0]:>22s} {cells[1]:>22s} {cells[2]:>22s}")
        h = cvs[unit]["pairwise"].get(HEADLINE, {})
        for k in ("published", "seven", "ten"):
            v = h.get(k)
            if not v:
                continue
            print(f"  {HEADLINE} {k:10s} {v['observed']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] "
                  f"P(>0)={v['p_gt_0']:.4f} {'EXCLUDES 0' if v['excludes_zero'] else 'includes 0'}")

    print("\n=== readability peak (linear readout of activations) ===")
    for m in METRICS:
        cells = []
        for k in ("published", "seven", "ten"):
            e = out["readability_peaks"][k].get(m)
            cells.append(f"{e['peak_mean']:.4f} @ {e['peak_step']}" if e else "n/a")
        print(f"{m:24s} published {cells[0]:>16s}   seven {cells[1]:>16s}   ten {cells[2]:>16s}")

    print("\n=== pre-registered verdict ===")
    for unit, v in verdict.items():
        print(f"{unit:8s} {v['claim']}: {v['gap_ten']:+.4f} "
              f"[{v['ci95_ten'][0]:+.4f},{v['ci95_ten'][1]:+.4f}] (published {v['gap_published']:+.4f})")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
