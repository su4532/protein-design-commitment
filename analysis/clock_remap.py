#!/usr/bin/env python
"""Does the interface-versus-fold spread gap survive being read on the model's own clock?

The capture step is an integer index into the sampler loop, and Complexa does not
advance its two data modes on the same time curve. The backbone alpha carbons
(``bb_ca``) use a ``log`` schedule and the local latents a ``power`` schedule, both
with p = 2. A coefficient of variation is invariant under a LINEAR rescaling of its
axis but not under a nonlinear one, so a quality whose settling point falls in a
steep part of a schedule spreads out in step units while one in a flat part
compresses. If fold identity and interface quality load on different modes, part of
the reported gap between their settling-step CVs could be two differently curved
rulers rather than the model.

This re-reads the SAME cells, the SAME commitment curves and the SAME bootstrap
replicates on remapped axes. Nothing is re-sampled and nothing is re-folded. The
only thing that changes is the x-coordinate the half-commitment crossing is
interpolated against, and therefore the units the spread is measured in.

Five axes, all applied to every quality unless stated:

``step_index``            the published axis: the integer capture step. Reproducing
                          the published numbers exactly on this axis is the check
                          that the remap harness is not doing anything else.
``time_uniform_reference``  t = step / nsteps. A LINEAR rescaling, so every CV must
                          come back bit-identical to ``step_index``. This is the
                          control that demonstrates the premise: linear rescalings
                          cannot move a CV, nonlinear ones can.
``time_bb_ca``            Complexa's backbone schedule, evaluated by Complexa's own
                          ``get_schedule``.
``time_local_latents``    Complexa's latent schedule, same function.
``time_native_assumed``   the sharp form of the objection: fold identity and
                          designability read on the backbone clock, structural
                          confidence and interface quality on the latent clock.
                          The loading assignment is an ASSUMPTION, not a
                          measurement, and is labelled as such everywhere.

Reporting both common-axis remaps rather than picking one is deliberate. We have
not established which mode each quality loads on, so bracketing is the honest
treatment, and the two axes answer different questions in any case: the step index
is what a search can act on, while the flow-matching time is the model's own
coordinate and runs 0 to 1 in both modes.

The estimator is imported unchanged from ``spread_stats2`` - the corrected CV, the
two matched analysis units, the bootstrap scheme, the draw count and the seed are
all the published ones.

    python schedule_remap.py <per_target_dir> <out.json> \
        --config <resolved config yaml> --clusters <target_clusters.json> \
        [--published <pooled.json>] [--boot 10000] [--seed 42]
"""

from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np
import yaml

from . import spread as S
from .spread import METRICS, SHORT, load_boot, spread, t_half_curve

#: the two data modes Complexa denoises, and the qualities we assume load on each
#: for the ``time_native_assumed`` sensitivity. Fold identity and designability are
#: read off the backbone (Foldseek over CA, redesign-and-refold backbone RMSD);
#: structural confidence and interface quality are interface/side-chain properties.
#: This is a plausibility argument about what each quality measures, NOT something
#: this experiment measured, and the verdict never rests on it alone.
NATIVE_MODE = {
    "fold_identity": "bb_ca",
    "designability": "bb_ca",
    "structural_confidence": "local_latents",
    "interface_quality": "local_latents",
}

#: the pre-registered post-hoc scorer, in ``spread_stats2``'s pair naming
GAP_KEY = f"cv({SHORT['interface_quality']}) - cv({SHORT['fold_identity']})"


# --------------------------------------------------------------------------- #
# Complexa's schedules
# --------------------------------------------------------------------------- #


def load_resolved_config(path: str) -> tuple[dict, int]:
    """Parse the resolved Hydra config, tolerating trailing job log lines.

    The captured config file is the config as it was resolved inside a real
    capture job, with that job's stdout tail still attached, so a strict parse
    fails on the last line or two. Trailing lines are dropped one at a time until
    the document parses, and how many were dropped is reported rather than
    silently swallowed.
    """
    lines = open(path).read().splitlines()
    for dropped in range(0, 25):
        text = "\n".join(lines[: len(lines) - dropped]) + "\n"
        try:
            cfg = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        if isinstance(cfg, dict) and "generation" in cfg:
            return cfg, dropped
    raise SystemExit(f"could not parse {path} after dropping up to 24 trailing lines")


def closed_form(mode: str, p: float, nsteps: int) -> np.ndarray:
    """The schedule written out algebraically, as an independent check.

    ``get_schedule`` builds these with ``torch.logspace``/``torch.linspace`` and a
    min-max normalisation. Writing the same thing in closed form and requiring the
    two to agree catches a misreading of the source as well as a version skew
    between the config and the installed code.
    """
    i = np.arange(nsteps + 1, dtype=float)
    if mode == "power":
        return (i / nsteps) ** p
    if mode == "log":
        raw = 1.0 - 10.0 ** (-p * i / nsteps)
        return raw / raw[-1]
    raise SystemExit(f"no closed form written for schedule mode {mode!r}")


def complexa_schedules(model_cfg: dict, nsteps: int) -> dict:
    """Evaluate Complexa's own ``get_schedule`` for each data mode at ``nsteps``.

    Returns ``mode -> {t: [nsteps+1], spec, check}``. ``t[s]`` is the flow-matching
    time of the state after ``s`` sampler steps, which is exactly the index the
    capture hook uses: the hooked forward pass at capture step ``s`` is evaluated
    at ``b["t"] = ts[data_mode][s]``.
    """
    from proteinfoundation.flow_matching.product_space_flow_matcher import (
        get_schedule,
        get_schedule_tsr_safe,
    )

    out = {}
    for dm, spec in model_cfg.items():
        if not isinstance(spec, dict) or "schedule" not in spec:
            continue
        sampling_mode = spec["simulation_step_params"]["sampling_mode"]
        # the sampler picks the tsr-safe variant only for vf_tsr; this run is "sc"
        fn = get_schedule_tsr_safe if sampling_mode == "vf_tsr" else get_schedule
        mode, p = spec["schedule"]["mode"], float(spec["schedule"]["p"])
        t = np.asarray(fn(mode=mode, nsteps=int(nsteps), p1=p).detach().cpu().numpy(), float)
        cf = closed_form(mode, p, int(nsteps)) if fn is get_schedule else None
        out[dm] = {
            "schedule_mode": mode,
            "p": p,
            "sampling_mode": sampling_mode,
            "schedule_function": fn.__name__,
            "n_points": int(t.size),
            "t_first": float(t[0]),
            "t_last": float(t[-1]),
            "monotone_increasing": bool(np.all(np.diff(t) > 0)),
            "closed_form_max_abs_diff": (float(np.max(np.abs(t - cf))) if cf is not None else None),
            "_t": t,
        }
    return out


def elasticity(t_full: np.ndarray, s: float) -> dict:
    """Local log-log slope of the schedule at (possibly fractional) step ``s``.

    ``d ln t / d ln s`` is the factor by which a small RELATIVE change in step
    becomes a relative change in model time, which is exactly the first-order
    factor by which the remap scales a CV. An elasticity below 1 compresses the
    spread, above 1 stretches it.
    """
    idx = np.arange(t_full.size, dtype=float)
    dtds = np.gradient(t_full, idx)
    t_s = float(np.interp(s, idx, t_full))
    d_s = float(np.interp(s, idx, dtds))
    return {
        "step": float(s),
        "t": t_s,
        "dt_dstep": d_s,
        "elasticity_dlnt_dlns": float(d_s * s / t_s) if t_s > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# crossings on a remapped axis
# --------------------------------------------------------------------------- #


def crossings(reps: dict, point_curves: dict, axis: dict) -> tuple[dict, dict]:
    """Re-interpolate every point and replicate crossing against ``axis``.

    ``axis`` maps quality -> the x-coordinate of each capture step. The remap is
    strictly increasing, so which pair of grid steps brackets the crossing never
    changes; only where inside that bracket the crossing lands, and in what units
    it is expressed. Interpolating in the target coordinate is the point: applying
    the remap to a step-interpolated crossing instead would carry the step axis's
    linearity assumption into the answer.
    """
    th_point, th_rep = {}, {}
    for t, per_m in reps.items():
        pt, rp = {}, {}
        for m in METRICS:
            if m not in per_m:
                raise SystemExit(f"{t} has no replicate curve for {m}; the panel is not complete")
            x = axis[m]
            pt[m] = t_half_curve(np.asarray(point_curves[t][m], float), steps=x)
            boot = per_m[m]
            rp[m] = np.array([t_half_curve(boot[:, b], steps=x) for b in range(boot.shape[1])])
        th_point[t], th_rep[t] = pt, rp
    return th_point, th_rep


# --------------------------------------------------------------------------- #
# the same bootstrap, re-derived so quantities beyond spread()'s own outputs can
# be put on it
# --------------------------------------------------------------------------- #


def _rows_for_mode(th_point: dict, th_rep: dict, clusters: dict, mode: str):
    """Rebuild spread()'s row matrices for one analysis unit.

    This mirrors ``spread_stats2.spread`` step for step. It exists only so that
    the bootstrap draw can be replayed and quantities that ``spread`` does not
    itself report - the fold-versus-interface CV ratio on every axis, and the
    paired change in the gap between two axes - can be put on exactly the same
    resampling rather than on a second, independent one. Every replay is checked
    against ``spread``'s own published intervals before anything is read off it,
    so a divergence here fails loudly instead of quietly reporting a different
    bootstrap.
    """
    targets = sorted(t for t in th_point
                     if all(np.isfinite(th_point[t].get(m, np.nan)) for m in METRICS))
    B = min(th_rep[t][m].shape[0] for t in targets for m in METRICS)
    # spread()'s bootstrap draw reads only the replicate cube and the per-row
    # estimation variances; its point-estimate matrix plays no part in a draw and
    # is not rebuilt here.
    Rrep = np.stack([np.stack([th_rep[t][m][:B] for m in METRICS], 1) for t in targets])
    within = np.nanvar(Rrep, axis=1, ddof=1)
    cluster_of = [clusters.get(t, t) for t in targets]
    if mode == "cluster":
        unit_members = {k: sorted(t for t, u in zip(targets, cluster_of) if u == k)
                        for k in sorted(set(cluster_of))}
        keys = sorted(unit_members)
        idx = {k: [targets.index(t) for t in unit_members[k]] for k in keys}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Rrep = np.stack([np.nanmean(Rrep[idx[k]], axis=0) for k in keys])
            within = np.nanvar(Rrep, axis=1, ddof=1)
        row_of_unit = list(keys)
    elif mode == "target":
        row_of_unit = list(targets)
    else:
        raise ValueError(f"replay not written for mode {mode!r}")
    return Rrep, within, row_of_unit, B


def replay_bootstrap(th_point: dict, th_rep: dict, clusters: dict, sp: dict,
                     boot: int, seed: int, mode: str) -> tuple[np.ndarray, float]:
    """Replay spread()'s corrected-CV bootstrap and verify it against spread()."""
    Rrep, within, row_of_unit, B = _rows_for_mode(th_point, th_rep, clusters, mode)
    rng = np.random.default_rng(seed)
    uniq = sorted(set(row_of_unit))
    members = {u: [i for i, x in enumerate(row_of_unit) if x == u] for u in uniq}
    nu = len(uniq)
    bs_cor = np.full((boot, 4), np.nan)
    for b in range(boot):
        drawn = [uniq[i] for i in rng.integers(0, nu, size=nu)]
        rows_b = [i for u in drawn for i in members[u]]
        reps = rng.integers(0, B, size=len(rows_b))
        draw = np.stack([Rrep[r, q] for r, q in zip(rows_b, reps)])
        wv = within[rows_b]
        for i in range(4):
            bs_cor[b, i] = S._cv_corrected(draw[:, i], wv[:, i])

    worst = 0.0
    for i, m in enumerate(METRICS):
        lo, hi = np.nanpercentile(bs_cor[:, i], [2.5, 97.5])
        ref = sp["per_metric"][m]["cv_corrected_ci95"]
        worst = max(worst, abs(lo - ref[0]), abs(hi - ref[1]))
    for i in range(4):
        for j in range(4):
            if i >= j:
                continue
            d = bs_cor[:, j] - bs_cor[:, i]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            ref = sp["pairwise_cv_differences"][f"cv({SHORT[METRICS[j]]}) - cv({SHORT[METRICS[i]]})"]
            worst = max(worst, abs(lo - ref["ci95"][0]), abs(hi - ref["ci95"][1]))
    if worst > 1e-12:
        raise SystemExit(f"bootstrap replay does not match spread() (max |diff| {worst:.3e})")
    return bs_cor, float(worst)


def ratio_ci(bs_cor: np.ndarray) -> dict:
    """cv(interface) / cv(fold) on the replayed draws.

    A corrected CV can legitimately be exactly zero when a quality's observed
    spread is no larger than its own estimation error; a ratio against zero is
    undefined rather than infinite, so those draws are dropped and counted, which
    is the convention spread() already uses for its own ratios.
    """
    num = bs_cor[:, METRICS.index("interface_quality")]
    den = bs_cor[:, METRICS.index("fold_identity")]
    r = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    lo, hi = (np.nanpercentile(r, [2.5, 97.5]) if np.isfinite(r).any() else (np.nan, np.nan))
    return {
        "ci95": [float(lo), float(hi)],
        "p_gt_1": float(np.nanmean(r > 1.0)),
        "n_defined": int(np.isfinite(r).sum()),
        "n_draws": int(r.size),
    }


def gap_row(sp: dict) -> dict:
    """The pre-registered scorer plus the extreme-pair ratio, out of one spread()."""
    pair = sp["pairwise_cv_differences"][GAP_KEY]
    ext = sp.get("extreme_contrast") or {}
    ratio_is_iface_over_fold = (
        ext.get("tightest") == "fold_identity" and ext.get("widest") == "interface_quality"
    )
    cvf = sp["per_metric"]["fold_identity"]["cv_corrected"]
    cvi = sp["per_metric"]["interface_quality"]["cv_corrected"]
    return {
        "cv_corrected_fold": cvf,
        "cv_corrected_interface": cvi,
        "cv_corrected_fold_ci95": sp["per_metric"]["fold_identity"]["cv_corrected_ci95"],
        "cv_corrected_interface_ci95": sp["per_metric"]["interface_quality"]["cv_corrected_ci95"],
        "gap": pair["observed"],
        "gap_ci95": pair["ci95"],
        "p_gap_gt_0": pair["p_gt_0"],
        "excludes_zero": pair["excludes_zero"],
        "survives": bool(pair["excludes_zero"] and pair["observed"] > 0),
        "ratio_point": float(cvi / cvf) if cvf > 0 else float("nan"),
        # spread() puts an interval on the ratio only for the extreme pair; on
        # every axis here that pair is expected to be fold vs interface, but it is
        # checked rather than assumed
        "ratio_ci95": ext.get("ratio_ci95") if ratio_is_iface_over_fold else None,
        "ratio_ci_pair_is_fold_vs_interface": bool(ratio_is_iface_over_fold),
        "extreme_pair": [ext.get("tightest"), ext.get("widest")],
        "cv_rank_order": sp["cv_rank_order"],
        "cv_resolved_metrics": sp["cv_resolved_metrics"],
        "mean_t_half": {m: sp["per_metric"][m]["mean_t_half"] for m in METRICS},
        "median_t_half": {m: sp["per_metric"][m]["median_t_half"] for m in METRICS},
    }


def _arg(name: str, default: str | None = None) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    per_target_dir, out_path = sys.argv[1], sys.argv[2]
    cfg_path = _arg("--config")
    clusters_path = _arg("--clusters", "")
    published_path = _arg("--published", "")
    boot = int(_arg("--boot", "10000"))
    seed = int(_arg("--seed", "42"))
    clusters = json.load(open(clusters_path)) if clusters_path and os.path.exists(clusters_path) else {}

    steps = list(S.STEPS)
    print(f"capture grid: {steps}", flush=True)

    # ---- Complexa's schedules, from Complexa ------------------------------- #
    cfg, dropped = load_resolved_config(cfg_path)
    nsteps = int(cfg["generation"]["args"]["nsteps"])
    sched = complexa_schedules(cfg["generation"]["model"], nsteps)
    print(f"resolved config parsed (dropped {dropped} trailing log line(s)); nsteps={nsteps}")
    for dm, d in sched.items():
        print(f"  {dm:16s} mode={d['schedule_mode']:6s} p={d['p']} via {d['schedule_function']}  "
              f"t[0]={d['t_first']:.6g} t[{nsteps}]={d['t_last']:.6g}  "
              f"monotone={d['monotone_increasing']}  "
              f"closed-form agrees to {d['closed_form_max_abs_diff']:.3e}")
        if not d["monotone_increasing"]:
            raise SystemExit(f"{dm} schedule is not strictly increasing")
        if d["closed_form_max_abs_diff"] is not None and d["closed_form_max_abs_diff"] > 1e-6:
            raise SystemExit(f"{dm} schedule disagrees with its closed form")
    for dm in ("bb_ca", "local_latents"):
        if dm not in sched:
            raise SystemExit(f"data mode {dm} missing from the resolved config")

    t_ca = sched["bb_ca"]["_t"]
    t_lat = sched["local_latents"]["_t"]
    ax_step = np.asarray(steps, float)
    ax_unif = ax_step / float(nsteps)
    ax_ca = np.asarray([t_ca[s] for s in steps], float)
    ax_lat = np.asarray([t_lat[s] for s in steps], float)

    AXES = {
        "step_index": {
            "desc": "the published axis: integer capture step",
            "unit": "denoiser step",
            "per_metric": {m: ax_step for m in METRICS},
        },
        "time_uniform_reference": {
            "desc": "step / nsteps, a linear rescaling; every CV must be unchanged",
            "unit": "fraction of the sampler loop",
            "per_metric": {m: ax_unif for m in METRICS},
        },
        "time_bb_ca": {
            "desc": "Complexa's backbone (log, p=2) flow-matching time, applied to all four qualities",
            "unit": "flow-matching time on the bb_ca clock",
            "per_metric": {m: ax_ca for m in METRICS},
        },
        "time_local_latents": {
            "desc": "Complexa's latent (power, p=2) flow-matching time, applied to all four qualities",
            "unit": "flow-matching time on the local_latents clock",
            "per_metric": {m: ax_lat for m in METRICS},
        },
        "time_native_assumed": {
            "desc": ("each quality on the clock of the data mode we ASSUME it loads on "
                     "(fold identity, designability -> bb_ca; structural confidence, "
                     "interface quality -> local_latents); the assignment is an assumption"),
            "unit": "flow-matching time on each quality's assumed own clock",
            "per_metric": {m: (ax_ca if NATIVE_MODE[m] == "bb_ca" else ax_lat) for m in METRICS},
        },
    }

    # ---- the cells, loaded once -------------------------------------------- #
    targets = sorted({os.path.basename(p).split("_step")[0]
                      for p in glob.glob(os.path.join(per_target_dir, "*_step*_boot.npz"))})
    reps = load_boot(per_target_dir, targets)
    print(f"{len(reps)} targets with complete replicate curves over {len(steps)} steps", flush=True)

    point_curves = {}
    for t in list(reps):
        p = os.path.join(per_target_dir, f"{t}.json")
        j = json.load(open(p))
        point_curves[t] = {
            m: np.asarray([j["steps"].get(str(s), {}).get(m, {})
                           .get("decomposition", {}).get("commitment", np.nan) for s in steps], float)
            for m in METRICS
        }

    # ---- run every axis at both matched units ------------------------------ #
    results, headline, bs, replay_worst = {}, {}, {}, []
    for name, spec in AXES.items():
        axis = spec["per_metric"]
        th_point, th_rep = crossings(reps, point_curves, axis)

        # spread() reports left censoring against a single module-level floor. On
        # the common axes that is the first capture step in the new units; on the
        # mixed axis there is no single floor, so it is disabled there and the
        # per-quality censoring is recorded here instead.
        firsts = {m: float(axis[m][0]) for m in METRICS}
        shared_floor = len(set(round(v, 15) for v in firsts.values())) == 1
        S.FLOOR = firsts[METRICS[0]] if shared_floor else float("-inf")
        censored_here = {m: sorted(t for t in th_point
                                   if np.isfinite(th_point[t][m]) and th_point[t][m] <= firsts[m])
                         for m in METRICS}

        by_cluster = spread(th_point, th_rep, clusters, boot=boot, seed=seed, mode="cluster")
        by_target = spread(th_point, th_rep, clusters, boot=boot, seed=seed, mode="target")
        bs_c, w_c = replay_bootstrap(th_point, th_rep, clusters, by_cluster, boot, seed, "cluster")
        bs_t, w_t = replay_bootstrap(th_point, th_rep, clusters, by_target, boot, seed, "target")
        bs[name] = {"cluster": bs_c, "target": bs_t}
        replay_worst.append(max(w_c, w_t))

        results[name] = {
            "description": spec["desc"],
            "axis_unit": spec["unit"],
            "axis_values": {m: [float(v) for v in axis[m]] for m in METRICS},
            "floor_shared_across_qualities": bool(shared_floor),
            "floor_value": firsts,
            "at_floor_targets": censored_here,
            "by_cluster": by_cluster,
            "by_target": by_target,
        }
        headline[name] = {
            "description": spec["desc"],
            "by_cluster": gap_row(by_cluster),
            "by_target": gap_row(by_target),
        }
        for unit, key in (("cluster", "by_cluster"), ("target", "by_target")):
            headline[name][key]["ratio_interface_over_fold"] = {
                "point": headline[name][key]["ratio_point"], **ratio_ci(bs[name][unit])}
        g = headline[name]["by_cluster"]
        print(f"\n=== {name} ===\n  {spec['desc']}")
        print(f"  cluster unit: cv(fold)={g['cv_corrected_fold']:.4f}  "
              f"cv(iface)={g['cv_corrected_interface']:.4f}  "
              f"gap={g['gap']:+.4f} [{g['gap_ci95'][0]:+.4f}, {g['gap_ci95'][1]:+.4f}]  "
              f"P(>0)={g['p_gap_gt_0']:.4f}  {'SURVIVES' if g['survives'] else 'does not clear zero'}")
        gt = headline[name]["by_target"]
        print(f"  target unit : cv(fold)={gt['cv_corrected_fold']:.4f}  "
              f"cv(iface)={gt['cv_corrected_interface']:.4f}  "
              f"gap={gt['gap']:+.4f} [{gt['gap_ci95'][0]:+.4f}, {gt['gap_ci95'][1]:+.4f}]  "
              f"P(>0)={gt['p_gap_gt_0']:.4f}  {'SURVIVES' if gt['survives'] else 'does not clear zero'}")

    # ---- why each axis moves the CV it moves ------------------------------- #
    #
    # To first order a remap multiplies a quality's CV by the local elasticity
    # d ln t / d ln step of the schedule at where that quality settles. Below 1
    # the axis compresses the spread, above 1 it stretches it. Evaluated at each
    # quality's step-axis mean settling step, which is a fixed reference point
    # that does not itself depend on the remap.
    elast = {}
    for m in METRICS:
        s_m = results["step_index"]["by_cluster"]["per_metric"][m]["mean_t_half"]
        elast[m] = {
            "mean_settling_step": float(s_m),
            "bb_ca": elasticity(t_ca, s_m),
            "local_latents": elasticity(t_lat, s_m),
            "assumed_own_mode": NATIVE_MODE[m],
        }
    print("\nschedule elasticity (d ln t / d ln step) at each quality's mean settling step:")
    for m in METRICS:
        e = elast[m]
        print(f"  {m:24s} step {e['mean_settling_step']:7.2f}   "
              f"bb_ca t={e['bb_ca']['t']:.4f} elast={e['bb_ca']['elasticity_dlnt_dlns']:.4f}   "
              f"latents t={e['local_latents']['t']:.4f} "
              f"elast={e['local_latents']['elasticity_dlnt_dlns']:.4f}")

    # ---- checks ------------------------------------------------------------ #
    checks = {
        "bootstrap_replay_matches_spread": {
            "max_abs_difference": float(max(replay_worst)),
            "tolerance": 1e-12,
            "passes": bool(max(replay_worst) <= 1e-12),
            "note": ("The CV ratio and the paired across-axis gap change are read off a replay of "
                     "spread()'s own bootstrap. Before anything is read off it, the replay's four "
                     "corrected-CV intervals and six pairwise-difference intervals are required to "
                     "match spread()'s reported ones on every axis and both units."),
            "n_replays_checked": len(replay_worst) * 2,
        }
    }
    print(f"bootstrap replay agrees with spread() to {max(replay_worst):.3e} "
          f"over {len(replay_worst) * 2} replays")

    # 1. the step axis must reproduce the published ten-step derivation exactly.
    #    Point estimates are compared always; the interval bounds only when this
    #    run uses the published bootstrap size, since a smaller draw count moves
    #    the percentiles for a reason that has nothing to do with the remap.
    if published_path and os.path.exists(published_path):
        pub = json.load(open(published_path))["pooled"]["t_half_spread_posthoc"]
        mine = results["step_index"]["by_cluster"]
        point_diffs, ci_diffs = {}, {}
        for m in METRICS:
            for k in ("cv_observed", "cv_corrected", "mean_t_half", "median_t_half"):
                point_diffs[f"{m}.{k}"] = abs(pub["per_metric"][m][k] - mine["per_metric"][m][k])
        pp, mp = pub["pairwise_cv_differences"][GAP_KEY], mine["pairwise_cv_differences"][GAP_KEY]
        point_diffs["gap.observed"] = abs(pp["observed"] - mp["observed"])
        ci_diffs["gap.ci_lo"] = abs(pp["ci95"][0] - mp["ci95"][0])
        ci_diffs["gap.ci_hi"] = abs(pp["ci95"][1] - mp["ci95"][1])
        boot_matches = int(pub.get("n_boot", -1)) == boot
        worst_pt = max(point_diffs.values())
        worst_ci = max(ci_diffs.values())
        ok = worst_pt < 1e-9 and (worst_ci < 1e-9 or not boot_matches)
        checks["reproduces_published_step_axis"] = {
            "source": published_path,
            "published_n_boot": int(pub.get("n_boot", -1)),
            "this_run_n_boot": boot,
            "bootstrap_size_matches_published": bool(boot_matches),
            "max_abs_difference_point_estimates": float(worst_pt),
            "max_abs_difference_interval_bounds": float(worst_ci),
            "interval_bounds_compared": bool(boot_matches),
            "tolerance": 1e-9,
            "passes": bool(ok),
            "per_quantity": {k: float(v) for k, v in {**point_diffs, **ci_diffs}.items()},
        }
        print(f"\nreproduction of the published step-axis derivation: "
              f"max |diff| point = {worst_pt:.3e}, interval = {worst_ci:.3e} "
              f"(bootstrap size {'matches' if boot_matches else 'differs from'} published) "
              f"({'PASS' if ok else 'FAIL'})")
        if not ok:
            raise SystemExit("step-axis remap does not reproduce the published numbers")

    # 2. a linear rescaling must leave every CV untouched
    lin = {}
    for unit in ("by_cluster", "by_target"):
        a, b = results["step_index"][unit], results["time_uniform_reference"][unit]
        for m in METRICS:
            for k in ("cv_observed", "cv_corrected"):
                lin[f"{unit}.{m}.{k}"] = abs(a["per_metric"][m][k] - b["per_metric"][m][k])
        lin[f"{unit}.gap"] = abs(a["pairwise_cv_differences"][GAP_KEY]["observed"]
                                 - b["pairwise_cv_differences"][GAP_KEY]["observed"])
    worst_lin = max(lin.values())
    checks["linear_rescaling_is_a_no_op"] = {
        "max_abs_difference": float(worst_lin),
        "tolerance": 1e-12,
        "passes": bool(worst_lin < 1e-12),
        "note": ("t = step/nsteps is a linear rescaling, so it cannot move a CV. This is the "
                 "control for the whole analysis: anything the curved schedules do move is "
                 "attributable to their curvature, not to remapping as such."),
    }
    print(f"linear-rescaling control: max |diff| = {worst_lin:.3e} "
          f"({'PASS' if worst_lin < 1e-12 else 'FAIL'})")

    # ---- how far each remap moves the gap, paired ------------------------- #
    #
    # Every axis is bootstrapped with the same seed over the same rows, so draw b
    # resamples the same clusters and the same replicate indices on each of them.
    # Differencing the gap across two axes within a draw therefore removes the
    # row-resampling uncertainty the two axes share, and leaves an interval on the
    # remap's own effect. Only the shared-row axes can be paired this way.
    ig, jf = METRICS.index("interface_quality"), METRICS.index("fold_identity")
    shift = {}
    for name in ("time_bb_ca", "time_local_latents", "time_native_assumed"):
        shift[name] = {}
        for unit, key in (("cluster", "by_cluster"), ("target", "by_target")):
            a = bs["step_index"][unit][:, ig] - bs["step_index"][unit][:, jf]
            b_ = bs[name][unit][:, ig] - bs[name][unit][:, jf]
            d = b_ - a
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            shift[name][unit] = {
                "gap_change_vs_step_axis": float(headline[name][key]["gap"]
                                                 - headline["step_index"][key]["gap"]),
                "ci95": [float(lo), float(hi)],
                "p_lt_0": float(np.nanmean(d < 0)),
                "fraction_of_step_axis_gap_retained": float(
                    headline[name][key]["gap"] / headline["step_index"][key]["gap"]),
            }
    print("\nchange in the gap relative to the step axis (paired over the same draws):")
    for name, per in shift.items():
        c = per["cluster"]
        print(f"  {name:24s} {c['gap_change_vs_step_axis']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  "
              f"retains {c['fraction_of_step_axis_gap_retained']*100:.1f}% of the step-axis gap")

    # ---- verdict ----------------------------------------------------------- #
    common = ["time_bb_ca", "time_local_latents"]
    surv_c = {a: headline[a]["by_cluster"]["survives"] for a in common}
    surv_t = {a: headline[a]["by_target"]["survives"] for a in common}
    n_surv = sum(surv_c.values())
    if n_surv == 2:
        verdict = "survives_both_remaps"
        text = ("The gap clears zero on both of Complexa's clocks at the lead unit. The schedule "
                "objection is closed: the interface-versus-fold spread difference is not an "
                "artifact of measuring two differently curved time axes in shared step units.")
    elif n_surv == 1:
        held = [a for a, v in surv_c.items() if v][0]
        lost = [a for a, v in surv_c.items() if not v][0]
        text = (f"The gap clears zero on {held} but not on {lost} at the lead unit. The claim is "
                f"therefore conditional on which mode the qualities load on, and that loading has "
                f"not been established here.")
        verdict = "survives_one_remap"
    else:
        verdict = "collapses_under_both"
        text = ("The gap does not clear zero on either of Complexa's clocks at the lead unit. Read "
                "on the model's own time coordinate the two qualities are not separated, so what "
                "the step-unit result describes is how the sampler allocates steps, not when the "
                "model settles each quality. This has to be said before the claim goes into a paper.")

    out = {
        "question": ("Does the interface-versus-fold settling-step spread gap survive being "
                     "re-expressed on Complexa's own flow-matching time, whose curve differs "
                     "between the two data modes?"),
        "no_new_compute": ("Same cells, same commitment curves, same stored bootstrap replicates as "
                           "the ten-step derivation. Only the axis the crossing is interpolated "
                           "against changes."),
        "scorer": GAP_KEY + " on corrected CVs, lead unit = structural cluster",
        "capture_steps": steps,
        "nsteps": nsteps,
        "config_source": cfg_path,
        "config_trailing_lines_dropped": dropped,
        "schedules": {dm: {k: v for k, v in d.items() if k != "_t"} for dm, d in sched.items()},
        "schedule_time_at_capture_steps": {
            "bb_ca": [float(v) for v in ax_ca],
            "local_latents": [float(v) for v in ax_lat],
            "uniform_reference": [float(v) for v in ax_unif],
        },
        "assumed_mode_loading": NATIVE_MODE,
        "assumed_mode_loading_status": ("ASSUMPTION, not a measurement. Used only for the "
                                        "time_native_assumed sensitivity, never for the headline."),
        "n_targets": len(reps),
        "boot": boot,
        "seed": seed,
        "schedule_elasticity_at_mean_settling_step": elast,
        "checks": checks,
        "headline": headline,
        "gap_change_vs_step_axis": shift,
        "verdict": {
            "code": verdict,
            "text": text,
            "survives_by_cluster": surv_c,
            "survives_by_target": surv_t,
            "native_assumed_by_cluster": headline["time_native_assumed"]["by_cluster"]["survives"],
            "native_assumed_by_target": headline["time_native_assumed"]["by_target"]["survives"],
        },
        "axes": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nVERDICT [{verdict}] {text}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
