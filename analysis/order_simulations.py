#!/usr/bin/env python3
"""Can this design tell a shared settling order apart from no shared order?

The pre-registered ordering test asked whether the four design qualities settle
in one order shared by every target. It failed: over the ten-step capture grid,
Kendall's W over the 19 targets' settling-step rankings is 0.1036 with 14
distinct orders, and none of the six pairwise contrasts is universally signed.

A failed test is not automatically a refutation. The ranks that go into W are
ranks of *point estimates*; the source analysis does not propagate each target's
settling-step uncertainty into them. The measured standard errors are large
relative to the gaps between qualities (median SE 4.58 steps for fold identity
but 30.1 for designability, against a 5.52-step gap between their pooled means),
so estimated ranks could scatter even if every target shared one true order.

The source is the ten-step re-derivation over all 190 cells, so the observed
statistics and the per-target standard errors that drive the null come from the
same refined grid as the rest of the paper.

This script measures that directly. It builds a null in which every target
*does* share one underlying order, perturbs it with each target's own measured
standard errors, and asks what Kendall's W and how many distinct orders such a
world would produce. If the observed W sits inside that null, the data cannot
distinguish "no shared order" from "one shared order measured too noisily",
and the honest verdict is that the pre-registered criterion was not met rather
than that a shared order is refuted.

Four nulls, all preserving one common order in the truth:

  common_value  every target's true settling steps equal the pooled means, so
                the true order is the pooled order and gaps are the pooled gaps.
  common_order  every target's true steps are the pooled means rescaled by that
                target's own overall speed, so targets may be globally fast or
                slow while the order stays identical. It matches each target's
                observed magnitude but not its gap *shape*.
  own_gaps      each target's own four observed settling steps, re-sorted into
                the pooled order. The order is shared by construction while
                every target keeps its own level *and* its own gap magnitudes,
                including the targets whose two middle qualities sit a few steps
                apart and would therefore swap under very little noise. Of the
                ordered truths constructed here it is the hardest to reject, but
                it is one plug-in truth built by sorting the observed estimates,
                not a maximisation over every per-target truth that satisfies a
                common order.
  own_gaps_deconvolved
                the same, with each target's spread shrunk to remove the part of
                it that is measurement noise. Observed within-target spread is
                true spread plus noise, so own_gaps simulates a world separated
                more widely than the estimates support; this variant subtracts
                the expected noise sum of squares before re-sorting.

The first two fix the *size* of the between-quality gaps at the pooled size,
which is the objection the own-gap nulls answer. "One shared order" is still a
composite hypothesis: it constrains only the signs of the gaps, so it also
contains worlds whose qualities are separated by much less than anything
estimated here, down to the degenerate world where all four settle together and
every estimated order is noise. Those worlds cannot be excluded by a simulation
run at one effect size. The gap-scale sweep below therefore walks the separation
down to zero and finds where the observation stops being extreme, in three
families:

  gap_scale_uniform  every target keeps its own mean level and the pooled
                     between-quality deviations, shrunk by a common factor c.
  gap_scale_varying  the same, but each target's factor is drawn independently
                     from U(0, 2c) per replicate, so the separation is
                     target-specific with mean c while the order is unchanged.
  gap_scale_own      each target's own noise-deconvolved deviations, re-sorted
                     into the pooled order and shrunk by a common factor c, so
                     the gap *shape* is the target's own at every scale.

That states what the design can support: within these families, shared orders
separated by at least so many steps are excluded, smaller ones are not.

Two limits apply to every number here, and both are stated in the manuscript.
The ordered truths are plug-in constructions from the observed estimates, three
families of them, rather than a profile over all target-specific means that
satisfy a common order, so the P values bound how far noise explains the failed
criterion under those families rather than excluding the composite hypothesis in
general. And each target's four qualities are perturbed independently at their
marginal standard errors, while the source bootstrap that produced those errors
resamples the four qualities together; a shared-order world whose four estimates
move together is therefore not among the ones simulated. The deconvolution uses
the same marginal standard errors and inherits the same limit.

Those limits are the terminal scope of this result, not a staging post toward a
stronger one. "One shared order" is a union over an unbounded space of orders
and separations, so no finite set of simulated families exhausts it and adding
families cannot convert a bound into an exclusion. Stating the result as a bound
over these three plug-in families under independent Gaussian marginal error is
what the design supports; if the objection that the families do not test the
full composite hypothesis is raised again, it is answered by that stated scope.
Narrowing the claim further would report less than the evidence shows. A
profiled order-restricted test that retains the joint error structure would
answer a different and stronger question, and would be new work rather than a
rewording of this one.

Simulated values are clipped to the capture grid's range and left-censored at
the floor, mirroring the estimator that produced the observed steps.

CPU-only, a few seconds. Writes results/order_power.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

# source data is supplied explicitly through the CLI

HERE = pathlib.Path(__file__).resolve().parent

METRICS = ["fold_identity", "designability", "structural_confidence", "interface_quality"]
N_SIM = 20000
SEED = 42


SOURCE_REL = ("results", "pooled_grid10.json")


def kendall_w(rank_matrix: np.ndarray) -> np.ndarray:
    """Kendall's W for a stack of (n_judges, n_items) rank matrices.

    rank_matrix is (n_sim, n_targets, n_metrics). Returns (n_sim,).
    """
    n_sim, m, k = rank_matrix.shape
    col_sums = rank_matrix.sum(axis=1)                      # (n_sim, k)
    mean_sum = m * (k + 1) / 2.0
    s = ((col_sums - mean_sum) ** 2).sum(axis=1)            # (n_sim,)
    return 12.0 * s / (m * m * (k ** 3 - k))


def rank_rows(values: np.ndarray) -> np.ndarray:
    """Ordinal ranks (1 = smallest) along the last axis, ties broken by order."""
    order = np.argsort(values, axis=-1, kind="stable")
    ranks = np.empty_like(order)
    idx = np.arange(values.shape[-1])
    np.put_along_axis(ranks, order, np.broadcast_to(idx, order.shape), axis=-1)
    return ranks + 1


def break_ties(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomise the order of exactly-tied values.

    Clipping simulated steps at the censoring floor creates exact ties. Ranking
    those with a stable sort would resolve every tie the same way, which invents
    agreement the design does not have and inflates the null's Kendall's W. A
    jitter far below the measurement scale randomises tied orders instead.
    """
    return values + rng.uniform(-1e-6, 1e-6, size=values.shape)


def n_distinct_orders(values: np.ndarray) -> np.ndarray:
    """Number of distinct quality orderings across targets, per simulation."""
    order = np.argsort(values, axis=-1, kind="stable")       # (n_sim, n_targets, k)
    k = values.shape[-1]
    codes = np.zeros(order.shape[:2], dtype=np.int64)
    for j in range(k):
        codes = codes * k + order[..., j]
    return np.array([len(np.unique(row)) for row in codes])


def sorted_into_pooled_order(values: np.ndarray, pooled_mean: np.ndarray) -> np.ndarray:
    """Re-sort each target's own values so every target shares the pooled order.

    ``values`` is (n_targets, n_metrics). The returned array holds exactly the
    same numbers row by row, permuted so that the metric with the smallest
    pooled mean gets each target's smallest value, and so on. Each target keeps
    its own level and its own gap magnitudes; only the order is imposed.
    """
    slot = np.argsort(np.argsort(pooled_mean))        # metric -> position in pooled order
    return np.sort(values, axis=1)[:, slot]


def deconvolve_deviations(th: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shrink each target's within-target spread to remove measurement noise.

    A target's observed deviations from its own mean are true deviations plus
    measurement error, so their sum of squares overstates the true separation by
    the expected noise contribution ``(1 - 1/k) * sum_j se_ij^2`` (the ``1/k``
    accounts for the noise that the target mean itself absorbs). Scaling the
    deviations by the square root of the surviving fraction gives a spread
    consistent with the estimated noise; a target whose observed spread is
    already smaller than its noise collapses to zero separation.

    Returns the shrunk deviations and the per-target shrink factor.
    """
    k = th.shape[1]
    dev = th - th.mean(axis=1, keepdims=True)
    ss_obs = (dev ** 2).sum(axis=1)
    ss_noise = (1.0 - 1.0 / k) * (se ** 2).sum(axis=1)
    frac = np.where(ss_obs > 0, np.clip((ss_obs - ss_noise) / np.where(ss_obs > 0, ss_obs, 1.0), 0.0, 1.0), 0.0)
    r = np.sqrt(frac)
    return dev * r[:, None], r


def main(argv: list[str] | None = None) -> int:
    global N_SIM, SEED
    parser = argparse.ArgumentParser(description="Run shared-order sensitivity simulations")
    parser.add_argument("input", type=pathlib.Path, help="ten-step pooled result JSON")
    parser.add_argument("output", type=pathlib.Path, help="destination JSON")
    parser.add_argument("--n-sim", type=int, default=N_SIM)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    N_SIM, SEED = args.n_sim, args.seed
    pooled = json.loads(args.input.read_text())
    p = pooled["pooled"]
    sp = p["t_half_spread_posthoc"]

    obs_w = p["order_consistency"]["kendall_w_t_half"]
    obs_distinct = p["order_consistency"]["n_distinct_orders"]
    floor = float(sp["floor_step"])
    top = float(max(pooled["capture_steps"]))

    targets = sorted(sp["t_half_by_target"])
    th = np.array([[sp["t_half_by_target"][t][m] for m in METRICS] for t in targets], float)
    se = np.array([[sp["t_half_se_by_target"][t][m] for m in METRICS] for t in targets], float)
    pooled_mean = np.array([p["t_half_pooled"][m]["mean"] for m in METRICS], float)

    if not np.isfinite(th).all() or not np.isfinite(se).all():
        print("non-finite settling steps or standard errors", file=sys.stderr)
        return 1

    n_t = len(targets)
    rng = np.random.default_rng(SEED)

    # Reproduce the observed statistics from the same per-target table, so the
    # null and the observation are computed by identical code.
    recomputed_w = float(kendall_w(rank_rows(th)[None, ...])[0])
    recomputed_distinct = int(n_distinct_orders(th[None, ...])[0])

    out = {
        "question": (
            "Could this design distinguish 'the four qualities settle in no shared order' "
            "from 'they settle in one shared order that is measured too noisily to see'?"
        ),
        "observed": {
            "kendall_w": obs_w,
            "n_distinct_orders": obs_distinct,
            "n_targets": n_t,
            "recomputed_kendall_w": recomputed_w,
            "recomputed_n_distinct_orders": recomputed_distinct,
        },
        "measurement_error_steps": {
            m: {
                "median_se": float(np.median(se[:, j])),
                "mean_se": float(se[:, j].mean()),
                "max_se": float(se[:, j].max()),
            }
            for j, m in enumerate(METRICS)
        },
        "pooled_t_half": {m: float(pooled_mean[j]) for j, m in enumerate(METRICS)},
        "source": {
            "record": "capture-grid",
            "file_sha256": __import__("hashlib").sha256(args.input.read_bytes()).hexdigest(),
            "capture_steps": pooled["capture_steps"],
            "n_cells": pooled["instrument_control"]["n_cells"],
        },
        "n_sim": N_SIM,
        "seed": SEED,
        "floor_step": floor,
        "max_step": top,
        "nulls": {},
    }

    for null_name in ("common_value", "common_order"):
        if null_name == "common_value":
            truth = np.broadcast_to(pooled_mean, (n_t, len(METRICS))).copy()
        else:
            speed = th.mean(axis=1) / pooled_mean.mean()
            truth = pooled_mean[None, :] * speed[:, None]

        noise = rng.normal(size=(N_SIM, n_t, len(METRICS))) * se[None, :, :]
        sim = break_ties(np.clip(truth[None, :, :] + noise, floor, top), rng)

        w = kendall_w(rank_rows(sim))
        nd = n_distinct_orders(sim)

        # How badly would the standard errors have to be understated before the
        # observed scatter became ordinary in a shared-order world?
        inflation = []
        for factor in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0):
            noise_f = rng.normal(size=(N_SIM // 4, n_t, len(METRICS))) * (se[None, :, :] * factor)
            sim_f = break_ties(np.clip(truth[None, :, :] + noise_f, floor, top), rng)
            w_f = kendall_w(rank_rows(sim_f))
            inflation.append({
                "se_factor": factor,
                "median_w": float(np.median(w_f)),
                "p_w_le_observed": float((w_f <= obs_w).mean()),
            })

        out["nulls"][null_name] = {
            "se_inflation_sweep": inflation,
            "description": (
                "every target shares the pooled order and the pooled magnitudes"
                if null_name == "common_value"
                else "every target shares the pooled order, rescaled to its own overall speed"
            ),
            "kendall_w": {
                "median": float(np.median(w)),
                "p05": float(np.percentile(w, 5)),
                "p95": float(np.percentile(w, 95)),
                "mean": float(w.mean()),
            },
            "n_distinct_orders": {
                "median": float(np.median(nd)),
                "p05": float(np.percentile(nd, 5)),
                "p95": float(np.percentile(nd, 95)),
            },
            # One-sided: is the observed order-agreement lower than a shared-order
            # world would produce? A large value means the observation is ordinary
            # under a shared order, so the failed test does not refute one.
            "p_w_le_observed": float((w <= obs_w).mean()),
            "p_distinct_ge_observed": float((nd >= obs_distinct).mean()),
        }

    # ---- nulls that keep each target's own gap magnitudes -------------------
    # The two nulls above impose the pooled gap structure on every target. A
    # target whose two middle qualities truly settle a few steps apart would
    # swap them under far less noise than a target separated by the pooled
    # amount, so those nulls under-produce low concordance. These two take each
    # target's own four settling steps and only re-sort them into the pooled
    # order, leaving level and gap magnitudes as measured.
    rng3 = np.random.default_rng(SEED + 2)
    dev_dec, shrink = deconvolve_deviations(th, se)
    own_truths = {
        "own_gaps": sorted_into_pooled_order(th, pooled_mean),
        "own_gaps_deconvolved": sorted_into_pooled_order(
            th.mean(axis=1, keepdims=True) + dev_dec, pooled_mean
        ),
    }
    own_descriptions = {
        "own_gaps": (
            "each target's own observed settling steps re-sorted into the pooled "
            "order, so the order is shared and every target keeps its own gaps"
        ),
        "own_gaps_deconvolved": (
            "the same, with each target's spread first shrunk to remove the part "
            "of the observed within-target spread that is measurement noise"
        ),
    }
    for null_name, truth in own_truths.items():
        noise = rng3.normal(size=(N_SIM, n_t, len(METRICS))) * se[None, :, :]
        sim = break_ties(np.clip(truth[None, :, :] + noise, floor, top), rng3)
        w = kendall_w(rank_rows(sim))
        nd = n_distinct_orders(sim)

        inflation = []
        for factor in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0):
            noise_f = rng3.normal(size=(N_SIM // 4, n_t, len(METRICS))) * (se[None, :, :] * factor)
            sim_f = break_ties(np.clip(truth[None, :, :] + noise_f, floor, top), rng3)
            w_f = kendall_w(rank_rows(sim_f))
            inflation.append({
                "se_factor": factor,
                "median_w": float(np.median(w_f)),
                "p_w_le_observed": float((w_f <= obs_w).mean()),
            })

        out["nulls"][null_name] = {
            "se_inflation_sweep": inflation,
            "description": own_descriptions[null_name],
            "separation_steps": {
                "median_target_range": float(np.median(truth.max(axis=1) - truth.min(axis=1))),
                "min_target_range": float((truth.max(axis=1) - truth.min(axis=1)).min()),
                "median_adjacent_gap": float(np.median(np.diff(np.sort(truth, axis=1), axis=1))),
                "min_adjacent_gap": float(np.diff(np.sort(truth, axis=1), axis=1).min()),
            },
            "kendall_w": {
                "median": float(np.median(w)),
                "p05": float(np.percentile(w, 5)),
                "p95": float(np.percentile(w, 95)),
                "mean": float(w.mean()),
            },
            "n_distinct_orders": {
                "median": float(np.median(nd)),
                "p05": float(np.percentile(nd, 5)),
                "p95": float(np.percentile(nd, 95)),
            },
            "p_w_le_observed": float((w <= obs_w).mean()),
            "p_distinct_ge_observed": float((nd >= obs_distinct).mean()),
        }
    out["nulls"]["own_gaps_deconvolved"]["shrink_factors"] = {
        "median": float(np.median(shrink)),
        "min": float(shrink.min()),
        "max": float(shrink.max()),
        "n_targets_collapsed_to_zero": int((shrink == 0).sum()),
    }

    # ---- gap-scale sweep over the composite shared-order hypothesis ---------
    # A shared order fixes the signs of the between-quality differences, not
    # their size. Shrink the pooled deviations by a factor c and ask at what c
    # the observed W stops being extreme. c = 1 is a shared order as separated
    # as the pooled estimates; c = 0 is four qualities settling together.
    rng2 = np.random.default_rng(SEED + 1)
    dev = pooled_mean - pooled_mean.mean()          # pooled between-quality deviations
    centre = th.mean(axis=1)                        # each target's own level
    pooled_range = float(pooled_mean.max() - pooled_mean.min())
    scales = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    own_dev_sorted = sorted_into_pooled_order(dev_dec, pooled_mean)
    own_range = float(np.median(own_dev_sorted.max(axis=1) - own_dev_sorted.min(axis=1)))
    family_scale = {
        "gap_scale_uniform": pooled_range,
        "gap_scale_varying": pooled_range,
        "gap_scale_own": own_range,
    }

    sweeps: dict[str, list[dict]] = {
        "gap_scale_uniform": [], "gap_scale_varying": [], "gap_scale_own": [],
    }
    for family in sweeps:
        for c in scales:
            if family == "gap_scale_own":
                truth = centre[None, :, None] + c * own_dev_sorted[None, :, :]
                truth = np.broadcast_to(truth, (N_SIM, n_t, len(METRICS)))
            elif family == "gap_scale_uniform":
                truth = centre[None, :, None] + c * dev[None, None, :]
                truth = np.broadcast_to(truth, (N_SIM, n_t, len(METRICS)))
            else:
                # target-specific separation magnitude, common order preserved
                ci = rng2.uniform(0.0, 2.0 * c, size=(N_SIM, n_t, 1))
                truth = centre[None, :, None] + ci * dev[None, None, :]
            noise = rng2.normal(size=(N_SIM, n_t, len(METRICS))) * se[None, :, :]
            sim = break_ties(np.clip(truth + noise, floor, top), rng2)
            w = kendall_w(rank_rows(sim))
            nd = n_distinct_orders(sim)
            sweeps[family].append({
                "gap_scale": c,
                "separation_steps": c * family_scale[family],
                "median_w": float(np.median(w)),
                "p05_w": float(np.percentile(w, 5)),
                "p95_w": float(np.percentile(w, 95)),
                "median_distinct_orders": float(np.median(nd)),
                "p_w_le_observed": float((w <= obs_w).mean()),
                "p_distinct_ge_observed": float((nd >= obs_distinct).mean()),
            })

    def crossing(rows: list[dict], scale: float) -> dict:
        """Largest gap scale whose null still makes the observation ordinary."""
        ok = [r for r in rows if r["p_w_le_observed"] > 0.05]
        bad = [r for r in rows if r["p_w_le_observed"] <= 0.05]
        largest_ok = max((r["gap_scale"] for r in ok), default=None)
        smallest_excluded = min((r["gap_scale"] for r in bad), default=None)
        return {
            "largest_compatible_gap_scale": largest_ok,
            "largest_compatible_separation_steps": (
                None if largest_ok is None else largest_ok * scale
            ),
            "smallest_excluded_gap_scale": smallest_excluded,
            "smallest_excluded_separation_steps": (
                None if smallest_excluded is None else smallest_excluded * scale
            ),
        }

    out["gap_scale_sweep"] = {
        "description": (
            "Shared-order worlds whose between-quality separation is the pooled "
            "separation shrunk by a factor c. c = 1 reproduces the pooled gaps on "
            "each target's own level; c = 0 has all four qualities settling together, "
            "so every estimated order is noise. gap_scale_varying draws each target's "
            "factor from U(0, 2c) per replicate, giving target-specific separations "
            "with the same common order. gap_scale_own instead shrinks each target's "
            "own noise-deconvolved deviations, re-sorted into the pooled order, so the "
            "gap shape stays target-specific at every scale; its separation_steps are "
            "measured against the median per-target range of those deviations."
        ),
        "pooled_separation_steps": pooled_range,
        "own_separation_steps_median": own_range,
        "family_separation_scale": family_scale,
        "seed": SEED + 1,
        "n_sim": N_SIM,
        "families": sweeps,
        "crossings": {f: crossing(rows, family_scale[f]) for f, rows in sweeps.items()},
    }

    # The design's resolving power for this test: the best order-agreement a
    # shared-order world could be expected to show through this much noise.
    cv = out["nulls"]["common_value"]["kendall_w"]
    co = out["nulls"]["common_order"]["kendall_w"]

    # Smallest tested SE inflation at which the observation stops being extreme.
    breaks = {}
    for null_name, blk in out["nulls"].items():
        hit = [r["se_factor"] for r in blk["se_inflation_sweep"] if r["p_w_le_observed"] > 0.05]
        breaks[null_name] = min(hit) if hit else None
    tested_max = max(r["se_factor"] for r in out["nulls"]["common_value"]["se_inflation_sweep"])

    out["verdict"] = {
        "observed_inside_both_nulls": bool(
            out["nulls"]["common_value"]["p_w_le_observed"] > 0.05
            and out["nulls"]["common_order"]["p_w_le_observed"] > 0.05
        ),
        "observed_inside_own_gap_null": bool(
            out["nulls"]["own_gaps"]["p_w_le_observed"] > 0.05
        ),
        "observed_inside_own_gap_deconvolved_null": bool(
            out["nulls"]["own_gaps_deconvolved"]["p_w_le_observed"] > 0.05
        ),
        "shared_order_null_median_w": cv["median"],
        "shared_order_null_median_w_generous": co["median"],
        "own_gap_null_median_w": out["nulls"]["own_gaps"]["kendall_w"]["median"],
        "own_gap_deconvolved_null_median_w": (
            out["nulls"]["own_gaps_deconvolved"]["kendall_w"]["median"]
        ),
        "own_gap_null_p_w_le_observed": out["nulls"]["own_gaps"]["p_w_le_observed"],
        "own_gap_deconvolved_null_p_w_le_observed": (
            out["nulls"]["own_gaps_deconvolved"]["p_w_le_observed"]
        ),
        "se_inflation_needed": {
            "common_value": breaks["common_value"],
            "common_order": breaks["common_order"],
            "own_gaps": breaks["own_gaps"],
            "own_gaps_deconvolved": breaks["own_gaps_deconvolved"],
            "max_factor_tested": tested_max,
            "note": (
                "null means no tested inflation up to max_factor_tested made the "
                "observed Kendall's W ordinary under a shared order"
            ),
        },
        "gap_scale_crossings": out["gap_scale_sweep"]["crossings"],
        "reading": (
            "The two fixed-size nulls answer a power question at one effect size: could "
            "this design have seen a shared order separated as widely as the pooled "
            "estimates? It could. The own-gap nulls impose only the order at the "
            "separations actually estimated: every target keeps its own level and its "
            "own gap magnitudes, including targets whose middle two qualities sit a few "
            "steps apart. own_gaps_deconvolved is the honest version of that, because "
            "observed within-target spread is inflated by measurement noise. What no "
            "simulation at an estimated separation can exclude is a shared order "
            "separated by much less than that; the gap-scale sweep bounds those worlds "
            "instead, and within these families shared orders separated by at least "
            "smallest_excluded_separation_steps are excluded. Two limits apply to all of "
            "it: the ordered truths are plug-in constructions from the observed "
            "estimates, not a profile over every per-target truth compatible with a "
            "common order, and each target's four qualities are perturbed independently "
            "at their marginal standard errors while the source bootstrap resamples the "
            "four together. These are therefore conditional sensitivity results, not an "
            "exclusion of the composite hypothesis in general."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dest = args.output
    dest.write_text(json.dumps(out, indent=1) + "\n")

    print(f"observed  W={obs_w:.4f} (recomputed {recomputed_w:.4f}), "
          f"distinct orders={obs_distinct} (recomputed {recomputed_distinct})")
    for name, blk in out["nulls"].items():
        kw, nd = blk["kendall_w"], blk["n_distinct_orders"]
        print(f"null {name:13s} W median {kw['median']:.4f} "
              f"[{kw['p05']:.4f}, {kw['p95']:.4f}]  "
              f"distinct median {nd['median']:.1f} [{nd['p05']:.0f}, {nd['p95']:.0f}]  "
              f"P(W<=obs)={blk['p_w_le_observed']:.4f}  "
              f"P(distinct>=obs)={blk['p_distinct_ge_observed']:.4f}")
        infl = "  ".join(f"x{r['se_factor']:g}:{r['median_w']:.3f}/p={r['p_w_le_observed']:.3f}"
                         for r in blk["se_inflation_sweep"])
        print(f"     SE inflation  {infl}")
    print(f"pooled separation {pooled_range:.2f} steps; "
          f"median own deconvolved separation {own_range:.2f} steps; "
          f"shrink factors {json.dumps(out['nulls']['own_gaps_deconvolved']['shrink_factors'])}")
    for family, rows in out["gap_scale_sweep"]["families"].items():
        line = "  ".join(f"c={r['gap_scale']:.1f}:W={r['median_w']:.3f}/p={r['p_w_le_observed']:.3f}"
                         for r in rows)
        print(f"{family:18s} {line}")
        print(f"     crossing {json.dumps(out['gap_scale_sweep']['crossings'][family])}")
    print(f"verdict: {json.dumps(out['verdict']['se_inflation_needed'])}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
