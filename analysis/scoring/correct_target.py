#!/usr/bin/env python
"""Re-derive one target's commitment curves with the ruler's error taken out.

Write a completion's reading as

    y[r,c,j] = mu + a_r + b_rc + e_rcj + eps_rcj

with ``eps`` the ruler's error. If ``eps`` were independent across completions the
correction would be one-sided: the nested ANOVA's root and child components would be
unbiased and the whole ruler error would land in the within-child term.

It is not independent. The ruler's error is a fixed function of the complex it reads,
and two completions of the same child at a late capture step are nearly the same
complex, so their errors are nearly the same number. ``eps`` therefore has its own
three-level structure, measured by ``nested_split.py``, and each outcome component
loses its matching share:

    Var(a)' = Var(a) - eps_root
    Var(b)' = Var(b) - eps_child
    Var(e)' = Var(e) - eps_within

The within-child subtraction raises commitment and the other two lower it, so the net
effect is a real question rather than a foregone conclusion. Subtracting the marginal
ruler variance from the within-child term alone, as a naive reading of the plan would
do, drives commitment to exactly 1.0 at late steps where the within-child spread has
collapsed; that variant is retained as ``commitment_within_only`` so the difference
is visible rather than asserted.

Ruler error by quality:

``interface quality, structural confidence, ipTM``
    from the AlphaFold ensemble members (``ruler.py``). Only the part that varies
    from completion to completion counts: the structure-by-member interaction, plus
    the variation of the recorded member's level-dependent offset across the cell's
    own score range. A member's constant offset is an additive constant inside a
    cell, cancels out of every variance component, and is deliberately excluded.
    Both terms are evaluated at each of the cell's own 1024 completion levels.
``designability``
    from the source analysis's own repeat-measure control, which re-ran the ProteinMPNN
    redesign and refold with an independent seed on a subset of states. That is a
    genuine repeat measurement, so no ensemble stand-in is needed. The 32 pairs in a
    single cell give a variance with about a 25 per cent relative standard error, so
    the level model is fitted on all 4256 pairs at once and evaluated at each cell's
    own scores rather than taken cell by cell; the per-cell figure is kept as a
    diagnostic.
``fold identity``
    zero. Clustering was run once per target over a fixed label space, so given a
    completed structure the assignment is deterministic and its scoring reliability
    is one by construction. Its curve is unchanged; whether the *label* discards
    structural variation is a separate question, answered by ``quantisation.py``.

Writes the corrected curves in exactly the layout the parent's ``spread_stats2.py``
reads, so the pooling, the cluster bootstrap and the settling-step interpolation
are the audited originals rather than a reimplementation.

    correct_target.py <target> <out_dir> --ruler <reliability.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

# Maintained estimators are imported from the package so every scoring path
# uses the same implementation.

from ..commitment.boot import _resample, as_rck, commitment_replicates, draw_design_boot  # noqa: E402
from ..commitment.variance import decompose, decompose_categorical  # noqa: E402

from . import ruler as ruler_mod  # noqa: E402

STEPS = (10, 25, 50, 75, 100, 200, 300)
N_BOOT = 2000
CONTINUOUS = {
    "interface_quality": "af2folding_i_pae",
    "structural_confidence": "af2folding_plddt_log",
    "iptm": "af2folding_i_ptm_log",
}
ALL_METRICS = ("fold_identity", "designability", "structural_confidence", "interface_quality", "iptm")


# --------------------------------------------------------------------------- #
# components under the shared resample
# --------------------------------------------------------------------------- #


def components_continuous(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nested-ANOVA components for a stack of resampled cells ``[B, R, C, k]``.

    Same expected-mean-square estimators as ``commitment.boot``; this returns the
    three components instead of collapsing them, so the ruler term can be removed
    from the within component before the share is formed.
    """
    B, R, C, k = A.shape
    child_mean = A.mean(3)
    root_mean = child_mean.mean(2)
    grand = root_mean.mean(1)
    ss_within = ((A - child_mean[..., None]) ** 2).sum((1, 2, 3))
    ss_child = k * ((child_mean - root_mean[..., None]) ** 2).sum((1, 2))
    ss_root = C * k * ((root_mean - grand[:, None]) ** 2).sum(1)
    ms_within = ss_within / (R * C * (k - 1))
    ms_child = ss_child / (R * (C - 1))
    ms_root = ss_root / (R - 1)
    return (ms_root - ms_child) / (C * k), (ms_child - ms_within) / k, ms_within


def share(s2a: np.ndarray, s2b: np.ndarray, s2e: np.ndarray,
          ruler: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Commitment after removing the ruler's own share of each component."""
    pa = np.clip(s2a - ruler[0], 0, None)
    pb = np.clip(s2b - ruler[1], 0, None)
    pe = np.clip(s2e - ruler[2], 0, None)
    tot = pa + pb + pe
    out = np.full(tot.shape, np.nan)
    ok = tot > 0
    out[ok] = (pa[ok] + pb[ok]) / tot[ok]
    return out


def replicate_commitments(a3: np.ndarray, u: np.ndarray, ruler: tuple[float, float, float],
                          chunk: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uncorrected, fully corrected, and within-only-corrected commitment per resample."""
    B = u.shape[0]
    raw = np.empty(B)
    cor = np.empty(B)
    win = np.empty(B)
    for s in range(0, B, chunk):
        A = _resample(a3, u[s:s + chunk])
        s2a, s2b, s2e = components_continuous(A)
        raw[s:s + chunk] = share(s2a, s2b, s2e)
        cor[s:s + chunk] = share(s2a, s2b, s2e, ruler)
        win[s:s + chunk] = share(s2a, s2b, s2e, (0.0, 0.0, sum(ruler)))
    return raw, cor, win


def split_ruler(total: float, fractions: dict, how: str = "pooled") -> tuple[float, float, float]:
    """Marginal ruler variance -> its root, child and within-child parts.

    ``pooled`` sums the unclipped components over the 19 cells at a capture step
    before forming the ratio, which is the estimate that does not inherit the
    upward bias of clipping a thin per-cell component at zero. ``mean`` is the
    mean of per-cell fractions and is carried as a sensitivity.
    """
    keys = (("frac_root_pooled", "frac_child_pooled", "frac_within_pooled") if how == "pooled"
            else ("frac_root_mean", "frac_child_mean", "frac_within_mean"))
    if how == "pooled" and not all(k in fractions and np.isfinite(fractions[k]) for k in keys):
        keys = ("frac_root_mean", "frac_child_mean", "frac_within_mean")
    return (total * fractions[keys[0]], total * fractions[keys[1]], total * fractions[keys[2]])


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def designability_cell_repeat(per_target2_dir: str, target: str, step: int) -> tuple[float, dict]:
    """One cell's own repeat-measure variance. Diagnostic only: 32 pairs is thin."""
    p = os.path.join(per_target2_dir, f"{target}_step{step}.json")
    if not os.path.exists(p):
        return 0.0, {"available": False}
    with open(p) as fh:
        j = json.load(fh)
    rep = j.get("cell", {}).get("designability", {}).get("repeat_measure", {})
    p1 = np.asarray(rep.get("pass1", []), float)
    p2 = np.asarray(rep.get("pass2", []), float)
    ok = np.isfinite(p1) & np.isfinite(p2)
    if ok.sum() < 3:
        return 0.0, {"available": False, "n": int(ok.sum())}
    d = p1[ok] - p2[ok]
    # two independent passes of the same measurement: Var(d) = 2 sigma^2
    s2 = float(np.var(d, ddof=1) / 2.0)
    return s2, {
        "available": True,
        "n_pairs": int(ok.sum()),
        "sigma2": s2,
        "sigma": float(np.sqrt(s2)),
        "mean_abs_diff": float(np.mean(np.abs(d))),
        "r": float(np.corrcoef(p1[ok], p2[ok])[0, 1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("out_dir")
    ap.add_argument("--ruler", required=True)
    ap.add_argument("--nesting", required=True,
                    help="nested_split.json: where the ruler error sits in the design")
    ap.add_argument("--harvest-root", default=os.environ.get("HARVEST_ROOT"))
    ap.add_argument("--fold-cache", default=os.environ.get("FOLD_CACHE_ROOT"))
    ap.add_argument("--stage2", default=os.environ.get("STAGE2_ROOT"))
    ap.add_argument("--per-target2", default=os.environ.get("PER_TARGET2"))
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    import analyze2  # the parent's loaders, used verbatim

    with open(args.ruler) as fh:
        ruler_cfg = json.load(fh)
    with open(args.nesting) as fh:
        nesting = json.load(fh)["per_quality"]

    os.makedirs(args.out_dir, exist_ok=True)
    steps_out: dict[str, dict] = {}
    diag: dict[str, dict] = {}

    seed = int(hashlib.sha1(f"boot|{args.target}".encode()).hexdigest()[:8], 16)
    u_cache: dict[int, np.ndarray] = {}

    for step in STEPS:
        shard_p = os.path.join(args.harvest_root, args.target, f"step{step}", "shard.pt")
        if not os.path.exists(shard_p):
            print(f"  no shard for step{step}, skipping")
            continue
        d = torch.load(shard_p, weights_only=False, map_location="cpu")
        root = np.asarray(d["root_idx"])
        child = np.asarray(d["child_idx"])
        R = len(np.unique(root))
        if R not in u_cache:
            u_cache[R] = draw_design_boot(args.boot, R, seed=seed)
        u = u_cache[R]
        m = d["metrics"]

        cell: dict[str, dict] = {}
        boots: dict[str, np.ndarray] = {}
        boots_raw: dict[str, np.ndarray] = {}
        boots_win: dict[str, np.ndarray] = {}
        cell_diag: dict[str, dict] = {}

        # ---- the three AlphaFold-scored continuous qualities -----------------
        for name, key in CONTINUOUS.items():
            y = np.asarray(m[key], float)
            q = ruler_cfg["per_quality"][name]
            parts = ruler_mod.cell_sigma2(q["level_model"], y.ravel())
            alt = ruler_mod.cell_sigma2(q["level_model_consensus_binned"], y.ravel())
            s2r = float(parts["total"])
            frac = nesting[name]["by_step"][str(step)]
            rsplit = split_ruler(s2r, frac)
            rsplit_alt = split_ruler(s2r, frac, how="mean")
            dec = decompose(y, root, child)
            a3 = as_rck(y, root, child)
            raw, cor, win = replicate_commitments(a3, u, rsplit)
            one = lambda v: np.array([v])  # noqa: E731
            c_point = float(share(one(dec.sigma2_root), one(dec.sigma2_child),
                                  one(dec.sigma2_within), rsplit)[0])
            c_win = float(share(one(dec.sigma2_root), one(dec.sigma2_child),
                                one(dec.sigma2_within), (0.0, 0.0, s2r))[0])
            cell[name] = {
                "kind": "continuous",
                "decomposition": {
                    **dec.as_dict(),
                    "sigma2_ruler": s2r,
                    "sigma2_ruler_root": rsplit[0],
                    "sigma2_ruler_child": rsplit[1],
                    "sigma2_ruler_within": rsplit[2],
                    "sigma2_within_corrected": float(max(dec.sigma2_within - rsplit[2], 0.0)),
                    "commitment_uncorrected": float(dec.commitment),
                    "commitment_within_only": c_win,
                    "commitment": c_point,
                },
            }
            boots[name] = cor
            boots_raw[name] = raw
            boots_win[name] = win
            cell_diag[name] = {
                "sigma2_ruler": s2r,
                "sigma2_ruler_interaction": float(parts["interaction"]),
                "sigma2_ruler_offset_variation": float(parts["offset_variation"]),
                "sigma2_ruler_consensus_binned": float(alt["total"]),
                "ruler_frac_within": frac.get("frac_within_pooled", frac["frac_within_mean"]),
                "ruler_frac_within_mean_of_cells": frac["frac_within_mean"],
                "sigma2_ruler_within_alt_mean_of_cells": rsplit_alt[2],
                "commitment_corrected_alt_nesting": float(share(
                    np.array([dec.sigma2_root]), np.array([dec.sigma2_child]),
                    np.array([dec.sigma2_within]), rsplit_alt)[0]),
                "sigma2_ruler_within": rsplit[2],
                "sigma2_within": float(dec.sigma2_within),
                "ruler_share_of_within": float(rsplit[2] / dec.sigma2_within)
                if dec.sigma2_within > 0 else float("nan"),
                "ruler_share_of_total": float(s2r / dec.total) if dec.total > 0 else float("nan"),
                "commitment_uncorrected": float(dec.commitment),
                "commitment_within_only": c_win,
                "commitment_corrected": c_point,
            }

        # ---- fold identity: categorical, deterministic given the label space --
        fold_lab, n_lab = analyze2.load_fold_cache(
            os.path.join(args.fold_cache, f"{args.target}.npz"), step)
        if fold_lab is not None and (fold_lab >= 0).all():
            dec = decompose_categorical(fold_lab, root, child)
            a3 = np.rint(as_rck(fold_lab, root, child)).astype(np.int64)
            rep = commitment_replicates(a3, u, kind="categorical", n_lab=n_lab)
            cell["fold_identity"] = {
                "kind": "categorical",
                "decomposition": {**dec.as_dict(), "sigma2_ruler": 0.0,
                                  "commitment_uncorrected": float(dec.commitment),
                                  "commitment_within_only": float(dec.commitment),
                                  "commitment": float(dec.commitment)},
            }
            boots["fold_identity"] = rep
            boots_raw["fold_identity"] = rep
            boots_win["fold_identity"] = rep
            cell_diag["fold_identity"] = {"sigma2_ruler": 0.0,
                                          "commitment_uncorrected": float(dec.commitment),
                                          "commitment_corrected": float(dec.commitment)}

        # ---- designability: repeat-measure control ---------------------------
        st2 = analyze2.load_stage2(args.stage2, args.target, step) if args.stage2 else None
        if st2 is not None:
            y = np.asarray(st2["y17"], float)
            qd = ruler_cfg["per_quality"].get("designability")
            if qd is None:
                raise SystemExit("reliability.json has no designability model; "
                                 "run ruler.py with --per-target2")
            parts = ruler_mod.cell_sigma2(qd["level_model"], y.ravel())
            alt = ruler_mod.cell_sigma2(qd["level_model_consensus_binned"], y.ravel())
            s2r = float(parts["total"])
            _, dg = designability_cell_repeat(args.per_target2, args.target, step)
            dec = decompose(y, st2["root"], st2["child"])
            a3 = as_rck(y, st2["root"], st2["child"])
            # designability's noise is a genuine lottery: ProteinMPNN resamples the
            # sequence and AlphaFold refolds it, so a repeat draws fresh randomness
            # and the error IS independent per measurement. The AlphaFold reading is
            # the opposite -- deterministic in the structure, hence correlated across
            # similar structures -- which is why only this quality keeps the
            # one-sided, all-in-the-within-term correction.
            rsplit = (0.0, 0.0, s2r)
            raw, cor, win = replicate_commitments(a3, u, rsplit)
            one = lambda v: np.array([v])  # noqa: E731
            c_point = float(share(one(dec.sigma2_root), one(dec.sigma2_child),
                                  one(dec.sigma2_within), rsplit)[0])
            cell["designability"] = {
                "kind": "continuous",
                "decomposition": {
                    **dec.as_dict(),
                    "sigma2_ruler": s2r,
                    "sigma2_ruler_within": s2r,
                    "noise_is_independent_by_construction": True,
                    "sigma2_within_corrected": float(max(dec.sigma2_within - s2r, 0.0)),
                    "commitment_uncorrected": float(dec.commitment),
                    "commitment_within_only": c_point,
                    "commitment": c_point,
                },
            }
            boots["designability"] = cor
            boots_raw["designability"] = raw
            boots_win["designability"] = win
            cell_diag["designability"] = {
                "sigma2_ruler": s2r,
                "sigma2_ruler_interaction": float(parts["interaction"]),
                "sigma2_ruler_offset_variation": float(parts["offset_variation"]),
                "sigma2_ruler_consensus_binned": float(alt["total"]),
                "ruler_frac_within": 1.0,
                "sigma2_ruler_within": s2r,
                "sigma2_within": float(dec.sigma2_within),
                "ruler_share_of_within": float(s2r / dec.sigma2_within) if dec.sigma2_within > 0 else float("nan"),
                "ruler_share_of_total": float(s2r / dec.total) if dec.total > 0 else float("nan"),
                "commitment_uncorrected": float(dec.commitment),
                "commitment_within_only": c_point,
                "commitment_corrected": c_point,
                "repeat_measure": dg,
            }

        # ---- does the uncorrected recomputation reproduce the source analysis? ----
        parent_p = os.path.join(args.per_target2, f"{args.target}_step{step}.json")
        if os.path.exists(parent_p):
            with open(parent_p) as fh:
                pj = json.load(fh).get("cell", {})
            parent_boot_p = os.path.join(args.per_target2, f"{args.target}_step{step}_boot.npz")
            pz = np.load(parent_boot_p) if os.path.exists(parent_boot_p) else {}
            for name in cell_diag:
                pc = pj.get(name, {}).get("decomposition", {}).get("commitment")
                if pc is None:
                    continue
                cell_diag[name]["parent_commitment"] = float(pc)
                cell_diag[name]["reproduction_abs_diff"] = abs(
                    float(pc) - cell_diag[name]["commitment_uncorrected"])
                if name in getattr(pz, "files", []):
                    a = np.asarray(pz[name], float)
                    b = np.asarray(boots_raw[name], float)
                    n = min(a.size, b.size)
                    cell_diag[name]["reproduction_boot_max_abs_diff"] = float(
                        np.nanmax(np.abs(a[:n] - b[:n])))

        np.savez_compressed(os.path.join(args.out_dir, f"{args.target}_step{step}_boot.npz"), **boots)
        np.savez_compressed(os.path.join(args.out_dir, f"{args.target}_step{step}_boot_raw.npz"), **boots_raw)
        np.savez_compressed(os.path.join(args.out_dir, f"{args.target}_step{step}_boot_win.npz"), **boots_win)
        steps_out[str(step)] = cell
        diag[str(step)] = cell_diag
        line = "  ".join(
            f"{k}: {v['commitment_uncorrected']:.4f}->{v['commitment_corrected']:.4f}"
            for k, v in cell_diag.items()
        )
        print(f"[{args.target} step{step}] {line}", flush=True)

    with open(os.path.join(args.out_dir, f"{args.target}.json"), "w") as fh:
        json.dump({"target": args.target, "steps": steps_out}, fh)
    with open(os.path.join(args.out_dir, f"{args.target}_diagnostics.json"), "w") as fh:
        json.dump({"target": args.target, "steps": diag}, fh, indent=2)
    print(f"-> {args.out_dir}/{args.target}.json")


if __name__ == "__main__":
    main()
