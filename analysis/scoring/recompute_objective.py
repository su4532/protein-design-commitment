#!/usr/bin/env python
"""Redo the action-step objective on the corrected variance components.

The action-timing experiment (the recorded source analysis) asks whether picking
the search's action step per target beats the best fixed step, and answers it with

    J(t) = r2_observed(t) * (sigma2_child(t) + sigma2_within(t)/k) / total(t)

capped at ``child_share = sigma2_child / total``. Its published conservative estimate is
that per-target timing gains ``-0.001390`` in J over the deployed policy, with a 75.31%
resolvable floor, so the recommendation is to leave the action step where it is.

Every variance component in that expression is one this experiment's sensitivity
analysis moves, and the readout score is not. ``r2_observed`` is the squared correlation
between a probe's predictions and the recorded outcome; treating part of the recorded
outcome's spread as scorer disagreement rather than design variance does not change the
fit or that correlation. It changes only the decomposition the correlation is converted
through.

That is not a per-cell rescaling, because a cell is not the unit the answer is computed
on. The source experiment's scored criterion comes from a three-way rotation of each
target's roots: one group fits the readout, a second selects the step, a third scores it.
Each of those scored subsets has its own nested-ANOVA decomposition, its own score
levels and so its own ruler variance, and its own admissibility cap; so does every
design-bootstrap replicate inside it, which resamples roots and children and recomputes
the components. This script therefore re-runs the source experiment's scoring stage
rather than rescaling its outputs. For each scored set of states -- the full cell, each
of the ``reps x 6`` rotation cells, and each of the 200 design-bootstrap replicates of
each rotation cell -- it

1. decomposes that set's own outcomes with the source's estimator,
2. evaluates the ruler variance on that set's own score levels and splits it across the
   three levels with the same nesting fractions ``correct_target.py`` uses,
3. subtracts, floors at zero, and re-forms ``J`` and the cap ``child_share`` from the
   corrected components.

The permutation floor and the squared correlation are the source experiment's own, at
the source's seeds, and the pooling that turns curves into the headline is its
``pool_gain.py`` run unchanged.

``--zero-ruler`` sets the subtraction to zero everywhere. That is a real check here, not
an identity: the whole scoring stage is re-run, so its output has to reproduce the source
experiment's stored per-cell J, its stored rotation-cell J and its stored bootstrap
replicate archives, and then its published pooled gain, before anything is subtracted.
The maximum absolute difference against every stored value is reported either way.

One honest limit remains, counted rather than assumed away. Where the corrected
components leave no variance at all, J is zero by the same convention the source
experiment uses for a degenerate cell.

    recompute_objective.py <labels_dir> <pred_dir> <src_curves_dir> <out_curves_dir>
        <out_json> --ruler <reliability.json> --nesting <nested_split.json>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.dirname(os.path.abspath(__file__))

from . import ruler as ruler_mod  # noqa: E402
from .correct_target import split_ruler  # noqa: E402

from ..action.objective import CAPTURE_STEPS, QUALITIES, stable_seed  # noqa: E402
from ..action.score_curves import _components, _rowwise_corr, permutation_floor_vec  # noqa: E402
from ..commitment.boot import as_rck, draw_design_boot  # noqa: E402
from ..commitment.variance import decompose  # noqa: E402

N_GROUPS = 3
NAN_CELL = {"J": float("nan"), "J_raw": float("nan"), "child_share": float("nan"),
            "r2_sibling": float("nan"), "r2_observed": float("nan"), "floor": float("nan"),
            "n_states": 0, "n_roots": 0}


# --------------------------------------------------------------------------- #
# the objective, with the ruler's share removed from each component
# --------------------------------------------------------------------------- #


def j_corrected(r2, s2a, s2b, s2e, k: int, ruler):
    """``J`` and its cap after subtracting the ruler's share of each component.

    With ``ruler = (0, 0, 0)`` this is character for character the source
    experiment's ``_J_from``: it clips the components at zero, forms
    ``child_share`` and the stable ``(s2b + s2e/k)/total`` product, and applies the
    admissibility cap. Subtracting first and clipping after is the same operation,
    because the subtracted quantity is non-negative.
    """
    ra, rb, re = ruler
    pa = np.clip(np.asarray(s2a, dtype=float) - ra, 0, None)
    pb = np.clip(np.asarray(s2b, dtype=float) - rb, 0, None)
    pe = np.clip(np.asarray(s2e, dtype=float) - re, 0, None)
    tot = pa + pb + pe
    good = tot > 0
    child_share = np.where(good, pb / np.where(good, tot, 1.0), 0.0)
    j = np.clip(np.asarray(r2, dtype=float), 0, None) * (pb + pe / k) / np.where(good, tot, 1.0)
    return np.where(good, np.minimum(j, child_share), 0.0), child_share, np.where(good, j, 0.0)


class Ruler:
    """The scorer's variance for an arbitrary set of states, split across levels.

    Same construction as ``correct_target.py``: the level model is evaluated on the
    states' own recorded values, so a subset of roots gets the ruler variance of the
    score range it actually holds; fold identity is deterministic given the label
    space and gets zero; designability's repeat-measure noise is independent by
    construction and lands entirely in the within-child term.
    """

    ZERO = (0.0, 0.0, 0.0)

    def __init__(self, cfg: dict, nesting: dict, zero: bool = False):
        self.cfg, self.nesting, self.zero = cfg, nesting, zero
        self._frac: dict[tuple[str, int], tuple[float, float, float]] = {}

    def has(self, q: str) -> bool:
        return not self.zero and q != "fold_identity" and q in self.cfg["per_quality"]

    def fractions(self, q: str, step: int) -> tuple[float, float, float]:
        key = (q, step)
        if key not in self._frac:
            if q == "designability":
                self._frac[key] = (0.0, 0.0, 1.0)
            else:
                self._frac[key] = split_ruler(1.0, self.nesting[q]["by_step"][str(step)])
        return self._frac[key]

    def point(self, q: str, step: int, y: np.ndarray) -> tuple[tuple[float, float, float], float]:
        if not self.has(q):
            return self.ZERO, 0.0
        model = self.cfg["per_quality"][q]["level_model"]
        s2r = float(ruler_mod.cell_sigma2(model, np.asarray(y, dtype=float).ravel())["total"])
        f = self.fractions(q, step)
        return (s2r * f[0], s2r * f[1], s2r * f[2]), s2r

    def batch(self, q: str, step: int, A: np.ndarray):
        """Per-replicate ruler variance for a stack of resampled cells ``[B, R, C, k]``."""
        B = A.shape[0]
        z = np.zeros(B)
        if not self.has(q):
            return (z, z, z), z
        model = self.cfg["per_quality"][q]["level_model"]
        c = np.asarray(model["centres"], dtype=float)
        if c.size == 0:
            return (z, z, z), z
        iv = np.asarray(model["interaction_var"], dtype=float)
        ro = np.asarray(model["recorded_offset"], dtype=float)
        X = np.asarray(A, dtype=float).reshape(B, -1)
        flat = X.ravel()
        inter = np.interp(flat, c, iv, left=iv[0], right=iv[-1]).reshape(X.shape).mean(axis=1)
        offs = np.interp(flat, c, ro, left=ro[0], right=ro[-1]).reshape(X.shape)
        off = np.var(offs, axis=1, ddof=1) if X.shape[1] > 1 else z
        s2r = inter + off
        f = self.fractions(q, step)
        return (s2r * f[0], s2r * f[1], s2r * f[2]), s2r


# --------------------------------------------------------------------------- #
# one scored set of states
# --------------------------------------------------------------------------- #


def score_subset(pred: np.ndarray, child_mean: np.ndarray, y: np.ndarray, root: np.ndarray,
                 child: np.ndarray, rows: np.ndarray, n_perm: int, seed: int,
                 ruler: Ruler, q: str, step: int) -> dict:
    """The source experiment's floor-corrected objective on ``rows``, on corrected components.

    Everything up to the objective is the source's: the same row filter, the same
    nested decomposition, the same parent-centred correlation and the same
    permutation floor at the same seed. Only the components ``J`` is formed from
    change, and the published-basis value is returned beside the corrected one so
    the per-subset effect of the subtraction is visible cell by cell.
    """
    rows = rows[np.isfinite(pred[rows])]
    if len(rows) < 8 or len(np.unique(root[rows])) < 2:
        return dict(NAN_CELL, n_states=int(len(rows)))
    try:
        dec = decompose(y[rows], root[rows], child[rows])
    except ValueError:
        return dict(NAN_CELL, n_states=int(len(rows)))

    p, ym, rt = pred[rows], child_mean[rows], root[rows]
    pc, yc = p.copy(), ym.copy()
    for r in np.unique(rt):
        m = rt == r
        pc[m] -= pc[m].mean()
        yc[m] -= yc[m].mean()
    r2 = 0.0 if (np.std(pc) == 0 or np.std(yc) == 0) else float(np.corrcoef(pc, yc)[0, 1] ** 2)
    floor = permutation_floor_vec(p, ym, rt, n_perm, seed)

    rs, s2r = ruler.point(q, step, y[rows])
    comp = (dec.sigma2_root, dec.sigma2_child, dec.sigma2_within)
    j_raw, cs, _ = j_corrected(r2, *comp, dec.k, rs)
    j, _, j_unc = j_corrected(max(r2 - floor, 0.0), *comp, dec.k, rs)
    j_pub, cs_pub, j_unc_pub = j_corrected(max(r2 - floor, 0.0), *comp, dec.k, Ruler.ZERO)
    return {
        "J": float(j), "J_raw": float(j_raw), "child_share": float(cs),
        "r2_sibling": float(max(r2 - floor, 0.0) / cs) if cs > 0 else 0.0,
        "r2_observed": r2, "floor": floor,
        "n_states": int(len(rows)), "n_roots": int(len(np.unique(rt))),
        "J_published_basis": float(j_pub), "child_share_published_basis": float(cs_pub),
        "sigma2_ruler": float(s2r),
        "sigma2_ruler_root": float(rs[0]), "sigma2_ruler_child": float(rs[1]),
        "sigma2_ruler_within": float(rs[2]),
        "capped": bool(float(j_unc) > float(cs)),
        "capped_published_basis": bool(float(j_unc_pub) > float(cs_pub)),
    }


def design_replicates(pred: np.ndarray, y: np.ndarray, root: np.ndarray, child: np.ndarray,
                      rows: np.ndarray, floor: float, u: np.ndarray,
                      ruler: Ruler, q: str, step: int) -> np.ndarray:
    """Replicate objectives under the source's two-stage resample of ``rows``.

    Roots then children are drawn with replacement with the source's shared draws,
    and every component of ``J`` is recomputed inside each replicate -- including
    the ruler's, which is evaluated on that replicate's own resampled score levels,
    so a replicate that happens to draw the cell's worst-read states is corrected by
    more than one that does not. The floor is held at its observed value, as in the
    source: it depends on the design's shape, which the resample preserves.
    """
    rows = rows[np.isfinite(pred[rows])]
    B = u.shape[0]
    if len(rows) < 8 or len(np.unique(root[rows])) < 2:
        return np.zeros(B)
    try:
        a = as_rck(y[rows], root[rows], child[rows])
        p = as_rck(pred[rows][:, None], root[rows], child[rows])[:, :, 0]
    except Exception:
        return np.zeros(B)
    R, C, k = a.shape
    uu = u[:, :R, :] if u.shape[1] >= R else np.tile(
        u, (1, int(np.ceil(R / u.shape[1])), 1))[:, :R, :]
    ri = np.minimum((uu[:, :, 0] * R).astype(np.int64), R - 1)
    ci = np.minimum((uu[:, :, 1:1 + C] * C).astype(np.int64), C - 1)
    A = a[ri[:, :, None], ci, :]
    P = p[ri[:, :, None], ci]
    s2a, s2b, s2e = _components(A)
    Y = A.mean(3)
    Yc = (Y - Y.mean(axis=2, keepdims=True)).reshape(B, -1)
    Pc = (P - P.mean(axis=2, keepdims=True)).reshape(B, -1)
    r2 = _rowwise_corr(Pc, Yc) ** 2
    rs, _ = ruler.batch(q, step, A)
    j, _, _ = j_corrected(np.clip(r2 - floor, 0, None), s2a, s2b, s2e, k, rs)
    return j


# --------------------------------------------------------------------------- #
# one target
# --------------------------------------------------------------------------- #


def rescore_target(target: str, labels_dir: str, pred_dir: str, out_dir: str, ruler: Ruler,
                   n_boot: int, n_perm: int, n_dboot: int) -> dict:
    z = np.load(os.path.join(pred_dir, f"{target}_pred.npz"))
    pred = {k: z[k] for k in z.files}
    for step in CAPTURE_STEPS:
        lab = np.load(os.path.join(labels_dir, f"{target}_step{step}.npz"))
        for q in QUALITIES:
            if f"{q}|{step}|oof_full" in pred:
                pred[f"{q}|{step}|y"] = lab[f"y_{q}"]

    u_full = draw_design_boot(n_boot, 16, seed=stable_seed(f"boot|{target}"))
    u_grp = draw_design_boot(n_dboot, 16, seed=stable_seed(f"dboot|{target}"))

    out: dict = {"target": target, "n_perm": n_perm, "n_design_boot": n_dboot, "steps": {}}
    jb_full: dict[str, np.ndarray] = {}
    jb_grp: dict[str, np.ndarray] = {}
    audit = {"n_cells": 0, "n_rotation_cells": 0, "n_factor_undefined": 0,
             "n_capped_full": 0, "n_capped_rotation": 0,
             "n_capped_rotation_published": 0,
             "factors_full": {q: [] for q in QUALITIES},
             "factors_rotation": {q: [] for q in QUALITIES},
             "factor_gap": {q: [] for q in QUALITIES}}
    t0 = time.time()

    for step in CAPTURE_STEPS:
        entry: dict = {}
        for q in QUALITIES:
            key = f"{q}|{step}"
            if f"{key}|oof_full" not in pred:
                continue
            y = pred[f"{key}|y"]
            child_mean = y.mean(axis=1)
            root, child = pred[f"{key}|root"], pred[f"{key}|child"]
            oof = pred[f"{key}|oof_full"]
            allrows = np.arange(len(root))
            seed = stable_seed(f"perm|{target}|{q}|{step}")

            full = score_subset(oof, child_mean, y, root, child, allrows, n_perm, seed,
                                ruler, q, step)
            dec = decompose(y, root, child)
            full.update({"commitment": dec.commitment, "root_share": dec.root_share,
                         "k": dec.k, "sigma2_child": dec.sigma2_child,
                         "sigma2_within": dec.sigma2_within})
            jb_full[key] = design_replicates(oof, y, root, child, allrows, full["floor"],
                                             u_full, ruler, q, step).astype(np.float32)
            audit["n_cells"] += 1
            audit["n_capped_full"] += int(bool(full.get("capped")))
            f_full = ratio(full)
            if f_full is None:
                audit["n_factor_undefined"] += 1
            else:
                audit["factors_full"][q].append(f_full)

            pg, gid = pred[f"{key}|pred_group"], pred[f"{key}|group_id"]
            reps = pg.shape[0]
            pair: dict[str, list] = {}
            rep_boot = np.zeros((reps, N_GROUPS, N_GROUPS, n_dboot), dtype=np.float32)
            for rep in range(reps):
                for gt in range(N_GROUPS):          # group the readout was fitted on
                    for gs in range(N_GROUPS):      # group the objective is scored on
                        if gt == gs:
                            continue
                        rows = np.where(gid[rep] == gs)[0]
                        sc = score_subset(pg[rep, gt], child_mean, y, root, child, rows,
                                          n_perm // 2, seed + 7 * rep + gs, ruler, q, step)
                        pair.setdefault(f"{gt}->{gs}", []).append(sc["J"])
                        rep_boot[rep, gt, gs] = design_replicates(
                            pg[rep, gt], y, root, child, rows, sc["floor"], u_grp,
                            ruler, q, step)
                        audit["n_rotation_cells"] += 1
                        audit["n_capped_rotation"] += int(bool(sc.get("capped")))
                        audit["n_capped_rotation_published"] += int(
                            bool(sc.get("capped_published_basis")))
                        f_rot = ratio(sc)
                        if f_rot is not None:
                            audit["factors_rotation"][q].append(f_rot)
                            if f_full is not None:
                                audit["factor_gap"][q].append(f_rot - f_full)
            entry[q] = {**full, "group_pairs": pair}
            jb_grp[key] = rep_boot
        out["steps"][str(step)] = entry

    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, f"{target}_curves.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed(os.path.join(out_dir, f"{target}_Jboot.npz"), **jb_full)
    np.savez_compressed(os.path.join(out_dir, f"{target}_Jgroup.npz"), **jb_grp)
    return audit


def ratio(cell: dict):
    """Corrected J over the same subset's published-basis J: the per-subset factor."""
    a, b = cell.get("J_published_basis"), cell.get("J")
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return float(b / a)


# --------------------------------------------------------------------------- #
# how far the re-derived curves sit from the source's stored ones
# --------------------------------------------------------------------------- #


def compare_curves(src_dir: str, new_dir: str, targets: list[str]) -> dict:
    worst = {"full_J": 0.0, "rotation_J": 0.0, "Jboot": 0.0, "Jgroup": 0.0,
             "r2_observed": 0.0, "floor": 0.0, "n_compared": 0}
    where: dict[str, str] = {}
    for t in targets:
        a = json.load(open(os.path.join(src_dir, f"{t}_curves.json")))
        b = json.load(open(os.path.join(new_dir, f"{t}_curves.json")))
        for s in CAPTURE_STEPS:
            for q, ca in a["steps"][str(s)].items():
                cb = b["steps"][str(s)].get(q)
                if cb is None:
                    continue
                for fld, key in (("full_J", "J"), ("r2_observed", "r2_observed"),
                                 ("floor", "floor")):
                    d = abs(float(ca[key]) - float(cb[key]))
                    if d > worst[fld]:
                        worst[fld], where[fld] = d, f"{t}|{q}|{s}"
                for pk, va in ca["group_pairs"].items():
                    vb = cb["group_pairs"][pk]
                    d = float(np.max(np.abs(np.asarray(va) - np.asarray(vb))))
                    worst["n_compared"] += len(va)
                    if d > worst["rotation_J"]:
                        worst["rotation_J"], where["rotation_J"] = d, f"{t}|{q}|{s}|{pk}"
        for fld, suffix in (("Jboot", "_Jboot.npz"), ("Jgroup", "_Jgroup.npz")):
            za, zb = (np.load(os.path.join(d, f"{t}{suffix}")) for d in (src_dir, new_dir))
            for k in za.files:
                if k not in zb.files:
                    continue
                d = float(np.max(np.abs(za[k].astype(float) - zb[k].astype(float))))
                if d > worst[fld]:
                    worst[fld], where[fld] = d, f"{t}|{k}"
    worst["worst_at"] = where
    return worst


# --------------------------------------------------------------------------- #


def summarise(v: list[float]) -> dict:
    a = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())}


def floor_stats(sc: dict) -> dict:
    """The source experiment's resolvable floor: what the interval could have excluded."""
    g = sc.get("gain_adaptive_vs_deployed", {})
    dep = sc.get("deployed", {}).get("mean")
    if not g or dep in (None, 0) or not np.isfinite(dep):
        return {}
    res_abs = float(max(abs(g["hi"]), abs(g["lo"])))
    return {"resolvable_abs": res_abs, "deployed_objective": float(dep),
            "resolvable_frac_of_deployed": float(res_abs / dep),
            "interval": [float(g["lo"]), float(g["hi"])]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir")
    ap.add_argument("pred_dir")
    ap.add_argument("src_curves_dir")
    ap.add_argument("out_curves_dir")
    ap.add_argument("out_json")
    ap.add_argument("--ruler", required=True)
    ap.add_argument("--nesting", required=True)
    ap.add_argument("--perm", type=int, default=400)
    ap.add_argument("--design-boot", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--pool-boot", type=int, default=10000)
    ap.add_argument("--targets", default="", help="comma-separated subset, for a smoke run")
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--zero-ruler", action="store_true",
                    help="set the subtraction to zero everywhere. The whole scoring stage "
                         "still runs, so its output must reproduce the source experiment's "
                         "stored curves and published pooled gain.")
    args = ap.parse_args()

    cfg = json.load(open(args.ruler))
    nesting = json.load(open(args.nesting))["per_quality"]
    ruler = Ruler(cfg, nesting, zero=args.zero_ruler)
    os.makedirs(args.out_curves_dir, exist_ok=True)

    targets = sorted(fn[: -len("_curves.json")] for fn in os.listdir(args.src_curves_dir)
                     if fn.endswith("_curves.json"))
    if args.targets:
        want = set(args.targets.split(","))
        targets = [t for t in targets if t in want]
    audit: dict = {"n_cells": 0, "n_rotation_cells": 0, "n_factor_undefined": 0,
                   "n_capped_full": 0, "n_capped_rotation": 0,
                   "n_capped_rotation_published": 0, "n_targets": len(targets),
                   "zero_ruler": bool(args.zero_ruler),
                   "n_perm": args.perm, "n_design_boot": args.design_boot,
                   "n_boot_full": args.boot}
    acc = {k: {q: [] for q in QUALITIES} for k in
           ("factors_full", "factors_rotation", "factor_gap")}
    for t in targets:
        a = rescore_target(t, args.labels_dir, args.pred_dir, args.out_curves_dir, ruler,
                           args.boot, args.perm, args.design_boot)
        for k in ("n_cells", "n_rotation_cells", "n_factor_undefined", "n_capped_full",
                  "n_capped_rotation", "n_capped_rotation_published"):
            audit[k] += a[k]
        for k in acc:
            for q in QUALITIES:
                acc[k][q].extend(a[k][q])
        print(f"  {t}: {a['n_cells']} cells, {a['n_rotation_cells']} rotation cells",
              flush=True)

    audit["factor_summary"] = {q: summarise(acc["factors_full"][q]) for q in QUALITIES}
    audit["factor_summary_rotation"] = {q: summarise(acc["factors_rotation"][q])
                                        for q in QUALITIES}
    audit["factor_rotation_minus_full"] = {q: summarise(acc["factor_gap"][q])
                                           for q in QUALITIES}
    audit["vs_source_curves"] = compare_curves(args.src_curves_dir, args.out_curves_dir,
                                               targets)
    print(f"vs source curves: {json.dumps(audit['vs_source_curves'], indent=1)}", flush=True)
    if args.no_pool:
        json.dump({"audit": audit}, open(args.out_json, "w"), indent=2)
        print(f"-> {args.out_json} (no pooling)")
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = VENDOR + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    pooled_p = args.out_json.replace(".json", "_pooled.json")
    r = subprocess.run(
        [sys.executable, os.path.join(VENDOR, "pool_gain.py"),
         args.out_curves_dir, pooled_p, "--boot", str(args.pool_boot)],
        env=env, capture_output=True, text=True,
    )
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"pool_gain failed rc={r.returncode}")

    pooled = json.load(open(pooled_p))
    published_p = os.path.join(HERE, "results", "sibling", "gain.json")
    published = json.load(open(published_p)) if os.path.exists(published_p) else None

    out = {"audit": audit, "corrected": {}, "published": {}}
    for q in QUALITIES:
        cq = pooled["by_quality"].get(q, {})
        out["corrected"][q] = {
            "scored_conservative": cq.get("scored_conservative", {}),
            "rotation": cq.get("rotation", {}),
            "regret": cq.get("regret", {}),
            "curves_full": cq.get("curves_full", {}),
        }
        if published:
            pq = published["by_quality"].get(q, {})
            out["published"][q] = {
                "scored_conservative": pq.get("scored_conservative", {}),
                "rotation": pq.get("rotation", {}),
                "regret": pq.get("regret", {}),
            }

    key = "gain_adaptive_vs_deployed"
    summary = {}
    for q in QUALITIES:
        c = out["corrected"][q]["scored_conservative"].get(key, {})
        p = out["published"].get(q, {}).get("scored_conservative", {}).get(key, {})
        summary[q] = {
            "published": {"mean": p.get("mean"), "ci": [p.get("lo"), p.get("hi")],
                          "p_gt_0": p.get("p_gt_0")},
            "corrected": {"mean": c.get("mean"), "ci": [c.get("lo"), c.get("hi")],
                          "p_gt_0": c.get("p_gt_0")},
            "recommendation_published": ("per-target timing" if (p.get("lo") or 0) > 0
                                         else "keep the fixed step"),
            "recommendation_corrected": ("per-target timing" if (c.get("lo") or 0) > 0
                                         else "keep the fixed step"),
        }
    out["headline_gain_adaptive_vs_deployed"] = summary
    out["recommendation_unchanged"] = all(
        v["recommendation_published"] == v["recommendation_corrected"] for v in summary.values())
    out["resolvable_floor"] = {
        q: {"published": floor_stats(out["published"].get(q, {}).get("scored_conservative", {})),
            "corrected": floor_stats(out["corrected"][q]["scored_conservative"])}
        for q in QUALITIES}

    # how far the re-derived pooled headline sits from the source's published one; in
    # ``--zero-ruler`` mode this must be zero, which is what makes the re-derivation
    # checkable rather than asserted
    worst = 0.0
    for q in QUALITIES:
        c = out["corrected"][q]["scored_conservative"]
        p = out["published"].get(q, {}).get("scored_conservative", {})
        for key in ("gain_adaptive_vs_deployed", "adaptive", "deployed", "global", "step100"):
            for f in ("mean", "lo", "hi", "p_gt_0"):
                a, b = p.get(key, {}).get(f), c.get(key, {}).get(f)
                if a is not None and b is not None:
                    worst = max(worst, abs(float(a) - float(b)))
    out["pooled_max_abs_diff_vs_published"] = float(worst)

    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n=== per-target action step: gain over the deployed policy ===")
    for q, v in summary.items():
        p, c = v["published"], v["corrected"]
        fl = out["resolvable_floor"][q]
        print(f"  {q:24s} published {p['mean']:+.6f} [{p['ci'][0]:+.5f}, {p['ci'][1]:+.5f}]"
              f"   corrected {c['mean']:+.6f} [{c['ci'][0]:+.5f}, {c['ci'][1]:+.5f}]"
              f"   -> {v['recommendation_corrected']}")
        if fl["published"] and fl["corrected"]:
            print(f"  {'':24s} resolvable floor {100*fl['published']['resolvable_frac_of_deployed']:.2f}%"
                  f" -> {100*fl['corrected']['resolvable_frac_of_deployed']:.2f}%"
                  f"   (deployed J {fl['published']['deployed_objective']:.4f}"
                  f" -> {fl['corrected']['deployed_objective']:.4f})")
    print(f"  recommendation unchanged: {out['recommendation_unchanged']}")
    print(f"  vs source curves: {json.dumps(audit['vs_source_curves'])[:400]}")
    print(f"-> {args.out_json}")


if __name__ == "__main__":
    main()
