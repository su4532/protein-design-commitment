#!/usr/bin/env python
"""Stage C1: turn stored predictions into floor-corrected objective curves.

Three things are computed per cell.

**The objective on the full 16 roots**, ``J = child_share * R2_sibling``, with
both factors taken from the same outcome variable.

**A noise floor, removed rather than merely reported.** A squared correlation is
non-negative, so it has a positive expectation even when the readout knows
nothing. That floor is about 0.01 with 128 states in 16 sibling groups and about
0.03 in a five-root scoring group, and because ``J`` multiplies it by a child
share that climbs from 0.02 to 0.76 across the capture steps, an uncorrected
floor tilts the whole curve towards later steps. Every scored subset therefore
gets its own within-root permutation null, computed with its own predictions
held fixed, and the floor is subtracted on the R² scale before ``J`` is formed.

**Group-pair curves for the leakage-free rotation.** For each root partition and
each ordered pair of distinct groups ``(trained_on, scored_on)``, the objective
on the scored group using predictions from the model fitted on the trained-on
group. A step chosen on one such curve can then be scored on a curve whose group
influenced neither the fit nor the choice. Design-bootstrap replicates of those
curves are stored so the panel interval can propagate each target's own
estimation error rather than treating it as measured exactly.

    python score_curves.py <target> <labels_dir> <pred_dir> <out_dir>
        [--boot 2000] [--perm 400] [--design-boot 200]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np


from .objective import CAPTURE_STEPS, QUALITIES, stable_seed  # noqa: E402
from ..commitment.boot import as_rck, draw_design_boot  # noqa: E402
from ..commitment.variance import decompose  # noqa: E402

N_GROUPS = 3


# --------------------------------------------------------------------------- #
# vectorised helpers
# --------------------------------------------------------------------------- #


def _components(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nested-ANOVA components for a stack ``[B, R, C, k]`` (clipped at zero)."""
    B, R, C, k = A.shape
    child_mean = A.mean(3)
    root_mean = child_mean.mean(2)
    grand = root_mean.mean(1)
    ms_within = ((A - child_mean[..., None]) ** 2).sum((1, 2, 3)) / (R * C * (k - 1))
    ms_child = k * ((child_mean - root_mean[..., None]) ** 2).sum((1, 2)) / (R * (C - 1))
    ms_root = C * k * ((root_mean - grand[:, None]) ** 2).sum(1) / (R - 1)
    s2_e = ms_within
    s2_b = (ms_child - ms_within) / k
    s2_a = (ms_root - ms_child) / (C * k)
    return np.clip(s2_a, 0, None), np.clip(s2_b, 0, None), np.clip(s2_e, 0, None)


def _rowwise_corr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    num = (Xc * Yc).sum(axis=1)
    den = np.sqrt((Xc**2).sum(axis=1) * (Yc**2).sum(axis=1))
    out = np.zeros(len(X))
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _J_from(r2: np.ndarray | float, s2a, s2b, s2e, k: int):
    """The stable form of J with its admissibility cap, vectorised.

    Components are clipped at zero first. The nested-ANOVA estimator can return a
    small negative sibling variance when the true component is near zero, and a
    negative variance share is not a quantity a selection can act on; without the
    clip it propagates into a negative objective.
    """
    s2a = np.clip(np.asarray(s2a, dtype=float), 0, None)
    s2b = np.clip(np.asarray(s2b, dtype=float), 0, None)
    s2e = np.clip(np.asarray(s2e, dtype=float), 0, None)
    tot = s2a + s2b + s2e
    good = tot > 0
    child_share = np.where(good, s2b / np.where(good, tot, 1.0), 0.0)
    j = np.clip(np.asarray(r2, dtype=float), 0, None) * (s2b + s2e / k) / np.where(good, tot, 1.0)
    return np.where(good, np.minimum(j, child_share), 0.0), child_share


def permutation_floor_vec(pred: np.ndarray, y: np.ndarray, root: np.ndarray,
                          n_perm: int, seed: int) -> float:
    """Mean squared correlation when outcomes are shuffled inside each sibling group.

    Predictions are held fixed, so this is the floor for *this* readout on *this*
    scoring set: everything about the design survives except the sibling-level
    signal the score is supposed to measure.
    """
    groups = [np.where(root == r)[0] for r in np.unique(root)]
    n = len(y)
    if n < 8 or len(groups) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    Y = np.repeat(y[None, :], n_perm, axis=0)
    for g in groups:
        order = np.argsort(rng.random((n_perm, len(g))), axis=1)
        Y[:, g] = y[g][order]
    # parent-centre both sides
    P = pred[None, :].copy()
    for g in groups:
        Y[:, g] -= Y[:, g].mean(axis=1, keepdims=True)
        P[:, g] -= P[:, g].mean(axis=1, keepdims=True)
    r = _rowwise_corr(np.repeat(P, n_perm, axis=0), Y)
    return float(np.mean(r**2))


def score_subset(pred: np.ndarray, child_mean: np.ndarray, y: np.ndarray,
                 root: np.ndarray, child: np.ndarray, rows: np.ndarray,
                 n_perm: int, seed: int) -> dict:
    """Floor-corrected objective on one set of states."""
    rows = rows[np.isfinite(pred[rows])]
    nan = {"J": float("nan"), "J_raw": float("nan"), "child_share": float("nan"),
           "r2_sibling": float("nan"), "r2_observed": float("nan"), "floor": float("nan"),
           "n_states": int(len(rows)), "n_roots": 0}
    if len(rows) < 8 or len(np.unique(root[rows])) < 2:
        return nan
    try:
        dec = decompose(y[rows], root[rows], child[rows])
    except ValueError:
        return nan

    p, ym, rt = pred[rows], child_mean[rows], root[rows]
    pc, yc = p.copy(), ym.copy()
    for r in np.unique(rt):
        m = rt == r
        pc[m] -= pc[m].mean()
        yc[m] -= yc[m].mean()
    if np.std(pc) == 0 or np.std(yc) == 0:
        r2 = 0.0
    else:
        r2 = float(np.corrcoef(pc, yc)[0, 1] ** 2)
    floor = permutation_floor_vec(p, ym, rt, n_perm, seed)
    j_raw, cs = _J_from(r2, dec.sigma2_root, dec.sigma2_child, dec.sigma2_within, dec.k)
    j, _ = _J_from(max(r2 - floor, 0.0), dec.sigma2_root, dec.sigma2_child,
                   dec.sigma2_within, dec.k)
    return {"J": float(j), "J_raw": float(j_raw), "child_share": float(cs),
            "r2_sibling": float(max(r2 - floor, 0.0) / cs) if cs > 0 else 0.0,
            "r2_observed": r2, "floor": floor,
            "n_states": int(len(rows)), "n_roots": int(len(np.unique(rt)))}


def design_replicates(pred: np.ndarray, y: np.ndarray, root: np.ndarray, child: np.ndarray,
                      rows: np.ndarray, floor: float, u: np.ndarray) -> np.ndarray:
    """Replicate objectives under a two-stage resample of the design on ``rows``.

    Roots then children are drawn with replacement, exactly as the parent
    experiment's shared draws do, and every component of ``J`` is recomputed
    inside each replicate. The floor is held at its observed value: it depends on
    the design's shape, which the resample preserves, not on the draw.
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
    j, _ = _J_from(np.clip(r2 - floor, 0, None), s2a, s2b, s2e, k)
    return j


# --------------------------------------------------------------------------- #


def main() -> None:
    target, labels_dir, pred_dir, out_dir = sys.argv[1:5]

    def arg(name, default):
        return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

    n_boot, n_perm, n_dboot = arg("--boot", 2000), arg("--perm", 400), arg("--design-boot", 200)
    os.makedirs(out_dir, exist_ok=True)

    z = np.load(f"{pred_dir}/{target}_pred.npz")
    pred = {k: z[k] for k in z.files}
    for step in CAPTURE_STEPS:
        lab = np.load(f"{labels_dir}/{target}_step{step}.npz")
        for q in QUALITIES:
            if f"{q}|{step}|oof_full" in pred:
                pred[f"{q}|{step}|y"] = lab[f"y_{q}"]

    u_full = draw_design_boot(n_boot, 16, seed=stable_seed(f"boot|{target}"))
    u_grp = draw_design_boot(n_dboot, 16, seed=stable_seed(f"dboot|{target}"))

    out: dict = {"target": target, "n_perm": n_perm, "n_design_boot": n_dboot, "steps": {}}
    jb_full: dict[str, np.ndarray] = {}
    jb_grp: dict[str, np.ndarray] = {}
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

            full = score_subset(oof, child_mean, y, root, child, allrows, n_perm, seed)
            dec = decompose(y, root, child)
            full.update({"commitment": dec.commitment, "root_share": dec.root_share,
                         "k": dec.k, "sigma2_child": dec.sigma2_child,
                         "sigma2_within": dec.sigma2_within})
            jb_full[key] = design_replicates(oof, y, root, child, allrows,
                                             full["floor"], u_full).astype(np.float32)

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
                                          n_perm // 2, seed + 7 * rep + gs)
                        pair.setdefault(f"{gt}->{gs}", []).append(sc["J"])
                        rep_boot[rep, gt, gs] = design_replicates(
                            pg[rep, gt], y, root, child, rows, sc["floor"], u_grp)
            entry[q] = {**full, "group_pairs": pair}
            jb_grp[key] = rep_boot
        out["steps"][str(step)] = entry
        js = {q: round(entry[q]["J"], 4) for q in entry}
        print(f"[score {target}] step {step:>3d}  J={js}", flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, f"{target}_curves.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, f"{target}_Jboot.npz"), **jb_full)
    np.savez_compressed(os.path.join(out_dir, f"{target}_Jgroup.npz"), **jb_grp)
    print(f"[score {target}] done in {out['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
