#!/usr/bin/env python
"""Corrected per-cell analysis: planned designability, nested readability, resampled curves.

Three things differ from the first pass.

**Designability is the planned quantity.** ``af2folding_rmsd`` - Complexa's own
self-consistency number for the generated sequence under an initial-guess
protocol - is replaced by the redesign-and-refold binder scRMSD produced by
``stage2_designability.py``: ProteinMPNN redesign, independent AlphaFold refold,
minimum CA RMSD to the generated backbone over 17 sequences. The best-of-8
reading is carried alongside as ``designability_bo8`` so the reduced-fidelity
variant can be calibrated against the paper's criterion. The old quantity is
kept as ``selfconsistency_rmsd`` because it is a perfectly good measurement of a
different thing, and the two disagreeing is itself worth seeing.

**Readability is selected nestedly and reported per family.** See
``commitment.nested``. The full readout x layer grid is still computed, because
the layer curve is a result in its own right, but it is a diagnostic rather than
the reported number.

**Commitment carries its resampling uncertainty forward.** Every cell stores
2000 replicate commitment values under a two-stage resample of roots and
children, using draws shared across all seven capture steps and all qualities,
so downstream code can interpolate a replicate settling step and propagate that
uncertainty into the across-target spread.

    python analyze2.py <harvest_root> <target> <out_dir> --step S \
        [--fold-cache PATH] [--stage2 DIR] [--cell-cache DIR] [--boot 2000]
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch


from .commitment.boot import as_rck, commitment_replicates, draw_design_boot  # noqa: E402
from .commitment.nested import candidate_families, family_readability  # noqa: E402
from .commitment.probes import make_feature_fn, readability, root_blocked_oof  # noqa: E402
from .commitment.variance import (  # noqa: E402
    bootstrap_over_roots,
    decompose,
    decompose_categorical,
    split_half_reliability,
)

from .commitment.grid import steps as _grid_steps  # noqa: E402

#: capture grid for this run; ``CAPTURE_STEPS`` in the environment overrides it
STEPS = _grid_steps()
N_BOOT_CURVE = 2000

#: name -> (source, lower_is_better, label)
CONTINUOUS = {
    "interface_quality": ("af2folding_i_pae", True, "interface PAE (normalised)"),
    "structural_confidence": ("af2folding_plddt_log", False, "complex pLDDT"),
    "selfconsistency_rmsd": ("af2folding_rmsd", True,
                             "AF2 self-consistency RMSD of the generated sequence (A)"),
}
#: qualities the report is about
PRIMARY = ("fold_identity", "designability", "structural_confidence", "interface_quality")


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def load_fold_cache(path: str, step: int) -> tuple[np.ndarray | None, int]:
    if not path or not os.path.exists(path):
        return None, 0
    z = np.load(path)
    key = f"s{step}"
    if key not in z.files:
        return None, 0
    return z[key], int(z["n_clusters_total"])


def load_stage2(stage2_dir: str, target: str, step: int) -> dict | None:
    """The redesign-and-refold labels for one cell, as a balanced sub-design.

    Returns per-state arrays over the measured ``(root, child)`` pairs: the
    best-of-17 scRMSD, the best-of-8 reading, the repeat-measure pass, and the
    lineage indices needed to line them up with the shard's rows.
    """
    p = os.path.join(stage2_dir, f"{target}_step{step}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    r, c, j = z["root"], z["child"], z["comp"]
    rmsd = z["rmsd_all"]                       # [M, n_seq]
    bo17 = np.nanmin(rmsd, axis=1)
    bo8 = np.nanmin(rmsd[:, :8], axis=1)

    pairs = sorted({(int(a), int(b)) for a, b in zip(r, c)})
    # renumber children within each root: which child index it is does not matter
    per_root: dict[int, list[int]] = {}
    for a, b in pairs:
        per_root.setdefault(a, []).append(b)
    k = min(len({int(x) for x in j[(r == a) & (c == b)]}) for a, b in pairs)
    n_c = min(len(v) for v in per_root.values())

    rows, roots, children, y17, y8 = [], [], [], [], []
    for a in sorted(per_root):
        for ci, b in enumerate(sorted(per_root[a])[:n_c]):
            sel = np.where((r == a) & (c == b))[0]
            sel = sel[np.argsort(j[sel])][:k]
            if len(sel) < k:
                continue
            rows.append((a, b))
            roots.append(a)
            children.append(ci)
            y17.append(bo17[sel])
            y8.append(bo8[sel])
    if not rows:
        return None

    rep = {}
    rr, rrow = z["rep_rmsd_all"], z["rep_row"]
    keep = rrow >= 0
    if keep.any():
        a1 = bo17[rrow[keep]]
        a2 = np.nanmin(rr[keep], axis=1)
        ok = np.isfinite(a1) & np.isfinite(a2)
        rep = {"n": int(ok.sum()),
               "pass1": a1[ok].tolist(), "pass2": a2[ok].tolist(),
               "r": float(np.corrcoef(a1[ok], a2[ok])[0, 1]) if ok.sum() > 2 else float("nan"),
               "mad": float(np.mean(np.abs(a1[ok] - a2[ok]))) if ok.any() else float("nan")}
    return {"y17": np.asarray(y17), "y8": np.asarray(y8),
            "root": np.asarray(roots), "child": np.asarray(children),
            "pairs": rows, "repeat": rep, "n_seq": int(z["n_seq"])}


def subset_shard(d: dict, rows: np.ndarray) -> dict:
    """A shard restricted to selected states, for probes on the designability sub-design."""
    per_state = ("partial_ca", "partial_valid", "binder_pool", "iface_pool",
                 "iface_res", "iface_valid", "iface_tok", "iface_kind", "root_idx",
                 "child_idx", "tags")
    out = dict(d)
    for k in per_state:
        if k in d:
            out[k] = np.asarray(d[k])[rows]
    return out


# --------------------------------------------------------------------------- #
# one cell
# --------------------------------------------------------------------------- #


def analyse_metric(name: str, y: np.ndarray, kind: str, lower: bool, shard: dict,
                   root: np.ndarray, child: np.ndarray, u: np.ndarray,
                   n_lab: int, with_probes: bool, n_layers: int) -> dict:
    try:
        dec = decompose_categorical(y, root, child) if kind == "categorical" else decompose(y, root, child)
    except ValueError as e:
        return {"error": str(e)}

    entry: dict = {
        "kind": kind,
        "lower_is_better": bool(lower),
        "decomposition": dec.as_dict(),
        "commitment_ci": bootstrap_over_roots(y, root, child, "commitment", kind=kind, n_boot=2000),
        "child_share_ci": bootstrap_over_roots(y, root, child, "child_share", kind=kind, n_boot=2000),
    }
    if kind == "continuous":
        entry["split_half"] = split_half_reliability(y)

    # replicate commitment under the shared two-stage resample
    a3 = as_rck(y, root, child)
    if kind == "categorical":
        a3 = np.rint(a3).astype(np.int64)
    entry["_boot"] = commitment_replicates(a3, u, kind=kind, n_lab=n_lab)

    # the probe target: a child's expected outcome (modal-cluster membership for fold identity)
    if kind == "categorical":
        vals, counts = np.unique(y[y >= 0], return_counts=True)
        modal = int(vals[np.argmax(counts)])
        y_probe = (y == modal).astype(float)
        probe_dec = decompose(y_probe, root, child)
        entry["probe_target"] = {"kind": "modal_fold_membership", "modal_cluster": modal,
                                 "modal_frac": float(np.mean(y == modal)),
                                 "ceiling": probe_dec.as_dict()}
    else:
        y_probe, probe_dec = y, dec

    if not with_probes:
        return entry

    child_mean = y_probe.mean(axis=1)
    fams = candidate_families(n_layers)

    # (a) reported number: configuration chosen on roots that never score it
    entry["nested"] = family_readability(shard, child_mean, root, probe_dec, fams, readability)

    # (b) the same protocol with the lineage block removed - this gap IS the leakage.
    #
    # The obvious way to write this, giving every state its own group id with
    # ``arange(N)``, is WRONG here and silently reproduces the root-blocked
    # split. GroupKFold assigns equal-sized groups to folds in label order, so
    # state ``i`` lands in fold ``i % n_folds``; the capture orders states
    # child-major with 16 roots, so ``i % 4`` equals ``root % 4`` and the
    # "random" split holds out exactly the same roots. A permutation of the
    # labels is what actually breaks the lineage block.
    fake = np.random.default_rng(20260825).permutation(len(child_mean))
    entry["nested_random_split"] = family_readability(
        shard, child_mean, fake, probe_dec, {"linear_activation": fams["linear_activation"]}, readability
    )

    # (c) the full grid, kept as the layer curve rather than as a maximum
    grid = {}
    for ro, layers in (("geom", (0,)), ("binder", tuple(range(n_layers))),
                       ("iface", tuple(range(n_layers))), ("iface_cov", tuple(range(n_layers))),
                       ("mlp", tuple(range(n_layers)))):
        for layer in layers:
            try:
                fn = make_feature_fn(shard, ro, layer)
                r = root_blocked_oof(fn, child_mean, root, readout=ro, n_splits=4)
                if r["oof"] is None:
                    continue
                grid[f"{ro}|{layer}"] = readability(r["oof"], child_mean, probe_dec)
            except Exception as e:
                grid[f"{ro}|{layer}"] = {"error": f"{type(e).__name__}: {e}"}
    entry["probe_grid"] = grid
    return entry


def _arg(name: str, default: str | None = None) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    harvest_root, target, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    step = int(_arg("--step"))
    fold_cache = _arg("--fold-cache", "")
    stage2_dir = _arg("--stage2", "")
    cell_cache = _arg("--cell-cache", "")
    n_boot = int(_arg("--boot", str(N_BOOT_CURVE)))

    shard_p = os.path.join(harvest_root, target, f"step{step}", "shard.pt")
    if not os.path.exists(shard_p):
        raise SystemExit(f"no shard at {shard_p}")
    print(f"[{target} step{step}] loading shard", flush=True)
    d = torch.load(shard_p, weights_only=False)
    root = np.asarray(d["root_idx"])
    child = np.asarray(d["child_idx"])
    n_layers = int(np.asarray(d["binder_pool"]).shape[1])
    R = len(np.unique(root))

    # One shared resampling draw per target, reused at every capture step so a
    # replicate is a coherent alternative version of the whole curve. The seed
    # must come from a stable hash: Python's built-in hash of a string is salted
    # per process, and these seven steps run as seven separate jobs.
    seed = int(hashlib.sha1(f"boot|{target}".encode()).hexdigest()[:8], 16)
    u = draw_design_boot(n_boot, R, seed=seed)

    fold_lab, n_lab = load_fold_cache(fold_cache, step)
    st2 = load_stage2(stage2_dir, target, step) if stage2_dir else None
    if st2 is None and stage2_dir:
        print(f"  WARNING: no stage-2 designability labels for {target} step{step}", flush=True)

    res: dict = {}
    m = d["metrics"]
    for name, (col, lower, _) in CONTINUOUS.items():
        if col in m:
            res[name] = analyse_metric(name, np.asarray(m[col], float), "continuous", lower,
                                       d, root, child, u, 0, True, n_layers)
    if fold_lab is not None and (fold_lab >= 0).all():
        res["fold_identity"] = analyse_metric("fold_identity", fold_lab, "categorical", False,
                                              d, root, child, u, n_lab, True, n_layers)

    # instrument control: binder length is fixed at initialisation
    ca = np.asarray(d["partial_ca"], dtype=np.float32)
    blen = (np.abs(ca).sum(axis=-1) > 1e-6).sum(axis=1).astype(float)
    kk = int(np.asarray(m["af2folding_i_pae"]).shape[1])
    res["control_binder_length"] = analyse_metric(
        "control_binder_length", np.repeat(blen[:, None], kk, axis=1), "continuous", False,
        d, root, child, u, 0, False, n_layers)

    # designability, the planned redesign-and-refold quantity
    if st2 is not None:
        lut = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(root, child))}
        rows = np.asarray([lut[p] for p in st2["pairs"] if p in lut])
        if len(rows) == len(st2["pairs"]):
            sub = subset_shard(d, rows)
            for nm, arr in (("designability", st2["y17"]), ("designability_bo8", st2["y8"])):
                res[nm] = analyse_metric(nm, arr, "continuous", True, sub,
                                         st2["root"], st2["child"], u, 0,
                                         nm == "designability", n_layers)
            res["designability"]["repeat_measure"] = st2["repeat"]
            res["designability"]["n_seq"] = st2["n_seq"]
            res["designability"]["subsample"] = {
                "n_states": int(len(rows)), "n_roots": int(len(np.unique(st2["root"]))),
                "n_children_per_root": int(len(np.unique(st2["child"]))),
                "k_completions": int(st2["y17"].shape[1]),
            }
        else:
            print("  WARNING: stage-2 lineage pairs do not all match the shard", flush=True)

    # ---- write --------------------------------------------------------------
    # One file per cell: the seven steps of a target run as seven separate jobs,
    # so a shared per-target file would be a write race. A cheap merge step
    # assembles them afterwards.
    os.makedirs(out_dir, exist_ok=True)
    boots = {k: v.pop("_boot") for k, v in res.items() if isinstance(v, dict) and "_boot" in v}
    np.savez_compressed(os.path.join(out_dir, f"{target}_step{step}_boot.npz"), **boots)
    p = os.path.join(out_dir, f"{target}_step{step}.json")
    with open(p, "w") as f:
        json.dump({"target": target, "step": step, "n_boot": n_boot, "cell": res}, f)

    if cell_cache:
        os.makedirs(cell_cache, exist_ok=True)
        cache = {
            "binder_pool": np.asarray(d["binder_pool"], dtype=np.float16),
            "iface_pool": np.asarray(d["iface_pool"], dtype=np.float16),
            "root_idx": root, "child_idx": child,
            "y_interface_quality": np.asarray(m["af2folding_i_pae"], float).mean(axis=1),
            "y_structural_confidence": np.asarray(m["af2folding_plddt_log"], float).mean(axis=1),
        }
        if fold_lab is not None:
            modal = int(np.bincount(fold_lab[fold_lab >= 0].ravel()).argmax())
            cache["y_fold_identity"] = (fold_lab == modal).astype(float).mean(axis=1)
        if st2 is not None and "designability" in res:
            cache["des_rows"] = rows
            cache["y_designability"] = st2["y17"].mean(axis=1)
        np.savez_compressed(os.path.join(cell_cache, f"{target}_step{step}.npz"), **cache)

    for nm in PRIMARY:
        e = res.get(nm)
        if e and "decomposition" in e:
            best = e.get("nested", {}).get("linear_activation", {}).get("readability", float("nan"))
            print(f"  {nm:24s} commitment {e['decomposition']['commitment']:.4f}  "
                  f"linear readability {best:.4f}", flush=True)
    print(f"[{target} step{step}] done -> {p}", flush=True)


if __name__ == "__main__":
    main()
