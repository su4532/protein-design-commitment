#!/usr/bin/env python
"""Stage 2: designability by redesign-and-refold (the plan's binder scRMSD).

The capture already records ``af2folding_rmsd``, but that is Complexa's own
self-consistency number for the *generated* sequence under an initial-guess
AlphaFold protocol - a different quantity from the designability criterion the
benchmark is scored on. This stage computes the real one:

  1. take the generated binder backbone,
  2. redesign its sequence with ProteinMPNN (17 samples, temperature 0.1),
  3. refold each sequence on its own with AlphaFold (single chain, no template,
     no MSA, model_1_ptm, 3 recycles),
  4. scRMSD = min over sequences of the CA RMSD between the refolded prediction
     and the generated backbone, after optimal superposition.

Two design points from the accepted plan:

``quarter subsample``
    A full pass over all 1024 completions per cell would cost four times the
    whole harvest's fold budget. The plan therefore specified a random quarter.
    The quarter is drawn to preserve the three-level nesting the analysis needs:
    every root is kept, 4 of its 8 children, and 4 of each child's 8 completions
    (16 x 4 x 4 = 256). The draw is seeded from the target name only, so the
    *same* children and completions are measured at every capture step and the
    curve is paired across steps.

``repeat-measure control``
    ProteinMPNN sampling and the refold are themselves stochastic, so part of
    the spread between completions is measurement noise rather than the model
    being undecided. A subset of states is measured a second time with an
    independent redesign seed, which separates the two.

All 17 per-sequence RMSDs are stored, so a best-of-8 reading (the reduced
fidelity the plan fell back to) is recoverable from the same file and can be
calibrated against the best-of-17 paper criterion without any extra compute.

    python stage2_designability.py <harvest_root> <target> <out_dir> \
        [--steps 10,25] [--n-seq 17] [--n-repeat 32] [--limit-states N]
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np


from .commitment.metrics import extract_binder_pdbs  # noqa: E402

from .commitment.grid import steps as _grid_steps  # noqa: E402

#: capture grid for this run; ``CAPTURE_STEPS`` in the environment overrides it
STEPS = _grid_steps()
N_SEQ = 17          # the paper's best-of-17 sweep
MPNN_TEMP = 0.1
N_RECYCLES = 3


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def ca_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD between two equal-length CA traces after optimal superposition."""
    if a.shape != b.shape or a.shape[0] < 3:
        return float("nan")
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def read_ca(pdb_path: str) -> np.ndarray:
    ca = [
        [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
        for ln in open(pdb_path)
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA"
    ]
    return np.asarray(ca, dtype=np.float64)


def binder_chain(pdb_path: str) -> str:
    """Chain letter of the extracted binder-only PDB.

    ``extract_binder_pdbs`` keeps the last chain of the complex and preserves its
    original letter, so it is ``B`` for a single-chain target and ``C`` for a
    dimeric one (36_VEGFA, 37_IL17A). Reading it back beats assuming.
    """
    for ln in open(pdb_path):
        if ln.startswith("ATOM"):
            return ln[21]
    raise ValueError(f"no ATOM records in {pdb_path}")


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


def patch_mpnn_weights(path: str) -> None:
    """Point colabdesign's bundled-weights import at our downloaded checkpoint.

    ``mk_mpnn_model`` resolves the checkpoint as
    ``dirname(colabdesign.mpnn.weights.__file__)``. The installed package ships that
    package empty, so install a stand-in module beside the weights we fetched.
    """
    import colabdesign.mpnn  # noqa: F401

    m = types.ModuleType("colabdesign.mpnn.weights")
    m.__file__ = os.path.join(path, "__init__.py")
    sys.modules["colabdesign.mpnn.weights"] = m


# --------------------------------------------------------------------------- #
# subsample
# --------------------------------------------------------------------------- #


def draw_subsample(target: str, n_roots: int, n_children: int, k: int,
                   frac: float = 0.25) -> list[tuple[int, int, int]]:
    """The planned random quarter, seeded from the target name only.

    Keeping every root and thinning children and completions instead preserves
    all three levels of the variance decomposition. Seeding on the target rather
    than the (target, step) pair means the same completions are measured at
    every capture step, so the commitment curve is paired across steps exactly
    the way the other three qualities are.
    """
    seed = int(hashlib.sha1(target.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    n_c = max(1, int(round(n_children * frac**0.5)))
    n_k = max(2, int(round(k * frac**0.5)))
    sel = []
    for r in range(n_roots):
        for c in sorted(rng.choice(n_children, size=n_c, replace=False).tolist()):
            for j in sorted(rng.choice(k, size=n_k, replace=False).tolist()):
                sel.append((r, c, j))
    return sel


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def _arg(name: str, default: str | None = None) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def run_step(target: str, step: int, harvest_root: str, out_dir: str, mpnn, af_state: dict,
             n_seq: int, n_repeat: int, limit_states: int | None) -> dict:
    cell = os.path.join(harvest_root, target, f"step{step}")
    tar = os.path.join(cell, "completions.tar.gz")
    man = json.load(open(os.path.join(cell, "manifest.json")))
    spec = man["spec"]
    n_roots, n_children, k = spec["n_roots"], spec["n_children"], spec["k_completions"]

    sel = draw_subsample(target, n_roots, n_children, k)
    if limit_states:
        sel = sel[:limit_states]
    # states measured twice with an independent redesign seed
    rep_every = max(1, len(sel) // max(n_repeat, 1))
    rep_set = set(range(0, len(sel), rep_every)[:n_repeat])

    work = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp"))
    t0 = time.time()
    names = set(extract_binder_pdbs(tar, work))
    print(f"  [{target} step{step}] extracted {len(names)} structures in {time.time() - t0:.1f}s; "
          f"{len(sel)} selected, {len(rep_set)} repeat-measured", flush=True)

    from colabdesign import mk_af_model

    rmsd = np.full((len(sel), n_seq), np.nan, dtype=np.float32)
    rep_rmsd = np.full((len(rep_set), n_seq), np.nan, dtype=np.float32)
    rep_row = np.full(len(rep_set), -1, dtype=np.int64)
    lengths = np.zeros(len(sel), dtype=np.int64)
    missing = 0
    t_start = time.time()
    rep_i = 0

    for i, (r, c, j) in enumerate(sel):
        nm = f"root{r}_child{c}_comp{j}"
        if nm not in names:
            missing += 1
            continue
        pdb = os.path.join(work, nm + ".pdb")
        ref = read_ca(pdb)
        L = int(ref.shape[0])
        lengths[i] = L
        if L < 3:
            missing += 1
            continue

        if af_state.get("af") is None:
            af_state["af"] = mk_af_model(protocol="hallucination", use_templates=False,
                                         data_dir=os.environ["AF2_DIR"])
        af = af_state["af"]
        if af_state.get("L") != L:
            af.prep_inputs(length=L)
            af_state["L"] = L

        mpnn.prep_inputs(pdb_filename=pdb, chain=binder_chain(pdb))
        for pass_id in (0, 1) if i in rep_set else (0,):
            out = mpnn.sample(num=1, batch=n_seq, temperature=MPNN_TEMP)
            vals = []
            for s in out["seq"]:
                s = s.replace("/", "")[:L]
                if len(s) != L:
                    vals.append(np.nan)
                    continue
                af.predict(seq=s, num_models=1, models=[0], num_recycles=N_RECYCLES, verbose=False)
                xyz = np.asarray(af.aux["atom_positions"])[:, 1, :].astype(np.float64)
                vals.append(ca_rmsd(xyz, ref))
            if pass_id == 0:
                rmsd[i] = vals
            else:
                rep_rmsd[rep_i] = vals
                rep_row[rep_i] = i
                rep_i += 1

        if (i + 1) % 32 == 0:
            el = time.time() - t_start
            print(f"    {i + 1}/{len(sel)} states, {el / (i + 1):.2f}s/state, "
                  f"eta {(len(sel) - i - 1) * el / (i + 1) / 60:.1f} min", flush=True)

    shutil.rmtree(work, ignore_errors=True)
    sel_arr = np.asarray(sel, dtype=np.int64)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{target}_step{step}.npz")
    np.savez_compressed(
        path,
        root=sel_arr[:, 0], child=sel_arr[:, 1], comp=sel_arr[:, 2],
        rmsd_all=rmsd, rep_rmsd_all=rep_rmsd, rep_row=rep_row, binder_len=lengths,
        n_roots=n_roots, n_children=n_children, k_completions=k, n_seq=n_seq,
    )
    dt = time.time() - t_start
    meta = {"target": target, "step": step, "n_selected": len(sel), "n_missing": missing,
            "n_repeat": int(rep_i), "n_seq": n_seq, "seconds": dt,
            "s_per_state": dt / max(len(sel), 1),
            "scrmsd_median": float(np.nanmedian(np.nanmin(rmsd, axis=1))),
            "frac_below_2A": float(np.nanmean(np.nanmin(rmsd, axis=1) < 2.0))}
    print(f"  [{target} step{step}] done in {dt / 60:.1f} min -> {path}\n"
          f"    median scRMSD {meta['scrmsd_median']:.3f} A, "
          f"frac<2A {meta['frac_below_2A']:.3f}, missing {missing}", flush=True)
    return meta


def main() -> None:
    harvest_root, target, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    n_seq = int(_arg("--n-seq", str(N_SEQ)))
    n_repeat = int(_arg("--n-repeat", "32"))
    limit_states = int(_arg("--limit-states", "0")) or None
    only = {int(x) for x in _arg("--steps", "").split(",") if x.strip()}
    steps = [s for s in STEPS
             if (not only or s in only)
             and os.path.exists(os.path.join(harvest_root, target, f"step{s}", "completions.tar.gz"))]
    if not steps:
        raise SystemExit(f"no completion archives for {target} under {harvest_root}")

    patch_mpnn_weights(os.environ["MPNN_WEIGHTS"])
    from colabdesign.mpnn import mk_mpnn_model

    print(f"[{target}] stage 2 over {len(steps)} steps: {steps} "
          f"(n_seq={n_seq}, recycles={N_RECYCLES})", flush=True)
    mpnn = mk_mpnn_model()
    af_state: dict = {"af": None, "L": None}

    os.makedirs(out_dir, exist_ok=True)
    metas = []
    for s in steps:
        out = os.path.join(out_dir, f"{target}_step{s}.npz")
        if os.path.exists(out) and "--force" not in sys.argv:
            print(f"  [{target} step{s}] already done, skipping", flush=True)
            continue
        metas.append(run_step(target, s, harvest_root, out_dir, mpnn, af_state,
                              n_seq, n_repeat, limit_states))
    tag = "_".join(str(s) for s in sorted(only)) if only else "all"
    with open(os.path.join(out_dir, f"meta_{target}_step{tag}.json"), "w") as f:
        json.dump(metas, f, indent=2)
    print(f"[{target}] stage 2 complete", flush=True)


if __name__ == "__main__":
    main()
