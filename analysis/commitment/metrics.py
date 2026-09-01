"""Derive the structural metrics from the saved completion structures (CPU).

Two of the four scored qualities come straight out of the AlphaFold reward the
capture already records (interface PAE, complex pLDDT), and a third - the
self-consistency RMSD of the generated sequence against the generated backbone -
is also already in the reward dict. The fourth, **fold identity**, has to be
computed from the structures themselves: cluster every completed binder with
Foldseek (TM-align, threshold 0.5) and use the cluster id as a categorical
outcome.

Clustering is done **once per target over all seven capture steps together**, so
cluster ids are comparable across steps within a target. Doing it per step would
give each step its own labelling and make the fold-identity curve meaningless.

Binder-chain extraction is plain text filtering rather than a structure parser:
Complexa writes ``[target..., binder...]`` so the binder is the last chain in
file order, and at 136k structures the parser is the bottleneck.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any

import numpy as np

__all__ = [
    "extract_binder_pdbs",
    "foldseek_cluster_dir",
    "complex_descriptors",
    "descriptors_from_tar",
]

# atom37 names we care about for the interface descriptors
_DONOR_ACCEPTOR = {"N", "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "ND1", "ND2", "NE", "NE1", "NE2",
                   "NH1", "NH2", "NZ", "SD", "SG", "OXT"}


def _atom_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith("ATOM")]


def extract_binder_pdbs(tar_path: str, out_dir: str) -> list[str]:
    """Write binder-chain-only PDBs from a completions archive. Returns member names.

    The binder is the LAST chain in file order (Complexa concatenates target then
    binder). Taking the first chain would collapse the identical target into one
    degenerate cluster.
    """
    os.makedirs(out_dir, exist_ok=True)
    names = []
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if not m.name.endswith(".pdb"):
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            lines = _atom_lines(fh.read().decode())
            if not lines:
                continue
            chains = list(dict.fromkeys(ln[21] for ln in lines))
            bc = chains[-1]
            keep = [ln for ln in lines if ln[21] == bc]
            if len(keep) < 10:
                continue
            stem = os.path.splitext(os.path.basename(m.name))[0]
            with open(os.path.join(out_dir, stem + ".pdb"), "w") as out:
                out.write("\n".join(keep) + "\nEND\n")
            names.append(stem)
    return names


def foldseek_cluster_dir(in_dir: str, foldseek: str, tm: float = 0.5, tmp_root: str | None = None) -> dict[str, str]:
    """``foldseek easy-cluster`` over a directory of binder PDBs -> {stem: representative}."""
    work = tempfile.mkdtemp(dir=tmp_root or tempfile.gettempdir())
    res = os.path.join(work, "clu")
    subprocess.run(
        [foldseek, "easy-cluster", in_dir, res, os.path.join(work, "tmp"),
         "--alignment-type", "1", "--tmscore-threshold", str(tm), "-c", "0.0", "-v", "0"],
        check=True,
        capture_output=True,
    )
    assign: dict[str, str] = {}
    with open(res + "_cluster.tsv") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, mem = parts[0], parts[1]
                assign[mem.replace(".pdb", "")] = rep.replace(".pdb", "")
    shutil.rmtree(work, ignore_errors=True)
    return assign


def complex_descriptors(text: str, n_hotspot_res: np.ndarray | None = None) -> dict[str, float]:
    """Interface and pose descriptors for one completed complex.

    Everything here comes from the generated geometry, not from a predictor:
    how many heavy-atom contacts the binder makes with the target, how many of
    those are donor-acceptor pairs at hydrogen-bond distance, and where and how
    compact the binder sits.
    """
    lines = _atom_lines(text)
    if not lines:
        return {}
    chain = np.array([ln[21] for ln in lines])
    aname = np.array([ln[12:16].strip() for ln in lines])
    resid = np.array([int(ln[22:26]) for ln in lines])
    xyz = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])] for ln in lines])
    chains = list(dict.fromkeys(chain.tolist()))
    if len(chains) < 2:
        return {}
    bmask = chain == chains[-1]
    tmask = ~bmask
    b, t = xyz[bmask], xyz[tmask]
    if len(b) < 5 or len(t) < 5:
        return {}

    d = np.linalg.norm(b[:, None, :] - t[None, :, :], axis=2)
    contacts = int((d < 4.5).sum())
    da_b = np.isin(aname[bmask], list(_DONOR_ACCEPTOR))
    da_t = np.isin(aname[tmask], list(_DONOR_ACCEPTOR))
    hb = int(((d < 3.5) & da_b[:, None] & da_t[None, :]).sum())

    bca = xyz[bmask & (aname == "CA")]
    cen = bca.mean(0) if len(bca) else b.mean(0)
    rg = float(np.sqrt(((bca - cen) ** 2).sum(1).mean())) if len(bca) > 1 else float("nan")
    # per-binder-residue burial: how many target atoms sit within 8 A
    bres = resid[bmask]
    near = (d < 8.0).sum(1)
    buried = float(np.mean([near[bres == r].sum() > 0 for r in np.unique(bres)]))
    return {
        "n_contacts_4p5": float(contacts),
        "n_hbond_3p5": float(hb),
        "min_interface_dist": float(d.min()),
        "binder_rg": rg,
        "binder_n_res": float(len(np.unique(bres))),
        "frac_binder_res_at_interface": buried,
        "centroid_dist_to_target": float(np.linalg.norm(cen - t.mean(0))),
    }


def descriptors_from_tar(tar_path: str, limit: int | None = None) -> dict[str, dict[str, float]]:
    """Descriptors for every completion in an archive, keyed by member stem."""
    out: dict[str, dict[str, float]] = {}
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".pdb")]
        if limit:
            members = members[:limit]
        for m in members:
            fh = tf.extractfile(m)
            if fh is None:
                continue
            desc = complex_descriptors(fh.read().decode())
            if desc:
                out[os.path.splitext(os.path.basename(m.name))[0]] = desc
    return out
