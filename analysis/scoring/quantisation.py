#!/usr/bin/env python
"""Does fold identity look tight because it is tight, or because its ruler is coarse?

Fold identity is the odd one out. It is a Foldseek cluster label over a label space
fixed once per target, so given a completed structure the assignment is deterministic
and its scoring reliability is one by construction. No variance subtraction applies
to it, and none is attempted.

The question that does apply is quantisation. A label rounds every structure inside
a cluster to the same value, so structural variation below the clustering threshold
never reaches the variance decomposition at all. If that discarded variation is
large, fold identity would look reproducible for a reason that has nothing to do
with the denoiser.

Two measurements, both on the structures the parent already produced.

``margin``
    For each completion, the TM-score to its own cluster's representative minus the
    best TM-score to any other cluster's representative. A completion sitting near a
    boundary would have been labelled differently by a slightly different clustering;
    a large margin means the label is not a close call.

``the same curve on a continuous ruler``
    The parent's fold-identity decomposition is a Gini-impurity one: impurity is the
    probability that two designs land in *different* clusters, so the quantity being
    averaged is a pairwise 0/1 distance, ``d(i,j) = 0`` when two designs share a label
    and ``1`` when they do not. Nothing in it refers to a modal cluster. Its continuous
    counterpart is therefore the same estimator with that 0/1 distance replaced by the
    structural distance the label is standing in for, ``d(i,j) = 1 - TM(i,j)``, over
    every pair of designs in the cell. This is label-free: no reference cluster has to
    be chosen, so nothing about the comparison can drift from one capture step to the
    next.

    An earlier version of this script used the TM-score to the modal cluster's
    representative instead. That is not usable as the primary ruler: the modal cluster
    is re-chosen in every capture-step cell, and for some targets it changes across
    steps, so the resulting curve compares different quantities over time. It is kept
    below only as a labelled sensitivity, with a flag recording whether the reference
    moved for that target.

``how much the label rounds away``
    Directly, and over all labels rather than one: the mean pairwise structural
    distance between designs that share a label, divided by the mean over all pairs.
    That is the share of structural variation the categorical ruler sets to zero.

The clustering is re-run with the parent's exact parameters (TM-align, threshold 0.5,
coverage 0) over all seven capture steps together, and the resulting partition is
checked against the cached labels before anything is computed from it.

    quantisation.py <target> <out_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

# Maintained estimators are imported from the package so every scoring path
# uses the same implementation.

from ..commitment.metrics import extract_binder_pdbs, foldseek_cluster_dir  # noqa: E402
from ..commitment.variance import decompose, decompose_categorical  # noqa: E402

STEPS = (10, 25, 50, 75, 100, 200, 300)


def build_pool(harvest_root: str, target: str, work: str) -> tuple[str, dict[int, list[str]]]:
    """Binder-only PDBs for every completion of every step, in one directory."""
    pdb_dir = os.path.join(work, "binders")
    os.makedirs(pdb_dir, exist_ok=True)
    present: dict[int, list[str]] = {}
    for s in STEPS:
        tar = os.path.join(harvest_root, target, f"step{s}", "completions.tar.gz")
        if not os.path.exists(tar):
            continue
        step_dir = os.path.join(work, f"s{s}")
        names = extract_binder_pdbs(tar, step_dir)
        for n in names:
            os.rename(os.path.join(step_dir, n + ".pdb"), os.path.join(pdb_dir, f"s{s}__{n}.pdb"))
        shutil.rmtree(step_dir, ignore_errors=True)
        present[s] = names
        print(f"  step{s}: {len(names)} binders", flush=True)
    return pdb_dir, present


def search_against_reps(pdb_dir: str, reps: list[str], foldseek: str, work: str) -> dict[str, dict[str, float]]:
    """TM-score of every structure in ``pdb_dir`` against every cluster representative."""
    rep_dir = os.path.join(work, "reps")
    os.makedirs(rep_dir, exist_ok=True)
    for r in reps:
        shutil.copy(os.path.join(pdb_dir, r + ".pdb"), os.path.join(rep_dir, r + ".pdb"))
    out_tsv = os.path.join(work, "search.tsv")
    subprocess.run(
        [foldseek, "easy-search", pdb_dir, rep_dir, out_tsv, os.path.join(work, "tmp_search"),
         "--alignment-type", "1", "-e", "inf", "--max-seqs", "100000",
         "--exhaustive-search", "1", "-c", "0.0",
         "--format-output", "query,target,alntmscore", "-v", "0"],
        check=True, capture_output=True,
    )
    tm: dict[str, dict[str, float]] = {}
    with open(out_tsv) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            q, t, v = parts[0].replace(".pdb", ""), parts[1].replace(".pdb", ""), float(parts[2])
            d = tm.setdefault(q, {})
            if v > d.get(t, -1.0):
                d[t] = v
    return tm


def search_all_pairs(pdb_dir: str, names: list[str], foldseek: str, work: str,
                     tag: str) -> np.ndarray:
    """Symmetric TM-score between every pair of the given structures.

    Foldseek's ``alntmscore`` is normalised by the query, so it is not symmetric; the
    two directions are combined with a max, matching the way clustering treats a hit
    found in either direction as a hit. A pair with no alignment at all is scored 0,
    the maximally-dissimilar end of the scale.
    """
    sub = os.path.join(work, f"pairs_{tag}")
    os.makedirs(sub, exist_ok=True)
    for n in names:
        shutil.copy(os.path.join(pdb_dir, n + ".pdb"), os.path.join(sub, n + ".pdb"))
    out_tsv = os.path.join(work, f"pairs_{tag}.tsv")
    subprocess.run(
        [foldseek, "easy-search", sub, sub, out_tsv, os.path.join(work, f"tmp_pairs_{tag}"),
         "--alignment-type", "1", "-e", "inf", "--max-seqs", "100000",
         "--exhaustive-search", "1", "-c", "0.0",
         "--format-output", "query,target,alntmscore", "-v", "0"],
        check=True, capture_output=True,
    )
    idx = {n: i for i, n in enumerate(names)}
    T = np.zeros((len(names), len(names)), float)
    with open(out_tsv) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            q = idx.get(parts[0].replace(".pdb", ""))
            t = idx.get(parts[1].replace(".pdb", ""))
            if q is None or t is None:
                continue
            v = float(parts[2])
            T[q, t] = max(T[q, t], v)
            T[t, q] = max(T[t, q], v)
    # foldseek's alntmscore can come back marginally above 1 for near-identical
    # structures; clip so the distance 1 - TM stays inside [0, 1]
    T = np.clip(T, 0.0, 1.0)
    np.fill_diagonal(T, 1.0)
    shutil.rmtree(sub, ignore_errors=True)
    return T


def nested_subsample(root: np.ndarray, n_roots: int, n_children: int,
                     seed: int) -> np.ndarray:
    """Row indices for a balanced nested subsample: some roots, some children of each.

    A cell holds 16 roots x 8 children x 8 completions = 1024 structures, and an
    all-against-all structural alignment over that is 1.05M alignments per cell. The
    pairwise means the decomposition needs are ordinary averages over pairs, so they can
    be estimated from a subsample; what matters is that the subsample stays *balanced*
    and keeps every completion of the children it keeps, because the within-child and
    within-root pair sets are exactly what the decomposition is made of. Dropping
    structures at random would thin those sets to nothing.
    """
    rng = np.random.default_rng(seed)
    roots = np.unique(root)
    keep_roots = rng.choice(roots, size=min(n_roots, roots.size), replace=False)
    rows = []
    for r in sorted(keep_roots):
        idx = np.where(root == r)[0]
        take = rng.choice(idx, size=min(n_children, idx.size), replace=False)
        rows.extend(sorted(take.tolist()))
    return np.asarray(sorted(rows), dtype=int)


def decompose_pairwise(D: np.ndarray, root: np.ndarray, child: np.ndarray,
                       k: int) -> dict:
    """The parent's Gini decomposition with a continuous pairwise distance.

    ``decompose_categorical`` averages the 0/1 distance ``labels[i] != labels[j]`` over
    pairs at each level of the hierarchy. This is the identical estimator over the same
    pairs with a continuous ``D[i, j]`` in place of that indicator, so the resulting
    commitment, root share and child share are on the same scale as the categorical
    ones and are directly comparable to them.

    ``D`` is ``[N*k, N*k]`` over the flattened states; ``root`` and ``child`` are the
    per-row labels of the ``[N, k]`` layout.
    """
    n = D.shape[0]
    N = n // k
    iu = np.triu_indices(n, k=1)
    state = np.repeat(np.arange(N), k)          # which row each flattened item is in
    root_of = np.repeat(np.asarray(root), k)

    g_total = float(np.mean(D[iu]))

    per_child = []
    for i in range(N):
        sel = np.where(state == i)[0]
        if sel.size > 1:
            sub = D[np.ix_(sel, sel)]
            per_child.append(float(np.mean(sub[np.triu_indices(sel.size, k=1)])))
    g_child = float(np.mean(per_child)) if per_child else float("nan")

    per_root = []
    for r in np.unique(root_of):
        sel = np.where(root_of == r)[0]
        if sel.size > 1:
            sub = D[np.ix_(sel, sel)]
            per_root.append(float(np.mean(sub[np.triu_indices(sel.size, k=1)])))
    g_root = float(np.mean(per_root)) if per_root else float("nan")

    if not np.isfinite(g_total) or g_total <= 0:
        commit = root_sh = child_sh = float("nan")
    else:
        commit = 1.0 - g_child / g_total
        root_sh = 1.0 - g_root / g_total
        child_sh = commit - root_sh
    return {
        "sigma2_root": float(g_total - g_root),
        "sigma2_child": float(g_root - g_child),
        "sigma2_within": float(g_child),
        "total": float(g_total),
        "commitment": float(commit),
        "root_share": float(root_sh),
        "child_share": float(child_sh),
        "n_roots": int(len(np.unique(root))),
        "n_children": int(N),
        "k": int(k),
        "n_obs": int(n),
        "kind": "pairwise_structural_distance",
    }


def rounded_away_share(D: np.ndarray, lab_flat: np.ndarray) -> dict:
    """How much pairwise structural distance the categorical label sets to zero.

    Over every pair of labelled designs in the cell, the mean structural distance
    between pairs that share a label, against the mean over all pairs. Unlike a
    within-modal-cluster figure this uses every label, so it is the share of structural
    variation the categorical ruler discards, not the share discarded inside one
    arbitrary cluster.
    """
    ok = np.where(lab_flat >= 0)[0]
    if ok.size < 3:
        return {}
    sub = D[np.ix_(ok, ok)]
    lv = lab_flat[ok]
    iu = np.triu_indices(ok.size, k=1)
    d = sub[iu]
    same = (lv[:, None] == lv[None, :])[iu]
    if same.sum() < 1 or d.size < 1:
        return {}
    mean_all = float(np.mean(d))
    mean_same = float(np.mean(d[same]))
    mean_diff = float(np.mean(d[~same])) if (~same).sum() else float("nan")
    return {
        "n_pairs": int(d.size),
        "n_pairs_same_label": int(same.sum()),
        "frac_pairs_same_label": float(np.mean(same)),
        "mean_distance_all_pairs": mean_all,
        "mean_distance_same_label": mean_same,
        "mean_distance_different_label": mean_diff,
        # the share of total pairwise structural distance that survives inside a label,
        # i.e. that the 0/1 ruler rounds to zero
        "share_rounded_away": (mean_same * float(same.sum()) / (mean_all * d.size))
        if mean_all > 0 else float("nan"),
        "mean_distance_ratio_same_over_all": (mean_same / mean_all) if mean_all > 0
        else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("out_dir")
    ap.add_argument("--harvest-root", default=os.environ.get("HARVEST_ROOT"))
    ap.add_argument("--fold-cache", default=os.environ.get("FOLD_CACHE_ROOT"))
    ap.add_argument("--foldseek", default=os.environ.get("FOLDSEEK_BIN", "foldseek"))
    ap.add_argument("--sub-roots", type=int, default=8,
                    help="roots kept per cell for the pairwise ruler (of 16)")
    ap.add_argument("--sub-children", type=int, default=4,
                    help="children kept per root for the pairwise ruler (of 8)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_root = os.environ.get("TMPDIR", "/tmp")
    work = tempfile.mkdtemp(prefix=f"quant_{args.target}_", dir=tmp_root)
    try:
        pdb_dir, present = build_pool(args.harvest_root, args.target, work)
        assign = foldseek_cluster_dir(pdb_dir, args.foldseek, tmp_root=tmp_root)
        reps = sorted(set(assign.values()))
        rep_id = {r: i for i, r in enumerate(reps)}
        print(f"[{args.target}] {len(reps)} clusters over {len(assign)} structures", flush=True)

        tm = search_against_reps(pdb_dir, reps, args.foldseek, work)

        cached = np.load(os.path.join(args.fold_cache, f"{args.target}.npz"))
        out: dict = {"target": args.target, "n_clusters": len(reps), "steps": {}}
        agree_num = agree_den = 0
        margins_all: list[float] = []

        for step in STEPS:
            shard_p = os.path.join(args.harvest_root, args.target, f"step{step}", "shard.pt")
            if not os.path.exists(shard_p):
                continue
            d = torch.load(shard_p, weights_only=False, map_location="cpu")
            tags = list(d["tags"])
            root = np.asarray(d["root_idx"])
            child = np.asarray(d["child_idx"])
            k = int(d["spec"]["k_completions"])
            N = len(tags)

            lab = np.full((N, k), -1, dtype=np.int64)
            margin = np.full((N, k), np.nan)
            tm_own = np.full((N, k), np.nan)
            for i, tag in enumerate(tags):
                for j in range(k):
                    key = f"s{step}__{tag}_comp{j}"
                    if key not in assign:
                        continue
                    own = assign[key]
                    lab[i, j] = rep_id[own]
                    row = tm.get(key, {})
                    if own in row:
                        tm_own[i, j] = row[own]
                        others = [v for r, v in row.items() if r != own]
                        margin[i, j] = row[own] - (max(others) if others else 0.0)

            cached_lab = cached[f"s{step}"] if f"s{step}" in cached.files else None
            if cached_lab is not None:
                # partitions are compared, not label ids: a re-run can permute ids
                a = lab.ravel()
                b = np.asarray(cached_lab).ravel()
                ok = (a >= 0) & (b >= 0)
                same_a = a[ok][:, None] == a[ok][None, :]
                same_b = b[ok][:, None] == b[ok][None, :]
                sub = np.random.default_rng(42).choice(int(ok.sum()), size=min(1500, int(ok.sum())),
                                                       replace=False)
                agree_num += int((same_a[np.ix_(sub, sub)] == same_b[np.ix_(sub, sub)]).sum())
                agree_den += sub.size ** 2

            # continuous counterpart of the parent's readout target
            vals, counts = np.unique(lab[lab >= 0], return_counts=True)
            modal = int(vals[np.argmax(counts)])
            modal_rep = reps[modal]
            y_tm = np.full((N, k), np.nan)
            for i, tag in enumerate(tags):
                for j in range(k):
                    key = f"s{step}__{tag}_comp{j}"
                    v = tm.get(key, {}).get(modal_rep)
                    if v is not None:
                        y_tm[i, j] = v
            # --- the primary continuous ruler: pairwise structural distance ---
            # Estimated on a balanced nested subsample of the cell; see nested_subsample.
            # The seed is fixed per target and step so the draw is reproducible.
            # a stable seed: Python's str hash is salted per process, so hash() here
            # would silently change the subsample from run to run
            seed = int(hashlib.sha1(f"quant|{args.target}|{step}".encode()).hexdigest()[:8], 16)
            sub_rows = nested_subsample(root, args.sub_roots, args.sub_children, seed=seed)
            flat_names = [f"s{step}__{tags[i]}_comp{j}" for i in sub_rows for j in range(k)]
            have = [n for n in flat_names if os.path.exists(os.path.join(pdb_dir, n + ".pdb"))]
            pair_entry: dict = {}
            rounded: dict = {}
            sub_cat: dict = {}
            if len(have) == len(flat_names) and sub_rows.size >= 4:
                Tm = search_all_pairs(pdb_dir, flat_names, args.foldseek, work, f"s{step}")
                D = 1.0 - Tm
                np.fill_diagonal(D, 0.0)
                sub_root, sub_child = root[sub_rows], child[sub_rows]
                pair_entry = decompose_pairwise(D, sub_root, sub_child, k)
                pair_entry["n_rows_subsampled"] = int(sub_rows.size)
                pair_entry["n_rows_total"] = int(N)
                rounded = rounded_away_share(D, lab[sub_rows].reshape(-1))
                # the categorical ruler on the *same* states, so the contrast between
                # the two rulers is matched and not partly a subsampling difference
                try:
                    sub_cat = decompose_categorical(lab[sub_rows], sub_root,
                                                    sub_child).as_dict()
                except Exception as e:  # noqa: BLE001
                    sub_cat = {"error": str(e)}
            else:
                print(f"  step{step}: {len(have)}/{len(flat_names)} binder PDBs present, "
                      f"skipping the pairwise ruler for this cell", flush=True)

            entry: dict = {
                "modal_cluster": modal,
                "modal_frac": float(np.mean(lab == modal)),
                "margin_mean": float(np.nanmean(margin)),
                "margin_median": float(np.nanmedian(margin)),
                "margin_frac_below_0p05": float(np.nanmean(margin < 0.05)),
                "margin_frac_below_0p10": float(np.nanmean(margin < 0.10)),
                "tm_to_own_rep_mean": float(np.nanmean(tm_own)),
                "tm_to_own_rep_sd": float(np.nanstd(tm_own)),
                "n_labelled": int((lab >= 0).sum()),
            }
            margins_all.extend(margin[np.isfinite(margin)].tolist())

            try:
                dec_cat = decompose_categorical(lab, root, child)
                entry["categorical"] = dec_cat.as_dict()
            except Exception as e:  # noqa: BLE001
                entry["categorical"] = {"error": str(e)}
            if pair_entry:
                entry["continuous_pairwise"] = pair_entry
            if sub_cat:
                entry["categorical_on_subsample"] = sub_cat
            if rounded:
                entry["rounded_away"] = rounded
            if np.isfinite(y_tm).all():
                # Sensitivity only. The reference is re-chosen per cell, so this curve
                # can compare different quantities across steps; `modal_cluster` above
                # records which one was used and pool_quant flags targets where it moved.
                dec_tm = decompose(y_tm, root, child)
                entry["continuous_tm_modal_rep"] = dec_tm.as_dict()
                same = lab == modal
                if same.sum() > 2:
                    entry["within_modal_tm_sd"] = float(np.nanstd(y_tm[same]))
                    entry["overall_tm_sd"] = float(np.nanstd(y_tm))
                    entry["frac_tm_variance_within_label"] = float(
                        np.nanvar(y_tm[same]) * same.sum() / (np.nanvar(y_tm) * y_tm.size))
            out["steps"][str(step)] = entry
            print(f"[{args.target} step{step}] margin mean {entry['margin_mean']:.4f} "
                  f"frac<0.05 {entry['margin_frac_below_0p05']:.4f} "
                  f"cat commitment {entry['categorical'].get('commitment', float('nan')):.4f} "
                  f"pairwise commitment {entry.get('continuous_pairwise', {}).get('commitment', float('nan')):.4f} "
                  f"rounded-away {entry.get('rounded_away', {}).get('share_rounded_away', float('nan')):.4f}",
                  flush=True)

        m = np.asarray(margins_all, float)
        out["margin_overall"] = {
            "n": int(m.size),
            "mean": float(m.mean()) if m.size else float("nan"),
            "median": float(np.median(m)) if m.size else float("nan"),
            "p05": float(np.percentile(m, 5)) if m.size else float("nan"),
            "frac_below_0p05": float(np.mean(m < 0.05)) if m.size else float("nan"),
            "frac_below_0p10": float(np.mean(m < 0.10)) if m.size else float("nan"),
        }
        out["clustering_reproduction"] = {
            "pair_agreement_with_cached_labels": float(agree_num / agree_den) if agree_den else float("nan"),
            "n_pairs_compared": int(agree_den),
        }
        p = os.path.join(args.out_dir, f"{args.target}.json")
        with open(p, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"-> {p}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
