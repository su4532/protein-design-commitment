#!/usr/bin/env python
"""Give the capacity control the same regularisation search the linear readout gets.

The readability comparison in the manuscript reports three readout families per
cell, each selected by the same nested protocol: candidates are ranked on a
root-blocked split of the training roots, and the winner is refit on all the
training roots and scored on held-out roots. For the linear family the winner's
ridge penalty is then chosen by a further search inside those same training
roots. For the MLP family it was not: the code selected the layer at a fixed
penalty and kept the initialiser's midpoint value (100.0) for the final fit. So
the comparator was tuned and the capacity control was not, in a regime with
roughly 112 training rows against 768 features.

This script removes that asymmetry at the layer the comparison is reported on.
For every cell (19 targets x 10 capture steps) and every reported quality it

  * reads the layer the published run's MLP family chose in each outer fold,
    leaving the layer sweep exactly as it was, then
  * reproduces that fold's published fit at the fixed penalty, as a check that
    this harness is the published one, and
  * refits it with the penalty chosen by the same 3-fold root-blocked inner
    search over the same five values the ridge searches.

Both readouts are put on the readability scale by the published function and
pooled over Foldseek clusters by the published block bootstrap, so the tuned
number is comparable to the numbers already in the paper.

Nothing else changes: same shards, same scored outcomes, same variance
components, same outer folds, same held-out roots.

    python retune_mlp.py <out_dir> [--workers N] [--limit N]

Writes <out_dir>/mlp_retune.json (pooled) and <out_dir>/cells/*.json (per cell).
CPU-only.
"""

from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from .commitment.probes import ALPHAS, _estimator, make_feature_fn, readability  # noqa: E402
from .commitment.variance import decompose  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from . import cell_analysis  # noqa: E402  (shared stage and cache loaders)

DATA_ROOT = os.environ.get("DENOISING_DATA_ROOT", "data")
HARVEST = f"{DATA_ROOT}/harvest"
STAGE2 = f"{DATA_ROOT}/stage2"
FOLD_CACHE = f"{DATA_ROOT}/fold_cache"
PER_TARGET = f"{DATA_ROOT}/per_target_grid"
#: the ten-step pooled derivation, as written by the source run on the cluster.
#: The manuscript's worktree cites the identical worktree copy, which is not
#: mounted inside a job, so the job reads the artifact tree instead.
POOLED = f"{DATA_ROOT}/analysis_grid10/pooled.json"

PRIMARY = ("fold_identity", "designability", "structural_confidence", "interface_quality")
N_OUTER, N_INNER = 4, 3
RANK_ALPHA = 100.0          # the fixed penalty the published MLP fits used
FAMILY = "mlp"


# --------------------------------------------------------------------------- #
# one cell
# --------------------------------------------------------------------------- #


def units_for_cell(target: str, step: int) -> dict:
    """Rebuild every quality's probe inputs for one cell, as ``analyze2`` does."""
    import torch

    d = torch.load(f"{HARVEST}/{target}/step{step}/shard.pt", weights_only=False)
    root = np.asarray(d["root_idx"])
    child = np.asarray(d["child_idx"])
    m = d["metrics"]
    out: dict[str, dict] = {}

    for name, (col, _lower, _lab) in cell_analysis.CONTINUOUS.items():
        if name not in PRIMARY or col not in m:
            continue
        y = np.asarray(m[col], float)
        dec = decompose(y, root, child)
        out[name] = {"shard": d, "root": root, "child_mean": y.mean(axis=1), "dec": dec}

    fold_lab, _n_lab = cell_analysis.load_fold_cache(f"{FOLD_CACHE}/{target}.npz", step)
    if fold_lab is not None and (fold_lab >= 0).all():
        vals, counts = np.unique(fold_lab[fold_lab >= 0], return_counts=True)
        modal = int(vals[np.argmax(counts)])
        # the probe target for fold identity is modal-cluster membership, and the
        # ceiling it is scored against is that indicator's own decomposition
        y_probe = (fold_lab == modal).astype(float)
        probe_dec = decompose(y_probe, root, child)
        out["fold_identity"] = {"shard": d, "root": root,
                                "child_mean": y_probe.mean(axis=1), "dec": probe_dec}

    st2 = cell_analysis.load_stage2(STAGE2, target, step)
    if st2 is not None:
        lut = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(root, child))}
        rows = np.asarray([lut[p] for p in st2["pairs"] if p in lut])
        if len(rows) == len(st2["pairs"]):
            sub = cell_analysis.subset_shard(d, rows)
            y = st2["y17"]
            dec = decompose(y, st2["root"], st2["child"])
            out["designability"] = {"shard": sub, "root": st2["root"],
                                    "child_mean": y.mean(axis=1), "dec": dec}
    return out


def _fit_predict(fn, alpha: float, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    Xa, Xb = fn(tr, te)
    return _estimator("mlp", alpha).fit(Xa, y[tr]).predict(Xb)


def retune_one(unit: dict, chosen_layers: list[int]) -> dict:
    """Published-penalty and inner-searched-penalty MLP fits at the published layers.

    ``chosen_layers[i]`` is the layer the published run's MLP family selected in
    outer fold ``i``; the layer sweep is therefore untouched and only the
    penalty of the final fit changes.
    """
    shard, root, y = unit["shard"], unit["root"], unit["child_mean"]
    n_outer = int(min(N_OUTER, len(np.unique(root))))
    oof_fixed = np.full(len(y), np.nan)
    oof_tuned = np.full(len(y), np.nan)
    picked: list[float] = []

    for i, (tr, te) in enumerate(GroupKFold(n_splits=n_outer).split(np.zeros(len(y)), y, root)):
        if i >= len(chosen_layers):
            break
        fn = make_feature_fn(shard, "mlp", int(chosen_layers[i]))
        inner_roots = np.unique(root[tr])
        n_in = int(min(N_INNER, len(inner_roots)))
        if n_in < 2:
            continue
        inner = list(GroupKFold(n_splits=n_in).split(np.zeros(len(tr)), y[tr], root[tr]))

        oof_fixed[te] = _fit_predict(fn, RANK_ALPHA, y, tr, te)

        alpha, a_score = ALPHAS[len(ALPHAS) // 2], -np.inf
        for a in ALPHAS:
            preds = np.full(len(tr), np.nan)
            for itr, ite in inner:
                preds[ite] = _fit_predict(fn, a, y, tr[itr], tr[ite])
            ok = np.isfinite(preds)
            if ok.sum() < 5:
                continue
            s = -float(np.mean((preds[ok] - y[tr][ok]) ** 2))
            if s > a_score:
                a_score, alpha = s, a
        oof_tuned[te] = _fit_predict(fn, alpha, y, tr, te)
        picked.append(float(alpha))

    return {
        "fixed": readability(oof_fixed, y, unit["dec"]),
        "tuned": readability(oof_tuned, y, unit["dec"]),
        "alphas_selected": picked,
        "layers": [int(x) for x in chosen_layers],
        "n_states": int(len(y)),
        "n_roots": int(len(np.unique(root))),
    }


def run_cell(args: tuple[str, int]) -> dict:
    target, step = args
    t0 = time.time()
    published = json.load(open(f"{PER_TARGET}/{target}_step{step}.json"))["cell"]
    units = units_for_cell(target, step)
    out = {"target": target, "step": step, "metrics": {}}
    for m in PRIMARY:
        pub = (published.get(m) or {}).get("nested", {}).get(FAMILY)
        if pub is None or m not in units:
            out["metrics"][m] = {"error": "missing published mlp block or unit"}
            continue
        layers = [int(c.split("|")[1]) for c in pub.get("chosen", [])]
        pub_alphas = sorted({c.split("alpha=")[1] for c in pub.get("chosen", []) if "alpha=" in c})
        if not layers:
            out["metrics"][m] = {"error": "no published chosen configuration"}
            continue
        r = retune_one(units[m], layers)
        r["published_readability"] = pub.get("readability")
        r["published_r2_corrected"] = pub.get("r2_corrected")
        r["published_alphas"] = pub_alphas
        r["published_chosen"] = pub.get("chosen", [])
        rep = r["fixed"].get("readability")
        r["reproduction_abs_diff"] = (
            None if (rep is None or pub.get("readability") is None
                     or not np.isfinite(rep) or not np.isfinite(pub["readability"]))
            else abs(float(rep) - float(pub["readability"]))
        )
        out["metrics"][m] = r
    out["seconds"] = time.time() - t0
    return out


# --------------------------------------------------------------------------- #
# pooling (the published block bootstrap over Foldseek clusters)
# --------------------------------------------------------------------------- #


def block_bootstrap(values_by_target: dict[str, float], clusters: dict[str, str],
                    n_boot: int = 5000, seed: int = 42) -> dict:
    """Verbatim from the source run's pool_targets.block_bootstrap."""
    by_cluster: dict[str, list[float]] = {}
    for t, v in values_by_target.items():
        if np.isfinite(v):
            by_cluster.setdefault(clusters.get(t, t), []).append(v)
    keys = sorted(by_cluster)
    if not keys:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_clusters": 0,
                "n_targets": 0, "n_targets_offered": len(values_by_target)}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        means.append(float(np.mean([np.mean(by_cluster[keys[i]]) for i in pick])))
    point = float(np.mean([np.mean(by_cluster[k]) for k in keys]))
    return {"mean": point, "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n_clusters": len(keys),
            "n_targets": sum(len(v) for v in by_cluster.values()),
            "n_targets_offered": len(values_by_target)}


def pool(cells: list[dict], pooled_src: dict) -> dict:
    clusters = pooled_src["target_clusters"]
    steps = pooled_src["capture_steps"]
    fam = pooled_src["pooled"]["readability_by_family"]
    lin = {m: fam[m]["linear_activation"] for m in PRIMARY}
    pub_mlp = {m: fam[m]["mlp"] for m in PRIMARY}

    by = {m: {"fixed": {}, "tuned": {}} for m in PRIMARY}
    for c in cells:
        for m, r in c["metrics"].items():
            if "error" in r:
                continue
            for which in ("fixed", "tuned"):
                v = r[which].get("readability")
                if v is not None and np.isfinite(v):
                    by[m][which].setdefault(str(c["step"]), {})[c["target"]] = float(v)

    out: dict = {"by_quality": {}}
    for m in PRIMARY:
        entry: dict = {"per_step": {}, "published_linear": {}, "published_mlp": {}}
        for s in steps:
            s = str(s)
            entry["per_step"][s] = {
                which: block_bootstrap(by[m][which].get(s, {}), clusters)
                for which in ("fixed", "tuned")
            }
            entry["published_linear"][s] = lin[m][s]
            entry["published_mlp"][s] = pub_mlp[m][s]

        def peak(d: dict, key: str | None = None) -> dict:
            best, bs = None, None
            for s in steps:
                v = (d[str(s)][key] if key else d[str(s)])["mean"]
                if v is not None and np.isfinite(v) and (best is None or v > best["mean"]):
                    best, bs = (d[str(s)][key] if key else d[str(s)]), s
            return {"step": bs, **(best or {})}

        entry["peak_tuned"] = peak(entry["per_step"], "tuned")
        entry["peak_fixed"] = peak(entry["per_step"], "fixed")
        entry["peak_published_mlp"] = peak(entry["published_mlp"])
        entry["peak_published_linear"] = peak(entry["published_linear"])
        entry["tuned_minus_linear_at_peaks"] = (
            entry["peak_tuned"]["mean"] - entry["peak_published_linear"]["mean"]
        )
        entry["tuned_beats_linear"] = bool(
            entry["peak_tuned"]["mean"] > entry["peak_published_linear"]["mean"]
        )
        # paired within target at the linear readout's peak step, resampled over
        # clusters: the honest test of "does the tuned MLP beat the ridge there"
        ps = str(entry["peak_published_linear"]["step"])
        entry["tuned_at_linear_peak_step"] = block_bootstrap(dict(by[m]["tuned"].get(ps, {})), clusters)
        entry["linear_peak_step"] = int(ps)
        out["by_quality"][m] = entry

    repro = [r["reproduction_abs_diff"] for c in cells for r in c["metrics"].values()
             if isinstance(r, dict) and r.get("reproduction_abs_diff") is not None]
    out["reproduction"] = {
        "n_compared": len(repro),
        "max_abs_diff": float(np.max(repro)) if repro else None,
        "median_abs_diff": float(np.median(repro)) if repro else None,
        "n_exact": int(sum(1 for x in repro if x == 0.0)),
        "n_within_1e_12": int(sum(1 for x in repro if x <= 1e-12)),
    }
    picked = [a for c in cells for r in c["metrics"].values()
              if isinstance(r, dict) and "alphas_selected" in r for a in r["alphas_selected"]]
    vals, counts = np.unique(picked, return_counts=True) if picked else ([], [])
    out["selected_penalties"] = {str(v): int(n) for v, n in zip(vals, counts)}
    out["n_cells"] = len(cells)
    out["n_fits_per_cell_quality"] = N_OUTER * (len(ALPHAS) * N_INNER + 2)
    return out


def main() -> int:
    out_dir = sys.argv[1]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    os.makedirs(os.path.join(out_dir, "cells"), exist_ok=True)

    pooled_src = json.load(open(POOLED))
    targets = sorted(pooled_src["targets"])
    steps = list(pooled_src["capture_steps"])
    todo = [(t, s) for t in targets for s in steps]
    if limit:
        todo = todo[:limit]

    print(f"tuned-MLP retune: {len(todo)} cells x {len(PRIMARY)} qualities, "
          f"{workers} workers, penalties {ALPHAS}", flush=True)
    print(f"harvest={HARVEST}\nper_target={PER_TARGET}\nout={out_dir}", flush=True)

    def report_progress(**kw):
        """Dependency-free progress hook for the public release."""
        return None

    t0 = time.time()
    cells: list[dict] = []
    with mp.get_context("fork").Pool(workers) as pool_:
        for i, res in enumerate(pool_.imap_unordered(run_cell, todo), start=1):
            cells.append(res)
            with open(os.path.join(out_dir, "cells", f"{res['target']}_step{res['step']}.json"), "w") as f:
                json.dump(res, f)
            report_progress(step=i, total_steps=len(todo), phase="retune")
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} cells  {el / 60:.1f} min elapsed  "
                      f"eta {el / i * (len(todo) - i) / 60:.1f} min", flush=True)

    print("--- pooling over Foldseek clusters ---", flush=True)
    res = pool(cells, pooled_src)
    res["wall_seconds"] = time.time() - t0
    dest = os.path.join(out_dir, "mlp_retune.json")
    with open(dest, "w") as f:
        json.dump(res, f, indent=1)

    print(f"reproduction of published fixed-penalty fits: {json.dumps(res['reproduction'])}", flush=True)
    print(f"penalties chosen: {json.dumps(res['selected_penalties'])}", flush=True)
    for m in PRIMARY:
        e = res["by_quality"][m]
        print(f"{m:24s} tuned peak {e['peak_tuned']['mean']:.4f} at step {e['peak_tuned']['step']}"
              f"  (published mlp {e['peak_published_mlp']['mean']:.4f}"
              f", linear {e['peak_published_linear']['mean']:.4f})"
              f"  beats_linear={e['tuned_beats_linear']}", flush=True)
    print(f"wrote {dest} in {res['wall_seconds'] / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
