#!/usr/bin/env python
"""Does a readout learned on other targets transfer to an unrelated fold?

The main readability numbers are fit and scored inside one target, with whole
roots held out. That answers "is this decodable here", not "is there a shared
direction". The plan asked for the transfer curve on top: hold out a whole
structural cluster of targets, fit on the rest, and score on the held-out
cluster. Nineteen benchmark targets fall into twelve Foldseek clusters and the
largest holds seven, so leaving out a *target* would leave near-homologues in
the training set; the cluster is the honest unit.

Two details make the comparison fair. Labels are z-scored within a target, since
the four qualities have different scales and different per-target distributions,
and a model that only learned "this target scores badly" would otherwise look
like a transferring readout. And the layer is chosen inside each fold's training
clusters, never on the held-out one.

Scoring is within the held-out target: the correlation between predicted and
true child means, averaged over the held-out cluster's targets. A transfer score
near zero with a healthy within-target score means the readout is target
specific.

    python loco_transfer.py <cell_cache_dir> <clusters.json> <out.json> [--pooling binder]
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .commitment.grid import steps as _grid_steps  # noqa: E402

#: capture grid for this run; ``CAPTURE_STEPS`` in the environment overrides it
STEPS = _grid_steps()
METRICS = ("fold_identity", "designability", "structural_confidence", "interface_quality")
ALPHAS = (10.0, 100.0, 1000.0, 10000.0)


def load_step(cell_dir: str, step: int, pooling: str) -> dict[str, dict]:
    out = {}
    for p in sorted(glob.glob(os.path.join(cell_dir, f"*_step{step}.npz"))):
        t = os.path.basename(p).split(f"_step{step}.npz")[0]
        z = np.load(p)
        if pooling not in z.files:
            continue
        e = {"X": np.asarray(z[pooling], dtype=np.float32), "y": {}}
        for m in METRICS:
            key = f"y_{m}"
            if key not in z.files:
                continue
            y = np.asarray(z[key], float)
            rows = np.asarray(z["des_rows"]) if (m == "designability" and "des_rows" in z.files) \
                else np.arange(e["X"].shape[0])
            if len(rows) != len(y):
                continue
            e["y"][m] = (rows, y)
        out[t] = e
    return out


def zscore(y: np.ndarray) -> np.ndarray:
    s = np.std(y)
    return (y - np.mean(y)) / s if s > 0 else np.zeros_like(y)


def fit_score(data: dict, clusters: dict, metric: str, n_layers: int) -> dict:
    """Leave-one-structural-cluster-out transfer for one metric at one step."""
    targets = [t for t in sorted(data) if metric in data[t]["y"]]
    if len(targets) < 4:
        return {"n_targets": len(targets), "insufficient": True}
    groups = {}
    for t in targets:
        groups.setdefault(clusters.get(t, t), []).append(t)
    if len(groups) < 3:
        return {"n_targets": len(targets), "n_clusters": len(groups), "insufficient": True}

    per_target_r: dict[str, float] = {}
    chosen: list[int] = []
    for held, members in sorted(groups.items()):
        train = [t for t in targets if t not in members]
        if not train:
            continue

        def build(ts, layer):
            X, y = [], []
            for t in ts:
                rows, yy = data[t]["y"][metric]
                X.append(data[t]["X"][rows, layer, :])
                y.append(zscore(yy))
            return np.concatenate(X), np.concatenate(y)

        # layer chosen on a split of the TRAINING clusters only
        inner = sorted({clusters.get(t, t) for t in train})
        hold = set(groups[inner[len(inner) // 2]])
        fit_ts = [t for t in train if t not in hold]
        val_ts = [t for t in train if t in hold]
        best_layer, best = 0, -np.inf
        for layer in range(n_layers):
            Xa, ya = build(fit_ts, layer)
            Xb, yb = build(val_ts, layer)
            m = make_pipeline(StandardScaler(), Ridge(alpha=1000.0)).fit(Xa, ya)
            p = m.predict(Xb)
            s = float(np.corrcoef(p, yb)[0, 1]) if np.std(p) > 0 else -np.inf
            if s > best:
                best, best_layer = s, layer
        chosen.append(best_layer)

        Xtr, ytr = build(train, best_layer)
        # regularisation on the training clusters, same inner split
        Xa, ya = build(fit_ts, best_layer)
        Xb, yb = build(val_ts, best_layer)
        alpha, a_best = ALPHAS[-1], -np.inf
        for a in ALPHAS:
            p = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xa, ya).predict(Xb)
            s = -float(np.mean((p - yb) ** 2))
            if s > a_best:
                a_best, alpha = s, a
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xtr, ytr)
        for t in members:
            rows, yy = data[t]["y"][metric]
            p = model.predict(data[t]["X"][rows, best_layer, :])
            z = zscore(yy)
            per_target_r[t] = float(np.corrcoef(p, z)[0, 1]) if np.std(p) > 0 and np.std(z) > 0 else np.nan

    vals = np.array([v for v in per_target_r.values() if np.isfinite(v)])
    return {
        "n_targets": len(targets), "n_clusters": len(groups),
        "mean_r": float(np.mean(vals)) if vals.size else float("nan"),
        "median_r": float(np.median(vals)) if vals.size else float("nan"),
        "mean_r2_signed": float(np.mean(np.sign(vals) * vals**2)) if vals.size else float("nan"),
        "frac_positive": float(np.mean(vals > 0)) if vals.size else float("nan"),
        "layers_chosen": chosen,
        "per_target_r": per_target_r,
    }


def _arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    cell_dir, clusters_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    pooling = _arg("--pooling", "binder_pool")
    clusters = json.load(open(clusters_path)) if os.path.exists(clusters_path) else {}

    out: dict = {"pooling": pooling, "steps": {}}
    for s in STEPS:
        data = load_step(cell_dir, s, pooling)
        if not data:
            continue
        n_layers = int(next(iter(data.values()))["X"].shape[1])
        out["steps"][str(s)] = {m: fit_score(data, clusters, m, n_layers) for m in METRICS}
        line = "  ".join(
            f"{m[:6]}={out['steps'][str(s)][m].get('mean_r', float('nan')):+.3f}" for m in METRICS)
        print(f"step {s:3d} ({len(data)} targets): {line}", flush=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
