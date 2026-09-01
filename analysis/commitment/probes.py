"""Readability: how much of what is settled a readout can actually see.

The decomposition in :mod:`commitment.variance` gives a *ceiling* - the share of
outcome variance the state at the capture step determines. This module measures
how much of that ceiling a readout recovers, on the same 0-1 scale, so the gap
between them is interpretable as "decided but not accessible".

Putting the two on one scale needs care. The probe is trained against a child's
mean over ``k`` completions, which still carries ``sigma2_within / k`` of
measurement noise, so an observed correlation understates the probe's agreement
with the child's *true* expected outcome. :func:`readability` divides that
attenuation out using the reliability implied by the variance components, then
multiplies by the ceiling, giving the fraction of **total** completion variance
the probe explains. By construction readability <= commitment.

The readout ladder, in increasing order of what it is allowed to see:

``geom``       geometric descriptors of the model's decoded clean prediction at
               the capture step - the partially built structure. This is the
               baseline an activation probe has to beat: if activations add
               nothing over the structure they encode, they are not telling us
               anything new.
``binder``     mean-pooled residual stream over the binder residues (the Phase-1
               selected readout).
``iface``      mean-pooled over the target hotspot residues.
``iface_cov``  the interface mean pool *plus* second-order structure: interface
               residue vectors projected to 32 dimensions (projection fit on
               training roots only) and their 528 unique covariance entries. It
               strictly extends ``iface``, so any gain is attributable to second
               order structure rather than to a different feature set.
``mlp``        a small MLP on the binder pool - a capacity control, not a
               separate feature set.

Every fit holds out **whole roots**. Children of one root share a trunk, so a
random split over states would let the probe memorise a lineage and report a
number that is mostly leakage - the failure mode that turned an earlier
within-target correlation of +0.4800 into -0.1000 once lineage was blocked.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "READOUTS",
    "geometry_features",
    "make_feature_fn",
    "root_blocked_oof",
    "readability",
    "probe_curve",
]

ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
READOUTS = ("geom", "binder", "iface", "iface_cov", "mlp")


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #


def geometry_features(partial_ca: np.ndarray, valid: np.ndarray, target_ca: np.ndarray, hotspot: np.ndarray,
                      root_idx: np.ndarray) -> np.ndarray:
    """Descriptors of the decoded clean prediction at the capture step.

    Everything here is computable from the partially built structure the model
    already believes in, with no access to activations: binder size and shape,
    how compact it is, and how it sits against the target's hotspots.
    """
    n = partial_ca.shape[0]
    feats = []
    for i in range(n):
        v = valid[i].astype(bool)
        ca = partial_ca[i][v].astype(np.float64)
        tgt = target_ca[root_idx[i] % target_ca.shape[0]].astype(np.float64)
        hot = hotspot[root_idx[i] % hotspot.shape[0]].astype(bool)
        if hot.sum() == 0:
            hot = np.ones(tgt.shape[0], dtype=bool)
        hc = tgt[hot]
        if ca.shape[0] < 3:
            feats.append(np.zeros(14))
            continue
        cen = ca.mean(0)
        d_cen = ca - cen
        rg = float(np.sqrt((d_cen**2).sum(1).mean()))
        # principal moments of the binder point cloud -> globular vs extended
        evals = np.linalg.eigvalsh(np.cov(d_cen.T) + 1e-9 * np.eye(3))[::-1]
        ete = float(np.linalg.norm(ca[0] - ca[-1]))
        # consecutive-CA spacing: how "assembled" the chain is at this step
        step = np.linalg.norm(np.diff(ca, axis=0), axis=1)
        d = np.linalg.norm(ca[:, None, :] - hc[None, :, :], axis=2)
        dmin = d.min(1)
        feats.append(
            np.array(
                [
                    float(v.sum()),
                    rg,
                    rg / max(float(v.sum()) ** 0.5, 1e-6),
                    evals[0], evals[1], evals[2],
                    float(evals[2] / max(evals[0], 1e-9)),
                    ete,
                    float(step.mean()), float(step.std()),
                    float(dmin.min()), float(dmin.mean()),
                    float((dmin < 10.0).sum()),
                    float(np.linalg.norm(cen - hc.mean(0))),
                ]
            )
        )
    return np.nan_to_num(np.stack(feats))


def _cov_upper(x: np.ndarray) -> np.ndarray:
    """Upper triangle (with diagonal) of a covariance matrix, flattened."""
    c = np.cov(x.T) if x.shape[0] > 1 else np.zeros((x.shape[1], x.shape[1]))
    iu = np.triu_indices(c.shape[0])
    return c[iu]


def make_feature_fn(shard: dict[str, Any], readout: str, layer: int, n_proj: int = 32) -> Callable:
    """Return ``fn(train_idx, test_idx) -> (X_train, X_test)``.

    Anything fitted from data (the covariance readout's projection) is fitted on
    the training rows only and applied unchanged to the test rows.
    """
    if readout == "geom":
        G = geometry_features(
            np.asarray(shard["partial_ca"], dtype=np.float32),
            np.asarray(shard["partial_valid"]),
            np.asarray(shard["target_ca"], dtype=np.float32),
            np.asarray(shard["hotspot_mask"]),
            np.asarray(shard["root_idx"]),
        )
        return lambda tr, te: (G[tr], G[te])

    if readout in ("binder", "mlp"):
        X = np.asarray(shard["binder_pool"], dtype=np.float32)[:, layer, :]
        return lambda tr, te: (X[tr], X[te])

    if readout == "iface":
        X = np.asarray(shard["iface_pool"], dtype=np.float32)[:, layer, :]
        return lambda tr, te: (X[tr], X[te])

    if readout == "iface_cov":
        pool = np.asarray(shard["iface_pool"], dtype=np.float32)[:, layer, :]
        res = np.asarray(shard["iface_res"], dtype=np.float32)[:, layer, :, :]  # [N, K, D]
        valid = np.asarray(shard["iface_valid"])

        def fn(tr, te):
            flat_tr = np.concatenate([res[i][valid[i].astype(bool)] for i in tr], axis=0)
            k = min(n_proj, flat_tr.shape[0], flat_tr.shape[1])
            p = PCA(n_components=k, random_state=0).fit(flat_tr)

            def build(idx):
                out = []
                for i in idx:
                    z = p.transform(res[i][valid[i].astype(bool)])
                    out.append(np.concatenate([pool[i], _cov_upper(z)]))
                return np.nan_to_num(np.stack(out))

            return build(tr), build(te)

        return fn

    raise ValueError(f"unknown readout {readout!r}")


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #


def _estimator(readout: str, alpha: float):
    if readout == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64,),
                alpha=alpha,
                max_iter=800,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=0,
            ),
        )
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def root_blocked_oof(
    feature_fn: Callable,
    y: np.ndarray,
    root: np.ndarray,
    readout: str = "binder",
    n_splits: int = 8,
    alphas: tuple[float, ...] = ALPHAS,
    seed: int = 42,
) -> dict[str, Any]:
    """Out-of-fold predictions with whole roots held out, regularisation chosen inside each fold.

    The outer loop holds out roots for scoring. Inside each outer fold the
    regularisation strength is chosen by a further root-blocked split of the
    training roots only, so no held-out root influences any fitted quantity -
    including the choice of ``alpha``.
    """
    roots = np.unique(root)
    n_splits = int(min(n_splits, len(roots)))
    if n_splits < 2:
        return {"oof": None, "n_splits": 0}
    oof = np.full(len(y), np.nan)
    chosen = []
    outer = GroupKFold(n_splits=n_splits)
    select_alpha = readout != "mlp" and len(alphas) > 1
    for tr, te in outer.split(np.zeros(len(y)), y, root):
        inner_roots = np.unique(root[tr])
        best, best_score = alphas[len(alphas) // 2], -np.inf
        if select_alpha and len(inner_roots) >= 3:
            inner = GroupKFold(n_splits=min(3, len(inner_roots)))
            # Build features ONCE per inner split and reuse them across alphas.
            # The covariance readout refits a projection per split, so rebuilding
            # inside the alpha loop dominated the whole analysis.
            splits = []
            for itr, ite in inner.split(np.zeros(len(tr)), y[tr], root[tr]):
                Xa, Xb = feature_fn(tr[itr], tr[ite])
                splits.append((itr, ite, Xa, Xb))
            for a in alphas:
                preds = np.full(len(tr), np.nan)
                for itr, ite, Xa, Xb in splits:
                    m = _estimator(readout, a).fit(Xa, y[tr][itr])
                    preds[ite] = m.predict(Xb)
                ok = ~np.isnan(preds)
                if ok.sum() > 3 and np.std(y[tr][ok]) > 0 and np.std(preds[ok]) > 0:
                    s = -np.mean((preds[ok] - y[tr][ok]) ** 2)
                    if s > best_score:
                        best_score, best = s, a
        Xtr, Xte = feature_fn(tr, te)
        m = _estimator(readout, best).fit(Xtr, y[tr])
        oof[te] = m.predict(Xte)
        chosen.append(best)
    return {"oof": oof, "n_splits": n_splits, "alphas": chosen}


def readability(oof: np.ndarray, child_mean: np.ndarray, decomp: Any) -> dict[str, float]:
    """Put a probe's out-of-fold predictions on the commitment scale.

    ``r2_observed`` is the plain squared correlation against the noisy child
    mean. ``reliability`` is the share of the child mean's variance that is real
    signal, ``S / (S + sigma2_within/k)``, implied by the variance components.
    Dividing the observed correlation by ``sqrt(reliability)`` removes the
    attenuation from finite ``k``; multiplying the result by the ceiling gives
    ``readability``, the fraction of *total* completion variance the probe
    explains. ``readability_frac_of_ceiling`` is the same thing expressed as a
    fraction of what a perfect predictor could do.
    """
    ok = np.isfinite(oof) & np.isfinite(child_mean)
    if ok.sum() < 5 or np.std(oof[ok]) == 0 or np.std(child_mean[ok]) == 0:
        return {k: float("nan") for k in
                ("r_observed", "r2_observed", "spearman", "reliability", "r2_corrected",
                 "readability", "readability_frac_of_ceiling")}
    r = float(np.corrcoef(oof[ok], child_mean[ok])[0, 1])
    rho = float(spearmanr(oof[ok], child_mean[ok]).correlation)
    s = max(decomp.sigma2_root, 0.0) + max(decomp.sigma2_child, 0.0)
    e = max(decomp.sigma2_within, 0.0)
    rel = s / (s + e / decomp.k) if (s + e / decomp.k) > 0 else float("nan")
    r_corr = r / np.sqrt(rel) if rel and rel > 0 else float("nan")
    r_corr = float(np.clip(r_corr, -1.0, 1.0)) if np.isfinite(r_corr) else float("nan")
    r2_corr = r_corr**2 if np.isfinite(r_corr) else float("nan")
    read = r2_corr * decomp.commitment if np.isfinite(r2_corr) else float("nan")
    return {
        "r_observed": r,
        "r2_observed": r * r,
        "spearman": rho,
        "reliability": float(rel),
        "r2_corrected": float(r2_corr),
        "readability": float(read),
        "readability_frac_of_ceiling": float(read / decomp.commitment) if decomp.commitment else float("nan"),
    }


def probe_curve(
    shard: dict[str, Any],
    y: np.ndarray,
    decomp: Any,
    readouts: tuple[str, ...] = READOUTS,
    layers: tuple[int, ...] | None = None,
    n_splits: int = 8,
    also_random_split: bool = True,
) -> dict[str, Any]:
    """Fit the whole readout x layer grid for one capture cell.

    ``y`` is ``[N, k]``; the probe target is the child mean. Both the
    root-blocked and the naive random split are reported, because the gap
    between them *is* the leakage measurement.
    """
    root = np.asarray(shard["root_idx"])
    child_mean = np.asarray(y, dtype=float).mean(axis=1)
    n_layers = np.asarray(shard["binder_pool"]).shape[1]
    layers = tuple(range(n_layers)) if layers is None else layers
    out: dict[str, Any] = {"n_layers": n_layers, "grid": {}}
    for ro in readouts:
        layer_iter = (0,) if ro == "geom" else layers
        for layer in layer_iter:
            fn = make_feature_fn(shard, ro, layer)
            res = root_blocked_oof(fn, child_mean, root, readout=ro, n_splits=n_splits)
            if res["oof"] is None:
                continue
            entry = {"blocked": readability(res["oof"], child_mean, decomp), "alphas": res["alphas"]}
            if also_random_split:
                fake = np.arange(len(child_mean))  # every state its own "root" -> random split
                rnd = root_blocked_oof(fn, child_mean, fake, readout=ro, n_splits=n_splits)
                if rnd["oof"] is not None:
                    entry["random_split"] = readability(rnd["oof"], child_mean, decomp)
            out["grid"][f"{ro}|{layer}"] = entry
    return out
