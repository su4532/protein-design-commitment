"""Readability with the configuration chosen on roots that never score it.

The first pass fitted every readout and layer with whole roots held out, then
reported the largest of those held-out scores. Holding out roots stops lineage
leakage inside a single fit, but taking a maximum over 46 fits reintroduces
selection on exactly the numbers being reported, and the winner was sometimes
the geometry baseline or the MLP capacity control rather than a linear readout
of activations at all.

This module does what the plan asked for instead. Roots are split into outer
folds; inside each outer fold the candidate configurations are ranked on a
further root-blocked split of the *training* roots only, and the winner is refit
and scored on outer roots that took no part in either the ranking or the fit.
Candidates are kept in separate families so a claim about a linear activation
readout is answered by linear activation readouts:

``linear_activation``  ridge on binder pooling, interface pooling, or interface
                       covariance features, over every layer - the family the
                       readability claim is about
``geometry``           descriptors of the decoded partial structure, the
                       baseline an activation probe has to beat
``mlp``                a small MLP on the binder pool over every layer, a
                       capacity control that says whether a non-linear readout
                       would see more

Regularisation is still chosen inside the training roots, so no held-out root
influences any fitted quantity, the configuration, or ``alpha``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from .probes import ALPHAS, _estimator, make_feature_fn

__all__ = ["candidate_families", "nested_oof", "family_readability"]

LINEAR_READOUTS = ("binder", "iface", "iface_cov")
RANK_ALPHA = 100.0  # fixed while ranking configurations; tuned again for the winner


def candidate_families(n_layers: int) -> dict[str, list[tuple[str, int]]]:
    return {
        "linear_activation": [(ro, l) for ro in LINEAR_READOUTS for l in range(n_layers)],
        "geometry": [("geom", 0)],
        "mlp": [("mlp", l) for l in range(n_layers)],
    }


def _fit_predict(fn: Callable, ro: str, alpha: float, y: np.ndarray,
                 tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """``tr``/``te`` index ``y`` directly; both must be absolute row indices."""
    Xa, Xb = fn(tr, te)
    return _estimator(ro, alpha).fit(Xa, y[tr]).predict(Xb)


def nested_oof(shard: dict[str, Any], y: np.ndarray, root: np.ndarray,
               candidates: list[tuple[str, int]], n_outer: int = 4, n_inner: int = 3,
               alphas: tuple[float, ...] = ALPHAS) -> dict[str, Any]:
    """Out-of-fold predictions whose configuration was selected without them.

    Returns the concatenated outer-fold predictions, the configuration chosen in
    each outer fold (its stability is itself informative), and the mean inner
    ranking score of every candidate.
    """
    roots = np.unique(root)
    n_outer = int(min(n_outer, len(roots)))
    if n_outer < 2 or len(candidates) == 0:
        return {"oof": None, "chosen": [], "n_outer": 0}

    fns = {}
    for cfg in candidates:
        try:
            fns[cfg] = make_feature_fn(shard, cfg[0], cfg[1])
        except Exception:
            continue
    if not fns:
        return {"oof": None, "chosen": [], "n_outer": 0}

    oof = np.full(len(y), np.nan)
    chosen: list[str] = []
    errors: list[str] = []
    rank_acc: dict[tuple[str, int], list[float]] = {c: [] for c in fns}

    for tr, te in GroupKFold(n_splits=n_outer).split(np.zeros(len(y)), y, root):
        inner_roots = np.unique(root[tr])
        n_in = int(min(n_inner, len(inner_roots)))
        if n_in < 2:
            continue
        inner = list(GroupKFold(n_splits=n_in).split(np.zeros(len(tr)), y[tr], root[tr]))

        best_cfg, best_score = None, -np.inf
        for cfg, fn in fns.items():
            preds = np.full(len(tr), np.nan)
            try:
                for itr, ite in inner:
                    preds[ite] = _fit_predict(fn, cfg[0], RANK_ALPHA, y, tr[itr], tr[ite])
            except Exception as e:
                errors.append(f"{cfg[0]}|{cfg[1]} rank: {type(e).__name__}: {e}")
                continue
            ok = np.isfinite(preds)
            if ok.sum() < 5 or np.std(y[tr][ok]) == 0:
                continue
            s = -float(np.mean((preds[ok] - y[tr][ok]) ** 2))
            rank_acc[cfg].append(s)
            if s > best_score:
                best_score, best_cfg = s, cfg
        if best_cfg is None:
            continue

        # regularisation for the winner, again on training roots only
        fn = fns[best_cfg]
        alpha, a_score = alphas[len(alphas) // 2], -np.inf
        if best_cfg[0] != "mlp" and len(alphas) > 1:
            for a in alphas:
                preds = np.full(len(tr), np.nan)
                try:
                    for itr, ite in inner:
                        preds[ite] = _fit_predict(fn, best_cfg[0], a, y, tr[itr], tr[ite])
                except Exception as e:
                    errors.append(f"alpha {a}: {type(e).__name__}: {e}")
                    continue
                ok = np.isfinite(preds)
                if ok.sum() < 5:
                    continue
                s = -float(np.mean((preds[ok] - y[tr][ok]) ** 2))
                if s > a_score:
                    a_score, alpha = s, a
        try:
            oof[te] = _fit_predict(fn, best_cfg[0], alpha, y, tr, te)
        except Exception as e:
            errors.append(f"final {best_cfg}: {type(e).__name__}: {e}")
            continue
        chosen.append(f"{best_cfg[0]}|{best_cfg[1]}|alpha={alpha:g}")

    if not np.isfinite(oof).any():
        # a family that produces nothing is a bug, not a finding; surface why
        return {"oof": None, "chosen": chosen, "n_outer": n_outer,
                "errors": errors[:5], "n_errors": len(errors)}
    return {
        "oof": oof,
        "chosen": chosen,
        "n_outer": n_outer,
        "n_inner": n_inner,
        "n_candidates": len(fns),
        "n_errors": len(errors),
        "rank_scores": {f"{c[0]}|{c[1]}": float(np.mean(v)) for c, v in rank_acc.items() if v},
    }


def family_readability(shard: dict[str, Any], y_probe: np.ndarray, root: np.ndarray,
                       decomp: Any, families: dict[str, list[tuple[str, int]]],
                       readability_fn: Callable) -> dict[str, Any]:
    """Run the nested protocol once per family and score each on the ceiling scale."""
    out: dict[str, Any] = {}
    for name, cands in families.items():
        r = nested_oof(shard, y_probe, root, cands)
        entry: dict[str, Any] = {"chosen": r.get("chosen", []), "n_candidates": r.get("n_candidates", 0),
                                 "n_errors": r.get("n_errors", 0)}
        if r.get("oof") is not None:
            entry.update(readability_fn(r["oof"], y_probe, decomp))
            entry["rank_scores"] = r.get("rank_scores", {})
        else:
            entry["readability"] = float("nan")
            entry["errors"] = r.get("errors", [])
        out[name] = entry
    return out
