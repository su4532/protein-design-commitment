"""The action-step objective: how much a selection at step ``t`` can act on.

``J(t) = child_share(t) * R2_sibling(t)``

The first factor is the share of *total completion variance* created among the
siblings of one initialisation - branches a search could actually choose between
at step ``t``. It comes from the package's nested-ANOVA decomposition. The second factor is the fraction of that
sibling-level variance a linear readout of the activations recovers.

The second factor is the one new estimator here, and it exists because the
source analysis's ``readability`` answers a different question. There the
probe is scored against a state's raw expected outcome, so a probe that only
learns "this whole trajectory family is good" scores well even though a search
branching within a family gains nothing from it. Three properties make the
sibling version mean what it says, and :mod:`verify_estimator` checks each one
numerically rather than by inspection:

1. **Both sides are parent-centred.** Outcomes and predictions each have their
   own root's mean subtracted before scoring (:func:`center_by_root`), so the
   score cannot be earned by ranking roots.
2. **The denominator is the parent-centred outcome variance.** The score is the
   squared Pearson correlation of the two centred vectors, whose denominator is
   ``var(y - mean_root(y))``. Scoring against total variance instead would
   re-count the sibling share that ``child_share`` already contributes, and
   :func:`sibling_score` returns that wrong variant as ``r2_vs_total_WRONG`` so
   the size of the double count is visible rather than hypothetical.
3. **The split is lineage-blocked.** Whole roots are held out, so no sibling of
   a scored state appears in training - the split discipline the parent
   experiment established after a random split inflated every one of its 35
   quality-by-step readability numbers.

Finite ``k`` attenuates the correlation exactly as it does in the parent's
``readability``, but the relevant reliability is the *sibling-level* one,
``sigma2_child / (sigma2_child + sigma2_within / k)``, because the signal being
ranked is the between-sibling signal alone.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold


from ..commitment.variance import decompose  # noqa: E402

from .features import FeatureStore, candidate_configs  # noqa: E402
from .ridge import ALPHAS, RidgeFold  # noqa: E402

__all__ = [
    "CAPTURE_STEPS",
    "QUALITIES",
    "LOWER_IS_BETTER",
    "RANK_ALPHA",
    "center_by_root",
    "nested_oof_full",
    "train_on_roots",
    "sibling_score",
    "objective",
    "three_way_roots",
]

CAPTURE_STEPS = tuple(
    int(x)
    for x in os.environ.get(
        "ACTION_STEPS", "10,25,50,75,100,150,200,250,300,350"
    ).split(",")
    if x
)
QUALITIES = ("interface_quality", "fold_identity", "structural_confidence", "designability")
#: sign convention of the raw outcome; the objective is sign-free (a squared
#: correlation), but per-quality summaries report it for readability
LOWER_IS_BETTER = {"interface_quality": True, "fold_identity": False,
                   "structural_confidence": False, "designability": True}
RANK_ALPHA = 100.0          # the parent's fixed alpha while ranking configurations


# --------------------------------------------------------------------------- #
# parent centring
# --------------------------------------------------------------------------- #


def center_by_root(v: np.ndarray, root: np.ndarray) -> np.ndarray:
    """Subtract each state's own parent (root) mean.

    Splits in this module always hold out whole roots, so a root's mean is
    computed entirely within whichever side of the split that root sits on: the
    centring never moves information across the split.
    """
    v = np.asarray(v, dtype=np.float64)
    root = np.asarray(root)
    out = np.empty_like(v)
    for r in np.unique(root):
        m = root == r
        out[m] = v[m] - v[m].mean()
    return out


def root_mean_abs(v: np.ndarray, root: np.ndarray) -> float:
    """Largest absolute per-root mean - zero iff ``v`` is parent-centred."""
    return float(max(abs(np.asarray(v, float)[root == r].mean()) for r in np.unique(root)))


# --------------------------------------------------------------------------- #
# candidate readouts
# --------------------------------------------------------------------------- #




# --------------------------------------------------------------------------- #
# nested, root-blocked fitting against parent-centred outcomes
# --------------------------------------------------------------------------- #


def _inner_splits(tr: np.ndarray, root: np.ndarray, n_inner: int) -> list[tuple[np.ndarray, np.ndarray]]:
    inner_roots = np.unique(root[tr])
    n = int(min(n_inner, len(inner_roots)))
    if n < 2:
        return []
    gk = GroupKFold(n_splits=n)
    return list(gk.split(np.zeros(len(tr)), np.zeros(len(tr)), root[tr]))


def _select_and_fit(
    store: FeatureStore,
    ys: dict[str, np.ndarray],
    root: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    cfgs: Sequence[tuple[str, int]],
    n_inner: int,
    alphas: tuple[float, ...],
) -> dict[str, dict[str, Any]]:
    """One outer fold: rank configurations inside ``tr``, refit the winner, predict ``te``.

    ``ys`` maps quality name -> parent-centred outcome vector. All qualities
    share the split, so each ``(configuration, inner split)`` factorisation is
    built once and reused across qualities and alphas; the *selection* is still
    made independently per quality.
    """
    inner = _inner_splits(tr, root, n_inner)
    if not inner:
        return {q: {"pred": None} for q in ys}

    # inner out-of-fold predictions: [quality][cfg][alpha] -> vector over tr
    acc: dict[str, dict[tuple[str, int], dict[float, np.ndarray]]] = {
        q: {c: {a: np.full(len(tr), np.nan) for a in alphas} for c in cfgs} for q in ys
    }
    for cfg in cfgs:
        for itr, ite in inner:
            X = store.get(cfg, tr[itr])
            fold = RidgeFold(X, tr[itr], tr[ite])
            for q, y in ys.items():
                pm = fold.predict_many(y, alphas)
                for a in alphas:
                    acc[q][cfg][a][ite] = pm[a]

    out: dict[str, dict[str, Any]] = {}
    for q, y in ys.items():
        ytr = y[tr]

        def score(v: np.ndarray) -> float:
            ok = np.isfinite(v)
            if ok.sum() < 5:
                return -np.inf
            return -float(np.mean((v[ok] - ytr[ok]) ** 2))

        ranked = {c: score(acc[q][c][RANK_ALPHA]) for c in cfgs}
        best_cfg = max(ranked, key=lambda c: ranked[c])
        a_scores = {a: score(acc[q][best_cfg][a]) for a in alphas}
        best_alpha = max(a_scores, key=lambda a: a_scores[a])
        fold = RidgeFold(store.get(best_cfg, tr), tr, te)
        out[q] = {
            "pred": fold.predict(y, best_alpha),
            "cfg": f"{best_cfg[0]}|{best_cfg[1]}",
            "alpha": float(best_alpha),
        }
    return out


def nested_oof_full(
    store: FeatureStore,
    ys: dict[str, np.ndarray],
    root: np.ndarray,
    n_outer: int = 4,
    n_inner: int = 3,
    alphas: tuple[float, ...] = ALPHAS,
    cfgs: Sequence[tuple[str, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Out-of-fold predictions over every root, configuration selected without them."""
    root = np.asarray(root)
    cfgs = candidate_configs(store.n_layers, store.available()) if cfgs is None else cfgs
    roots = np.unique(root)
    n_outer = int(min(n_outer, len(roots)))
    oof = {q: np.full(len(root), np.nan) for q in ys}
    chosen: dict[str, list[str]] = {q: [] for q in ys}
    for tr, te in GroupKFold(n_splits=n_outer).split(np.zeros(len(root)), np.zeros(len(root)), root):
        res = _select_and_fit(store, ys, root, tr, te, cfgs, n_inner, alphas)
        for q in ys:
            if res[q]["pred"] is not None:
                oof[q][te] = res[q]["pred"]
                chosen[q].append(f"{res[q]['cfg']}|alpha={res[q]['alpha']:g}")
    return {q: {"oof": oof[q], "chosen": chosen[q], "n_candidates": len(cfgs)} for q in ys}


def train_on_roots(
    store: FeatureStore,
    ys: dict[str, np.ndarray],
    root: np.ndarray,
    train_roots: Iterable,
    n_inner: int = 3,
    alphas: tuple[float, ...] = ALPHAS,
    cfgs: Sequence[tuple[str, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit on ``train_roots`` only and predict every state outside them."""
    root = np.asarray(root)
    cfgs = candidate_configs(store.n_layers, store.available()) if cfgs is None else cfgs
    trmask = np.isin(root, list(train_roots))
    tr, te = np.where(trmask)[0], np.where(~trmask)[0]
    res = _select_and_fit(store, ys, root, tr, te, cfgs, n_inner, alphas)
    out = {}
    for q in ys:
        pred = np.full(len(root), np.nan)
        if res[q]["pred"] is not None:
            pred[te] = res[q]["pred"]
        out[q] = {"oof": pred, "chosen": f"{res[q].get('cfg')}|alpha={res[q].get('alpha')}",
                  "test_rows": te}
    return out


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sibling_score(
    oof: np.ndarray,
    child_mean: np.ndarray,
    root: np.ndarray,
    sigma2_child: float,
    sigma2_within: float,
    k: int,
    rows: np.ndarray | None = None,
) -> dict[str, float]:
    """Parent-centred R^2: how well the readout ranks siblings of one initialisation.

    ``rows`` restricts scoring to a subset of states (a half-split evaluation
    side); the parent centring is then computed within that subset, which is
    exactly right because the subset holds whole roots.
    """
    oof = np.asarray(oof, dtype=np.float64)
    y = np.asarray(child_mean, dtype=np.float64)
    root = np.asarray(root)
    idx = np.arange(len(y)) if rows is None else np.asarray(rows)
    idx = idx[np.isfinite(oof[idx]) & np.isfinite(y[idx])]
    nan = {k_: float("nan") for k_ in
           ("r_sibling_observed", "r2_sibling_observed", "reliability_sibling",
            "r2_sibling", "r2_strict", "r2_vs_total_WRONG", "n_states", "n_roots")}
    if len(idx) < 8:
        return nan

    yc = center_by_root(y[idx], root[idx])
    pc = center_by_root(oof[idx], root[idx])
    r = _corr(pc, yc)
    if not np.isfinite(r):
        return nan

    s2b = max(float(sigma2_child), 0.0)
    s2e = max(float(sigma2_within), 0.0)
    denom = s2b + s2e / max(int(k), 1)
    rel = s2b / denom if denom > 0 else float("nan")
    r_corr = (float(np.clip(r / np.sqrt(rel), -1.0, 1.0))
              if (rel is not None and np.isfinite(rel) and rel > 0) else float("nan"))

    # strict variant: the prediction is taken as-is rather than optimally rescaled
    sse = float(np.sum((yc - pc) ** 2))
    sst = float(np.sum(yc**2))
    r2_strict = 1.0 - sse / sst if sst > 0 else float("nan")

    # the denominator error this estimator exists to avoid: scoring the centred
    # prediction against the *uncentred* outcome re-counts the sibling share
    y_tot = y[idx] - y[idx].mean()
    r_wrong = _corr(pc, y_tot)

    return {
        "r_sibling_observed": r,
        "r2_sibling_observed": r * r,
        "reliability_sibling": float(rel),
        "r2_sibling": float(r_corr**2) if np.isfinite(r_corr) else float("nan"),
        "r2_strict": float(r2_strict),
        "r2_vs_total_WRONG": float(r_wrong**2) if np.isfinite(r_wrong) else float("nan"),
        "n_states": int(len(idx)),
        "n_roots": int(len(np.unique(root[idx]))),
    }


def objective(r2_observed: float, sigma2_root: float, sigma2_child: float,
              sigma2_within: float, k: int) -> dict[str, float]:
    """``J = [sigma2_child / sigma2_total] * R2_sibling``, computed stably.

    Written out, the definition is

    ``J = (s2b / total) * (r2_obs / reliability)``  with ``reliability = s2b / (s2b + s2e/k)``

    and ``s2b`` cancels::

        J = r2_obs * (s2b + s2e/k) / total

    The two expressions are the same quantity, but the first divides by a
    sibling variance that is legitimately near zero at early capture steps -
    where the attenuation correction then explodes and the raw ratio has to be
    clipped at an ``R2`` of one, turning "nothing has been decided yet" into
    ``J = child_share``. The second form has no such division and degrades to
    ``J = 0`` exactly when there is no sibling variance to act on.

    One admissibility cap remains: a selection cannot act on more than the
    branch-controllable share, so ``J <= child_share``. The cap binding is
    recorded rather than silently applied - it means the observed correlation
    exceeded what the cell's reliability can support, which is a sampling
    accident, not skill.
    """
    s2a, s2b, s2e = (max(float(sigma2_root), 0.0), max(float(sigma2_child), 0.0),
                     max(float(sigma2_within), 0.0))
    total = s2a + s2b + s2e
    # Degenerate cells are zero, not missing. An outcome with no variance left
    # (late-step fold identity, where every completion lands in the modal
    # cluster) offers a selection nothing to act on, and so does a readout whose
    # predictions are constant. Both are J = 0; propagating NaN instead would
    # silently drop those cells out of every panel average.
    if total <= 0:
        return {"J": 0.0, "J_uncapped": 0.0, "child_share": 0.0,
                "r2_sibling": float("nan"), "capped": False, "degenerate": True}
    child_share = s2b / total
    if not np.isfinite(r2_observed):
        return {"J": 0.0, "J_uncapped": 0.0, "child_share": float(child_share),
                "r2_sibling": 0.0, "capped": False, "degenerate": True}
    j_raw = float(r2_observed) * (s2b + s2e / max(int(k), 1)) / total
    j = min(j_raw, child_share)
    return {"J": float(j), "J_uncapped": float(j_raw), "child_share": float(child_share),
            "r2_sibling": float(j / child_share) if child_share > 0 else 0.0,
            "capped": bool(j_raw > child_share), "degenerate": False}


def decompose_subset(y: np.ndarray, root: np.ndarray, child: np.ndarray,
                     rows: np.ndarray | None = None) -> Any:
    """The maintained nested-ANOVA decomposition, optionally on a subset of states."""
    if rows is None:
        return decompose(y, root, child)
    return decompose(np.asarray(y)[rows], np.asarray(root)[rows], np.asarray(child)[rows])


# --------------------------------------------------------------------------- #
# half splits
# --------------------------------------------------------------------------- #


def three_way_roots(roots: np.ndarray, rep: int, seed_base: int) -> list[np.ndarray]:
    """Split a target's roots into three disjoint groups.

    Three, not two, because choosing a step needs a set that influenced neither
    the readout that produced the predictions nor the score it is judged on. One
    group trains the readout, a second selects the step, and the third scores it;
    rotating over the six orderings uses every group in every role.

    The same partition is used at every capture step and for every quality of a
    target, so a replicate is a coherent alternative version of the whole curve.
    """
    rng = np.random.default_rng(seed_base + 1009 * rep)
    perm = rng.permutation(np.asarray(roots))
    return [np.sort(a) for a in np.array_split(perm, 3)]


def stable_seed(text: str) -> int:
    """Process-independent seed (Python's ``hash`` of a str is salted per run)."""
    import hashlib

    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


def curve_interpolate(steps: Sequence[float], values: Sequence[float],
                      grid: np.ndarray) -> np.ndarray:
    """Shape-preserving interpolation of an objective curve onto a dense grid.

    PCHIP rather than a cubic spline: a spline overshoots between knots and can
    invent a peak where the seven measured points show none.
    """
    from scipy.interpolate import PchipInterpolator

    s = np.asarray(steps, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.full(len(grid), np.nan)
    return PchipInterpolator(s[ok], v[ok], extrapolate=False)(grid)
