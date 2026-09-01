"""Three-level variance decomposition of a completion-level outcome.

The capture design is a balanced nested layout::

    y[r, c, j]   r = 1..R roots, c = 1..C children, j = 1..k completions

with the model ``y = mu + a_r + b_rc + e_rcj``. The three components answer
three different questions:

``sigma2_root``   what the *initialisation* fixed (binder length, initial noise)
``sigma2_child``  what the *trajectory up to the capture step* added
``sigma2_within`` what is still open after the capture step

**Commitment** at the capture step is ``(root + child) / total``: the share of
outcome variance already determined by the state at that step. The **child**
share alone is what a search branching at initialisation and selecting at that
step could act on - if it is small, branch selection is limited no matter how
good the value model is.

Finite ``k`` matters. The naive statistic "variance of child means / variance of
all completions" is biased *upward*, because a child mean over ``k`` draws still
carries ``sigma2_within / k`` of measurement noise. The balanced nested ANOVA
estimators used here remove that bias exactly rather than approximately;
:func:`split_half_reliability` provides an independent check of the same
quantity by correlating two disjoint halves of the completions.

Categorical outcomes (Foldseek fold identity) get the same decomposition applied
to Gini impurity instead of variance, with the unbiased ``n(n-1)/k(k-1)``
estimator of the collision probability at each level, so all four metrics land on
one comparable 0-1 scale.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

__all__ = [
    "VarianceDecomposition",
    "decompose",
    "decompose_categorical",
    "split_half_reliability",
    "bootstrap_over_roots",
]


@dataclass
class VarianceDecomposition:
    """Nested variance components and the shares derived from them."""

    sigma2_root: float
    sigma2_child: float
    sigma2_within: float
    total: float
    commitment: float
    root_share: float
    child_share: float
    n_roots: int
    n_children: int
    k: int
    n_obs: int
    mean: float
    kind: str = "continuous"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_rck(y: np.ndarray, root: np.ndarray, child: np.ndarray) -> np.ndarray:
    """Reshape a flat ``[N, k]`` label matrix into ``[R, C, k]``.

    ``root``/``child`` are the per-state lineage indices written by the capture.
    Requires a balanced design (which the no-search capture always produces).
    """
    roots = np.unique(root)
    children = np.unique(child)
    R, C, k = len(roots), len(children), y.shape[1]
    out = np.full((R, C, k), np.nan, dtype=float)
    rpos = {v: i for i, v in enumerate(roots)}
    cpos = {v: i for i, v in enumerate(children)}
    for i in range(y.shape[0]):
        out[rpos[root[i]], cpos[child[i]], :] = y[i]
    return out


def decompose(y: np.ndarray, root: np.ndarray, child: np.ndarray, kind: str = "continuous") -> VarianceDecomposition:
    """Balanced three-level nested ANOVA on ``y[N, k]`` with lineage ``root``/``child``.

    Uses the classical expected-mean-square estimators, which are unbiased for
    each component:

    ``E[MS_within] = s2_e``,
    ``E[MS_child]  = s2_e + k*s2_b``,
    ``E[MS_root]   = s2_e + k*s2_b + C*k*s2_a``.

    Components can come out slightly negative when the true component is near
    zero; they are reported as estimated and only clipped when forming shares.
    """
    a = _as_rck(np.asarray(y, dtype=float), np.asarray(root), np.asarray(child))
    if np.isnan(a).any():
        # tolerate failed folds by dropping states with any NaN completion
        ok = ~np.isnan(a).any(axis=2)
        if not ok.all():
            keep_children = ok.all(axis=0)
            a = a[:, keep_children, :]
    R, C, k = a.shape
    if R < 2 or C < 2 or k < 2:
        raise ValueError(f"need R>=2, C>=2, k>=2; got {R},{C},{k}")

    child_mean = a.mean(axis=2)  # [R, C]
    root_mean = child_mean.mean(axis=1)  # [R]
    grand = root_mean.mean()

    ss_within = float(((a - child_mean[:, :, None]) ** 2).sum())
    ss_child = float(k * ((child_mean - root_mean[:, None]) ** 2).sum())
    ss_root = float(C * k * ((root_mean - grand) ** 2).sum())

    ms_within = ss_within / (R * C * (k - 1))
    ms_child = ss_child / (R * (C - 1))
    ms_root = ss_root / (R - 1)

    s2_e = ms_within
    s2_b = (ms_child - ms_within) / k
    s2_a = (ms_root - ms_child) / (C * k)

    pa, pb, pe = max(s2_a, 0.0), max(s2_b, 0.0), max(s2_e, 0.0)
    tot = pa + pb + pe
    if tot <= 0:
        commit = root_sh = child_sh = float("nan")
    else:
        commit = (pa + pb) / tot
        root_sh = pa / tot
        child_sh = pb / tot
    return VarianceDecomposition(
        sigma2_root=float(s2_a),
        sigma2_child=float(s2_b),
        sigma2_within=float(s2_e),
        total=float(tot),
        commitment=float(commit),
        root_share=float(root_sh),
        child_share=float(child_sh),
        n_roots=int(R),
        n_children=int(C),
        k=int(k),
        n_obs=int(R * C * k),
        mean=float(grand),
        kind=kind,
    )


def _unbiased_collision(labels: np.ndarray) -> float:
    """Unbiased estimate of ``sum_c p_c^2`` (the probability two draws collide).

    ``sum n_c(n_c - 1) / (n(n - 1))`` is unbiased for the collision probability,
    whereas the plug-in ``sum (n_c/n)^2`` is biased upward at small ``n`` - the
    categorical analogue of the finite-``k`` variance bias.
    """
    n = len(labels)
    if n < 2:
        return float("nan")
    _, counts = np.unique(labels, return_counts=True)
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def decompose_categorical(labels: np.ndarray, root: np.ndarray, child: np.ndarray) -> VarianceDecomposition:
    """Gini-impurity decomposition for a categorical outcome (e.g. fold cluster).

    ``labels`` is ``[N, k]`` of cluster ids. Impurity ``G = 1 - sum p^2`` plays
    the role of variance; the collision probability at each level is estimated
    with the unbiased ``n(n-1)`` form. Shares are the proportional reductions in
    impurity, so they are directly comparable to the continuous ``commitment``.
    """
    labels = np.asarray(labels)
    root = np.asarray(root)
    child = np.asarray(child)
    R = len(np.unique(root))
    C = len(np.unique(child))
    k = labels.shape[1]

    flat = labels.reshape(-1)
    g_total = 1.0 - _unbiased_collision(flat)

    g_child = float(np.nanmean([1.0 - _unbiased_collision(labels[i]) for i in range(labels.shape[0])]))
    g_root = float(
        np.nanmean([1.0 - _unbiased_collision(labels[root == r].reshape(-1)) for r in np.unique(root)])
    )

    if not np.isfinite(g_total) or g_total <= 0:
        commit = root_sh = child_sh = float("nan")
    else:
        commit = 1.0 - g_child / g_total
        root_sh = 1.0 - g_root / g_total
        child_sh = commit - root_sh
    return VarianceDecomposition(
        sigma2_root=float(g_total - g_root),
        sigma2_child=float(g_root - g_child),
        sigma2_within=float(g_child),
        total=float(g_total),
        commitment=float(commit),
        root_share=float(root_sh),
        child_share=float(child_sh),
        n_roots=int(R),
        n_children=int(C),
        k=int(k),
        n_obs=int(labels.size),
        mean=float("nan"),
        kind="categorical",
    )


def split_half_reliability(y: np.ndarray, seed: int = 42, n_rep: int = 200) -> dict[str, float]:
    """Independent check on the finite-``k`` correction.

    Splits each state's ``k`` completions into two disjoint halves, correlates
    the two half-means across states, and Spearman-Brown-corrects to the full
    ``k``. The result estimates ``reliability = S / (S + sigma2_within/k)``,
    which the ANOVA components predict independently; agreement between the two
    is evidence the decomposition is not an artefact of the estimator.
    """
    y = np.asarray(y, dtype=float)
    n, k = y.shape
    if k < 4:
        return {"reliability_half": float("nan"), "reliability_full": float("nan"), "n_rep": 0}
    rng = np.random.default_rng(seed)
    rs = []
    half = k // 2
    for _ in range(n_rep):
        perm = np.argsort(rng.random((n, k)), axis=1)
        a = np.take_along_axis(y, perm[:, :half], axis=1).mean(axis=1)
        b = np.take_along_axis(y, perm[:, half : 2 * half], axis=1).mean(axis=1)
        if np.std(a) > 0 and np.std(b) > 0:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    if not rs:
        return {"reliability_half": float("nan"), "reliability_full": float("nan"), "n_rep": 0}
    r_half = float(np.mean(rs))
    # Spearman-Brown from half-length (k/2) up to full k
    r_full = 2 * r_half / (1 + r_half) if r_half > -1 else float("nan")
    return {"reliability_half": r_half, "reliability_full": float(r_full), "n_rep": len(rs)}


def bootstrap_over_roots(
    y: np.ndarray,
    root: np.ndarray,
    child: np.ndarray,
    stat: str = "commitment",
    kind: str = "continuous",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile CI for a decomposition statistic, resampling whole roots.

    The root is the independent unit here: children of one root share a trunk and
    are not exchangeable with children of another. Resampling states instead of
    roots would understate the interval.
    """
    y = np.asarray(y)
    root = np.asarray(root)
    child = np.asarray(child)
    roots = np.unique(root)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(roots, size=len(roots), replace=True)
        ys, rs, cs = [], [], []
        for new_r, r in enumerate(pick):
            m = root == r
            ys.append(y[m])
            rs.append(np.full(int(m.sum()), new_r))
            cs.append(child[m])
        try:
            d = (
                decompose_categorical(np.concatenate(ys), np.concatenate(rs), np.concatenate(cs))
                if kind == "categorical"
                else decompose(np.concatenate(ys), np.concatenate(rs), np.concatenate(cs))
            )
        except ValueError:
            continue
        v = getattr(d, stat)
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    return {
        "lo": float(np.percentile(vals, 2.5)),
        "hi": float(np.percentile(vals, 97.5)),
        "median": float(np.median(vals)),
        "n_boot": len(vals),
    }
