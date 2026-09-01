"""Resampling that carries commitment uncertainty all the way to the spread.

The first pass through this experiment estimated each target's settling step
once and then bootstrapped only the 19 target-level numbers. That treats a
half-commitment step as if it were measured exactly, which it is not: fold
identity is estimated with a categorical Gini decomposition and the other three
with a continuous nested ANOVA, the curves have different slopes near the
crossing, and the within-completion noise differs by quality. A quality whose
half-time is simply *harder to pin down* would then look like a quality that
genuinely varies more from target to target.

This module fixes that by resampling the design itself. One replicate draws
roots with replacement and then children within each drawn root - the standard
multistage cluster bootstrap for a nested layout - and recomputes the whole
commitment curve from the resampled data. Crucially the *same* draw is applied
at every capture step and to every quality, because the capture reuses the same
16 roots at all seven steps. That makes a replicate a coherent alternative
version of the whole experiment for one target, so interpolating it gives a
replicate settling step, and differences between qualities keep their pairing.

Everything here works on the balanced ``[R, C, k]`` layout rather than on the
flat state list, so 2000 replicates of a cell cost about a second.
"""

from __future__ import annotations

import numpy as np

__all__ = ["as_rck", "draw_design_boot", "commitment_replicates"]


def as_rck(y: np.ndarray, root: np.ndarray, child: np.ndarray) -> np.ndarray:
    """Flat ``[N, k]`` plus lineage indices -> balanced ``[R, C, k]``.

    Children are renumbered per root, so a design that measured a different
    subset of children in each root (which the designability subsample does)
    still forms a balanced array. Child index is arbitrary within a root, so
    the renumbering changes nothing the decomposition depends on.
    """
    y = np.asarray(y, dtype=float)
    root = np.asarray(root)
    child = np.asarray(child)
    roots = np.unique(root)
    per = [np.where(root == r)[0][np.argsort(child[root == r])] for r in roots]
    C = min(len(p) for p in per)
    return np.stack([y[p[:C]] for p in per], axis=0)


def draw_design_boot(n_boot: int, R: int, seed: int) -> np.ndarray:
    """Uniform draws for a two-stage resample, shared across steps and qualities.

    Returns ``u`` of shape ``[n_boot, R, 9]``: column 0 selects the root, the
    remaining columns select children within it. Uniforms rather than indices,
    so the same draw maps onto designs with different numbers of children (the
    designability subsample keeps 4 of 8) while staying coupled to the other
    qualities' draws.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n_boot, R, 9))


def _resample(a: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Apply the shared uniform draws to one ``[R, C, k]`` cell -> ``[B, R, C, k]``."""
    R, C, _ = a.shape
    B = u.shape[0]
    # the shared draw is sized for the full design; a sub-design (the
    # designability quarter keeps fewer children, and can keep fewer roots if a
    # cell is incomplete) reads the first R columns of the same draw, which
    # keeps it coupled to the other qualities' resamples
    uu = u[:, :R, :] if u.shape[1] >= R else np.tile(u, (1, int(np.ceil(R / u.shape[1])), 1))[:, :R, :]
    ri = np.minimum((uu[:, :, 0] * R).astype(np.int64), R - 1)          # [B, R]
    ci = np.minimum((uu[:, :, 1:1 + C] * C).astype(np.int64), C - 1)    # [B, R, C]
    return a[ri[:, :, None], ci, :].reshape(B, R, C, -1)


def _commitment_continuous(A: np.ndarray) -> np.ndarray:
    """Nested-ANOVA commitment for a stack of resampled cells ``[B, R, C, k]``."""
    B, R, C, k = A.shape
    child_mean = A.mean(3)
    root_mean = child_mean.mean(2)
    grand = root_mean.mean(1)
    ss_within = ((A - child_mean[..., None]) ** 2).sum((1, 2, 3))
    ss_child = k * ((child_mean - root_mean[..., None]) ** 2).sum((1, 2))
    ss_root = C * k * ((root_mean - grand[:, None]) ** 2).sum(1)
    ms_within = ss_within / (R * C * (k - 1))
    ms_child = ss_child / (R * (C - 1))
    ms_root = ss_root / (R - 1)
    s2_e = ms_within
    s2_b = (ms_child - ms_within) / k
    s2_a = (ms_root - ms_child) / (C * k)
    pa, pb, pe = np.clip(s2_a, 0, None), np.clip(s2_b, 0, None), np.clip(s2_e, 0, None)
    tot = pa + pb + pe
    out = np.full(B, np.nan)
    ok = tot > 0
    out[ok] = (pa[ok] + pb[ok]) / tot[ok]
    return out


def _coll(x: np.ndarray, n_lab: int) -> float:
    """Unbiased collision probability ``sum_c p_c^2`` for one flat label vector."""
    m = x.size
    if m < 2:
        return np.nan
    n = np.bincount(x.ravel(), minlength=n_lab)
    return float((n * (n - 1)).sum() / (m * (m - 1)))


def _commitment_categorical(A: np.ndarray, n_lab: int) -> np.ndarray:
    """Gini-impurity commitment for a stack of resampled label cells."""
    B, R, C, k = A.shape
    out = np.full(B, np.nan)
    for b in range(B):
        a = A[b]
        g_total = 1.0 - _coll(a, n_lab)
        g_child = float(np.mean([1.0 - _coll(a[r, c], n_lab) for r in range(R) for c in range(C)]))
        if not np.isfinite(g_total) or g_total <= 0:
            continue
        out[b] = 1.0 - g_child / g_total
    return out


def commitment_replicates(a: np.ndarray, u: np.ndarray, kind: str = "continuous",
                          n_lab: int | None = None, chunk: int = 500) -> np.ndarray:
    """Commitment under each resample of one capture cell.

    ``a`` is the cell's balanced ``[R, C, k]`` outcome array and ``u`` the shared
    uniform draws from :func:`draw_design_boot`. Returns ``[B]``.
    """
    B = u.shape[0]
    out = np.empty(B)
    for s in range(0, B, chunk):
        A = _resample(a, u[s:s + chunk])
        if kind == "categorical":
            out[s:s + chunk] = _commitment_categorical(A.astype(np.int64), int(n_lab))
        else:
            out[s:s + chunk] = _commitment_continuous(A)
    return out
