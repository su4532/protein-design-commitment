"""The linear-activation readout family, matching the source analysis's.

Three readouts over every layer:

``binder``     mean-pooled residual stream over the binder residues
``iface``      mean-pooled over the target hotspot residues
``iface_cov``  the interface mean pool plus second-order structure - interface
               residue vectors projected to 32 dimensions and their 528 unique
               covariance entries

``iface_cov`` is the reason this module exists. Its projection is fitted on the
training rows of whichever fold is being run, so its feature matrix is not a
fixed array but a function of the training set, and it cannot be precomputed the
way the two pooling readouts can. An earlier version of this analysis dropped it
for that reason; omitting a readout the parent's nested selection chose about a
fifth of the time for interface quality can move the achieved R² differently at
different steps, which is exactly what the objective's shape depends on.

Matrices are cached per ``(configuration, training set)`` because one fold reuses
the same matrix across five regularisation strengths and every quality.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA

__all__ = ["POOLS", "FeatureStore", "candidate_configs"]

POOLS = ("binder", "iface", "iface_cov")
N_PROJ = 32


def candidate_configs(n_layers: int, pools: Sequence[str] = POOLS) -> list[tuple[str, int]]:
    return [(p, layer) for p in pools for layer in range(n_layers)]


def _cov_upper(x: np.ndarray) -> np.ndarray:
    c = np.cov(x.T) if x.shape[0] > 1 else np.zeros((x.shape[1], x.shape[1]))
    return c[np.triu_indices(c.shape[0])]


class FeatureStore:
    """Feature matrices for one cell, built on demand and cached per training set.

    ``get(cfg, train)`` returns the full ``[n_states, d]`` matrix for that
    configuration with anything data-fitted (the covariance readout's
    projection) estimated from ``train`` rows only.
    """

    def __init__(self, binder: np.ndarray, iface: np.ndarray,
                 iface_res: np.ndarray | None, iface_valid: np.ndarray | None) -> None:
        self._binder = binder
        self._iface = iface
        self._res = iface_res
        self._valid = None if iface_valid is None else iface_valid.astype(bool)
        self.n_layers = int(binder.shape[1])
        self.has_cov = iface_res is not None
        self._cache: dict[tuple, np.ndarray] = {}

    def available(self) -> tuple[str, ...]:
        return POOLS if self.has_cov else POOLS[:2]

    def get(self, cfg: tuple[str, int], train: np.ndarray) -> np.ndarray:
        pool, layer = cfg
        if pool == "binder":
            return self._binder[:, layer, :]
        if pool == "iface":
            return self._iface[:, layer, :]
        if pool != "iface_cov":
            raise ValueError(f"unknown readout {pool!r}")
        key = (layer, train.tobytes())
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        res = self._res[:, layer, :, :].astype(np.float32)
        flat = np.concatenate([res[i][self._valid[i]] for i in train], axis=0)
        k = int(min(N_PROJ, flat.shape[0], flat.shape[1]))
        proj = PCA(n_components=k, random_state=0).fit(flat)
        pool_mat = self._iface[:, layer, :]
        out = np.empty((res.shape[0], pool_mat.shape[1] + k * (k + 1) // 2), dtype=np.float32)
        for i in range(res.shape[0]):
            z = proj.transform(res[i][self._valid[i]])
            out[i] = np.concatenate([pool_mat[i], _cov_upper(z)])
        out = np.nan_to_num(out)
        # one cell runs many folds; keep the cache from growing without bound
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = out
        return out
