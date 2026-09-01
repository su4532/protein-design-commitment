"""Ridge with the training-fold factorisation computed once.

The refit in this experiment runs the same nested root-blocked protocol as the
source analysis, but many more times: once on the full 16 roots and once per
half-split repetition, at seven capture steps, for four qualities, on nineteen
targets. Every one of those fits reuses a small number of distinct
``(feature matrix, training rows)`` pairs across five regularisation strengths
and four outcome vectors.

So the factorisation is done once per pair. With ``n_train <= 128`` rows and
``D = 768`` features the dual (kernel) form is the cheap one::

    w      = X^T (X X^T + a I)^-1 y_c
    y_hat  = X_te X^T (X X^T + a I)^-1 y_c + mean(y_train)

Eigendecomposing ``K = X X^T = U diag(s) U^T`` turns every additional alpha and
every additional outcome vector into two matrix-vector products. This is
algebraically identical to ``make_pipeline(StandardScaler(), Ridge(alpha))``,
which :func:`check_against_sklearn` asserts to floating-point tolerance.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ALPHAS", "RidgeFold", "check_against_sklearn"]

#: the source analysis's grid, unchanged
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


class RidgeFold:
    """Standardise on the training rows, then factorise once.

    Parameters
    ----------
    X
        ``[n_samples, n_features]`` feature matrix for *all* rows.
    train, test
        absolute row indices; ``train`` fixes the scaler and the factorisation.
    """

    def __init__(self, X: np.ndarray, train: np.ndarray, test: np.ndarray) -> None:
        Xtr = np.asarray(X[train], dtype=np.float64)
        Xte = np.asarray(X[test], dtype=np.float64)
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0                      # sklearn's StandardScaler convention
        A = (Xtr - mu) / sd
        B = (Xte - mu) / sd
        K = A @ A.T
        # symmetric PSD: eigh is stable and gives the full spectrum in one shot
        s, U = np.linalg.eigh(K)
        s = np.clip(s, 0.0, None)
        self._s = s
        self._U = U
        self._MU = (B @ A.T) @ U               # [n_test, n_train]
        self.train = np.asarray(train)
        self.test = np.asarray(test)

    def predict(self, y: np.ndarray, alpha: float) -> np.ndarray:
        """Predictions on ``test`` for outcome vector ``y`` (indexed like ``X``)."""
        ytr = np.asarray(y, dtype=np.float64)[self.train]
        ybar = ytr.mean()
        v = self._U.T @ (ytr - ybar)
        return self._MU @ (v / (self._s + alpha)) + ybar

    def predict_many(self, y: np.ndarray, alphas: tuple[float, ...] = ALPHAS) -> dict[float, np.ndarray]:
        ytr = np.asarray(y, dtype=np.float64)[self.train]
        ybar = ytr.mean()
        v = self._U.T @ (ytr - ybar)
        return {a: self._MU @ (v / (self._s + a)) + ybar for a in alphas}


def check_against_sklearn(seed: int = 0, n: int = 128, d: int = 768, tol: float = 1e-7) -> dict[str, float]:
    """Assert :class:`RidgeFold` reproduces sklearn's standardised Ridge.

    Returns the worst absolute deviation over the alpha grid so the number can
    be recorded rather than merely asserted.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[:, 3] = 2.5                                    # a constant column, as real pools have
    y = X[:, :5] @ rng.normal(size=5) + rng.normal(size=n)
    tr = np.arange(0, 96)
    te = np.arange(96, n)
    fold = RidgeFold(X, tr, te)
    worst = 0.0
    for a in ALPHAS:
        ref = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X[tr], y[tr]).predict(X[te])
        got = fold.predict(y, a)
        worst = max(worst, float(np.max(np.abs(ref - got))))
    if not worst < tol:
        raise AssertionError(f"RidgeFold deviates from sklearn by {worst:.3e} (tol {tol:.0e})")
    return {"max_abs_deviation": worst, "tol": tol, "n": n, "d": d}
