"""Recommenders. Every model scores from a sparse history matrix, so training users and
brand-new demo visitors go through the same fold-in code path."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp


class Recommender(Protocol):
    def score(self, hist: sp.csr_matrix) -> np.ndarray:
        """Return an (n_users, n_items) score matrix for signed-history rows."""


def zscore_rows(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score each row using statistics from ``mask`` items only. Constant rows become 0."""
    s = scores[:, mask]
    mean = s.mean(axis=1, keepdims=True)
    std = s.std(axis=1, keepdims=True)
    std[std < 1e-9] = 1.0
    return (scores - mean) / std


@dataclass
class Popularity:
    """Non-personalised baseline: item counts (all-time or windowed)."""

    counts: np.ndarray

    def score(self, hist: sp.csr_matrix) -> np.ndarray:
        return np.tile(self.counts.astype(np.float32), (hist.shape[0], 1))


@dataclass
class EASE:
    """Closed-form linear autoencoder (Steck, 2019) on signed confidences."""

    l2: float = 500.0
    B: np.ndarray | None = None

    def fit(self, X: sp.csr_matrix) -> "EASE":
        n_items = X.shape[1]
        active = np.flatnonzero(np.asarray(X.getnnz(axis=0)) > 0)
        Xa = X[:, active]
        G = np.asarray((Xa.T @ Xa).todense(), dtype=np.float64)
        G[np.diag_indices_from(G)] += self.l2
        P = np.linalg.inv(G)
        Ba = P / (-np.diag(P))[:, None].T
        Ba[np.diag_indices_from(Ba)] = 0.0
        B = np.zeros((n_items, n_items), dtype=np.float32)
        B[np.ix_(active, active)] = Ba.astype(np.float32)
        self.B = B
        return self

    def score(self, hist: sp.csr_matrix) -> np.ndarray:
        return np.asarray(hist @ self.B, dtype=np.float32)


def fold_in_als(
    hist: sp.csr_matrix, Y: np.ndarray, YtY: np.ndarray, regularization: float, alpha: float
) -> np.ndarray:
    """Weighted ridge fold-in, matching ``implicit``'s least-squares user update.

    Dropped titles carry negative confidence: target 0 with confidence |c|. This is plain
    NumPy so the demo does not need the ``implicit`` package at serving time.
    """
    k = Y.shape[1]
    out = np.zeros((hist.shape[0], k), dtype=np.float32)
    base = YtY + regularization * np.eye(k)
    for u in range(hist.shape[0]):
        idx = hist.indices[hist.indptr[u] : hist.indptr[u + 1]]
        if idx.size == 0:
            continue
        c = hist.data[hist.indptr[u] : hist.indptr[u + 1]] * alpha
        Yu = Y[idx]
        A = base + (Yu.T * (np.abs(c) - 1.0)) @ Yu  # YtY already carries weight 1 for each title
        b = Yu[c > 0].T @ c[c > 0]
        out[u] = np.linalg.solve(A, b)
    return out


@dataclass
class ALS:
    """Implicit-feedback matrix factorisation (Hu et al., 2008) via the ``implicit`` package."""

    factors: int = 128
    regularization: float = 10.0
    alpha: float = 10.0
    iterations: int = 15
    seed: int = 42
    Y: np.ndarray | None = None
    YtY: np.ndarray | None = None

    def fit(self, X: sp.csr_matrix) -> "ALS":
        from implicit.als import AlternatingLeastSquares

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=self.seed,
            use_gpu=False,
            num_threads=0,
        )
        model.fit(X.astype(np.float32), show_progress=False)
        self.Y = np.asarray(model.item_factors, dtype=np.float32)
        self.YtY = self.Y.T @ self.Y
        return self

    def user_factors(self, hist: sp.csr_matrix) -> np.ndarray:
        return fold_in_als(hist, self.Y, self.YtY, self.regularization, self.alpha)

    def score(self, hist: sp.csr_matrix) -> np.ndarray:
        return self.user_factors(hist) @ self.Y.T


@dataclass
class ContentCentroid:
    """Cosine similarity between each title and the signed mean of the user's history."""

    C: np.ndarray

    def score(self, hist: sp.csr_matrix) -> np.ndarray:
        prof = np.asarray(hist @ self.C, dtype=np.float32)
        norm = np.linalg.norm(prof, axis=1, keepdims=True)
        norm[norm < 1e-9] = 1.0
        return (prof / norm) @ self.C.T


@dataclass
class ContentToCF:
    """Ridge map from content features to ALS item factors (usable on zero-history titles)."""

    C: np.ndarray
    als: ALS
    l2: float = 10.0
    Y_hat: np.ndarray | None = None

    def fit(self, warm_mask: np.ndarray) -> "ContentToCF":
        C, Y = self.C[warm_mask], self.als.Y[warm_mask]
        A = C.T @ C + self.l2 * np.eye(C.shape[1])
        W = np.linalg.solve(A, C.T @ Y)
        self.Y_hat = (self.C @ W).astype(np.float32)
        return self

    def score(self, hist: sp.csr_matrix) -> np.ndarray:
        return self.als.user_factors(hist) @ self.Y_hat.T


@dataclass
class Hybrid:
    """Blend whose content share grows smoothly as a title's training history shrinks.

    ``alpha_i = n_i / (n_i + kappa)``, so ``alpha`` is 0 for a title with no training
    interactions and approaches 1 for established titles::

        s = alpha * (z(cf) + pop_weight * z(recent_pop)) + (1 - alpha) * (z(content) + cold_bias)

    Popularity sits inside the alpha term because a title with no history has no popularity
    evidence, and should not be penalised for that.
    """

    cf: Recommender
    content: Recommender
    item_counts: np.ndarray
    eligible: np.ndarray
    kappa: float = 20.0
    cold_bias: float = 0.0
    pop_weight: float = 0.0
    recent_counts: np.ndarray | None = None
    _alpha: np.ndarray = field(init=False, repr=False)
    _pop_z: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = self.item_counts.astype(np.float32)
        self._alpha = n / (n + self.kappa)
        if self.recent_counts is None:
            self._pop_z = np.zeros(n.shape, dtype=np.float32)
        else:
            p = np.log1p(self.recent_counts.astype(np.float32))
            s = p[self.eligible]
            self._pop_z = (p - s.mean()) / (s.std() + 1e-9)

    def score(self, hist: sp.csr_matrix) -> np.ndarray:
        z_cf = zscore_rows(self.cf.score(hist), self.eligible)
        z_cb = zscore_rows(self.content.score(hist), self.eligible)
        a = self._alpha
        return a * (z_cf + self.pop_weight * self._pop_z) + (1 - a) * (z_cb + self.cold_bias)
