"""Model behaviour, seen-item filtering, and cold-title handling."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from animerec.evaluation import rank_top_k, tiebreak_noise
from animerec.models import ALS, EASE, ContentCentroid, ContentToCF, Hybrid, Popularity, fold_in_als, zscore_rows
from animerec.serving import SparseEASE, truncate_matrix


def _toy_matrix(n_users=400, n_items=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for u in range(n_users):
        block = range(0, 6) if u % 2 == 0 else range(6, 12)
        for i in block:
            if rng.random() < 0.6:
                rows.append(u)
                cols.append(i)
    X = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_users, n_items))
    X.sort_indices()
    return X


def test_ease_diagonal_zero_and_learns_blocks():
    X = _toy_matrix()
    m = EASE(l2=10).fit(X)
    assert np.allclose(np.diag(m.B), 0)
    s = m.score(sp.csr_matrix(np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.float32))).ravel()
    assert s[2:6].min() > s[6:].max()


def test_ease_leaves_zero_interaction_items_at_zero():
    X = _toy_matrix().tolil()
    X[:, 11] = 0
    X = X.tocsr()
    X.eliminate_zeros()
    m = EASE(l2=10).fit(X)
    assert np.all(m.B[11] == 0) and np.all(m.B[:, 11] == 0)


def test_numpy_fold_in_matches_implicit():
    pytest.importorskip("implicit")
    X = _toy_matrix()
    m = ALS(factors=4, regularization=1.0, alpha=2.0, iterations=3).fit(X)
    from implicit.als import AlternatingLeastSquares

    ref = AlternatingLeastSquares(factors=4, regularization=1.0, alpha=2.0, use_gpu=False, use_native=True, use_cg=False)
    ref.item_factors = m.Y
    ref.user_factors = np.zeros((1, 4), np.float32)
    ref._YtY = None
    h = sp.csr_matrix(np.array([[1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]], np.float32))
    expected = ref.recalculate_user(0, h)
    got = fold_in_als(h, m.Y, m.YtY, 1.0, 2.0)
    assert np.allclose(got.ravel(), np.asarray(expected).ravel(), atol=1e-3)


def test_negative_history_lowers_related_scores():
    Y = np.eye(4, dtype=np.float32) + 0.05
    # Confidence must exceed 1 to differ from an unobserved title, hence alpha=5.
    als = ALS(factors=4, regularization=0.1, alpha=5.0, Y=Y, YtY=Y.T @ Y)
    pos = sp.csr_matrix(np.array([[1, 0, 0, 0]], np.float32))
    neg = sp.csr_matrix(np.array([[1, -1, 0, 0]], np.float32))
    assert als.score(neg)[0, 1] < als.score(pos)[0, 1]


def test_rank_top_k_never_returns_seen_or_ineligible():
    pop = Popularity(np.array([10, 9, 8, 7, 6, 5], float))
    hist = sp.csr_matrix(np.zeros((2, 6), np.float32))
    seen = sp.csr_matrix(np.array([[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], np.float32))
    eligible = np.array([1, 1, 1, 0, 1, 1], bool)
    top = rank_top_k(pop, hist, seen, eligible, 3, tiebreak_noise(6))
    assert top[0].tolist() == [2, 4, 5]
    assert top[1].tolist() == [0, 1, 2]


def test_rank_top_k_marks_short_lists():
    pop = Popularity(np.arange(3, 0, -1, dtype=float))
    top = rank_top_k(pop, sp.csr_matrix((1, 3), dtype=np.float32), sp.csr_matrix(np.array([[1, 1, 0]], np.float32)), np.ones(3, bool), 3, tiebreak_noise(3))
    assert top[0].tolist() == [2, -1, -1]


def test_zscore_ignores_ineligible_and_handles_constant_rows():
    z = zscore_rows(np.array([[1.0, 2.0, 100.0], [5.0, 5.0, 5.0]]), np.array([True, True, False]))
    assert z[0, 0] == pytest.approx(-1.0) and z[0, 1] == pytest.approx(1.0)
    assert np.allclose(z[1], 0)


class _Fixed:
    def __init__(self, s):
        self.s = np.asarray(s, np.float32)

    def score(self, hist):
        return np.tile(self.s, (hist.shape[0], 1))


def _hybrid(counts, cf, content, **kw):
    return Hybrid(_Fixed(cf), _Fixed(content), np.asarray(counts), np.ones(len(cf), bool), **kw)


def test_hybrid_alpha_is_zero_for_cold_and_near_one_for_established():
    h = _hybrid([0, 100000], [0, 1], [1, 0], kappa=20)
    assert h._alpha[0] == 0.0
    assert h._alpha[1] > 0.99


def test_hybrid_shift_is_monotonic_in_history():
    h = _hybrid([0, 1, 10, 100, 1000], [0] * 5, [0] * 5, kappa=20)
    assert np.all(np.diff(h._alpha) > 0)


def test_hybrid_scores_cold_item_from_content_only():
    hist = sp.csr_matrix((1, 3), dtype=np.float32)
    content = [3.0, 1.0, 2.0]
    h = _hybrid([0, 50, 50], [9, 1, 2], content, kappa=5, cold_bias=0.5)
    z = zscore_rows(np.array([content]), np.ones(3, bool))
    assert h.score(hist)[0, 0] == pytest.approx(z[0, 0] + 0.5, rel=1e-5)


def test_cold_title_can_reach_top_k():
    h = _hybrid([0, 500, 500, 500], [0, 2, 1, 0], [5, 0, 0, 0], kappa=20, cold_bias=1.0)
    top = rank_top_k(h, sp.csr_matrix((1, 4), dtype=np.float32), sp.csr_matrix((1, 4), dtype=np.float32), np.ones(4, bool), 2, tiebreak_noise(4))
    assert 0 in top[0].tolist()


def test_content_centroid_scores_cold_item():
    C = np.array([[1, 0], [0.9, 0.1], [0, 1]], np.float32)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    s = ContentCentroid(C).score(sp.csr_matrix(np.array([[1, 0, 0]], np.float32))).ravel()
    assert s[1] > s[2]


def test_content_to_cf_predicts_for_unseen_items():
    C = np.random.default_rng(0).normal(size=(20, 5)).astype(np.float32)
    Y = C[:, :3].copy()
    als = ALS(factors=3, regularization=1.0, alpha=1.0, Y=Y, YtY=Y.T @ Y)
    warm = np.ones(20, bool)
    warm[-1] = False
    m = ContentToCF(C, als, l2=0.01).fit(warm)
    assert np.allclose(m.Y_hat[-1], Y[-1], atol=0.05)


def test_truncation_keeps_strongest_entries_and_scores_match():
    B = np.array([[0, 5, 1], [2, 0, 9], [7, 1, 0]], np.float32)
    S = truncate_matrix(B, 1)
    assert S.nnz == 3
    assert S[2, 0] == 7 and S[0, 1] == 5 and S[1, 2] == 9
    h = sp.csr_matrix(np.array([[1, 0, 0]], np.float32))
    assert SparseEASE(truncate_matrix(B, 3)).score(h).tolist() == [[0.0, 5.0, 1.0]]
