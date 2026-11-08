from __future__ import annotations

import math

import numpy as np
import pytest

from animerec import metrics as M


def test_ndcg_perfect_and_empty():
    assert M.ndcg_at_k([1, 2, 3], [1, 2, 3], 3) == pytest.approx(1.0)
    assert M.ndcg_at_k([1, 2, 3], [], 3) == 0.0
    assert M.ndcg_at_k([4, 5], [1], 2) == 0.0


def test_ndcg_known_value():
    # Relevant item at rank 2, single relevant item: DCG = 1/log2(3), IDCG = 1.
    assert M.ndcg_at_k([9, 1, 8], [1], 3) == pytest.approx(1 / math.log2(3))


def test_ndcg_ideal_is_capped_at_k():
    assert M.ndcg_at_k([1, 2], [1, 2, 3, 4], 2) == pytest.approx(1.0)


def test_recall_precision():
    assert M.recall_at_k([1, 2, 3], [1, 9], 3) == 0.5
    assert M.precision_at_k([1, 2, 3, 4], [1, 2], 4) == 0.5


def test_coverage():
    assert M.catalog_coverage([1, 1, 2], 4) == 0.5
    with pytest.raises(ValueError):
        M.catalog_coverage([1], 0)


def test_diversity_identical_vs_orthogonal():
    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert M.intra_list_diversity([0, 1], v) == pytest.approx(0.0)
    assert M.intra_list_diversity([0, 2], v) == pytest.approx(1.0)


def test_novelty_prefers_rare_and_defines_cold():
    counts = np.array([100, 1, 0])
    assert M.novelty([1], counts, 100) > M.novelty([0], counts, 100)
    assert np.isfinite(M.novelty([2], counts, 100))


def test_bootstrap_ci_contains_true_mean_difference():
    rng = np.random.default_rng(0)
    base = rng.random(3000)
    model = base + 0.1
    out = M.bootstrap_lift_ci(model, base, n_boot=300, seed=1)
    assert out["diff_ci95"][0] == pytest.approx(0.1, abs=1e-9)
    assert out["lift_pct"] == pytest.approx(0.1 / base.mean() * 100)
    assert out["lift_pct_ci95"][0] <= out["lift_pct"] <= out["lift_pct_ci95"][1]


def test_bootstrap_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        M.bootstrap_lift_ci(np.ones(3), np.ones(4))
