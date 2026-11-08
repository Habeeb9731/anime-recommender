"""Ranking and beyond-accuracy metrics, written as small pure functions."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def dcg_gains(k: int) -> np.ndarray:
    """Binary-relevance discount for ranks 1..k."""
    return 1.0 / np.log2(np.arange(2, k + 2))


def ndcg_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int = 10) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    gains = dcg_gains(k)
    dcg = sum(gains[r] for r, item in enumerate(list(recommended)[:k]) if item in relevant)
    ideal = gains[: min(len(relevant), k)].sum()
    return float(dcg / ideal)


def recall_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int = 10) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(relevant.intersection(list(recommended)[:k])) / len(relevant)


def precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int = 10) -> float:
    relevant = set(relevant)
    return len(relevant.intersection(list(recommended)[:k])) / k


def catalog_coverage(all_recommended: Iterable[int], catalog_size: int) -> float:
    if catalog_size <= 0:
        raise ValueError("catalog_size must be positive")
    return len(set(all_recommended)) / catalog_size


def intra_list_diversity(items: Sequence[int], item_vectors: np.ndarray) -> float:
    """1 - mean pairwise cosine similarity. Vectors must be L2-normalised."""
    items = list(items)
    if len(items) < 2:
        return 0.0
    v = item_vectors[items]
    sim = v @ v.T
    n = len(items)
    return float(1.0 - (sim.sum() - np.trace(sim)) / (n * (n - 1)))


def novelty(items: Sequence[int], item_counts: np.ndarray, n_users: int) -> float:
    """Mean self-information -log2(p(i)), add-one smoothed so cold titles are defined."""
    items = list(items)
    if not items:
        return 0.0
    p = (item_counts[items] + 1.0) / (n_users + 1.0)
    return float(np.mean(-np.log2(p)))


def bootstrap_lift_ci(
    model_scores: np.ndarray,
    baseline_scores: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    chunk: int = 100,
) -> dict:
    """Paired user-level bootstrap for the mean difference and the percent lift.

    Users are resampled together across both models, so the interval reflects the
    paired comparison.
    """
    model_scores = np.asarray(model_scores, dtype=np.float64)
    baseline_scores = np.asarray(baseline_scores, dtype=np.float64)
    if model_scores.shape != baseline_scores.shape:
        raise ValueError("score arrays must be aligned per user")
    n = model_scores.size
    if n == 0:
        return {"n_users": 0}
    rng = np.random.default_rng(seed)
    means_m, means_b = [], []
    remaining = n_boot
    while remaining > 0:
        batch = min(chunk, remaining)
        idx = rng.integers(0, n, size=(batch, n))
        means_m.append(model_scores[idx].mean(axis=1))
        means_b.append(baseline_scores[idx].mean(axis=1))
        remaining -= batch
    bm, bb = np.concatenate(means_m), np.concatenate(means_b)
    diff = bm - bb
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = diff / bb * 100.0
    lift = lift[np.isfinite(lift)]
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    base_mean = float(baseline_scores.mean())
    return {
        "n_users": int(n),
        "n_boot": int(n_boot),
        "model_mean": float(model_scores.mean()),
        "baseline_mean": base_mean,
        "diff_mean": float(model_scores.mean() - base_mean),
        "diff_ci95": [float(x) for x in np.percentile(diff, q)],
        "lift_pct": float((model_scores.mean() - base_mean) / base_mean * 100.0) if base_mean > 0 else None,
        "lift_pct_ci95": [float(x) for x in np.percentile(lift, q)] if lift.size else None,
    }
