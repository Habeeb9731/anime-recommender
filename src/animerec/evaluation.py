"""Full-catalog top-K evaluation with per-user outputs for paired bootstrapping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import metrics as M
from .folds import Fold
from .models import Recommender

SEGMENTS = ["all", "little_history", "warm_titles", "cold_titles", "cold_titles_within_cold"]


def tiebreak_noise(n_items: int, seed: int = 42) -> np.ndarray:
    """Tiny fixed per-title noise so ties (e.g. all-zero popularity) break arbitrarily, not by id."""
    return np.random.default_rng(seed).random(n_items).astype(np.float32) * 1e-6


def rank_top_k(
    model: Recommender,
    hist: sp.csr_matrix,
    seen: sp.csr_matrix,
    candidates: np.ndarray,
    k: int,
    noise: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Return top-k item indices per row, drawn from ``candidates`` minus each row's seen titles."""
    out = np.full((hist.shape[0], k), -1, dtype=np.int32)
    blocked = ~candidates
    for start in range(0, hist.shape[0], batch_size):
        end = min(start + batch_size, hist.shape[0])
        s = np.array(model.score(hist[start:end]), dtype=np.float64) + noise
        s[:, blocked] = -np.inf
        sb = seen[start:end]
        rows = np.repeat(np.arange(end - start), np.diff(sb.indptr))
        s[rows, sb.indices] = -np.inf
        part = np.argpartition(-s, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(s, part, axis=1)
        order = np.argsort(-vals, axis=1, kind="stable")
        ranked = np.take_along_axis(part, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        ranked[~np.isfinite(vals)] = -1  # fewer than k candidates available
        out[start:end] = ranked
    return out


@dataclass
class EvalResult:
    users: np.ndarray
    top: np.ndarray
    per_user: dict[str, dict[str, np.ndarray]]  # segment -> metric -> aligned per-user array
    segment_users: dict[str, np.ndarray]  # segment -> user indices aligned with per_user arrays
    summary: dict[str, dict[str, float]]


def _per_user(top: np.ndarray, rel: list[np.ndarray], k: int) -> dict[str, np.ndarray]:
    n = len(rel)
    out = {"ndcg": np.zeros(n), "recall": np.zeros(n), "precision": np.zeros(n)}
    for j, r in enumerate(rel):
        items = [int(x) for x in top[j] if x >= 0]
        out["ndcg"][j] = M.ndcg_at_k(items, r, k)
        out["recall"][j] = M.recall_at_k(items, r, k)
        out["precision"][j] = M.precision_at_k(items, r, k)
    return out


def evaluate(
    model: Recommender,
    fold: Fold,
    item_vectors: np.ndarray,
    users: np.ndarray | None = None,
    k: int = 10,
    little_history_max: int = 5,
    seed: int = 42,
    with_segments: bool = True,
) -> EvalResult:
    users = fold.eval_users if users is None else users
    noise = tiebreak_noise(fold.n_items, seed)
    hist = fold.X[users]
    seen = fold.seen[users]
    top = rank_top_k(model, hist, seen, fold.eligible, k, noise)
    rel = [fold.relevant[int(u)] for u in users]

    per_user: dict[str, dict[str, np.ndarray]] = {"all": _per_user(top, rel, k)}
    seg_users: dict[str, np.ndarray] = {"all": users}

    if with_segments:
        hist_len = np.asarray(fold.seen.getnnz(axis=1)).ravel()[users]
        idx = np.flatnonzero(hist_len <= little_history_max)
        per_user["little_history"] = {m: v[idx] for m, v in per_user["all"].items()}
        seg_users["little_history"] = users[idx]

        for name, mask in (("warm_titles", ~fold.cold), ("cold_titles", fold.cold)):
            keep = [j for j, r in enumerate(rel) if mask[r].any()]
            sub = [rel[j][mask[rel[j]]] for j in keep]
            per_user[name] = _per_user(top[keep], sub, k) if keep else {m: np.zeros(0) for m in per_user["all"]}
            seg_users[name] = users[keep]

        keep = [j for j, r in enumerate(rel) if fold.cold[r].any()]
        cold_candidates = fold.eligible & fold.cold
        if keep and int(cold_candidates.sum()) > 0:
            k_c = min(k, int(cold_candidates.sum()))
            top_c = rank_top_k(model, hist[keep], seen[keep], cold_candidates, k_c, noise)
            sub = [rel[j][fold.cold[rel[j]]] for j in keep]
            per_user["cold_titles_within_cold"] = _per_user(top_c, sub, k)
            seg_users["cold_titles_within_cold"] = users[keep]

    n_cat = int(fold.eligible.sum())
    flat = top[top >= 0]
    div = [M.intra_list_diversity([int(x) for x in row if x >= 0], item_vectors) for row in top[: 5000]]
    nov = [M.novelty([int(x) for x in row if x >= 0], fold.item_counts_any, fold.n_users) for row in top[: 5000]]
    cold_share = float(fold.cold[flat].mean()) if flat.size else 0.0

    summary: dict[str, dict[str, float]] = {}
    for seg, vals in per_user.items():
        summary[seg] = {
            "n_users": int(vals["ndcg"].size),
            "ndcg@10": float(vals["ndcg"].mean()) if vals["ndcg"].size else float("nan"),
            "recall@10": float(vals["recall"].mean()) if vals["ndcg"].size else float("nan"),
            "precision@10": float(vals["precision"].mean()) if vals["ndcg"].size else float("nan"),
        }
    summary["beyond_accuracy"] = {
        "catalog_coverage": M.catalog_coverage(flat.tolist(), n_cat),
        "diversity": float(np.mean(div)),
        "novelty_bits": float(np.mean(nov)),
        "share_of_rec_slots_cold": cold_share,
        "eligible_catalog": n_cat,
    }
    return EvalResult(users=users, top=top, per_user=per_user, segment_users=seg_users, summary=summary)
