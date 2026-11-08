"""Shared helpers for tuning, refitting and evaluating models from a frozen configuration."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import content as content_mod
from .folds import Fold, TargetDef, build_fold, recent_positive_counts
from .models import ALS, EASE, ContentCentroid, ContentToCF, Hybrid, Popularity, Recommender


ALL_TIME_DAYS = 40000  # longer than the dataset, i.e. no recency window


def load_processed(cfg: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, int]:
    p = cfg["paths"]["processed_dir"]
    interactions = pl.read_parquet(p / "interactions.parquet")
    items = pl.read_parquet(p / "items.parquet")
    emb = np.load(p / "synopsis_embeddings.npy")
    return interactions, items, emb, int(interactions["user_idx"].max()) + 1


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def target_from(d: dict[str, Any]) -> TargetDef:
    return TargetDef(
        threshold=int(d.get("threshold", 7)),
        use_user_mean=bool(d.get("use_user_mean", True)),
        any_completed=bool(d.get("any_completed", False)),
    )


def validation_fold(
    cfg: dict[str, Any], interactions: pl.DataFrame, items: pl.DataFrame, n_users: int, **kw: Any
) -> Fold:
    """Fit on train, score on validation. Test-window rows are removed before anything else."""
    s = cfg["split"]
    visible = interactions.filter(pl.col("ts") < dt(s["test_start"]))
    return build_fold(visible, items, n_users, dt(s["val_start"]), dt(s["val_start"]), dt(s["test_start"]), **kw)


def test_fold(
    cfg: dict[str, Any], interactions: pl.DataFrame, items: pl.DataFrame, n_users: int, **kw: Any
) -> Fold:
    """Fit on train + validation, score on the held-out test window."""
    s = cfg["split"]
    return build_fold(interactions, items, n_users, dt(s["test_start"]), dt(s["test_start"]), dt(s["test_end"]), **kw)


def fold_kwargs(frozen: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "train_target": target_from(frozen["train_target"]),
        "eval_target": target_from(cfg["target"]),
        "dropped_weight": float(frozen["dropped_weight"]),
        "decay_days": frozen["decay_days"],
    }


def item_vectors(
    items: pl.DataFrame, emb: np.ndarray, cfg: dict[str, Any], weights: dict[str, Any] | None = None
) -> np.ndarray:
    blocks = content_mod.structured_blocks(items, cfg["content"]["top_studios"], cfg["content"]["year_bucket"])
    return content_mod.build_item_vectors(emb, blocks, weights or cfg["content"]["block_weights"])


class CachedArm:
    """Memoise a model's scores per history row, so hybrid grids avoid re-scoring."""

    def __init__(self, model: Recommender, hist: sp.csr_matrix) -> None:
        self.scores = np.asarray(model.score(hist), dtype=np.float32)
        self.index = {self._key(hist, r): r for r in range(hist.shape[0])}

    @staticmethod
    def _key(hist: sp.csr_matrix, r: int) -> bytes:
        a, b = hist.indptr[r], hist.indptr[r + 1]
        return hist.indices[a:b].tobytes() + hist.data[a:b].tobytes()

    def score(self, hist: sp.csr_matrix) -> np.ndarray:
        return self.scores[[self.index[self._key(hist, r)] for r in range(hist.shape[0])]]


def score_in_batches(model: Recommender, hist: sp.csr_matrix, batch: int = 1024) -> np.ndarray:
    return np.vstack([model.score(hist[i : i + batch]) for i in range(0, hist.shape[0], batch)])


def fit_models(
    fold: Fold,
    interactions: pl.DataFrame,
    vectors: np.ndarray,
    frozen: dict[str, Any],
) -> dict[str, Recommender]:
    """Fit every compared model on the fold's fit window using the frozen configuration."""
    # Baselines count titles by the label definition, whatever target the CF arms train on.
    label = target_from(frozen["label_target"])
    recent = recent_positive_counts(interactions, fold.fit_end, int(frozen["recent_days"]), fold.n_items, label)
    all_time = recent_positive_counts(interactions, fold.fit_end, ALL_TIME_DAYS, fold.n_items, label)
    ease = EASE(l2=float(frozen["ease_l2"])).fit(fold.X)
    als = ALS(
        factors=int(frozen["als_factors"]),
        regularization=float(frozen["als_regularization"]),
        alpha=float(frozen["als_alpha"]),
        iterations=int(frozen["als_iterations"]),
        seed=int(frozen["seed"]),
    ).fit(fold.X)
    warm = fold.item_counts_cf > 0
    centroid = ContentCentroid(vectors)
    content_arms: dict[str, Recommender] = {"centroid": centroid}
    content_arms["content_to_cf"] = ContentToCF(vectors, als, l2=float(frozen["content_to_cf_l2"])).fit(warm)
    cf_arms: dict[str, Recommender] = {"ease": ease, "als": als}
    hybrid = Hybrid(
        cf=cf_arms[frozen["hybrid_cf"]],
        content=content_arms[frozen["hybrid_content"]],
        item_counts=fold.item_counts_cf,
        eligible=fold.eligible,
        kappa=float(frozen["hybrid_kappa"]),
        cold_bias=float(frozen["hybrid_cold_bias"]),
        pop_weight=float(frozen["hybrid_pop_weight"]),
        recent_counts=recent,
    )
    return {
        "popularity": Popularity(all_time),
        "recent_popularity": Popularity(recent),
        "ease": ease,
        "als": als,
        "content": content_arms[frozen["hybrid_content"]],
        "hybrid": hybrid,
    }
