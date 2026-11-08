"""Coordinate-wise hyperparameter search on the validation window only."""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import polars as pl

from .evaluation import evaluate
from .folds import Fold, recent_positive_counts
from .models import ALS, EASE, ContentCentroid, ContentToCF, Hybrid, Popularity, Recommender
from .pipeline import CachedArm, item_vectors, target_from, validation_fold
from .utils import get_logger

log = get_logger(__name__)


def _ndcg(model: Recommender, fold: Fold, vec: np.ndarray, users: np.ndarray, k: int) -> float:
    return evaluate(model, fold, vec, users=users, k=k, with_segments=False).summary["all"]["ndcg@10"]


def _pick(
    trace: list[dict[str, Any]], stage: str, grid: list[Any], fn: Callable[[Any], float]
) -> Any:
    best, best_v = None, -1.0
    for v in grid:
        t = time.time()
        score = fn(v)
        trace.append({"stage": stage, "value": v, "val_ndcg@10": score, "seconds": round(time.time() - t, 1)})
        log.info("%-22s %-55s ndcg@10=%.5f (%.0fs)", stage, str(v), score, time.time() - t)
        if score > best_v:
            best, best_v = v, score
    return best


def run_tuning(
    cfg: dict[str, Any], interactions: pl.DataFrame, items: pl.DataFrame, emb: np.ndarray, n_users: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t = cfg["tuning"]
    k = cfg["evaluation"]["k"]
    seed = cfg["seed"]
    trace: list[dict[str, Any]] = []
    vec = item_vectors(items, emb, cfg)
    eval_target = target_from(cfg["target"])

    def fold_of(**kw: Any) -> Fold:
        return validation_fold(cfg, interactions, items, n_users, eval_target=eval_target, **kw)

    base = fold_of()
    rng = np.random.default_rng(seed)
    cap = cfg["evaluation"]["val_user_cap"]
    users = np.sort(rng.choice(base.eval_users, size=min(cap, base.eval_users.size), replace=False))
    frozen: dict[str, Any] = {"seed": seed, "label_target": dict(cfg["target"])}

    pop_score = _ndcg(Popularity(base.pos_counts), base, vec, users, k)
    trace.append({"stage": "popularity", "value": None, "val_ndcg@10": pop_score})
    log.info("popularity baseline ndcg@10=%.5f", pop_score)

    def recent_counts(days: int, fold: Fold) -> np.ndarray:
        return recent_positive_counts(interactions.filter(pl.col("ts") < fold.fit_end), fold.fit_end, days, fold.n_items, eval_target)

    frozen["recent_days"] = _pick(
        trace, "recent_days", t["recent_days"],
        lambda d: _ndcg(Popularity(recent_counts(d, base)), base, vec, users, k),
    )

    l2_probe = 300.0
    frozen["train_target"] = _pick(
        trace, "train_target", t["train_targets"],
        lambda d: (lambda f: _ndcg(EASE(l2_probe).fit(f.X), f, vec, users, k))(fold_of(train_target=target_from(d), dropped_weight=0.0)),
    )
    tgt = target_from(frozen["train_target"])

    frozen["dropped_weight"] = _pick(
        trace, "dropped_weight", t["dropped_weight"],
        lambda d: (lambda f: _ndcg(EASE(l2_probe).fit(f.X), f, vec, users, k))(fold_of(train_target=tgt, dropped_weight=d)),
    )
    frozen["decay_days"] = _pick(
        trace, "decay_days", t["decay_days"],
        lambda d: (lambda f: _ndcg(EASE(l2_probe).fit(f.X), f, vec, users, k))(
            fold_of(train_target=tgt, dropped_weight=frozen["dropped_weight"], decay_days=d)
        ),
    )
    fold = fold_of(train_target=tgt, dropped_weight=frozen["dropped_weight"], decay_days=frozen["decay_days"])

    frozen["ease_l2"] = _pick(trace, "ease_l2", t["ease_l2"], lambda v: _ndcg(EASE(float(v)).fit(fold.X), fold, vec, users, k))

    iters = int(t["als_iterations"])
    frozen["als_iterations"] = iters

    def als(factors: int, reg: float, alpha: float) -> ALS:
        return ALS(factors=factors, regularization=reg, alpha=alpha, iterations=iters, seed=seed).fit(fold.X)

    frozen["als_factors"] = _pick(trace, "als_factors", t["als_factors"], lambda v: _ndcg(als(int(v), 10.0, 10.0), fold, vec, users, k))
    frozen["als_regularization"] = _pick(
        trace, "als_regularization", t["als_regularization"],
        lambda v: _ndcg(als(frozen["als_factors"], float(v), 10.0), fold, vec, users, k),
    )
    frozen["als_alpha"] = _pick(
        trace, "als_alpha", t["als_alpha"],
        lambda v: _ndcg(als(frozen["als_factors"], frozen["als_regularization"], float(v)), fold, vec, users, k),
    )

    frozen["block_weights"] = _pick(
        trace, "content_block_weights", t["content_block_weights"],
        lambda w: _ndcg(ContentCentroid(item_vectors(items, emb, cfg, w)), fold, vec, users, k),
    )
    vec_best = item_vectors(items, emb, cfg, frozen["block_weights"])

    ease = EASE(float(frozen["ease_l2"])).fit(fold.X)
    best_als = als(frozen["als_factors"], frozen["als_regularization"], frozen["als_alpha"])
    warm = fold.item_counts_cf > 0
    frozen["content_to_cf_l2"] = _pick(
        trace, "content_to_cf_l2", t["content_to_cf_l2"],
        lambda v: _ndcg(ContentToCF(vec_best, best_als, l2=float(v)).fit(warm), fold, vec, users, k),
    )

    recent = recent_counts(int(frozen["recent_days"]), fold)
    hist = fold.X[users]
    arms: dict[str, CachedArm] = {
        "ease": CachedArm(ease, hist),
        "als": CachedArm(best_als, hist),
        "centroid": CachedArm(ContentCentroid(vec_best), hist),
        "content_to_cf": CachedArm(ContentToCF(vec_best, best_als, l2=float(frozen["content_to_cf_l2"])).fit(warm), hist),
    }
    del ease

    def hybrid(p: dict[str, Any]) -> dict[str, dict[str, float]]:
        m = Hybrid(
            cf=arms[p["cf"]], content=arms[p["content"]], item_counts=fold.item_counts_cf, eligible=fold.eligible,
            kappa=p["kappa"], cold_bias=p["bias"], pop_weight=p["pop"], recent_counts=recent,
        )
        return evaluate(m, fold, vec, users=users, k=k).summary

    params: dict[str, Any] = {"cf": "ease", "content": "centroid", "kappa": 20.0, "bias": 0.0, "pop": 1.0}
    grids: dict[str, list[Any]] = {
        "cf": ["ease", "als"],
        "content": ["centroid", "content_to_cf"],
        "kappa": t["hybrid_kappa"],
        "bias": t["hybrid_cold_bias"],
        "pop": t["hybrid_pop_weight"],
    }
    # Overall NDCG barely reacts to cold titles, so among configurations within 1% of the
    # best overall NDCG, keep the one that ranks cold titles best.
    seen_configs: list[dict[str, Any]] = []
    for sweep in range(2):
        for name, grid in grids.items():
            best_v, best_s = params[name], -1.0
            for v in grid:
                p = {**params, name: v}
                s = hybrid(p)
                row = {
                    "stage": f"hybrid_{name}", "sweep": sweep, "value": v, "params": p,
                    "val_ndcg@10": s["all"]["ndcg@10"],
                    "val_cold_ndcg@10": s.get("cold_titles", {}).get("ndcg@10", 0.0),
                    "cold_share": s["beyond_accuracy"]["share_of_rec_slots_cold"],
                }
                trace.append(row)
                seen_configs.append(row)
                log.info("hybrid %-8s %-16s ndcg=%.5f cold_ndcg=%.5f cold_slots=%.3f", name, v, row["val_ndcg@10"], row["val_cold_ndcg@10"], row["cold_share"])
                if row["val_ndcg@10"] > best_s:
                    best_v, best_s = v, row["val_ndcg@10"]
            params[name] = best_v

    best_overall = max(r["val_ndcg@10"] for r in seen_configs)
    eligible_rows = [r for r in seen_configs if r["val_ndcg@10"] >= 0.99 * best_overall]
    chosen = max(eligible_rows, key=lambda r: (round(r["val_cold_ndcg@10"], 6), r["val_ndcg@10"]))
    trace.append({"stage": "hybrid_selected", "value": chosen["params"], "val_ndcg@10": chosen["val_ndcg@10"], "val_cold_ndcg@10": chosen["val_cold_ndcg@10"], "best_overall_ndcg@10": best_overall})
    log.info("Selected %s", chosen)
    p = chosen["params"]
    frozen.update(
        hybrid_cf=p["cf"], hybrid_content=p["content"], hybrid_kappa=float(p["kappa"]),
        hybrid_cold_bias=float(p["bias"]), hybrid_pop_weight=float(p["pop"]),
    )
    return frozen, trace
