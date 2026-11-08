"""Stage 2: tune on validation, freeze, refit on train+validation, export artifacts.

This stage never loads test-window rows: they are removed right after reading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp

from animerec import data, pipeline, tuning
from animerec.evaluation import evaluate
from animerec.metrics import bootstrap_lift_ci
from animerec.models import Hybrid, Recommender
from animerec.serving import SparseEASE, truncate_matrix
from animerec.utils import get_logger, load_config, set_seed, write_json

log = get_logger("train")
logging.getLogger("httpx2").setLevel(logging.WARNING)


def config_hash(frozen: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]


def validation_report(cfg, interactions, items, emb, n_users, frozen) -> dict[str, Any]:
    """Compare all models on every validation user, then run threshold sensitivity and bundle parity."""
    kw = pipeline.fold_kwargs(frozen, cfg)
    fold = pipeline.validation_fold(cfg, interactions, items, n_users, **kw)
    vec_cfg = pipeline.item_vectors(items, emb, cfg)
    vec = pipeline.item_vectors(items, emb, cfg, frozen["block_weights"])
    visible = interactions.filter(pl.col("ts") < pipeline.dt(cfg["split"]["test_start"]))
    models = pipeline.fit_models(fold, visible, vec, frozen)
    res = {}
    results = {}
    for name, m in models.items():
        r = evaluate(m, fold, vec_cfg, k=cfg["evaluation"]["k"], little_history_max=cfg["evaluation"]["little_history_max"])
        results[name] = r
        res[name] = r.summary
        log.info("val %-18s ndcg@10=%.5f", name, r.summary["all"]["ndcg@10"])

    base = results["popularity"].per_user["all"]["ndcg"]
    hyb = results["hybrid"].per_user["all"]["ndcg"]
    out: dict[str, Any] = {
        "models": res,
        "hybrid_vs_popularity": bootstrap_lift_ci(hyb, base, cfg["evaluation"]["bootstrap_samples"], cfg["seed"]),
        "fold": {"eval_users": fold.eval_users.size, "eligible": int(fold.eligible.sum()), "cold_eligible": int((fold.eligible & fold.cold).sum())},
    }

    sens = []
    for td in cfg["tuning"]["train_targets"]:
        alt = pipeline.validation_fold(cfg, interactions, items, n_users, **{**kw, "eval_target": pipeline.target_from(td)})
        row = {"label_definition": td, "eval_users": int(alt.eval_users.size)}
        scores = {}
        for name in ("popularity", "hybrid"):
            scores[name] = evaluate(models[name], alt, vec_cfg, with_segments=False).summary["all"]["ndcg@10"]
        row.update(scores, lift_pct=(scores["hybrid"] - scores["popularity"]) / scores["popularity"] * 100)
        sens.append(row)
        log.info("label sensitivity %s", row)
    out["label_threshold_sensitivity"] = sens

    if frozen["hybrid_cf"] == "ease":
        h: Hybrid = models["hybrid"]
        full = h.cf
        h.cf = SparseEASE(truncate_matrix(full.B, cfg["demo"]["ease_neighbors"]))
        t = evaluate(h, fold, vec_cfg, with_segments=False).summary["all"]["ndcg@10"]
        h.cf = full
        out["demo_truncation_parity"] = {
            "full_ndcg@10": res["hybrid"]["all"]["ndcg@10"], "truncated_ndcg@10": t, "neighbors": cfg["demo"]["ease_neighbors"],
        }
        log.info("Truncation parity: %s", out["demo_truncation_parity"])
    return out


def export_demo(cfg, items, fold, models: dict[str, Recommender], vectors: np.ndarray, emb_dim: int, frozen: dict[str, Any], recent: np.ndarray) -> None:
    d = cfg["paths"]["demo_dir"]
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    from datetime import timedelta

    snapshot = pipeline.dt(cfg["split"]["test_start"])
    new_from = snapshot - timedelta(days=cfg["demo"]["new_title_days"])
    # Demo catalog is the set of titles released by the end of the data window.
    keep = np.flatnonzero(fold.eligible)
    pos = {int(old): new for new, old in enumerate(keep)}
    frame = items.with_row_index("_row").filter(pl.col("_row").is_in(keep.tolist())).drop("_row")
    frame = frame.with_columns(
        pl.coalesce([pl.col("title_english"), pl.col("title")]).alias("display_title"),
        pl.Series("is_cold", fold.cold[keep]),
        pl.col("synopsis").fill_null("").str.slice(0, 600).alias("synopsis_short"),
    ).with_columns((pl.col("is_cold") | (pl.col("start_date") >= new_from)).alias("is_new"))
    frame = frame.join(data.load_display_scores(cfg["paths"]["synopsis_zip"]), on="anime_id", how="left")
    frame = frame.select(["anime_id", "display_score", "title", "display_title", "genres", "year", "type", "franchise_id", "is_cold", "is_new", "synopsis_short", "cover_url"])
    frame.write_parquet(d / "items.parquet", compression="zstd")

    arrays: dict[str, np.ndarray] = {
        "vectors": vectors[keep].astype(np.float16),
        "item_counts_cf": fold.item_counts_cf[keep].astype(np.int32),
        "recent_counts": recent[keep].astype(np.int32),
    }
    h: Hybrid = models["hybrid"]
    als = models["als"]
    if frozen["hybrid_cf"] == "als" or frozen["hybrid_content"] == "content_to_cf":
        arrays["als_Y"] = als.Y[keep].astype(np.float16)
    if frozen["hybrid_content"] == "content_to_cf":
        arrays["content_Y_hat"] = models["content"].Y_hat[keep].astype(np.float16)
    if frozen["hybrid_cf"] == "ease":
        B = models["ease"].B[np.ix_(keep, keep)]
        sp.save_npz(d / "ease_B.npz", truncate_matrix(B, cfg["demo"]["ease_neighbors"]), compressed=True)
    np.savez_compressed(d / "arrays.npz", **arrays)

    meta = {
        "snapshot": snapshot.strftime("%B %Y"),
        "fit_end": str(fold.fit_end),
        "text_dim": emb_dim,
        "n_titles": int(keep.size),
        "frozen_config_hash": config_hash(frozen),
        "hybrid": {
            "cf": frozen["hybrid_cf"], "content": frozen["hybrid_content"], "kappa": frozen["hybrid_kappa"],
            "cold_bias": frozen["hybrid_cold_bias"], "pop_weight": frozen["hybrid_pop_weight"],
            "als_regularization": frozen["als_regularization"], "als_alpha": frozen["als_alpha"],
        },
    }
    write_json(d / "metadata.json", meta)
    size = sum(f.stat().st_size for f in d.iterdir())
    log.info("Demo bundle: %.1f MB, %d titles", size / 1e6, keep.size)
    _ = pos, h


def main(config: str | None = None) -> None:
    cfg = load_config(config)
    set_seed(cfg["seed"])
    p = cfg["paths"]
    interactions, items, emb, n_users = pipeline.load_processed(cfg)
    gen = p["reports_dir"] / "generated"

    # Everything below sees training and validation rows only.
    test_start = pipeline.dt(cfg["split"]["test_start"])
    visible = interactions.filter(pl.col("ts") < test_start)
    del interactions

    frozen, trace = tuning.run_tuning(cfg, visible, items, emb, n_users)
    write_json(p["artifacts_dir"] / "frozen_config.json", frozen)
    write_json(gen / "tuning_trace.json", trace)
    log.info("Frozen configuration %s: %s", config_hash(frozen), frozen)

    write_json(gen / "validation_results.json", validation_report(cfg, visible, items, emb, n_users, frozen))

    vec = pipeline.item_vectors(items, emb, cfg, frozen["block_weights"])
    fold = pipeline.test_fold(cfg, visible, items, n_users, **pipeline.fold_kwargs(frozen, cfg))
    models = pipeline.fit_models(fold, visible, vec, frozen)

    m_dir = p["artifacts_dir"] / "models"
    m_dir.mkdir(parents=True, exist_ok=True)
    recent = models["recent_popularity"].counts
    np.save(m_dir / "ease_B.npy", models["ease"].B)
    np.save(m_dir / "als_Y.npy", models["als"].Y)
    np.save(m_dir / "content_vectors.npy", vec)
    if frozen["hybrid_content"] == "content_to_cf":
        np.save(m_dir / "content_Y_hat.npy", models["content"].Y_hat)
    np.save(m_dir / "recent_counts.npy", recent)
    np.save(m_dir / "popularity_counts.npy", models["popularity"].counts)
    write_json(
        m_dir / "metadata.json",
        {
            "config_hash": config_hash(frozen),
            "frozen": frozen,
            "fit_end": str(fold.fit_end),
            "seed": cfg["seed"],
            "n_users": n_users,
            "n_items": fold.n_items,
            "fit_nnz": int(fold.X.nnz),
            "eligible_titles": int(fold.eligible.sum()),
            "cold_eligible_titles": int((fold.eligible & fold.cold).sum()),
        },
    )
    export_demo(cfg, items, fold, models, vec, emb.shape[1], frozen, recent)
    log.info("Training stage finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
