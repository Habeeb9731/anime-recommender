"""Stage 3: evaluate the frozen models once on the held-out test window."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import numpy as np
import polars as pl

from animerec import pipeline
from animerec.evaluation import evaluate
from animerec.metrics import bootstrap_lift_ci
from animerec.models import ALS, EASE, ContentCentroid, ContentToCF, Hybrid, Popularity, Recommender
from animerec.utils import get_logger, load_config, read_json, set_seed, write_json

log = get_logger("evaluate")


def load_models(m_dir, fold, frozen: dict[str, Any]) -> dict[str, Recommender]:
    vec = np.load(m_dir / "content_vectors.npy")
    ease = EASE(l2=float(frozen["ease_l2"]), B=np.load(m_dir / "ease_B.npy"))
    Y = np.load(m_dir / "als_Y.npy")
    als = ALS(factors=Y.shape[1], regularization=float(frozen["als_regularization"]), alpha=float(frozen["als_alpha"]), Y=Y, YtY=Y.T @ Y)
    if frozen["hybrid_content"] == "centroid":
        content: Recommender = ContentCentroid(vec)
    else:
        c = ContentToCF(vec, als)
        c.Y_hat = np.load(m_dir / "content_Y_hat.npy")
        content = c
    recent = np.load(m_dir / "recent_counts.npy")
    hybrid = Hybrid(
        cf={"ease": ease, "als": als}[frozen["hybrid_cf"]], content=content, item_counts=fold.item_counts_cf,
        eligible=fold.eligible, kappa=float(frozen["hybrid_kappa"]), cold_bias=float(frozen["hybrid_cold_bias"]),
        pop_weight=float(frozen["hybrid_pop_weight"]), recent_counts=recent,
    )
    return {
        "popularity": Popularity(np.load(m_dir / "popularity_counts.npy")),
        "recent_popularity": Popularity(recent),
        "ease": ease,
        "als": als,
        "content": content,
        "hybrid": hybrid,
    }


def failure_analysis(results, fold, items: pl.DataFrame) -> dict[str, Any]:
    hyb, pop, rec = (results[m].per_user["all"]["ndcg"] for m in ("hybrid", "popularity", "recent_popularity"))
    def outcome(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        return {"better": float((a > b).mean()), "tie": float((a == b).mean()), "worse": float((a < b).mean())}

    users = results["hybrid"].users
    hist = np.asarray(fold.seen.getnnz(axis=1)).ravel()[users]
    buckets = []
    for lo, hi in [(1, 5), (6, 20), (21, 100), (101, 500), (501, 10**9)]:
        m = (hist >= lo) & (hist <= hi)
        if not m.any():
            continue
        buckets.append(
            {
                "history": f"{lo}-{hi}" if hi < 10**9 else f"{lo}+",
                "n_users": int(m.sum()),
                "hybrid_ndcg@10": float(hyb[m].mean()),
                "popularity_ndcg@10": float(pop[m].mean()),
                "recent_popularity_ndcg@10": float(rec[m].mean()),
            }
        )

    top = results["hybrid"].top
    relevant_counts: Counter[int] = Counter()
    hit_counts: Counter[int] = Counter()
    cold_hit, cold_rel, cold_text_hit, cold_text_rel = 0, 0, 0, 0
    has_text = items["has_synopsis"].to_numpy()
    for j, u in enumerate(users):
        rel = fold.relevant[int(u)]
        rec_set = set(top[j].tolist())
        for it in rel.tolist():
            relevant_counts[it] += 1
            hit = it in rec_set
            hit_counts[it] += int(hit)
            if fold.cold[it]:
                cold_rel += 1
                cold_hit += int(hit)
                if has_text[it]:
                    cold_text_rel += 1
                    cold_text_hit += int(hit)
    titles = items["title"].to_list()
    missed = sorted(relevant_counts, key=lambda i: relevant_counts[i] - hit_counts[i], reverse=True)[:10]
    cold_no_text_rel = cold_rel - cold_text_rel
    return {
        "hybrid_vs_popularity_user_outcomes": outcome(hyb, pop),
        "hybrid_vs_recent_popularity_user_outcomes": outcome(hyb, rec),
        "by_history_length": buckets,
        "most_missed_titles": [
            {"title": titles[i], "relevant_users": relevant_counts[i], "recommended_to_those_users": hit_counts[i], "cold": bool(fold.cold[i])}
            for i in missed
        ],
        "cold_relevant_pairs": cold_rel,
        "cold_hit_rate_with_synopsis": cold_text_hit / cold_text_rel if cold_text_rel else None,
        "cold_hit_rate_without_synopsis": (cold_hit - cold_text_hit) / cold_no_text_rel if cold_no_text_rel else None,
        "cold_pairs_without_synopsis": cold_no_text_rel,
    }


def main(config: str | None = None) -> None:
    cfg = load_config(config)
    set_seed(cfg["seed"])
    p = cfg["paths"]
    m_dir = p["artifacts_dir"] / "models"
    frozen = read_json(p["artifacts_dir"] / "frozen_config.json")
    meta = read_json(m_dir / "metadata.json")
    gen = p["reports_dir"] / "generated"

    lock = gen / "test_run_lock.json"
    if lock.exists() and read_json(lock)["config_hash"] != meta["config_hash"]:
        raise SystemExit("The test set was already evaluated with a different frozen configuration; refusing to re-evaluate.")

    interactions, items, emb, n_users = pipeline.load_processed(cfg)
    fold = pipeline.test_fold(cfg, interactions, items, n_users, **pipeline.fold_kwargs(frozen, cfg))
    vec_cfg = pipeline.item_vectors(items, emb, cfg)
    models = load_models(m_dir, fold, frozen)
    log.info("Test: %d users, %d eligible titles (%d cold)", fold.eval_users.size, fold.eligible.sum(), (fold.eligible & fold.cold).sum())

    k, lh = cfg["evaluation"]["k"], cfg["evaluation"]["little_history_max"]
    results = {}
    summaries = {}
    for name, m in models.items():
        results[name] = evaluate(m, fold, vec_cfg, k=k, little_history_max=lh, seed=cfg["seed"])
        summaries[name] = results[name].summary
        log.info("test %-18s %s", name, json.dumps(summaries[name]["all"]))

    n_boot = cfg["evaluation"]["bootstrap_samples"]
    ci: dict[str, Any] = {}
    for seg in results["hybrid"].per_user:
        ci[seg] = {
            base: bootstrap_lift_ci(results["hybrid"].per_user[seg]["ndcg"], results[base].per_user[seg]["ndcg"], n_boot, cfg["seed"])
            for base in ("popularity", "recent_popularity")
        }

    zero_hist: dict[str, Any] = {"n_users": int(fold.zero_history_users.size)}
    if fold.zero_history_users.size:
        z = {n: evaluate(models[n], fold, vec_cfg, users=fold.zero_history_users, k=k, with_segments=False, seed=cfg["seed"]) for n in ("hybrid", "popularity", "recent_popularity")}
        zero_hist.update({n: r.summary["all"] for n, r in z.items()})
        zero_hist["hybrid_vs_popularity"] = bootstrap_lift_ci(z["hybrid"].per_user["all"]["ndcg"], z["popularity"].per_user["all"]["ndcg"], n_boot, cfg["seed"])

    write_json(
        gen / "test_results.json",
        {
            "config_hash": meta["config_hash"],
            "frozen": frozen,
            "fold": {
                "eval_users": int(fold.eval_users.size),
                "eligible_titles": int(fold.eligible.sum()),
                "cold_eligible_titles": int((fold.eligible & fold.cold).sum()),
                "positives_excluded_ineligible_or_seen": fold.n_positive_dropped_ineligible,
            },
            "models": summaries,
            "bootstrap": ci,
            "zero_history_users": zero_hist,
            "failure_analysis": failure_analysis(results, fold, items),
        },
    )
    np.savez_compressed(
        p["artifacts_dir"] / "test_per_user_ndcg.npz",
        users=results["hybrid"].users,
        **{m: r.per_user["all"]["ndcg"] for m, r in results.items()},
    )
    write_json(lock, {"config_hash": meta["config_hash"]})
    log.info("Hybrid vs popularity: %s", ci["all"]["popularity"])
    log.info("Hybrid vs recent popularity: %s", ci["all"]["recent_popularity"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
