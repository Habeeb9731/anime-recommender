"""Stage 1: download, clean, sample, index, EDA, and synopsis embeddings."""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import requests

from animerec import content, data
from animerec.charts import run_eda
from animerec.utils import get_logger, load_config, set_seed, write_json

log = get_logger("prepare_data")
logging.getLogger("httpx2").setLevel(logging.WARNING)


def download(dataset: str, dest: Path) -> None:
    """Fetch a public Kaggle archive. Requires no credentials and skips existing files."""
    if dest.exists():
        log.info("Using existing %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    log.info("Downloading %s", dataset)
    with requests.get(f"https://www.kaggle.com/api/v1/datasets/download/{dataset}", stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f, length=1 << 22)
    tmp.rename(dest)


def main(config: str | None = None) -> None:
    cfg = load_config(config)
    set_seed(cfg["seed"])
    p = cfg["paths"]
    download(cfg["sources"]["interactions_dataset"], p["interactions_zip"])
    download(cfg["sources"]["synopsis_dataset"], p["synopsis_zip"])
    p["processed_dir"].mkdir(parents=True, exist_ok=True)

    catalog, join_stats = data.build_catalog(p["interactions_zip"], p["synopsis_zip"], cfg["data"])
    log.info("Catalog: %s", join_stats)

    raw = data.load_raw_interactions(p["interactions_zip"])
    clean, filter_rows = data.clean_interactions(raw, catalog, cfg["data"])
    del raw
    n_clean_users = clean["user"].n_unique()
    sampled = data.sample_users(clean, cfg["data"]["user_sample"], cfg["seed"])
    del clean
    interactions, n_users = data.index_users(sampled)
    del sampled

    interactions.write_parquet(p["processed_dir"] / "interactions.parquet")
    catalog.write_parquet(p["processed_dir"] / "items.parquet")
    report = {
        "catalog_join": join_stats,
        "filters": filter_rows,
        "users_after_cleaning": n_clean_users,
        "user_sample_seed": cfg["seed"],
        "users_sampled": n_users,
        "interactions_sampled": interactions.height,
    }
    report["rejected_source_timestamp_audit"] = data.audit_rejected_timestamps(p["synopsis_zip"])
    report["eda"] = run_eda(interactions, n_users, catalog.height, cfg["split"], p["figures_dir"])
    write_json(p["reports_dir"] / "generated" / "data_report.json", report)

    emb_path = p["processed_dir"] / "synopsis_embeddings.npy"
    emb = content.embed_synopses(catalog["synopsis"].to_list(), cfg["content"]["model"])
    np.save(emb_path, emb)
    log.info("Prepared %d users, %d titles, %d interactions", n_users, catalog.height, interactions.height)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
