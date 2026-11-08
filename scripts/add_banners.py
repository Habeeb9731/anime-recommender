"""Add wide banner-image links from AniList's public API to a demo bundle.

Only links are stored; no image files are downloaded. Results are cached under data/raw so
reruns do not repeat the requests. Display only: banners never reach a model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests

from animerec.utils import get_logger, load_config

log = get_logger("add_banners")
API = "https://graphql.anilist.co"
QUERY = "query($ids:[Int]){Page(page:1,perPage:50){media(idMal_in:$ids,type:ANIME){idMal bannerImage}}}"
PAUSE_SECONDS = 2.1  # stays under AniList's reduced limit of 30 requests per minute


def fetch_batch(ids: list[int]) -> dict[str, str | None]:
    for attempt in range(6):
        r = requests.post(API, json={"query": QUERY, "variables": {"ids": ids}}, headers={"User-Agent": "animerec-demo"}, timeout=30)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 60)) + 1)
            continue
        if r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        found = {str(m["idMal"]): m["bannerImage"] for m in r.json()["data"]["Page"]["media"]}
        return {str(i): found.get(str(i)) for i in ids}
    raise RuntimeError("AniList kept refusing requests")


def main(bundle: str | None, limit: int | None) -> None:
    cfg = load_config()
    d = Path(bundle) if bundle else cfg["paths"]["demo_dir"]
    cache_path = cfg["paths"]["raw_dir"] / "anilist_banners.json"
    cache: dict[str, str | None] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    items = pl.read_parquet(d / "items.parquet")
    counts = np.load(d / "arrays.npz")["item_counts_cf"]
    ordered = items["anime_id"].to_numpy()[np.argsort(-counts, kind="stable")].tolist()  # most-watched first
    todo = [int(i) for i in ordered if str(i) not in cache][:limit]
    for start in range(0, len(todo), 50):
        cache.update(fetch_batch(todo[start : start + 50]))
        cache_path.write_text(json.dumps(cache))
        log.info("Fetched %d of %d", min(start + 50, len(todo)), len(todo))
        time.sleep(PAUSE_SECONDS)

    banners = pl.DataFrame(
        {"anime_id": [int(k) for k in cache], "banner_url": list(cache.values())},
        schema={"anime_id": pl.Int64, "banner_url": pl.Utf8},
    )
    out = items.drop("banner_url", strict=False).join(banners, on="anime_id", how="left")
    out.write_parquet(d / "items.parquet", compression="zstd")
    log.info("Banner coverage: %.1f%% of %d titles", 100 * out["banner_url"].is_not_null().mean(), out.height)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.bundle, args.limit)
