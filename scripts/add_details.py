"""Add display-only title details to a demo bundle. These columns never reach a model."""
from __future__ import annotations

import argparse
import html
import io
import zipfile
from pathlib import Path

import pandas as pd
import polars as pl

from animerec.utils import get_logger, load_config

log = get_logger("add_details")
DETAIL_COLUMNS = ["title_japanese", "episodes", "duration", "age_rating", "aired", "source", "studios_text", "premiered", "synopsis_full"]


def clean(value: object) -> str | None:
    """Treat the dataset's placeholder values as missing."""
    if not isinstance(value, str):
        return None
    text = html.unescape(value).strip()
    return None if not text or text.lower() in {"unknown", "none", "not available", "?"} else text


def main(bundle: str | None) -> None:
    cfg = load_config()
    d = Path(bundle) if bundle else cfg["paths"]["demo_dir"]
    with zipfile.ZipFile(cfg["paths"]["interactions_zip"]) as z:
        raw = pd.read_csv(
            io.BytesIO(z.read("AnimeList.csv")),
            usecols=["anime_id", "title_japanese", "episodes", "duration", "rating", "aired_string", "source", "studio", "premiered"],
        ).drop_duplicates("anime_id")
    catalog = pl.read_parquet(cfg["paths"]["processed_dir"] / "items.parquet", columns=["anime_id", "synopsis"]).to_pandas()
    merged = raw.merge(catalog, on="anime_id", how="left")
    details = pl.DataFrame(
        {
            "anime_id": merged["anime_id"].astype(int).tolist(),
            "title_japanese": [clean(v) for v in merged["title_japanese"]],
            "episodes": [int(v) if pd.notna(v) and int(v) > 0 else None for v in merged["episodes"]],
            "duration": [clean(v) for v in merged["duration"]],
            "age_rating": [clean(v) for v in merged["rating"]],
            "aired": [clean(v) for v in merged["aired_string"]],
            "source": [clean(v) for v in merged["source"]],
            "studios_text": [clean(v) for v in merged["studio"]],
            "premiered": [clean(v) for v in merged["premiered"]],
            "synopsis_full": [clean(v) for v in merged["synopsis"]],
        },
        schema_overrides={"anime_id": pl.Int64, "episodes": pl.Int64},
    )
    items = pl.read_parquet(d / "items.parquet")
    out = items.drop(DETAIL_COLUMNS, strict=False).join(details, on="anime_id", how="left")
    out.write_parquet(d / "items.parquet", compression="zstd")
    log.info("Details added for %d titles; bundle items file is %.1f MB", out.height, (d / "items.parquet").stat().st_size / 1e6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=None)
    main(parser.parse_args().bundle)
