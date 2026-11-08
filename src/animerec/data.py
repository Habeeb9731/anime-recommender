"""Raw data loading, catalog construction, and logged cleaning filters."""
from __future__ import annotations

import ast
import io
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from .utils import get_logger

log = get_logger(__name__)

STATUS_MAP = {1: "watching", 2: "completed", 3: "on_hold", 4: "dropped", 6: "plan_to_watch"}
FRANCHISE_RELATIONS = {
    "Sequel", "Prequel", "Parent story", "Side story", "Full story", "Summary", "Alternative version",
}
# Popularity-style columns are crawl-time aggregates and would leak the future.
LEAKY_COLUMNS = {
    "score", "scored_by", "rank", "popularity", "members", "favorites",
    "score_count", "score_rank", "popularity_rank", "members_count", "favorites_count",
    "watching_count", "completed_count", "on_hold_count", "dropped_count",
    "plan_to_watch_count", "total_count",
}
CATALOG_COLUMNS = [
    "item_idx", "anime_id", "title", "title_english", "type", "genres", "studios",
    "year", "start_date", "synopsis", "has_synopsis", "franchise_id", "cover_url",
]


def load_raw_interactions(zip_path: Path) -> pl.DataFrame:
    """Read every field as text; type validation is a logged filter, not a silent cast."""
    cols = ["username", "anime_id", "my_score", "my_status", "my_last_updated"]
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read("UserAnimeList.csv")
    df = pl.read_csv(
        io.BytesIO(raw),
        columns=cols,
        schema_overrides={c: pl.Utf8 for c in cols},
        ignore_errors=True,
        truncate_ragged_lines=True,
    )
    return df.rename(
        {"username": "user", "my_score": "score", "my_status": "status_code", "my_last_updated": "epoch"}
    )


def clean_synopsis(text: Any, min_chars: int) -> str | None:
    """Drop source credits and placeholders such as '.' or 'Second season of .'."""
    if not isinstance(text, str):
        return None
    t = re.sub(r"\s*[\(\[](Source|Written by)[^\)\]]*[\)\]]\s*$", "", text.strip(), flags=re.I).strip()
    if len(t) < min_chars or t.lower().startswith("no synopsis"):
        return None
    return t


def _aired_from(s: Any) -> datetime | None:
    try:
        v = ast.literal_eval(s).get("from")
        return datetime.strptime(v, "%Y-%m-%d") if v else None
    except (ValueError, SyntaxError, AttributeError, TypeError):
        return None


def _franchise_ids(anime_ids: list[int], related: list[Any]) -> list[int]:
    """Union-find over sequel/prequel-style links; the root id labels the franchise."""
    known = set(anime_ids)
    parent = {a: a for a in anime_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, rel in zip(anime_ids, related):
        try:
            links = ast.literal_eval(rel) if isinstance(rel, str) else {}
        except (ValueError, SyntaxError):
            continue
        if not isinstance(links, dict):
            continue
        for kind, targets in links.items():
            if kind not in FRANCHISE_RELATIONS or not isinstance(targets, list):
                continue
            for t in targets:
                if isinstance(t, dict) and t.get("type") == "anime" and t.get("mal_id") in known:
                    ra, rb = find(a), find(t["mal_id"])
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    return [find(a) for a in anime_ids]


def build_catalog(
    interactions_zip: Path, synopsis_zip: Path, cfg: dict[str, Any]
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Join metadata and synopses by anime id, keeping only allow-listed, non-behavioural columns."""
    with zipfile.ZipFile(interactions_zip) as z:
        a = pd.read_csv(
            io.BytesIO(z.read("AnimeList.csv")),
            usecols=["anime_id", "title", "title_english", "type", "aired", "genre", "studio", "related", "image_url"],
        )
    with zipfile.ZipFile(synopsis_zip) as z:
        syn = pd.read_csv(
            io.BytesIO(z.read("anime.csv")), sep="\t", usecols=["anime_id", "synopsis", "main_pic"], low_memory=False
        )
    n_raw = len(a)
    a = a.drop_duplicates("anime_id").merge(syn.drop_duplicates("anime_id"), on="anime_id", how="left")
    in_source = a["synopsis"].notna()
    # Prefer the newer cover link; older links point at a retired CDN host with the same path.
    legacy = a["image_url"].str.replace(r"^https?://[^/]+/", "https://cdn.myanimelist.net/", regex=True)
    a["cover_url"] = a["main_pic"].fillna(legacy)
    a["synopsis"] = [clean_synopsis(s, cfg["min_synopsis_chars"]) for s in a["synopsis"]]
    a["has_synopsis"] = a["synopsis"].notna()

    join_stats = {
        "catalog_rows_raw": n_raw,
        "synopsis_id_match_rate": float(in_source.mean()),
        "usable_synopsis_rate_after_placeholder_removal": float(a["has_synopsis"].mean()),
    }

    genres = a["genre"].fillna("").str.split(r",\s*", regex=True).map(lambda g: [x for x in g if x])
    bad_genre = genres.map(lambda g: bool(set(g) & set(cfg["excluded_genres"])))
    bad_type = a["type"].isin(cfg["excluded_types"]) | a["type"].isna()
    join_stats["titles_removed_excluded_genre"] = int(bad_genre.sum())
    join_stats["titles_removed_excluded_type"] = int((bad_type & ~bad_genre).sum())
    keep = ~(bad_genre | bad_type)
    a, genres = a[keep].copy(), genres[keep]

    a["genres"] = genres.map(lambda g: "|".join(g))
    a["studios"] = a["studio"].fillna("").str.replace(r",\s*", "|", regex=True)
    a["start_date"] = a["aired"].map(_aired_from)
    a["year"] = a["start_date"].map(lambda d: d.year if d is not None else None).astype("Int64")
    a = a.sort_values("anime_id").reset_index(drop=True)
    a["item_idx"] = np.arange(len(a))
    a["franchise_id"] = _franchise_ids(a["anime_id"].tolist(), a["related"].tolist())
    for c in ("title_english", "synopsis", "cover_url"):
        a[c] = a[c].astype("object").where(a[c].notna(), None)

    catalog = pl.from_pandas(a[CATALOG_COLUMNS]).with_columns(pl.col("start_date").cast(pl.Datetime("us")))
    join_stats["catalog_rows_final"] = catalog.height
    join_stats["synopsis_coverage_final_catalog"] = float(catalog["has_synopsis"].mean())
    join_stats["franchises"] = int(catalog["franchise_id"].n_unique())
    return catalog, join_stats


def audit_rejected_timestamps(zip_path: Path, n_shards: int = 3) -> dict[str, Any]:
    """Quantify why the newer crawl's ``last_interaction_date`` cannot support a temporal split."""
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.startswith("user_anime0"))[:n_shards]
        cols = ["user_id", "anime_id", "last_interaction_date"]
        shards = [
            pl.read_csv(
                io.BytesIO(z.read(n)), separator="\t", quote_char=None, columns=cols,
                schema_overrides={"user_id": pl.Utf8, "anime_id": pl.Int64, "last_interaction_date": pl.Utf8},
            )
            for n in names
        ]
        meta = pd.read_csv(io.BytesIO(z.read("anime.csv")), sep="\t", usecols=["anime_id", "start_date", "end_date"], low_memory=False)
    fmt = "%Y-%m-%d %H:%M:%S"
    dates = pl.from_pandas(meta).with_columns(
        pl.col("start_date").str.strptime(pl.Datetime, fmt, strict=False),
        pl.col("end_date").str.strptime(pl.Datetime, fmt, strict=False),
    )
    df = (
        pl.concat(shards)
        .with_columns(pl.col("last_interaction_date").str.strptime(pl.Datetime, fmt, strict=False).alias("ts"))
        .drop_nulls("ts")
        .join(dates, on="anime_id", how="left")
    )
    placeholder = df["ts"].mode().sort()[0]
    eq_end = (df["ts"] == df["end_date"]).fill_null(False)
    eq_start = (df["ts"] == df["start_date"]).fill_null(False)
    is_placeholder = df["ts"] == placeholder
    return {
        "shards_sampled": len(names),
        "rows_sampled": df.height,
        "share_equal_to_title_end_date": float(eq_end.mean()),
        "share_equal_to_air_date_or_placeholder": float((eq_end | eq_start | is_placeholder).mean()),
        "most_common_timestamp": str(placeholder),
        "share_most_common_timestamp": float(is_placeholder.mean()),
    }


def load_display_scores(synopsis_zip: Path) -> pl.DataFrame:
    """Community scores shown on demo cards. Display only: never joined into model inputs."""
    with zipfile.ZipFile(synopsis_zip) as z:
        s = pd.read_csv(io.BytesIO(z.read("anime.csv")), sep="\t", usecols=["anime_id", "score"], low_memory=False)
    return pl.from_pandas(s.drop_duplicates("anime_id")).rename({"score": "display_score"}).with_columns(
        pl.col("anime_id").cast(pl.Int64), pl.col("display_score").cast(pl.Float64, strict=False)
    )


class FilterLog:
    """Records users, titles and interactions removed by each cleaning step."""

    def __init__(self, df: pl.DataFrame) -> None:
        self.rows: list[dict[str, Any]] = []
        self._prev = self._counts(df)
        self.rows.append({"filter": "raw", **self._prev, "users_removed": 0, "titles_removed": 0, "interactions_removed": 0})

    @staticmethod
    def _counts(df: pl.DataFrame) -> dict[str, int]:
        return {"users": df["user"].n_unique(), "titles": df["anime_id"].n_unique(), "interactions": df.height}

    def step(self, name: str, df: pl.DataFrame) -> pl.DataFrame:
        cur = self._counts(df)
        row = {
            "filter": name,
            **cur,
            "users_removed": self._prev["users"] - cur["users"],
            "titles_removed": self._prev["titles"] - cur["titles"],
            "interactions_removed": self._prev["interactions"] - cur["interactions"],
        }
        self.rows.append(row)
        log.info("%-28s removed %8d users %6d titles %10d rows", name, row["users_removed"], row["titles_removed"], row["interactions_removed"])
        self._prev = cur
        return df


def bot_like_users(df: pl.DataFrame, bots: dict[str, Any]) -> dict[str, pl.Series]:
    """Return the user set flagged by each rule. Expects typed columns (user, ts, score, status)."""
    list_size = df.group_by("user").len()
    too_long = list_size.filter(pl.col("len") > bots["max_list_size"])["user"]

    per_sec = df.group_by(["user", "ts"]).len().group_by("user").agg(pl.col("len").max().alias("mode_n"))
    bulk = per_sec.join(list_size, on="user").filter(
        (pl.col("len") >= bots["bulk_min_rows"])
        & (pl.col("mode_n") / pl.col("len") >= bots["bulk_second_share"])
    )["user"]

    flat = (
        df.filter(pl.col("score") > 0)
        .group_by("user")
        .agg(pl.len().alias("n"), pl.col("score").n_unique().alias("distinct"))
        .filter((pl.col("n") >= bots["flat_min_rated"]) & (pl.col("distinct") == 1))["user"]
    )

    daily = (
        df.filter(pl.col("status") == "completed")
        .group_by(["user", pl.col("ts").dt.date().alias("d")])
        .len()
        .filter(pl.col("len") >= bots["max_completed_per_day"])["user"]
        .unique()
    )
    return {"list_too_long": too_long, "bulk_import": bulk, "flat_scores": flat, "binge_rate": daily}


def clean_interactions(
    raw: pl.DataFrame, catalog: pl.DataFrame, cfg: dict[str, Any]
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Apply each filter in turn and log what it removed. No sampling happens here."""
    min_ts = datetime.fromisoformat(cfg["min_valid_ts"])
    study_end = datetime.fromisoformat(cfg["study_end"])
    flog = FilterLog(raw)

    df = raw.with_columns(
        pl.col("anime_id").cast(pl.Int64, strict=False),
        pl.col("score").cast(pl.Int64, strict=False),
        pl.col("status_code").cast(pl.Int64, strict=False),
        pl.col("epoch").cast(pl.Int64, strict=False),
    )
    df = flog.step(
        "null_or_unparseable",
        df.filter(pl.all_horizontal(pl.col(["user", "anime_id", "score", "status_code", "epoch"]).is_not_null())),
    )
    df = df.with_columns(pl.from_epoch("epoch", time_unit="s").dt.cast_time_unit("us").alias("ts")).drop("epoch")
    df = flog.step("placeholder_or_pre_launch_ts", df.filter(pl.col("ts") >= min_ts))
    df = flog.step("after_study_end", df.filter(pl.col("ts") < study_end))
    df = flog.step("invalid_status", df.filter(pl.col("status_code").is_in(list(STATUS_MAP))))
    df = df.with_columns(
        pl.col("status_code").replace_strict(STATUS_MAP, return_dtype=pl.Utf8).alias("status")
    ).drop("status_code")
    df = flog.step("invalid_score", df.filter(pl.col("score").is_between(0, 10)))
    df = flog.step(
        "duplicate_user_title",
        df.sort("ts", descending=True).unique(["user", "anime_id"], keep="first", maintain_order=True),
    )

    meta = catalog.select(["anime_id", "item_idx", "start_date"])
    df = flog.step("title_not_in_catalog", df.join(meta, on="anime_id", how="inner"))
    df = flog.step(
        "activity_before_release",
        df.filter(
            (pl.col("status") == "plan_to_watch")
            | pl.col("start_date").is_null()
            | (pl.col("ts") >= pl.col("start_date"))
        ),
    ).drop("start_date")

    for name, users in bot_like_users(df, cfg["bots"]).items():
        df = flog.step(f"bot_{name}", df.filter(~pl.col("user").is_in(users.to_list())))
    return df, flog.rows


def sample_users(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    users = np.sort(df["user"].unique().to_numpy().astype(str))
    if n >= users.size:
        return df
    picked = np.random.default_rng(seed).choice(users, size=n, replace=False)
    return df.filter(pl.col("user").is_in(picked.tolist()))


def index_users(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Assign contiguous user ids and drop usernames, so no handle reaches processed files."""
    users = df.select("user").unique().sort("user").with_row_index("user_idx")
    out = df.join(users, on="user").with_columns(pl.col("user_idx").cast(pl.Int64)).drop(["user", "anime_id"])
    return out, users.height
