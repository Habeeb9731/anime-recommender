"""Synthetic fixtures for unit tests only. None of this data reaches a reported result."""
from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest


def row(user: int, item: int, status: str, score: int, ts: datetime) -> dict:
    return {"user_idx": user, "item_idx": item, "status": status, "score": score, "ts": ts}


@pytest.fixture
def items() -> pl.DataFrame:
    # item 3 is unreleased at the evaluation end, item 4 has no release date, item 2 is cold.
    return pl.DataFrame(
        {
            "item_idx": [0, 1, 2, 3, 4],
            "start_date": [datetime(2010, 1, 1), datetime(2012, 1, 1), datetime(2018, 1, 5), datetime(2030, 1, 1), None],
        },
        schema={"item_idx": pl.Int64, "start_date": pl.Datetime("us")},
    )


FIT_END = datetime(2018, 1, 1)
EVAL_END = datetime(2018, 2, 1)


@pytest.fixture
def interactions() -> pl.DataFrame:
    return pl.DataFrame(
        [
            row(0, 0, "completed", 9, datetime(2017, 5, 1)),
            row(0, 1, "plan_to_watch", 0, datetime(2017, 6, 1)),
            row(1, 0, "dropped", 2, datetime(2017, 7, 1)),
            row(1, 1, "completed", 8, datetime(2017, 8, 1)),
            row(2, 0, "watching", 0, datetime(2017, 8, 2)),
            row(0, 2, "completed", 10, datetime(2018, 1, 10)),   # cold positive in the eval window
            row(1, 2, "completed", 3, datetime(2018, 1, 11)),    # low score, not positive
            row(2, 1, "completed", 9, datetime(2018, 1, 12)),
            row(3, 1, "completed", 9, datetime(2018, 1, 13)),    # zero-history user
            row(1, 3, "completed", 9, datetime(2018, 1, 14)),  # ineligible title
            row(4, 1, "completed", 9, datetime(2018, 3, 1)),     # after the eval window
        ],
        schema={"user_idx": pl.Int64, "item_idx": pl.Int64, "status": pl.Utf8, "score": pl.Int64, "ts": pl.Datetime("us")},
    )
