"""Data splits, leakage prevention, targets, and cold-title handling."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from animerec.data import CATALOG_COLUMNS, LEAKY_COLUMNS, clean_interactions, clean_synopsis
from animerec.folds import TargetDef, build_fold, recent_positive_counts
from conftest import EVAL_END, FIT_END, row


def fold(interactions, items, **kw):
    return build_fold(interactions, items, 5, FIT_END, FIT_END, EVAL_END, **kw)


def test_split_order_is_enforced(interactions, items):
    with pytest.raises(ValueError):
        build_fold(interactions, items, 5, EVAL_END, FIT_END, EVAL_END)


def test_fit_matrix_uses_only_pre_cutoff_rows(interactions, items):
    f = fold(interactions, items)
    assert f.X[0, 2] == 0  # the 2018-01-10 positive is outside the fit window
    assert f.item_counts_any[2] == 0
    assert f.X.nnz == 3


def test_future_rows_do_not_change_fit_artifacts(interactions, items):
    a = fold(interactions, items)
    future = pl.concat([interactions, pl.DataFrame([row(0, 1, "completed", 10, datetime(2018, 1, 20))], schema=interactions.schema)])
    b = fold(future, items)
    assert (a.X != b.X).nnz == 0
    assert (a.seen != b.seen).nnz == 0
    assert np.array_equal(a.item_counts_any, b.item_counts_any)


def test_labels_are_inside_the_eval_window(interactions, items):
    f = fold(interactions, items)
    assert 4 not in f.relevant


def test_catalog_comes_from_metadata_not_eval_activity(interactions, items):
    f = fold(interactions, items)
    assert f.eligible.tolist() == [True, True, True, False, False]
    without = interactions.filter(pl.col("item_idx") != 2)
    assert fold(without, items).eligible.tolist() == f.eligible.tolist()


def test_ineligible_positive_is_excluded_and_counted(interactions, items):
    f = fold(interactions, items)
    assert 1 not in f.relevant  # user 1's only eval positive is unreleased
    assert f.n_positive_dropped_ineligible == 1


def test_user_mean_comes_from_fit_window_only():
    schema = {"user_idx": pl.Int64, "item_idx": pl.Int64, "status": pl.Utf8, "score": pl.Int64, "ts": pl.Datetime("us")}
    # Score 6 clears the fit-window mean (5) but would not clear a mean that included
    # the later score of 10 (7.0), so this fails if the mean leaks from the eval window.
    df2 = pl.DataFrame(
        [
            row(0, 0, "completed", 5, datetime(2017, 1, 1)),
            row(0, 1, "completed", 6, datetime(2018, 1, 5)),
            row(0, 2, "completed", 10, datetime(2018, 1, 6)),
        ],
        schema=schema,
    )
    three = pl.DataFrame({"item_idx": [0, 1, 2], "start_date": [datetime(2010, 1, 1)] * 3}, schema={"item_idx": pl.Int64, "start_date": pl.Datetime("us")})
    assert build_fold(df2, three, 1, FIT_END, FIT_END, EVAL_END).relevant[0].tolist() == [1, 2]


def test_positive_definition(interactions, items):
    f = fold(interactions, items)
    assert f.X[1, 1] > 0  # completed, score 8
    assert f.relevant[0].tolist() == [2]


def test_dropped_is_negative_and_plan_to_watch_is_not_positive(interactions, items):
    f = fold(interactions, items, dropped_weight=0.5)
    assert f.X[1, 0] == pytest.approx(-0.5)
    assert f.X[0, 1] == 0
    assert f.seen[0, 1] == 0  # plan-to-watch does not count as seen
    assert f.seen[1, 0] == 1


def test_any_completed_target_includes_low_scores(interactions, items):
    strict = fold(interactions, items)
    loose = fold(interactions, items, eval_target=TargetDef(any_completed=True))
    assert 2 not in strict.relevant.get(1, [])
    assert 2 in loose.relevant[1].tolist()


def test_cold_titles_have_no_fit_interactions(interactions, items):
    f = fold(interactions, items)
    assert f.cold.tolist() == [False, False, True, True, True]


def test_zero_history_users_are_kept_separate(interactions, items):
    f = fold(interactions, items)
    assert 3 in f.zero_history_users
    assert 3 not in f.eval_users


def test_recent_counts_respect_the_window(interactions):
    c = recent_positive_counts(interactions, FIT_END, 170, 5, TargetDef())
    assert c.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]


def test_no_behavioural_metadata_in_catalog():
    assert not (set(CATALOG_COLUMNS) & LEAKY_COLUMNS)


def _catalog():
    return pl.DataFrame(
        {"anime_id": [1, 2], "item_idx": [0, 1], "start_date": [datetime(2010, 1, 1), datetime(2015, 1, 1)]},
        schema={"anime_id": pl.Int64, "item_idx": pl.Int64, "start_date": pl.Datetime("us")},
    )


CLEAN_CFG = {
    "min_valid_ts": "2004-11-05",
    "study_end": "2018-05-01",
    "bots": {"max_list_size": 5000, "bulk_min_rows": 20, "bulk_second_share": 0.5, "flat_min_rated": 50, "max_completed_per_day": 100},
}


def _raw(rows):
    return pl.DataFrame(rows, schema=["user", "anime_id", "score", "status_code", "epoch"], orient="row")


def test_cleaning_removes_invalid_records_and_logs_them():
    good = int(datetime(2015, 6, 1).timestamp())
    df, log_rows = clean_interactions(
        _raw(
            [
                ("a", "1", "8", "2", str(good)),
                ("a", "1", "7", "2", str(good - 100)),            # duplicate, older
                ("b", "1", "8", "2", "0"),                         # placeholder timestamp
                ("c", "1", "8", "33", str(good)),                  # invalid status
                ("d", "99", "8", "2", str(good)),                  # not in catalog
                ("e", "2", "8", "2", str(int(datetime(2012, 1, 1).timestamp()))),  # before release
                ("f", "2", "0", "6", str(int(datetime(2012, 1, 1).timestamp()))),  # plan-to-watch before release is fine
            ]
        ),
        _catalog(),
        CLEAN_CFG,
    )
    assert sorted(df["user"].to_list()) == ["a", "f"]
    assert df.filter(pl.col("user") == "a")["score"].to_list() == [8]
    removed = {r["filter"]: r["interactions_removed"] for r in log_rows}
    assert removed["placeholder_or_pre_launch_ts"] == 1
    assert removed["invalid_status"] == 1
    assert removed["duplicate_user_title"] == 1
    assert removed["title_not_in_catalog"] == 1
    assert removed["activity_before_release"] == 1


def test_bulk_import_users_are_removed():
    base = int(datetime(2015, 6, 1).timestamp())
    catalog = pl.DataFrame(
        {"anime_id": list(range(30)), "item_idx": list(range(30)), "start_date": [datetime(2010, 1, 1)] * 30},
        schema={"anime_id": pl.Int64, "item_idx": pl.Int64, "start_date": pl.Datetime("us")},
    )
    rows = [("bot", str(i), "8", "2", str(base)) for i in range(25)] + [("human", str(i), "8", "2", str(base + i * 86400)) for i in range(25)]
    df, _ = clean_interactions(_raw(rows), catalog, CLEAN_CFG)
    assert set(df["user"].to_list()) == {"human"}


def test_placeholder_synopses_are_rejected():
    assert clean_synopsis("Second season of .", 40) is None
    assert clean_synopsis(None, 40) is None
    text = "A young pilot joins a crew exploring the outer edge of the known galaxy. (Source: MAL Rewrite)"
    assert clean_synopsis(text, 40).endswith("galaxy.")
