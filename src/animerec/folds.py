"""Turn cleaned interactions into a leakage-safe fold.

All behavioural inputs (matrices, user means, popularity, warm/cold flags) are computed
from rows with ``ts < fit_end``. The eligible catalog comes from metadata only, never from
which titles happen to receive interactions in the evaluation window.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
import scipy.sparse as sp

PTW = "plan_to_watch"
DROPPED = "dropped"
COMPLETED = "completed"


@dataclass(frozen=True)
class TargetDef:
    """Positive = completed and (score >= threshold or score > the user's fit-window mean)."""

    threshold: int = 7
    use_user_mean: bool = True
    any_completed: bool = False


def positive_expr(t: TargetDef) -> pl.Expr:
    completed = pl.col("status") == COMPLETED
    if t.any_completed:
        return completed
    rated = pl.col("score") > 0
    cond = pl.col("score") >= t.threshold
    if t.use_user_mean:
        cond = cond | (pl.col("score") > pl.col("user_mean").fill_null(99.0))
    return completed & rated & cond


@dataclass
class Fold:
    n_users: int
    n_items: int
    X: sp.csr_matrix  # signed fit-window confidences (positive +, dropped -)
    seen: sp.csr_matrix  # every non-plan-to-watch fit-window title; excluded from recommendations
    item_counts_any: np.ndarray  # fit-window rows of any status, defines warm vs cold
    item_counts_cf: np.ndarray  # rows that entered X, drives the hybrid's alpha
    pos_counts: np.ndarray
    eligible: np.ndarray
    cold: np.ndarray
    eval_users: np.ndarray  # users with >=1 fit-window seen title and >=1 eval positive
    zero_history_users: np.ndarray  # users with an eval positive and no fit-window history
    relevant: dict[int, np.ndarray]  # user index -> eligible, unseen eval positives
    fit_end: datetime
    eval_end: datetime
    n_positive_dropped_ineligible: int


def eligible_mask(items: pl.DataFrame, eval_end: datetime, n_items: int) -> np.ndarray:
    """Titles released by the end of the evaluation window; metadata only."""
    mask = np.zeros(n_items, dtype=bool)
    idx = items.filter(pl.col("start_date").is_not_null() & (pl.col("start_date") < eval_end))["item_idx"]
    mask[idx.to_numpy()] = True
    return mask


def _csr(u: np.ndarray, i: np.ndarray, v: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    m = sp.csr_matrix((v.astype(np.float32), (u, i)), shape=shape)
    m.sort_indices()
    return m


def recent_positive_counts(
    interactions: pl.DataFrame, fit_end: datetime, days: int, n_items: int, target: TargetDef
) -> np.ndarray:
    fit = with_user_mean(interactions.filter(pl.col("ts") < fit_end))
    start = fit_end - _days(days)
    rows = fit.filter((pl.col("ts") >= start) & positive_expr(target))
    return np.bincount(rows["item_idx"].to_numpy(), minlength=n_items).astype(np.float64)


def _days(n: int):
    from datetime import timedelta

    return timedelta(days=n)


def with_user_mean(fit: pl.DataFrame) -> pl.DataFrame:
    means = fit.filter(pl.col("score") > 0).group_by("user_idx").agg(pl.col("score").mean().alias("user_mean"))
    return fit.join(means, on="user_idx", how="left")


def build_fold(
    interactions: pl.DataFrame,
    items: pl.DataFrame,
    n_users: int,
    fit_end: datetime,
    eval_start: datetime,
    eval_end: datetime,
    train_target: TargetDef = TargetDef(),
    eval_target: TargetDef = TargetDef(),
    dropped_weight: float = 1.0,
    decay_days: float | None = None,
) -> Fold:
    if not fit_end <= eval_start < eval_end:
        raise ValueError("require fit_end <= eval_start < eval_end")
    n_items = items.height
    shape = (n_users, n_items)
    fit = with_user_mean(interactions.filter(pl.col("ts") < fit_end))

    item_counts_any = np.bincount(fit["item_idx"].to_numpy(), minlength=n_items).astype(np.int64)

    weight = pl.lit(1.0)
    if decay_days:
        age = (pl.lit(fit_end) - pl.col("ts")).dt.total_seconds() / 86400.0
        weight = (-age / decay_days).exp()

    fit = fit.with_columns(positive_expr(train_target).alias("is_pos"))
    signal = fit.filter(pl.col("is_pos") | (pl.col("status") == DROPPED)).with_columns(
        pl.when(pl.col("is_pos")).then(weight).otherwise(-dropped_weight * weight).alias("conf")
    )
    X = _csr(signal["user_idx"].to_numpy(), signal["item_idx"].to_numpy(), signal["conf"].to_numpy(), shape)
    X.eliminate_zeros()  # dropped_weight=0 must not leave explicit zeros in the matrix
    item_counts_cf = np.asarray(X.getnnz(axis=0)).ravel().astype(np.int64)
    pos = signal.filter(pl.col("is_pos"))
    pos_counts = np.bincount(pos["item_idx"].to_numpy(), minlength=n_items).astype(np.float64)

    seen_rows = fit.filter(pl.col("status") != PTW)
    seen = _csr(
        seen_rows["user_idx"].to_numpy(),
        seen_rows["item_idx"].to_numpy(),
        np.ones(seen_rows.height),
        shape,
    )

    eligible = eligible_mask(items, eval_end, n_items)
    cold = item_counts_any == 0

    # User means come from the fit window only; labels never look at the eval-window mean.
    means = fit.select(["user_idx", "user_mean"]).unique("user_idx")
    window = interactions.filter((pl.col("ts") >= eval_start) & (pl.col("ts") < eval_end)).join(
        means, on="user_idx", how="left"
    )
    positives = window.filter(positive_expr(eval_target)).select(["user_idx", "item_idx"])

    seen_counts = np.asarray(seen.getnnz(axis=1)).ravel()
    relevant: dict[int, np.ndarray] = {}
    dropped = 0
    for (uid,), g in positives.group_by(["user_idx"], maintain_order=True):
        arr = g["item_idx"].to_numpy()
        seen_items = seen.indices[seen.indptr[uid] : seen.indptr[uid + 1]]
        keep = eligible[arr] & ~np.isin(arr, seen_items)
        dropped += int((~keep).sum())
        if keep.any():
            relevant[int(uid)] = np.sort(arr[keep])

    users = np.array(sorted(relevant), dtype=np.int64)
    has_history = seen_counts[users] > 0 if users.size else np.zeros(0, dtype=bool)
    return Fold(
        n_users=n_users,
        n_items=n_items,
        X=X,
        seen=seen,
        item_counts_any=item_counts_any,
        item_counts_cf=item_counts_cf,
        pos_counts=pos_counts,
        eligible=eligible,
        cold=cold,
        eval_users=users[has_history] if users.size else users,
        zero_history_users=users[~has_history] if users.size else users,
        relevant=relevant,
        fit_end=fit_end,
        eval_end=eval_end,
        n_positive_dropped_ineligible=dropped,
    )
