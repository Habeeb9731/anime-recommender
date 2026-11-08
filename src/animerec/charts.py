"""Shared chart style and the EDA figures."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
PRIMARY = "#2a78d6"
ACCENT = "#eb6834"
NEUTRAL = "#898781"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.bottom": False,
            "ytick.left": False,
        }
    )


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def run_eda(df: pl.DataFrame, n_users: int, n_items: int, split: dict[str, str], fig_dir: Path) -> dict[str, Any]:
    """Create the EDA figures and return the statistics they show."""
    apply_style()
    stats: dict[str, Any] = {
        "users": n_users,
        "catalog_titles": n_items,
        "interactions": df.height,
        "titles_with_interactions": int(df["item_idx"].n_unique()),
        "density_pct": 100 * df.height / (n_users * n_items),
    }

    counts = np.sort(df.group_by("item_idx").len()["len"].to_numpy())[::-1]
    share = np.cumsum(counts) / counts.sum()
    x = np.arange(1, counts.size + 1) / counts.size * 100
    top1 = max(1, int(counts.size * 0.01))
    top10 = max(1, int(counts.size * 0.10))
    stats["share_top_1pct_titles"] = float(share[top1 - 1])
    stats["share_top_10pct_titles"] = float(share[top10 - 1])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(x, share * 100, color=PRIMARY, lw=2)
    ax.axvline(10, color=NEUTRAL, lw=1, ls="--")
    ax.annotate(f"Top 10% of titles: {share[top10 - 1] * 100:.0f}% of interactions",
                xy=(10, share[top10 - 1] * 100), xytext=(18, share[top10 - 1] * 100 - 22), color=INK,
                arrowprops={"arrowstyle": "-", "color": NEUTRAL})
    ax.set(title="A small share of titles receives most interactions",
           xlabel="Titles, most to least popular (%)", ylabel="Cumulative interactions (%)", ylim=(0, 102), xlim=(0, 100))
    _save(fig, fig_dir / "eda_long_tail.png")

    st = df.group_by("status").len().sort("len")
    total = st["len"].sum()
    stats["status_counts"] = dict(zip(st["status"].to_list(), st["len"].to_list()))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    labels = [s.replace("_", " ") for s in st["status"].to_list()]
    bars = ax.barh(labels, st["len"].to_numpy() / 1e6, color=PRIMARY, height=0.55)
    for b, v in zip(bars, st["len"].to_list()):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {v / total * 100:.1f}%", va="center", color=INK)
    ax.grid(axis="y", visible=False)
    ax.set(title="Most list entries are completed titles", xlabel="Interactions (millions)")
    _save(fig, fig_dir / "eda_status.png")

    sc = df.filter(pl.col("score") > 0).group_by("score").len().sort("score")
    stats["share_unrated"] = float((df["score"] == 0).mean())
    stats["score_counts"] = {str(k): v for k, v in zip(sc["score"].to_list(), sc["len"].to_list())}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(sc["score"].to_numpy(), sc["len"].to_numpy() / 1e6, color=PRIMARY, width=0.7)
    ax.set_xticks(range(1, 11))
    ax.grid(axis="x", visible=False)
    ax.set(title=f"Scores skew high ({stats['share_unrated'] * 100:.0f}% of entries are unrated)",
           xlabel="User score", ylabel="Interactions (millions)")
    _save(fig, fig_dir / "eda_scores.png")

    m = df.group_by(pl.col("ts").dt.truncate("1mo").alias("m")).len().sort("m")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(m["m"].to_list(), m["len"].to_numpy() / 1e3, color=PRIMARY, lw=2)
    for key, label in (("val_start", "validation"), ("test_start", "test")):
        d = datetime.fromisoformat(split[key])
        ax.axvline(d, color=ACCENT, lw=1.2, ls="--")
        ax.text(d, ax.get_ylim()[1], f" {label} starts", color=INK, va="top", ha="left" if key == "test_start" else "right")
    ax.set(title="Interactions per month, by last-update time", ylabel="Interactions (thousands)")
    _save(fig, fig_dir / "eda_over_time.png")
    stats["monthly_interactions"] = {d.strftime("%Y-%m"): v for d, v in zip(m["m"].to_list(), m["len"].to_list())}
    return stats
