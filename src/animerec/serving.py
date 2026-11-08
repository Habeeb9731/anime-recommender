"""Demo inference: instant fold-in over a small precomputed bundle, no retraining.

The heavy third-party training stack (torch, implicit, sentence-transformers) is not
imported here, so the app runs on free hosting.
"""
from __future__ import annotations

import difflib
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
import scipy.sparse as sp

from .models import ALS, ContentCentroid, ContentToCF, Hybrid, Popularity, Recommender


@dataclass
class SparseEASE:
    """EASE with each item's weights truncated to the strongest neighbours."""

    B: sp.csr_matrix

    def score(self, hist: sp.csr_matrix) -> np.ndarray:
        return np.asarray((hist @ self.B).todense(), dtype=np.float32)


def truncate_matrix(B: np.ndarray, neighbors: int) -> sp.csr_matrix:
    """Keep the largest-magnitude ``neighbors`` entries in each column."""
    n = B.shape[0]
    keep = min(neighbors, n)
    rows, cols, vals = [], [], []
    for j in range(n):
        col = B[:, j]
        idx = np.argpartition(-np.abs(col), keep - 1)[:keep] if keep < n else np.arange(n)
        idx = idx[col[idx] != 0]
        rows.append(idx)
        cols.append(np.full(idx.size, j))
        vals.append(col[idx])
    out = sp.csr_matrix(
        (np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    out.sort_indices()
    return out


@dataclass
class Recommendation:
    item_idx: int
    title: str
    genres: list[str]
    year: int | None
    type: str
    explanation: str
    is_cold: bool
    is_new: bool
    score: float
    synopsis: str = ""
    cover_url: str | None = None
    mal_id: int | None = None
    banner_url: str | None = None
    rating: float | None = None


@dataclass
class RecommendationResult:
    items: list[Recommendation]
    unknown: list[str] = field(default_factory=list)
    message: str = ""
    personalized: bool = False


class DemoRecommender:
    def __init__(
        self,
        items: pl.DataFrame,
        model: Recommender,
        vectors: np.ndarray,
        text_dim: int,
        cf_similarity: np.ndarray | sp.csr_matrix,
        recent_counts: np.ndarray,
        alpha: np.ndarray,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.items = items
        self.model = model
        self.vectors = vectors
        self.text_dim = text_dim
        self.cf_similarity = cf_similarity
        self.recent_counts = recent_counts
        self.alpha = alpha
        self.meta = meta or {}
        self.n = items.height
        self.titles = [html.unescape(t) for t in items["display_title"].to_list()]
        self.genres = [g.split("|") if g else [] for g in items["genres"].to_list()]
        self.years = items["year"].to_list()
        self.types = items["type"].to_list()
        self.franchise = items["franchise_id"].to_numpy()
        self.is_cold = items["is_cold"].to_numpy()
        self.is_new = items["is_new"].to_numpy()
        self.synopses = [html.unescape(s) if s else s for s in items["synopsis_short"].to_list()]
        self.mal_ids = items["anime_id"].to_list() if "anime_id" in items.columns else [None] * self.n
        self.ratings = items["display_score"].to_list() if "display_score" in items.columns else [None] * self.n
        self.banners = items["banner_url"].to_list() if "banner_url" in items.columns else [None] * self.n
        self.covers = items["cover_url"].to_list() if "cover_url" in items.columns else [None] * self.n
        self._by_title = {t.lower(): i for i, t in reversed(list(enumerate(self.titles)))}
        for i, t in reversed(list(enumerate(items["title"].to_list()))):
            self._by_title.setdefault(t.lower(), i)
        text = vectors[:, :text_dim].astype(np.float32)
        norm = np.linalg.norm(text, axis=1, keepdims=True)
        self._has_text = norm.ravel() > 1e-6
        norm[norm < 1e-6] = 1.0
        self._text = text / norm

    def resolve(self, names: Iterable[str]) -> tuple[list[int], list[str]]:
        """Map titles to indices, tolerating case and small typos; report what is unknown."""
        found: list[int] = []
        unknown: list[str] = []
        for raw in names:
            name = str(raw).strip().lower()
            if not name:
                continue
            if name in self._by_title:
                found.append(self._by_title[name])
                continue
            close = difflib.get_close_matches(name, list(self._by_title), n=1, cutoff=0.85)
            if close:
                found.append(self._by_title[close[0]])
            else:
                unknown.append(str(raw).strip())
        return list(dict.fromkeys(found)), unknown

    def _history(self, liked: list[int], disliked: list[int]) -> sp.csr_matrix:
        idx = liked + disliked
        vals = [1.0] * len(liked) + [-1.0] * len(disliked)
        m = sp.csr_matrix((np.array(vals, np.float32), (np.zeros(len(idx), int), idx)), shape=(1, self.n))
        m.sort_indices()
        return m

    def _candidates(
        self,
        genres: list[str] | None,
        year_range: tuple[int, int] | None,
        types: list[str] | None,
        new_only: bool,
    ) -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        if genres:
            wanted = set(genres)
            mask &= np.fromiter((bool(wanted & set(g)) for g in self.genres), bool, self.n)
        if year_range:
            lo, hi = year_range
            mask &= np.fromiter((y is not None and lo <= y <= hi for y in self.years), bool, self.n)
        if types:
            wanted_t = set(types)
            mask &= np.fromiter((t in wanted_t for t in self.types), bool, self.n)
        if new_only:
            mask &= self.is_new
        return mask

    def _cf_affinity(self, liked: list[int], item: int) -> tuple[int, float] | None:
        """Most influential liked title for this item under the collaborative arm."""
        if isinstance(self.cf_similarity, sp.spmatrix):
            vals = np.asarray(self.cf_similarity[liked, item].todense()).ravel()
        else:
            Y = self.cf_similarity
            denom = np.linalg.norm(Y[liked], axis=1) * max(float(np.linalg.norm(Y[item])), 1e-9) + 1e-9
            vals = (Y[liked] @ Y[item]) / denom
        j = int(np.argmax(vals))
        return (liked[j], float(vals[j])) if vals[j] > 0 else None

    def explain(self, item: int, liked: list[int]) -> str:
        if not liked:
            return "Popular with viewers recently"
        if self.alpha[item] >= 0.5:
            hit = self._cf_affinity(liked, item)
            if hit:
                return f"Because you liked {self.titles[hit[0]]}"
        anchors = [l for l in liked if self._has_text[l]]
        if self._has_text[item] and anchors:
            sims = self._text[anchors] @ self._text[item]
            j = int(np.argmax(sims))
            if sims[j] >= 0.3:
                return f"Similar synopsis to {self.titles[anchors[j]]}"
        best, shared = None, []
        for l in liked:
            common = [g for g in self.genres[item] if g in self.genres[l]]
            if len(common) > len(shared):
                best, shared = l, common
        if best is not None:
            return f"Shares {', '.join(shared[:3])} with {self.titles[best]}"
        hit = self._cf_affinity(liked, item)
        if hit:
            return f"Because you liked {self.titles[hit[0]]}"
        return "Matches the overall profile of your favorites"

    def recommend(
        self,
        liked: Iterable[str] = (),
        disliked: Iterable[str] = (),
        genres: list[str] | None = None,
        year_range: tuple[int, int] | None = None,
        types: list[str] | None = None,
        new_only: bool = False,
        k: int = 10,
        include_same_franchise: bool = False,
    ) -> RecommendationResult:
        liked_idx, unknown_l = self.resolve(liked)
        disliked_idx, unknown_d = self.resolve(disliked)
        disliked_idx = [d for d in disliked_idx if d not in set(liked_idx)]
        unknown = unknown_l + unknown_d
        personalized = bool(liked_idx or disliked_idx)

        if personalized:
            s = np.asarray(self.model.score(self._history(liked_idx, disliked_idx)), dtype=np.float64).ravel()
        else:
            s = np.log1p(self.recent_counts.astype(np.float64))

        mask = self._candidates(genres, year_range, types, new_only)
        mask[liked_idx + disliked_idx] = False
        order = np.argsort(-s, kind="stable")
        order = order[mask[order]]

        used = set() if include_same_franchise else {int(self.franchise[i]) for i in liked_idx + disliked_idx}
        out: list[Recommendation] = []
        for i in order:
            i = int(i)
            f = int(self.franchise[i])
            if f in used:
                continue
            used.add(f)
            y = self.years[i]
            out.append(
                Recommendation(
                    item_idx=i, title=self.titles[i], genres=self.genres[i], year=int(y) if y is not None else None,
                    type=self.types[i], explanation=self.explain(i, liked_idx), is_cold=bool(self.is_cold[i]),
                    is_new=bool(self.is_new[i]), score=float(s[i]), synopsis=self.synopses[i] or "", cover_url=self.covers[i], banner_url=self.banners[i], mal_id=self.mal_ids[i], rating=self.ratings[i],
                )
            )
            if len(out) == k:
                break

        msg = ""
        if not out:
            msg = "No titles match these filters. Try widening the genre, year or type filters."
        elif not personalized:
            msg = "Pick a few favorites for personal picks. Until then, these are popular recent titles."
        return RecommendationResult(items=out, unknown=unknown, message=msg, personalized=personalized)


def load_demo(bundle_dir: str | Path) -> DemoRecommender:
    d = Path(bundle_dir)
    meta = json.loads((d / "metadata.json").read_text())
    items = pl.read_parquet(d / "items.parquet")
    arrays = np.load(d / "arrays.npz")
    vectors = arrays["vectors"].astype(np.float32)
    eligible = np.ones(items.height, dtype=bool)
    hp = meta["hybrid"]

    als: ALS | None = None
    if "als_Y" in arrays:
        Y = arrays["als_Y"].astype(np.float32)
        als = ALS(factors=Y.shape[1], regularization=hp["als_regularization"], alpha=hp["als_alpha"], Y=Y, YtY=Y.T @ Y)

    cf: Recommender
    sim: np.ndarray | sp.csr_matrix
    if hp["cf"] == "ease":
        sim = sp.load_npz(d / "ease_B.npz")
        cf = SparseEASE(sim)
    else:
        assert als is not None
        cf, sim = als, als.Y

    content: Recommender
    if hp["content"] == "centroid":
        content = ContentCentroid(vectors)
    else:
        assert als is not None
        c2 = ContentToCF(vectors, als)
        c2.Y_hat = arrays["content_Y_hat"].astype(np.float32)
        content = c2

    counts = arrays["item_counts_cf"]
    recent = arrays["recent_counts"]
    model = Hybrid(
        cf=cf, content=content, item_counts=counts, eligible=eligible,
        kappa=hp["kappa"], cold_bias=hp["cold_bias"], pop_weight=hp["pop_weight"], recent_counts=recent,
    )
    alpha = counts.astype(np.float32) / (counts.astype(np.float32) + hp["kappa"])
    return DemoRecommender(items, model, vectors, int(meta["text_dim"]), sim, recent, alpha, meta)


__all__ = ["DemoRecommender", "RecommendationResult", "Recommendation", "SparseEASE", "truncate_matrix", "load_demo", "Popularity"]
