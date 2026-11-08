"""Demo recommendation logic on a synthetic bundle."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from animerec.models import ContentCentroid, Hybrid
from animerec.serving import DemoRecommender, SparseEASE


def _vec(n):
    rng = np.random.default_rng(1)
    v = rng.normal(size=(n, 8)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def demo() -> DemoRecommender:
    n = 8
    items = pl.DataFrame(
        {
            "title": ["Alpha", "Alpha 2", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Fresh"],
            "display_title": ["Alpha", "Alpha 2", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Fresh"],
            "genres": ["Action|Drama", "Action|Drama", "Action", "Comedy", "Drama", "Comedy", "Action", "Action|Drama"],
            "year": [2010, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
            "type": ["TV", "TV", "Movie", "TV", "OVA", "TV", "TV", "TV"],
            "franchise_id": [1, 1, 2, 3, 4, 5, 6, 7],
            "is_cold": [False] * 7 + [True],
            "is_new": [False] * 6 + [True, True],
            "synopsis_short": ["text"] * n,
        },
        schema_overrides={"year": pl.Int64},
    )
    B = np.zeros((n, n), np.float32)
    B[0, [1, 2, 3, 6]] = [9, 5, 3, 1]
    Bs = sp.csr_matrix(B)
    counts = np.array([500, 400, 300, 200, 100, 50, 25, 0])
    vec = _vec(n)
    vec[7] = vec[0]  # the cold title copies Alpha's content
    model = Hybrid(SparseEASE(Bs), ContentCentroid(vec), counts, np.ones(n, bool), kappa=20.0, cold_bias=1.0, pop_weight=0.1, recent_counts=counts.astype(float))
    return DemoRecommender(items, model, vec, 8, Bs, counts.astype(float), counts / (counts + 20.0))


def test_new_user_gets_explained_recommendations(demo):
    r = demo.recommend(liked=["Alpha"], k=5)
    assert r.personalized and len(r.items) == 5
    assert all(x.explanation for x in r.items)
    assert any("Alpha" in x.explanation for x in r.items)


def test_input_titles_are_never_recommended(demo):
    r = demo.recommend(liked=["Alpha"], disliked=["Gamma"], k=7)
    titles = {x.title for x in r.items}
    assert "Alpha" not in titles and "Gamma" not in titles


def test_franchise_of_liked_title_is_excluded_by_default(demo):
    assert "Alpha 2" not in {x.title for x in demo.recommend(liked=["Alpha"], k=7).items}
    assert "Alpha 2" in {x.title for x in demo.recommend(liked=["Alpha"], k=7, include_same_franchise=True).items}


def test_only_one_title_per_franchise(demo):
    r = demo.recommend(k=8)
    fr = [int(demo.franchise[x.item_idx]) for x in r.items]
    assert len(fr) == len(set(fr))


def test_empty_input_falls_back_to_popular(demo):
    r = demo.recommend()
    assert not r.personalized and r.items and r.message
    assert r.items[0].title == "Alpha"


def test_unknown_titles_are_reported_not_fatal(demo):
    r = demo.recommend(liked=["Totally Made Up Show"])
    assert r.unknown == ["Totally Made Up Show"]
    assert r.items and not r.personalized


def test_typo_and_case_are_tolerated(demo):
    assert demo.resolve(["alpha", "Epsilom"])[0] == [0, 5]


def test_blank_input_is_ignored(demo):
    assert demo.resolve(["", "   "]) == ([], [])


def test_filters(demo):
    assert all("Comedy" in x.genres for x in demo.recommend(liked=["Alpha"], genres=["Comedy"]).items)
    assert all(x.type == "Movie" for x in demo.recommend(liked=["Alpha"], types=["Movie"]).items)
    ys = demo.recommend(liked=["Alpha"], year_range=(2013, 2015)).items
    assert ys and all(2013 <= x.year <= 2015 for x in ys)
    assert {x.title for x in demo.recommend(liked=["Alpha"], new_only=True).items} == {"Zeta", "Fresh"}


def test_filters_with_no_match_give_a_message(demo):
    r = demo.recommend(liked=["Alpha"], genres=["Comedy"], types=["Movie"])
    assert r.items == [] and "filters" in r.message


def test_cold_title_is_recommended_with_content_explanation(demo):
    r = demo.recommend(liked=["Alpha"], new_only=True)
    fresh = next(x for x in r.items if x.title == "Fresh")
    assert fresh.is_cold
    assert "Alpha" in fresh.explanation and "Because you liked" not in fresh.explanation


def test_html_entities_in_titles_are_decoded(demo):
    items = demo.items.with_columns(pl.col("display_title").str.replace("Beta", "Beta&#039;s"))
    d = DemoRecommender(items, demo.model, demo.vectors, demo.text_dim, demo.cf_similarity, demo.recent_counts, demo.alpha)
    assert "Beta's" in d.titles


def test_dislike_lowers_score(demo):
    base = demo.recommend(liked=["Beta"], k=8)
    with_dislike = demo.recommend(liked=["Beta"], disliked=["Alpha"], k=8)
    b = {x.title: x.score for x in base.items}
    d = {x.title: x.score for x in with_dislike.items}
    assert d["Fresh"] < b["Fresh"]
