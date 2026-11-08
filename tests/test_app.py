"""End-to-end checks of the Streamlit page on a synthetic bundle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
TITLES = ["Alpha", "Beta", "Gamma", "Delta <b>Show</b>", "Epsilon", "Zeta", "Eta", "Fresh"]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("bundle")
    n = len(TITLES)
    pl.DataFrame(
        {
            "title": TITLES,
            "display_title": TITLES,
            "genres": ["Action", "Action", "Comedy", "Drama", "Comedy", "Action", "Drama", "Action"],
            "year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2018],
            "type": ["TV", "TV", "Movie", "TV", "OVA", "TV", "TV", "TV"],
            "franchise_id": list(range(n)),
            "is_cold": [False] * (n - 1) + [True],
            "is_new": [False] * (n - 1) + [True],
            "synopsis_short": ["A story."] * n,
            "anime_id": list(range(1, n + 1)),
            "banner_url": ["https://s4.anilist.co/file/banner/1.jpg"] + [None] * (n - 1),
            "display_score": [8.1] * n,
            "cover_url": ["https://cdn.myanimelist.net/images/x.jpg"] * n,
        }
    ).write_parquet(d / "items.parquet")
    v = np.random.default_rng(0).normal(size=(n, 6)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    np.savez_compressed(
        d / "arrays.npz",
        vectors=v.astype(np.float16),
        item_counts_cf=np.array([900, 800, 700, 600, 500, 400, 300, 0], np.int32),
        recent_counts=np.array([90, 80, 70, 60, 50, 40, 30, 0], np.int32),
    )
    B = np.zeros((n, n), np.float32)
    B[0, 1:7] = [6, 5, 4, 3, 2, 1]
    sp.save_npz(d / "ease_B.npz", sp.csr_matrix(B))
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "snapshot": "February 2018",
                "text_dim": 6,
                "hybrid": {"cf": "ease", "content": "centroid", "kappa": 20.0, "cold_bias": 1.0, "pop_weight": 0.1, "als_regularization": 1.0, "als_alpha": 1.0},
            }
        )
    )
    return d


@pytest.fixture
def app(bundle, monkeypatch) -> AppTest:
    monkeypatch.setenv("ANIMEREC_DEMO_DIR", str(bundle))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def page(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_first_visit_shows_popular_titles_and_starters(app):
    assert "Popular right now" in page(app)
    assert any(b.key and b.key.startswith("starter_") for b in app.button)


def test_picking_a_favorite_personalizes(app):
    app.multiselect(key="liked").set_value(["Alpha"]).run()
    assert not app.exception
    text = page(app)
    assert "Your top 10" in text and "Because you liked Alpha" in text


def test_starter_button_adds_a_favorite(app):
    next(b for b in app.button if b.key == "starter_0").click().run()
    assert not app.exception
    assert app.multiselect(key="liked").value == ["Alpha"]
    assert "Your top 10" in page(app)


def test_new_only_filter_surfaces_cold_title(app):
    app.multiselect(key="liked").set_value(["Alpha"])
    app.toggle(key="new_only").set_value(True).run()
    assert not app.exception
    text = page(app)
    assert "Fresh" in text and "No viewer history yet" in text


def test_no_matches_shows_guidance(app):
    app.button_group(key="genre_chip").set_value("Drama")
    app.button_group(key="nav").set_value("Movie").run()
    assert not app.exception
    assert any("widening" in i.value for i in app.info)


def test_real_covers_show_by_default(app):
    assert "cdn.myanimelist.net" in page(app)
    assert "https://myanimelist.net/anime/1" in page(app)


def test_hero_prefers_wide_banner(app):
    assert 'class="sharp banner" src="https://s4.anilist.co/file/banner/1.jpg"' in page(app)


def test_covers_can_be_turned_off(bundle, monkeypatch):
    monkeypatch.setenv("ANIMEREC_DEMO_DIR", str(bundle))
    monkeypatch.setenv("ANIMEREC_SHOW_COVERS", "0")
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    assert "cdn.myanimelist.net" not in page(at)


def test_dataset_text_is_escaped(app):
    assert "Delta &lt;b&gt;Show&lt;/b&gt;" in page(app)
    assert "<b>Show</b>" not in page(app)
