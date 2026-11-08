"""End-to-end checks of the Streamlit page on a synthetic bundle."""
from __future__ import annotations

import base64
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
            "episodes": [12] * n,
            "studios_text": ["Madhouse"] * n,
            "synopsis_full": ["A long story."] * n,
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
    at.session_state["onboarded"] = True
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
    at.session_state["onboarded"] = True
    at.run()
    assert not at.exception
    assert "cdn.myanimelist.net" not in page(at)


@pytest.fixture
def fresh(bundle, monkeypatch) -> AppTest:
    monkeypatch.setenv("ANIMEREC_DEMO_DIR", str(bundle))
    monkeypatch.setenv("ANIMEREC_LOADER_SECONDS", "0")
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def press(at: AppTest, key: str) -> None:
    next(b for b in at.button if b.key == key).click().run()
    assert not at.exception


def test_new_visitor_sees_onboarding_first(fresh):
    assert "What do you like to watch?" in page(fresh)
    assert "Popular right now" not in page(fresh)


def test_onboarding_answers_shape_recommendations(fresh):
    fresh.button_group(key="ob_genres").set_value(["Action"])
    press(fresh, "ob_next")
    assert "Which of these have you loved?" in page(fresh)
    press(fresh, "ob_pick_0")
    press(fresh, "ob_next")
    fresh.button_group(key="ob_avoid").set_value(["Comedy"])
    press(fresh, "ob_next")
    press(fresh, "ob_next")
    text = page(fresh)
    assert "Your top 10" in text and "Because you liked Alpha" in text
    assert "Gamma" not in text and "Epsilon" not in text


def test_skip_goes_straight_to_the_feed(fresh):
    press(fresh, "ob_skip")
    assert "Popular right now" in page(fresh)


def test_genres_filter_feed_when_no_titles_picked(fresh):
    fresh.button_group(key="ob_genres").set_value(["Drama"])
    press(fresh, "ob_skip")
    text = page(fresh)
    assert "Delta" in text and "Eta" in text and "Alpha" not in text


def test_format_answer_filters_feed(fresh):
    for _ in range(3):
        press(fresh, "ob_next")
    fresh.button_group(key="ob_format").set_value("Movies")
    press(fresh, "ob_next")
    text = page(fresh)
    assert "Gamma" in text and "Alpha" not in text


def token(**kw) -> str:
    return base64.urlsafe_b64encode(json.dumps(kw).encode()).decode().rstrip("=")


def test_saved_answers_skip_onboarding_on_reload(bundle, monkeypatch):
    monkeypatch.setenv("ANIMEREC_DEMO_DIR", str(bundle))
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["p"] = token(l=["Alpha"], d=[], g=[], a=[], f="Both", e="Any time")
    at.run()
    assert not at.exception
    assert "Because you liked Alpha" in page(at)
    assert "What do you like to watch?" not in page(at)


def test_corrupt_saved_answers_fall_back_to_onboarding(bundle, monkeypatch):
    monkeypatch.setenv("ANIMEREC_DEMO_DIR", str(bundle))
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["p"] = "not-valid!!"
    at.run()
    assert not at.exception
    assert "What do you like to watch?" in page(at)


def test_finishing_saves_answers_in_the_url(fresh):
    press(fresh, "ob_skip")
    assert fresh.query_params["p"]


def test_loading_screen_shows_after_last_question(bundle, monkeypatch):
    monkeypatch.setenv("ANIMEREC_DEMO_DIR", str(bundle))
    monkeypatch.setenv("ANIMEREC_LOADER_SECONDS", "0")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["ob_step"] = 3
    at.run()
    press(at, "ob_next")
    assert at.session_state["onboarded"] and not at.session_state["ob_loading"]


def test_hero_entrance_plays_once_after_onboarding(fresh):
    press(fresh, "ob_skip")
    assert 'class="hero intro"' in page(fresh)
    fresh.run()
    assert 'class="hero intro"' not in page(fresh)


def test_select_all_cannot_add_many_titles_at_once(app):
    app.multiselect(key="liked").set_value(["Alpha", "Beta", "Gamma"]).run()
    assert not app.exception
    assert app.multiselect(key="liked").value == ["Alpha"]


def test_titles_can_still_be_added_one_by_one(app):
    app.multiselect(key="liked").set_value(["Alpha"]).run()
    app.multiselect(key="liked").set_value(["Alpha", "Beta"]).run()
    assert not app.exception
    assert app.multiselect(key="liked").value == ["Alpha", "Beta"]


def test_logo_appears_on_onboarding_and_main_screen(app, fresh):
    assert 'alt="AnimeRec"' in page(app)
    assert 'alt="AnimeRec"' in page(fresh)


def test_cards_open_info_popup_instead_of_leaving_the_site(app):
    text = page(app)
    assert 'class="tile" href="#title-1"' in text
    assert 'class="modal" id="title-1"' in text
    assert 'class="tile" href="https://myanimelist.net' not in text


def test_popup_has_details_and_a_redirect_button(app):
    text = page(app)
    assert "12 episodes" not in text
    assert "<span>Studio</span>Madhouse" in text
    assert "<span>Episodes</span>12" in text
    assert 'class="out" href="https://myanimelist.net/anime/1" target="_blank"' in text


def test_retake_returns_to_onboarding(app):
    press(app, "retake")
    assert "What do you like to watch?" in page(app)


def test_dataset_text_is_escaped(app):
    assert "Delta &lt;b&gt;Show&lt;/b&gt;" in page(app)
    assert "<b>Show</b>" not in page(app)
