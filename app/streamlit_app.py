"""Anime recommender demo: a streaming-style page with instant, explained recommendations."""
from __future__ import annotations

import hashlib
import html
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from animerec.serving import DemoRecommender, Recommendation, load_demo  # noqa: E402

BUNDLE = Path(os.environ.get("ANIMEREC_DEMO_DIR", ROOT / "artifacts" / "demo"))
# Cover art belongs to its rights holders; set this to 0 to use generated art instead.
SHOW_COVERS = os.environ.get("ANIMEREC_SHOW_COVERS", "1") != "0"
COVER_PREFIX = "https://cdn.myanimelist.net/"
BANNER_PREFIX = "https://s4.anilist.co/"

GENRE_HUES = {
    "Action": 8, "Adventure": 32, "Comedy": 44, "Drama": 280, "Fantasy": 262, "Horror": 350, "Mystery": 228,
    "Psychological": 300, "Romance": 335, "Sci-Fi": 190, "Slice of Life": 140, "Sports": 96, "Supernatural": 245,
    "Mecha": 205, "Music": 165, "School": 215, "Historical": 25, "Military": 75, "Kids": 55,
}

NAV = ["All", "TV", "Movie", "OVA", "ONA", "Special"]
NAV_LABELS = {"All": "Home", "TV": "TV Shows", "Movie": "Movies"}
GENRES = ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Slice of Life", "Sports", "Supernatural"]

st.set_page_config(page_title="AnimeRec", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
:root { --gap: .55rem; --blue: #2a1fd1; --blue2: #3a2fe6; --purple: #8a12b8; }
.stApp { background: #000; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stHeaderActionElements"] { display: none !important; }
.block-container { max-width: 100%; padding: var(--gap) var(--gap) 1rem !important; }
.block-container [data-testid="stVerticalBlock"] { gap: var(--gap); }
.st-key-topbar { background: var(--blue); border-radius: 18px; padding: .55rem 1.1rem; }
.brand { font-size: 1.02rem; font-weight: 700; color: #fff; letter-spacing: -.01em; }
.st-key-nav button { background: rgba(255,255,255,.13) !important; border: 0 !important; box-shadow: none !important; border-radius: 999px !important; padding: 0 .95rem !important; min-height: 34px !important; }
.st-key-nav button p { color: #fff; font-size: .8rem; font-weight: 500; text-transform: uppercase; letter-spacing: .01em; }
.st-key-nav button[aria-checked="true"] { background: rgba(255,255,255,.34) !important; }
.st-key-topbar [data-testid="stMultiSelect"] div[data-baseweb="select"] > div { background: rgba(255,255,255,.13) !important; border: 0 !important; border-radius: 999px !important; }
[data-testid="stPopover"] button { background: rgba(255,255,255,.13) !important; border: 0 !important; border-radius: 999px !important; color: #fff !important; }
.herorail { display: flex; gap: var(--gap); overflow-x: auto; scroll-snap-type: x mandatory; scrollbar-width: none; }
.herorail::-webkit-scrollbar { display: none; }
.hero { position: relative; flex: 0 0 calc(100% - 8.5rem); height: clamp(460px, 72vh, 720px); border-radius: 22px; overflow: hidden; background: #15151b; scroll-snap-align: start; }
.hero .blur { position: absolute; inset: -40px; background-size: cover; background-position: center 25%; filter: blur(30px) brightness(.62) saturate(1.25); }
.hero .sharp { position: absolute; right: 0; top: 0; width: 70%; height: 100%; object-fit: cover; object-position: center 18%; -webkit-mask-image: linear-gradient(90deg, transparent 0, #000 34%); mask-image: linear-gradient(90deg, transparent 0, #000 34%); }
.hero .sharp.banner { width: 100%; object-position: center; -webkit-mask-image: none; mask-image: none; }
.hero::after { content: ""; position: absolute; inset: 0; background: linear-gradient(0deg, rgba(0,0,0,.82) 0%, rgba(0,0,0,.25) 42%, rgba(0,0,0,0) 62%), linear-gradient(90deg, rgba(0,0,0,.42) 0%, rgba(0,0,0,0) 50%); }
.hero .eyebrow { position: absolute; z-index: 2; top: 1.1rem; right: 1.3rem; color: #fff; font-size: .76rem; }
.hero .inner { position: absolute; z-index: 2; left: 1.5rem; bottom: 1.35rem; width: min(44%, 560px); }
.hero h2 { color: #fff; font-size: clamp(2.2rem, 4.2vw, 4rem); line-height: 1.02; font-weight: 400; letter-spacing: -.025em; margin: 0 0 .7rem; padding: 0; overflow-wrap: anywhere; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.hero .why { color: #fff; font-size: .95rem; font-weight: 600; margin-bottom: .3rem; }
.hero .syn { color: rgba(255,255,255,.86); font-size: .82rem; line-height: 1.38; margin-bottom: .75rem; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.pill { display: inline-block; font-size: .76rem; color: #111; background: #fff; border-radius: 999px; padding: .3rem .85rem; margin: 0 .3rem .25rem 0; }
.pill.ghost { color: #fff; background: transparent; border: 1px solid rgba(255,255,255,.65); }
.hero .side { position: absolute; z-index: 2; right: 1.3rem; bottom: 1.35rem; text-align: right; }
.hero .score { color: #fff; font-size: clamp(2rem, 3.3vw, 3.1rem); font-weight: 300; letter-spacing: -.03em; margin-bottom: .6rem; }
.hero .score sup { font-size: .5em; }
.go { display: inline-block; background: var(--blue2); color: #fff !important; text-decoration: none !important; font-size: .78rem; font-weight: 600; text-transform: uppercase; border-radius: 999px; padding: .68rem 1.2rem; white-space: nowrap; }
.flag { display: inline-block; font-size: .66rem; font-weight: 700; color: #111; background: #8ff0bd; border-radius: 999px; padding: .08rem .5rem; margin-left: .4rem; }
.flag.cold { background: #ffc46b; }
.st-key-panel { background: #161616; border-radius: 22px; padding: 1.15rem 1.4rem 1.2rem; }
.row-title { color: #fff; font-size: clamp(1.45rem, 2vw, 2rem); font-weight: 400; letter-spacing: -.02em; text-transform: uppercase; margin: .15rem 0 .6rem; }
.st-key-love_top button { background: var(--purple) !important; border: 0 !important; border-radius: 999px !important; color: #fff !important; }
.rail { display: flex; gap: .6rem; overflow-x: auto; padding-bottom: .5rem; scroll-snap-type: x proximity; }
.rail::-webkit-scrollbar { height: 5px; } .rail::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 9px; }
.tile { flex: 0 0 clamp(250px, 21vw, 390px); scroll-snap-align: start; text-decoration: none !important; display: block; color: #fff; }
.poster { position: relative; aspect-ratio: 16 / 9; border-radius: 14px; overflow: hidden; transition: transform .16s ease; }
.tile:hover .poster { transform: translateY(-3px); }
.poster img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: center 22%; }
.poster .letter { position: absolute; inset: 0; display: grid; place-items: center; font-size: 5rem; font-weight: 800; color: rgba(255,255,255,.3); }
.poster .shade { position: absolute; inset: 0; background: linear-gradient(200deg, rgba(0,0,0,.6), rgba(0,0,0,0) 45%); }
.poster .rate { position: absolute; top: .45rem; right: .7rem; color: #fff; font-size: 1.55rem; font-weight: 300; letter-spacing: -.02em; }
.poster .rate sup { font-size: .55em; }
.tile .name { color: #fff; font-size: .92rem; font-weight: 500; margin-top: .5rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tile .meta { color: #a4a4a4; font-size: .75rem; margin-top: .1rem; }
.tile .why { color: #8f8f8f; font-size: .74rem; margin-top: .12rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.foot { color: #6f6f6f; font-size: .72rem; margin-top: .9rem; padding: 0 .6rem; }
@media (max-width: 640px) {
  .hero { flex-basis: calc(100% - 2.2rem); height: 560px; } .hero .inner { width: auto; right: 1.1rem; left: 1.1rem; bottom: 4.7rem; }
  .hero .side { right: 1.1rem; left: 1.1rem; bottom: 1.1rem; display: flex; justify-content: space-between; align-items: center; } .hero .score { margin: 0; font-size: 2rem; }
  .hero .sharp, .hero .sharp.banner { width: 100%; height: 66%; -webkit-mask-image: linear-gradient(0deg, transparent 0, #000 40%); mask-image: linear-gradient(0deg, transparent 0, #000 40%); }
  .st-key-panel { padding: 1rem; } .tile { flex-basis: 78vw; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading recommender...")
def get_model(path: str) -> DemoRecommender:
    return load_demo(path)


def art_style(r: Recommendation) -> str:
    """Generated fallback artwork for titles without a cover."""
    d = hashlib.sha256(r.title.encode()).digest()
    h = d[0] * 360 // 255
    return f"background-image: linear-gradient(150deg, hsl({h} 55% 42%), hsl({(h + 50) % 360} 50% 20%));"


def cover(r: Recommendation) -> str | None:
    if SHOW_COVERS and r.cover_url and r.cover_url.startswith(COVER_PREFIX):
        return html.escape(r.cover_url, quote=True)
    return None


def banner(r: Recommendation) -> str | None:
    if SHOW_COVERS and r.banner_url and r.banner_url.startswith(BANNER_PREFIX):
        return html.escape(r.banner_url, quote=True)
    return None


def large(url: str) -> str:
    """MyAnimeList serves a bigger version of each cover with an 'l' before the extension."""
    return url[:-4] + "l.jpg" if url.endswith(".jpg") and not url.endswith("l.jpg") else url


def initial(r: Recommendation) -> str:
    return html.escape(next((c.upper() for c in r.title if c.isalnum()), "?"))


def flag(r: Recommendation) -> str:
    if r.is_cold:
        return '<span class="flag cold">No viewer history yet</span>'
    return '<span class="flag">New</span>' if r.is_new else ""


def page_url(r: Recommendation) -> str | None:
    return f"https://myanimelist.net/anime/{int(r.mal_id)}" if r.mal_id is not None else None


def score_text(r: Recommendation) -> str:
    return f"{r.rating:.1f}<sup>&#9733;</sup>" if r.rating is not None and r.rating == r.rating else ""


def hero(r: Recommendation, label: str) -> str:
    """Featured pick. Every dataset string is escaped before it reaches HTML."""
    url = cover(r)
    bg = f"background-image: url('{url}');" if url else art_style(r)
    wide_art = banner(r)
    if wide_art:
        sharp = f'<img class="sharp banner" src="{wide_art}" alt="" referrerpolicy="no-referrer">'
    else:
        sharp = f'<img class="sharp" src="{large(url)}" alt="" referrerpolicy="no-referrer">' if url else ""
    pills = "".join(f'<span class="pill">{html.escape(g)}</span>' for g in r.genres[:3])
    if r.year:
        pills += f'<span class="pill ghost">{r.year}</span>'
    if r.type:
        pills += f'<span class="pill ghost">{html.escape(r.type)}</span>'
    link = page_url(r)
    go = f'<a class="go" href="{link}" target="_blank" rel="noopener">View on MyAnimeList &#8599;</a>' if link else ""
    score = score_text(r)
    score_html = f'<div class="score">{score}</div>' if score else ""
    return (
        f'<div class="hero"><div class="blur" style="{bg}"></div>{sharp}<div class="eyebrow">{label}{flag(r)}</div>'
        f'<div class="inner"><h2>{html.escape(r.title)}</h2><div class="why">{html.escape(r.explanation)}</div>'
        f'<div class="syn">{html.escape(r.synopsis)}</div>{pills}</div><div class="side">{score_html}{go}</div></div>'
    )


def tile(r: Recommendation) -> str:
    url = cover(r)
    art = f'<div class="letter">{initial(r)}</div>'
    wide_art = banner(r) or (large(url) if url else None)
    if wide_art:
        art += f'<img src="{wide_art}" alt="" loading="lazy" referrerpolicy="no-referrer">'
    score = score_text(r)
    rate = f'<div class="shade"></div><div class="rate">{score}</div>' if score else ""
    meta = " · ".join(x for x in [html.escape(r.type or ""), str(r.year) if r.year else ""] if x)
    href = page_url(r)
    tag, end = (f'<a class="tile" href="{href}" target="_blank" rel="noopener">', "</a>") if href else ('<div class="tile">', "</div>")
    return (
        f'{tag}<div class="poster" style="{art_style(r)}">{art}{rate}</div><div class="name">{html.escape(r.title)}</div>'
        f'<div class="meta">{meta}{flag(r)}</div><div class="why">{html.escape(r.explanation)}</div>{end}'
    )


def rail(items: list[Recommendation]) -> None:
    st.markdown(f'<div class="rail">{"".join(tile(r) for r in items)}</div>', unsafe_allow_html=True)


def row_title(title: str) -> None:
    st.markdown(f'<div class="row-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def starter_titles(model: DemoRecommender, n: int = 6) -> list[str]:
    """Most popular recent titles, one per franchise."""
    picks: list[str] = []
    franchises: set[int] = set()
    for i in np.argsort(-model.recent_counts, kind="stable"):
        f = int(model.franchise[i])
        if f not in franchises:
            franchises.add(f)
            picks.append(model.titles[int(i)])
        if len(picks) == n:
            break
    return picks


def add_favorite(title: str) -> None:
    current = st.session_state.get("liked", [])
    if title not in current:
        st.session_state["liked"] = [*current, title]


def clear_all() -> None:
    st.session_state["liked"] = []
    st.session_state["disliked"] = []


if not (BUNDLE / "metadata.json").exists():
    st.error("The model bundle was not found. Run `make all` to build it, then reload this page.")
    st.stop()

model = get_model(str(BUNDLE))
options = sorted(set(model.titles), key=str.lower)
years = [int(y) for y in model.years if y is not None]
y_min, y_max = min(years), max(years)
snapshot = html.escape(str(model.meta.get("snapshot", "the data snapshot")))

with st.container(key="topbar"):
    t1, t2, t3, t4 = st.columns([1, 4.4, 3.2, 0.8], vertical_alignment="center")
    t1.markdown('<div class="brand">AnimeRec</div>', unsafe_allow_html=True)
    with t2:
        nav = st.pills("Type", NAV, default="All", selection_mode="single", key="nav", label_visibility="collapsed", format_func=lambda n: NAV_LABELS.get(n, n))
    liked = t3.multiselect("Titles you loved", options, key="liked", placeholder="Search titles you love...", label_visibility="collapsed")
    with t4, st.popover("Refine", use_container_width=True):
        if not st.session_state.get("liked") and not st.session_state.get("disliked"):
            st.caption("Quick start")
            for i, title in enumerate(starter_titles(model)):
                st.button(title, key=f"starter_{i}", on_click=add_favorite, args=(title,), use_container_width=True)
        chip = st.pills("Genre", GENRES, selection_mode="single", key="genre_chip")
        disliked = st.multiselect("Titles you disliked", options, key="disliked", placeholder="Search for a title")
        year_range = st.slider("Release year", y_min, y_max, (y_min, y_max), key="years")
        new_only = st.toggle("New titles only", key="new_only", help=f"Released in the year before {snapshot}, or with no viewer history yet.")
        same_franchise = st.toggle("Include sequels of titles I picked", key="same_franchise")
        if liked or disliked:
            st.button("Start over", key="reset", on_click=clear_all, use_container_width=True)

filters = {
    "genres": [chip] if chip else None,
    "year_range": None if year_range == (y_min, y_max) else year_range,
    "types": [nav] if nav and nav != "All" else None,
    "include_same_franchise": same_franchise,
}
result = model.recommend(liked=liked, disliked=disliked, new_only=new_only, **filters)

for name in result.unknown:
    st.warning(f"Not in the catalog, so it was ignored: {name}")

label = "Your top 10" if result.personalized else "Popular right now"
if not result.items:
    st.info(result.message)
else:
    heroes = "".join(hero(r, f"{label} · #{i}") for i, r in enumerate(result.items[:3], start=1))
    st.markdown(f'<div class="herorail">{heroes}</div>', unsafe_allow_html=True)
    with st.container(key="panel"):
        topic = chip or "Anime"
        h1, h2 = st.columns([6, 1.3], vertical_alignment="center")
        with h1:
            row_title(f"Top picks in {topic}" if result.personalized else f"Recommended in {topic}")
        h2.button("♡ Love the #1 pick", key="love_top", on_click=add_favorite, args=(result.items[0].title,), use_container_width=True)
        if len(result.items) > 3:
            rail(result.items[3:])
        if result.personalized and not new_only:
            fresh = model.recommend(liked=liked, disliked=disliked, new_only=True, **filters)
            if fresh.items:
                row_title("New titles you might like")
                rail(fresh.items)

art_note = (
    "Cover and banner art load from MyAnimeList and AniList and belong to their rights holders."
    if SHOW_COVERS else "Poster art is generated."
)
st.markdown(
    f'<div class="foot">Built on MyAnimeList activity up to {snapshot}, so newer shows are not included. Nothing you select is stored. '
    f"Star ratings are MyAnimeList community scores, shown for reference and not used by the recommender. "
    f"Data: MyAnimeList Dataset by azathoth42 (ODbL database, DbCL contents) and MyAnimeList Dataset by svanoo (CC0). {art_note}</div>",
    unsafe_allow_html=True,
)
