"""Anime recommender demo: a streaming-style page with instant, explained recommendations."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

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
NAV_ICONS = {"All": "home", "TV": "tv", "Movie": "movie", "OVA": "video_library", "ONA": "public", "Special": "star"}
GENRES = ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Slice of Life", "Sports", "Supernatural"]

FAVICON = ROOT / "app" / "assets" / "favicon.png"
st.set_page_config(page_title="AnimeRec", page_icon=str(FAVICON) if FAVICON.exists() else None, layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap");
body *:not([data-testid="stIconMaterial"]):not([role="img"]) { font-family: "Inter", system-ui, sans-serif !important; }
b, strong { font-weight: 500 !important; }
:root { --rail: 76px; --left: calc(var(--rail) + 1.6rem); --gap: .55rem; }
.stApp { background: #000; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stHeaderActionElements"] { display: none !important; }
.block-container { max-width: 100%; padding: 0 0 1rem !important; position: relative; }
.st-key-siderail { position: fixed; left: 0; top: 0; bottom: 0; width: var(--rail) !important; z-index: 40; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: .35rem; background: linear-gradient(90deg, rgba(0,0,0,.72), rgba(0,0,0,0)); }
.st-key-siderail button { background: transparent !important; border: 0 !important; box-shadow: none !important; color: #fff !important; min-height: 42px !important; width: 46px !important; padding: 0 !important; justify-content: center; }
.st-key-siderail button p, .st-key-siderail button span { font-size: 1.45rem !important; }
.st-key-pop_search button [data-testid="stIconMaterial"], .st-key-pop_refine button [data-testid="stIconMaterial"] { display: none; }
.st-key-nav [role="radiogroup"], .st-key-nav [data-testid="stButtonGroup"] > div { flex-direction: column !important; flex-wrap: nowrap !important; gap: .35rem !important; align-items: center; }
.st-key-nav button { border-radius: 0 !important; color: rgba(255,255,255,.72) !important; }
.st-key-nav button[aria-checked="true"] { color: #fff !important; box-shadow: inset 0 -2px 0 #e50914 !important; }
.st-key-siderail > div, .st-key-pop_search, .st-key-pop_refine, .st-key-pop_search [data-testid="stPopover"], .st-key-pop_refine [data-testid="stPopover"], .st-key-pop_search [data-testid="stElementContainer"], .st-key-pop_refine [data-testid="stElementContainer"] { width: 46px !important; min-width: 46px !important; display: flex; justify-content: center; align-items: center; }
.st-key-pop_search button > div, .st-key-pop_refine button > div { justify-content: center; margin: 0 !important; gap: 0 !important; }
.st-key-pop_search button [aria-hidden="true"], .st-key-pop_refine button [aria-hidden="true"] { display: none !important; }
.brandmark { display: flex; justify-content: center; margin-bottom: .9rem; }
.brandmark img { width: 32px; height: 32px; display: block; }
.herorail { margin-top: -2rem; }
.hero-logo { display: none; }
.hero { position: relative; height: 100vh; height: 100svh; min-height: 560px; overflow: hidden; background: #111; }
.hero .blur { position: absolute; inset: -40px; background-size: cover; background-position: center 25%; filter: blur(30px) brightness(.55) saturate(1.2); }
.hero .sharp { position: absolute; right: 0; top: 0; width: 72%; height: 100%; object-fit: cover; object-position: center 18%; -webkit-mask-image: linear-gradient(90deg, transparent 0, #000 34%); mask-image: linear-gradient(90deg, transparent 0, #000 34%); }
.hero .sharp.banner { width: 100%; object-position: center 25%; -webkit-mask-image: none; mask-image: none; }
.hero::after { content: ""; position: absolute; inset: 0; background: linear-gradient(90deg, rgba(0,0,0,.94) 0%, rgba(0,0,0,.72) 26%, rgba(0,0,0,0) 62%), linear-gradient(0deg, #000 0%, rgba(0,0,0,.6) 24%, rgba(0,0,0,0) 46%); }
.hero .inner { position: absolute; z-index: 2; left: var(--left); bottom: 37vh; bottom: 37svh; width: min(36vw, 500px); }
.hero .eyebrow { color: rgba(255,255,255,.82); font-size: .78rem; font-weight: 500; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .45rem; }
.hero h2 { color: #fff; font-size: clamp(2.2rem, 4.3vw, 4.3rem); line-height: 1; font-weight: 500; letter-spacing: -.02em; margin: 0 0 .75rem; padding: 0; overflow-wrap: anywhere; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.hero .meta { color: rgba(255,255,255,.76); font-size: .84rem; margin-bottom: .45rem; }
.hero .why { color: #fff; font-size: .98rem; font-weight: 500; margin-bottom: .35rem; }
.hero .syn { color: #fff; font-size: clamp(.92rem, 1.1vw, 1.08rem); line-height: 1.38; margin-bottom: 1rem; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.go { display: inline-flex; align-items: center; justify-content: center; gap: .55rem; width: 226px; height: 46px; box-sizing: border-box; background: #fff; color: #000 !important; text-decoration: none !important; font-size: .98rem; font-weight: 500; border-radius: 5px; }
.flag { display: inline-block; font-size: .64rem; font-weight: 500; letter-spacing: 0; text-transform: none; color: #111; background: #ffc46b; border-radius: 4px; padding: .05rem .36rem; margin-left: .4rem; }
.flag.new { background: #8ff0bd; }
.st-key-love { position: absolute; z-index: 15; left: calc(var(--left) + 226px + .65rem); top: calc(63vh - 46px); top: calc(63svh - 46px); width: 150px !important; }
.st-key-love button { height: 46px; background: rgba(120,120,120,.72) !important; border: 0 !important; border-radius: 5px !important; color: #fff !important; font-weight: 500; }
.st-key-rows { position: relative; z-index: 10; margin-top: -31vh; margin-top: -31svh; padding-left: var(--left); }
.row-title { color: #fff; font-size: 1.32rem; font-weight: 500; margin: .1rem 0 .55rem; }
.rail { display: flex; gap: .45rem; overflow-x: auto; padding: 0 1rem .55rem 0; scroll-snap-type: x proximity; scrollbar-width: none; }
.rail::-webkit-scrollbar { display: none; }
.tile { --tile: clamp(150px, 11.2vw, 230px); flex: 0 0 var(--tile); width: var(--tile); min-width: 0; max-width: var(--tile); scroll-snap-align: start; text-decoration: none !important; display: block; color: #fff; }
.poster { position: relative; aspect-ratio: 2 / 3; border-radius: 4px; overflow: hidden; transition: transform .16s ease; }
.tile:hover .poster { transform: scale(1.04); }
.poster img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.poster .letter { position: absolute; inset: 0; display: grid; place-items: center; font-size: 5rem; font-weight: 500; color: rgba(255,255,255,.3); }
.tile .name { color: #fff; font-size: .84rem; font-weight: 500; margin-top: .4rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tile .meta, .tile .why { color: #8f8f8f; font-size: .72rem; margin-top: .08rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tile .why { color: #b7b7b7; }
.foot { color: #6f6f6f; font-size: .72rem; margin-top: 1rem; padding: 0 1.6rem 0 var(--left); }
.empty { height: 11rem; }
@media (max-width: 640px) {
  :root { --left: 1rem; }
  .st-key-siderail { --cell: 36px; --cgap: 6px; top: auto; left: 0; right: 0; bottom: calc(.75rem + env(safe-area-inset-bottom)); width: fit-content !important; max-width: calc(100vw - 1rem); margin: 0 auto; height: 52px; flex-direction: row; justify-content: center; align-items: center; gap: var(--cgap); padding: 0 8px; background: #0b0b0b; border: 1px solid rgba(255,255,255,.1); border-radius: 999px; box-shadow: 0 8px 24px rgba(0,0,0,.45); }
  .block-container { padding-bottom: 6.5rem !important; }
  .st-key-nav button { color: rgba(255,255,255,.86) !important; }
  .st-key-nav button[aria-checked="true"] { box-shadow: none !important; background: #fff !important; border-radius: 50% !important; }
  .st-key-nav button[aria-checked="true"] [role="img"], .st-key-nav button[aria-checked="true"] [data-testid="stIconMaterial"], .st-key-nav button[aria-checked="true"] p { color: #000 !important; }
  .st-key-siderail button [role="img"], .st-key-siderail button [data-testid="stIconMaterial"] { font-variation-settings: "wght" 300 !important; font-weight: 300 !important; }
  .st-key-pop_refine { position: relative; }
  .st-key-pop_refine::before { content: ""; position: absolute; left: -3.5px; top: 9px; bottom: 9px; width: 1px; background: rgba(255,255,255,.2); }
  .brandmark { display: none; }
  .st-key-siderail > div:has(.brandmark) { display: none !important; }
  .st-key-siderail > div, .st-key-pop_search, .st-key-pop_refine, .st-key-pop_search [data-testid="stPopover"], .st-key-pop_refine [data-testid="stPopover"], .st-key-pop_search [data-testid="stElementContainer"], .st-key-pop_refine [data-testid="stElementContainer"] { width: var(--cell) !important; min-width: var(--cell) !important; height: var(--cell); display: flex; align-items: center; justify-content: center; margin: 0 !important; }
  .st-key-siderail > .st-key-nav { width: max-content !important; min-width: max-content !important; flex: none !important; }
  .st-key-nav [role="radiogroup"], .st-key-nav [data-testid="stButtonGroup"], .st-key-nav [data-testid="stButtonGroup"] > div { width: max-content !important; gap: var(--cgap) !important; margin: 0 !important; align-items: center; }
  .st-key-siderail button { width: var(--cell) !important; min-width: var(--cell) !important; height: var(--cell) !important; min-height: var(--cell) !important; flex: none !important; display: flex !important; align-items: center !important; justify-content: center !important; padding: 0 !important; }
  .st-key-siderail button p, .st-key-siderail button span { font-size: 1.2rem !important; line-height: 1 !important; margin: 0 !important; }
  .st-key-siderail button > div, .st-key-siderail button [data-testid="stMarkdownContainer"] { display: flex !important; align-items: center; justify-content: center; margin: 0 !important; }
  .st-key-nav button [role="img"], .st-key-nav button [data-testid="stIconMaterial"] { position: relative; top: -4px; }  /* the icon font sits low in these buttons */
  .hero-logo { display: block; position: absolute; z-index: 3; top: .85rem; left: 1rem; }
  .hero-logo img { width: 34px; height: 34px; display: block; }
  .st-key-nav [role="radiogroup"], .st-key-nav [data-testid="stButtonGroup"] > div { flex-direction: row !important; }
  .st-key-siderail > .st-key-nav { width: max-content !important; min-width: max-content !important; flex: none !important; }
  .st-key-nav [role="radiogroup"], .st-key-nav [data-testid="stButtonGroup"] { width: max-content !important; }
  .hero .inner { width: auto; right: 1rem; }
  .hero .sharp, .hero .sharp.banner { width: 100%; height: 60%; -webkit-mask-image: linear-gradient(0deg, transparent 0, #000 40%); mask-image: linear-gradient(0deg, transparent 0, #000 40%); }
  .go { width: 200px; } .st-key-love { left: calc(1rem + 200px + .5rem); width: 136px !important; }
  .tile { --tile: 40vw; }
}
.ob-head { max-width: 860px; margin: 0 auto; padding: 3.2rem 1.2rem .6rem; text-align: center; }
.ob-brand { color: #fff; font-size: 1.3rem; font-weight: 500; letter-spacing: -.01em; margin-bottom: 1.5rem; display: flex; justify-content: center; align-items: center; gap: .55rem; }
.ob-brand img { width: 34px; height: 34px; display: block; }
.ob-dots { display: flex; justify-content: center; gap: .4rem; margin-bottom: 1.8rem; }
.ob-dots span { width: 34px; height: 3px; border-radius: 3px; background: rgba(255,255,255,.2); }
.ob-dots span.on { background: #e50914; }
.ob-head h1 { color: #fff; font-size: clamp(1.9rem, 3.4vw, 3rem); line-height: 1.1; font-weight: 500; letter-spacing: -.02em; margin: 0 0 .5rem; padding: 0; }
.ob-head p { color: #d8d8d8; font-size: 1rem; margin: 0; }
.st-key-ob_body { max-width: 1040px; margin: 1.6rem auto .4rem; padding: 0 1.2rem; }
.st-key-ob_body [data-testid="stButtonGroup"] { display: flex; justify-content: center; }
.st-key-ob_body [role="radiogroup"], .st-key-ob_body [data-testid="stButtonGroup"] > div { justify-content: center; }
.st-key-ob_body [data-testid="stButtonGroup"] button { background: rgba(28,28,28,.88) !important; border: 1px solid #333 !important; border-radius: 999px !important; min-height: 44px !important; padding: 0 1.2rem !important; }
.st-key-ob_body [data-testid="stButtonGroup"] button[aria-checked="true"], .st-key-ob_body [data-testid="stButtonGroup"] button[aria-pressed="true"], .st-key-ob_body [data-testid="stButtonGroup"] button[data-selected="true"] { background: #fff !important; border-color: #fff !important; }
.st-key-ob_body [data-testid="stButtonGroup"] button[aria-checked="true"] p, .st-key-ob_body [data-testid="stButtonGroup"] button[aria-pressed="true"] p, .st-key-ob_body [data-testid="stButtonGroup"] button[data-selected="true"] p { color: #000 !important; }
.st-key-ob_body [data-testid="stElementContainer"]:has([data-testid="stButtonGroup"]) { display: flex; justify-content: center; width: 100% !important; }
.st-key-ob_body [data-testid="stElementContainer"]:has([data-testid="stButtonGroup"]) > * { width: fit-content !important; margin: 0 auto; }
.st-key-ob_grid [data-testid="stColumn"] { position: relative; }
.st-key-ob_grid [data-testid="stColumn"] div:not(.ob-poster):not(.ob-check) { position: static !important; }
.st-key-ob_grid button::before { content: ""; position: absolute; inset: 0; cursor: pointer; z-index: 3; }
.ob-q { color: #fff; text-align: center; font-size: 1.05rem; font-weight: 500; margin: .9rem 0 .3rem; }
.ob-poster { position: relative; aspect-ratio: 2 / 3; border-radius: 6px; overflow: hidden; background: #1c1c1c; border: 2px solid transparent; }
.ob-poster.on { border-color: #fff; }
.ob-poster img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.ob-poster.on img { opacity: .55; }
.ob-check { position: absolute; top: .45rem; right: .45rem; z-index: 2; width: 26px; height: 26px; border-radius: 50%; background: #fff; color: #000; display: grid; place-items: center; font-size: .85rem; }
.st-key-ob_grid button { background: transparent !important; border: 0 !important; min-height: 30px !important; padding: 0 !important; }
.st-key-ob_grid button p { color: #f0f0f0; font-size: .76rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.st-key-ob_body [data-testid="stMultiSelect"] { max-width: 560px; margin: 1.2rem auto 0; }
.st-key-ob_body [data-testid="stMultiSelect"] label { justify-content: center; width: 100%; }
.st-key-ob_body [data-testid="stMultiSelect"] label p { color: rgba(255,255,255,.72) !important; font-size: .9rem; }
.st-key-ob_body [data-testid="stMultiSelect"] [role="group"] { min-height: 54px; padding: .25rem 1rem .25rem 3rem; background-color: rgba(255,255,255,.08) !important; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' stroke='rgba(255,255,255,.65)' stroke-width='2' stroke-linecap='round'%3E%3Ccircle cx='9' cy='9' r='6'/%3E%3Cpath d='m14 14 4 4'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: 1.15rem center; border: 1px solid rgba(255,255,255,.18) !important; border-radius: 999px !important; backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); transition: border-color .15s ease, background-color .15s ease; box-shadow: none !important; }
.st-key-ob_body [data-testid="stMultiSelect"] [role="group"]:hover { background-color: rgba(255,255,255,.12) !important; }
.st-key-ob_body [data-testid="stMultiSelect"] [role="group"]:focus-within { border-color: rgba(255,255,255,.6) !important; }
.st-key-ob_body [data-testid="stMultiSelect"] input { color: #fff !important; }
.st-key-ob_body [data-testid="stMultiSelect"] span[data-baseweb="tag"] { background: #fff !important; border-radius: 999px !important; }
.st-key-ob_body [data-testid="stMultiSelect"] span[data-baseweb="tag"] span { color: #000 !important; }
.st-key-ob_body [data-testid="stMultiSelect"] span[data-baseweb="tag"] svg { color: #000 !important; fill: #000 !important; }
[role="listbox"] { background: #161616 !important; border: 1px solid rgba(255,255,255,.14); border-radius: 18px; padding: .35rem !important; }
[role="option"] { border-radius: 12px; }
[role="option"]:hover, [role="option"][aria-selected="true"] { background: rgba(255,255,255,.1) !important; }
[data-testid="stPopoverBody"] { background: rgba(18,18,18,.92) !important; backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px); border: 1px solid rgba(255,255,255,.14) !important; border-radius: 22px !important; }
.st-key-liked label p, .st-key-disliked label p { color: rgba(255,255,255,.72) !important; font-size: .86rem; }
.st-key-liked [role="group"], .st-key-disliked [role="group"] { min-height: 48px; padding: .2rem .8rem .2rem 2.6rem; background-color: rgba(255,255,255,.08) !important; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 20 20' fill='none' stroke='rgba(255,255,255,.65)' stroke-width='2' stroke-linecap='round'%3E%3Ccircle cx='9' cy='9' r='6'/%3E%3Cpath d='m14 14 4 4'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: .95rem center; border: 1px solid rgba(255,255,255,.18) !important; border-radius: 999px !important; box-shadow: none !important; transition: border-color .15s ease, background-color .15s ease; }
.st-key-liked [role="group"]:hover, .st-key-disliked [role="group"]:hover { background-color: rgba(255,255,255,.12) !important; }
.st-key-liked [role="group"]:focus-within, .st-key-disliked [role="group"]:focus-within { border-color: rgba(255,255,255,.6) !important; }
.st-key-liked input, .st-key-disliked input { color: #fff !important; }
.st-key-reset button, [class*="st-key-starter_"] button { border-radius: 999px !important; background: rgba(255,255,255,.1) !important; border: 1px solid rgba(255,255,255,.2) !important; color: #fff !important; min-height: 44px; box-shadow: none !important; }
.st-key-reset button:hover, [class*="st-key-starter_"] button:hover { background: rgba(255,255,255,.18) !important; }
.st-key-ob_actions { max-width: 620px; margin: 1.4rem auto 2.4rem; padding: 0 1.2rem; }
.st-key-ob_actions [data-testid="stHorizontalBlock"] { justify-content: center; align-items: center; gap: .6rem; flex-direction: row; flex-wrap: nowrap; }
.st-key-ob_actions [data-testid="stColumn"] { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
.st-key-ob_actions button { height: 52px; min-height: 52px; border-radius: 999px !important; padding: 0 1.5rem !important; white-space: nowrap; }
.st-key-ob_actions button p { font-size: .98rem; font-weight: 500; }
.st-key-ob_skip button { background: transparent !important; border: 0 !important; color: rgba(255,255,255,.66) !important; }
.st-key-ob_skip button p { color: rgba(255,255,255,.66) !important; transition: color .15s ease; }
.st-key-ob_skip button:hover p { color: #fff !important; }
.st-key-ob_back button { background: rgba(255,255,255,.1) !important; border: 1px solid rgba(255,255,255,.22) !important; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); color: #fff !important; }
.st-key-ob_back button:hover { background: rgba(255,255,255,.18) !important; }
.st-key-ob_back button p::before { content: "←"; display: inline-block; margin-right: .5rem; transition: transform .18s ease; }
.st-key-ob_back button:hover p::before { transform: translateX(-4px); }
.st-key-ob_next button { background: #fff !important; border: 0 !important; color: #000 !important; min-width: 168px; padding: 0 1.9rem !important; box-shadow: none !important; }
.st-key-ob_next button p { color: #000 !important; }
.st-key-ob_actions button, .st-key-ob_actions button * { overflow: visible !important; }
.st-key-siderail button, .st-key-siderail button * { overflow: visible !important; }
.st-key-siderail button [role="img"], .st-key-siderail button [data-testid="stIconMaterial"] { display: inline-block; width: auto !important; min-width: 1.5em; line-height: 1.2 !important; text-align: center; }
.st-key-ob_next button p::after { content: "→"; display: inline-block; margin-left: .55rem; transition: transform .18s ease; }
.st-key-ob_next button:hover p::after { transform: translateX(5px); }
@media (max-width: 640px) {
  .ob-head { padding-top: 1.6rem; }
  .st-key-ob_actions { padding: 0 .75rem; }
  .st-key-ob_actions [data-testid="stHorizontalBlock"] { gap: .2rem; }
  .st-key-ob_actions button p { font-size: .9rem; }
  .st-key-ob_skip button { padding: 0 .55rem !important; }
  .st-key-ob_back button { padding: 0 .95rem !important; }
  .st-key-ob_next button { min-width: 0; padding: 0 1.15rem !important; }
  .st-key-ob_grid [data-testid="stHorizontalBlock"] { flex-wrap: wrap; flex-direction: row; gap: .5rem; }
  .st-key-ob_grid [data-testid="stColumn"] { flex: 1 1 30% !important; min-width: 30% !important; width: 30% !important; }
}
@keyframes ob-rise { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: none; } }
@keyframes ob-pop { 0% { opacity: 0; transform: scale(.72); } 70% { opacity: 1; transform: scale(1.06); } 100% { opacity: 1; transform: scale(1); } }
@keyframes ob-drift { from { transform: scale(1.02) translate(0, 0); } to { transform: scale(1.14) translate(-2.5%, -2%); } }
@keyframes ob-check { 0% { transform: scale(0) rotate(-40deg); } 70% { transform: scale(1.25) rotate(6deg); } 100% { transform: scale(1); } }
@keyframes ob-fill { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes ob-spin { to { transform: rotate(360deg); } }
@keyframes ob-msg { 0%, 4% { opacity: 0; transform: translateY(14px); } 9%, 28% { opacity: 1; transform: none; } 33%, 100% { opacity: 0; transform: translateY(-14px); } }
@keyframes ob-card-a { 0%, 100% { transform: translateX(-112px) rotate(-11deg) scale(.92); } 50% { transform: translateX(-96px) translateY(-14px) rotate(-7deg) scale(.95); } }
@keyframes ob-card-b { 0%, 100% { transform: translateY(0) scale(1.04); } 50% { transform: translateY(-18px) scale(1.08); } }
@keyframes ob-card-c { 0%, 100% { transform: translateX(112px) rotate(11deg) scale(.92); } 50% { transform: translateX(96px) translateY(-14px) rotate(7deg) scale(.95); } }
.ob-head .ob-brand { animation: ob-rise .5s ease both; }
.ob-head .ob-dots { animation: ob-rise .5s .05s ease both; }
.ob-head h1 { animation: ob-rise .6s .1s cubic-bezier(.2,.8,.2,1) both; }
.ob-head p { animation: ob-rise .6s .2s cubic-bezier(.2,.8,.2,1) both; }
.ob-q { animation: ob-rise .5s .25s ease both; }
.ob-dots span { position: relative; overflow: hidden; }
.ob-dots span.on { background: rgba(255,255,255,.2); }
.ob-dots span.on::after { content: ""; position: absolute; inset: 0; background: #e50914; transform-origin: left; animation: ob-fill .55s ease both; }
.st-key-ob_body [data-testid="stButtonGroup"] button { animation: ob-pop .45s cubic-bezier(.2,.9,.3,1.2) both; transition: transform .15s ease, background .15s ease; }
.st-key-ob_body [data-testid="stButtonGroup"] button:hover { transform: translateY(-3px) scale(1.04); }
.st-key-ob_body [data-testid="stButtonGroup"] button:nth-child(2) { animation-delay: .04s; } .st-key-ob_body [data-testid="stButtonGroup"] button:nth-child(3) { animation-delay: .08s; }
.st-key-ob_body [data-testid="stButtonGroup"] button:nth-child(4) { animation-delay: .12s; } .st-key-ob_body [data-testid="stButtonGroup"] button:nth-child(5) { animation-delay: .16s; }
.st-key-ob_body [data-testid="stButtonGroup"] button:nth-child(6) { animation-delay: .2s; } .st-key-ob_body [data-testid="stButtonGroup"] button:nth-child(7) { animation-delay: .24s; }
.st-key-ob_body [data-testid="stButtonGroup"] button:nth-child(8) { animation-delay: .28s; } .st-key-ob_body [data-testid="stButtonGroup"] button:nth-child(9) { animation-delay: .32s; }
.st-key-ob_body [data-testid="stButtonGroup"] button:nth-child(10) { animation-delay: .36s; } .st-key-ob_body [data-testid="stButtonGroup"] button:nth-child(11) { animation-delay: .4s; }
.st-key-ob_body [data-testid="stButtonGroup"] button:nth-child(12) { animation-delay: .44s; }
.st-key-ob_grid [data-testid="stColumn"] { animation: ob-rise .5s cubic-bezier(.2,.8,.2,1) both; }
.st-key-ob_grid [data-testid="stColumn"]:nth-child(2) { animation-delay: .05s; } .st-key-ob_grid [data-testid="stColumn"]:nth-child(3) { animation-delay: .1s; }
.st-key-ob_grid [data-testid="stColumn"]:nth-child(4) { animation-delay: .15s; } .st-key-ob_grid [data-testid="stColumn"]:nth-child(5) { animation-delay: .2s; }
.st-key-ob_grid [data-testid="stColumn"]:nth-child(6) { animation-delay: .25s; }
.ob-poster { transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
.st-key-ob_grid [data-testid="stColumn"]:hover .ob-poster { transform: translateY(-6px) rotate(-1.2deg) scale(1.03); box-shadow: 0 16px 34px rgba(0,0,0,.6); }
.ob-check { animation: ob-check .35s cubic-bezier(.2,.9,.3,1.3) both; }
.st-key-ob_actions button { transition: transform .15s ease, box-shadow .15s ease; animation: ob-rise .5s .3s ease both; }
.st-key-ob_actions button:hover { transform: translateY(-2px); }
.st-key-ob_next button:hover { box-shadow: none !important; background: #e9e9e9 !important; }
.ob-loader { position: fixed; inset: 0; z-index: 9999; background: #000; display: flex; flex-direction: column; align-items: center; justify-content: center; overflow: hidden; }
.ob-loader .bg { position: absolute; inset: -6%; background-position: center; background-size: cover; opacity: .2; animation: ob-drift 6s ease-out both; }
.ob-loader .stack, .ob-loader .ring, .ob-loader .msgs, .ob-loader .bar { position: relative; z-index: 1; }
.ob-loader .stack { width: 170px; height: 244px; margin-bottom: 2.4rem; }
.ob-loader .card { position: absolute; inset: 0; border-radius: 10px; background: #222 center / cover; box-shadow: 0 18px 44px rgba(0,0,0,.65); }
.ob-loader .card.a { animation: ob-card-a 2s ease-in-out infinite; } .ob-loader .card.c { animation: ob-card-c 2s .2s ease-in-out infinite; }
.ob-loader .card.b { z-index: 2; animation: ob-card-b 2s .1s ease-in-out infinite; }
.ob-loader .ring { width: 34px; height: 34px; border-radius: 50%; border: 3px solid rgba(255,255,255,.2); border-top-color: #e50914; animation: ob-spin .8s linear infinite; margin-bottom: 1.2rem; }
.ob-loader .msgs { height: 2.4rem; width: min(90vw, 620px); text-align: center; }
.ob-loader .msgs span { position: absolute; inset: 0; color: #fff; font-size: clamp(1.25rem, 2.2vw, 1.7rem); font-weight: 500; opacity: 0; animation: ob-msg 6s linear both; }
.ob-loader .msgs span:nth-child(2) { animation-delay: 2s; } .ob-loader .msgs span:nth-child(3) { animation-delay: 4s; }
.ob-loader .bar { width: min(70vw, 340px); height: 3px; border-radius: 3px; background: rgba(255,255,255,.18); overflow: hidden; margin-top: 1.1rem; }
.ob-loader .bar i { display: block; height: 100%; background: #e50914; transform-origin: left; animation: ob-fill 6s linear both; }
.stApp:has(.ob-head)::before { content: ""; position: fixed; inset: -8%; z-index: 0; background-image: linear-gradient(rgba(0,0,0,.76), rgba(0,0,0,.86)), var(--ob-bg); background-size: cover; background-position: center; animation: ob-drift 26s ease-in-out infinite alternate; }
.stApp:has(.ob-head) > * { position: relative; z-index: 1; }
@media (prefers-reduced-motion: reduce) { .ob-head *, .st-key-ob_body *, .st-key-ob_grid *, .st-key-ob_actions *, .ob-check, .ob-loader * , .stApp::before { animation-duration: .001s !important; animation-iteration-count: 1 !important; animation-delay: 0s !important; } }
@keyframes hero-zoom { 0% { opacity: 0; transform: scale(.62) rotate(-1.6deg); } 55% { opacity: 1; transform: scale(1.09) rotate(.5deg); } 78% { transform: scale(.975) rotate(-.15deg); } 100% { transform: scale(1) rotate(0); } }
@keyframes hero-pop { 0% { opacity: 0; transform: translateY(30px) scale(.9); } 70% { opacity: 1; transform: translateY(-4px) scale(1.02); } 100% { opacity: 1; transform: none; } }
.hero.intro .sharp, .hero.intro .blur { animation: hero-zoom 1.5s cubic-bezier(.2,.8,.25,1) both; }
.hero.intro .inner > * { animation: hero-pop .7s cubic-bezier(.2,.9,.3,1.15) both; }
.hero.intro .inner > .eyebrow { animation-delay: .55s; } .hero.intro .inner > h2 { animation-delay: .67s; } .hero.intro .inner > .meta { animation-delay: .79s; }
.hero.intro .inner > .why { animation-delay: .89s; } .hero.intro .inner > .syn { animation-delay: .99s; } .hero.intro .inner > .go { animation-delay: 1.09s; }
.stApp:has(.hero.intro) .st-key-love { animation: hero-pop .7s 1.17s cubic-bezier(.2,.9,.3,1.15) both; }
.stApp:has(.hero.intro) .st-key-rows { animation: hero-pop .8s 1.25s cubic-bezier(.2,.8,.25,1) both; }
.stApp:has(.hero.intro) .st-key-siderail { animation: hero-pop .6s .9s ease both; }
@media (prefers-reduced-motion: reduce) { .hero.intro *, .stApp:has(.hero.intro) .st-key-love, .stApp:has(.hero.intro) .st-key-rows, .stApp:has(.hero.intro) .st-key-siderail { animation: none !important; } }
@keyframes modal-in { from { opacity: 0; transform: translateY(26px) scale(.95); } to { opacity: 1; transform: none; } }
@keyframes modal-fade { from { opacity: 0; } to { opacity: 1; } }
.modal { display: none; position: fixed; inset: 0; z-index: 9000; align-items: center; justify-content: center; padding: 1rem; }
.modal:target { display: flex; animation: modal-fade .2s ease both; }
.modal .scrim { position: absolute; inset: 0; background: rgba(0,0,0,.76); backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px); }
.modal .sheet { position: relative; width: min(760px, 100%); max-height: 90vh; overflow-y: auto; background: #161616; border: 1px solid rgba(255,255,255,.12); border-radius: 22px; animation: modal-in .32s cubic-bezier(.2,.9,.3,1.1) both; scrollbar-width: thin; }
.modal .art { position: relative; height: 230px; background-size: cover; background-position: center 22%; }
.modal .art::after { content: ""; position: absolute; inset: 0; background: linear-gradient(0deg, #161616 2%, rgba(22,22,22,0) 62%); }
.modal .x { position: absolute; top: .85rem; right: .85rem; z-index: 3; width: 38px; height: 38px; display: grid; place-items: center; border-radius: 50%; background: rgba(0,0,0,.6); color: #fff !important; text-decoration: none !important; font-size: 1.05rem; }
.modal .body { padding: 0 1.7rem 1.6rem; margin-top: -2.2rem; position: relative; z-index: 2; }
.modal h3 { color: #fff; font-size: clamp(1.6rem, 3vw, 2.2rem); line-height: 1.1; font-weight: 500; letter-spacing: -.02em; margin: 0; padding: 0; }
.modal .jp { color: #9b9b9b; font-size: .9rem; margin: .3rem 0 .75rem; }
.modal .why { color: #fff; font-size: .95rem; font-weight: 500; margin: .7rem 0 .55rem; }
.modal .syn { color: #d2d2d2; font-size: .93rem; line-height: 1.55; margin: 0 0 1.1rem; white-space: pre-line; }
.modal .facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .75rem 1.2rem; border-top: 1px solid rgba(255,255,255,.1); padding-top: 1rem; margin-bottom: 1.3rem; }
.modal .facts div span { display: block; color: #8b8b8b; font-size: .73rem; margin-bottom: .12rem; }
.modal .facts div { color: #efefef; font-size: .9rem; }
.modal .pill { display: inline-block; font-size: .76rem; color: #fff; background: rgba(255,255,255,.12); border-radius: 999px; padding: .28rem .8rem; margin: 0 .35rem .35rem 0; }
.modal .out { position: sticky; bottom: 1rem; z-index: 4; display: inline-flex; align-items: center; justify-content: center; height: 48px; padding: 0 1.6rem; border-radius: 999px; background: #fff; color: #000 !important; text-decoration: none !important; font-size: .95rem; font-weight: 500; }
@media (max-width: 640px) { .modal { padding: 0; align-items: flex-end; } .modal .sheet { max-height: 92vh; border-radius: 22px 22px 0 0; } .modal .art { height: 190px; } .modal .body { padding: 0 1.1rem 1.3rem; } }
@media (max-width: 380px) { .st-key-siderail { --cell: 30px; --cgap: 3px; padding: 0 6px; height: 46px; } .st-key-siderail button p, .st-key-siderail button span { font-size: 1.1rem !important; } }
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
    return '<span class="flag new">New</span>' if r.is_new else ""


def page_url(r: Recommendation) -> str | None:
    return f"https://myanimelist.net/anime/{int(r.mal_id)}" if r.mal_id is not None else None


def modal_id(r: Recommendation) -> str:
    return f"title-{r.item_idx}"


def modal(r: Recommendation) -> str:
    """Info popup for one title. Every dataset string is escaped before it reaches HTML."""
    url = cover(r)
    art = banner(r) or (large(url) if url else None)
    art_css = f"background-image: url('{art}');" if art else art_style(r)
    d = r.details or {}
    episodes = d.get("episodes")
    rating = f"&#9733; {r.rating:.1f}" if r.rating is not None and r.rating == r.rating else None
    facts = [
        ("Type", html.escape(r.type) if r.type else None),
        ("Score", rating),
        ("Episodes", str(episodes) if episodes and r.type != "Movie" else None),
        ("Runtime", html.escape(d["duration"]) if d.get("duration") else None),
        ("Aired", html.escape(d["aired"]) if d.get("aired") else (str(r.year) if r.year else None)),
        ("Studio", html.escape(d["studios_text"]) if d.get("studios_text") else None),
        ("Source", html.escape(d["source"]) if d.get("source") else None),
        ("Age rating", html.escape(d["age_rating"]) if d.get("age_rating") else None),
    ]
    fact_html = "".join(f"<div><span>{k}</span>{v}</div>" for k, v in facts if v)
    pills = "".join(f'<span class="pill">{html.escape(g)}</span>' for g in r.genres)
    text = d.get("synopsis_full") or r.synopsis
    syn = f'<p class="syn">{html.escape(text)}</p>' if text else ""
    jp = f'<div class="jp">{html.escape(d["title_japanese"])}</div>' if d.get("title_japanese") else '<div class="jp"></div>'
    link = page_url(r)
    out = f'<a class="out" href="{link}" target="_blank" rel="noopener">View on MyAnimeList &#8599;</a>' if link else ""
    return (
        f'<div class="modal" id="{modal_id(r)}"><a class="scrim" href="#close" aria-label="Close"></a><div class="sheet">'
        f'<a class="x" href="#close" aria-label="Close">&#10005;</a><div class="art" style="{art_css}"></div><div class="body">'
        f'<h3>{html.escape(r.title)}</h3>{jp}{pills}<div class="why">{html.escape(r.explanation)}</div>{syn}'
        f'<div class="facts">{fact_html}</div>{out}</div></div></div>'
    )


def hero(r: Recommendation, label: str, intro: bool = False) -> str:
    """Featured pick. Every dataset string is escaped before it reaches HTML."""
    url = cover(r)
    bg = f"background-image: url('{url}');" if url else art_style(r)
    wide_art = banner(r)
    if wide_art:
        sharp = f'<img class="sharp banner" src="{wide_art}" alt="" referrerpolicy="no-referrer">'
    else:
        sharp = f'<img class="sharp" src="{large(url)}" alt="" referrerpolicy="no-referrer">' if url else ""
    rating = f"&#9733; {r.rating:.1f}" if r.rating is not None and r.rating == r.rating else ""
    meta = " &nbsp;·&nbsp; ".join(x for x in [html.escape(r.type or ""), str(r.year) if r.year else "", rating, html.escape(", ".join(r.genres[:3]))] if x)
    go = f'<a class="go" href="#{modal_id(r)}">&#9432;&nbsp; More info</a>'
    return (
        f'<div class="hero{" intro" if intro else ""}"><div class="blur" style="{bg}"></div>{sharp}<div class="hero-logo">{logo_img()}</div><div class="inner"><div class="eyebrow">{label}{flag(r)}</div>'
        f'<h2>{html.escape(r.title)}</h2><div class="meta">{meta}</div><div class="why">{html.escape(r.explanation)}</div>'
        f'<div class="syn">{html.escape(r.synopsis)}</div>{go}</div></div>'
    )


def tile(r: Recommendation) -> str:
    url = cover(r)
    art = f'<div class="letter">{initial(r)}</div>'
    if url:
        art += f'<img src="{large(url)}" alt="" loading="lazy" referrerpolicy="no-referrer">'
    meta = " · ".join(x for x in [str(r.year) if r.year else "", html.escape(", ".join(r.genres[:2]))] if x)
    tag, end = f'<a class="tile" href="#{modal_id(r)}">', "</a>"
    return (
        f'{tag}<div class="poster" style="{art_style(r)}">{art}</div><div class="name">{html.escape(r.title)}</div>'
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
    if title not in current and len(current) < MAX_PICKS:
        st.session_state["liked"] = [*current, title]


def clear_all() -> None:
    st.session_state["liked"] = []
    st.session_state["disliked"] = []


OB_STEPS = 4
FORMATS = {"Both": "All", "Series": "TV", "Movies": "Movie"}
ERAS = ["Any time", "Last 10 years", "Recent releases"]


def title_pool(model: DemoRecommender, genres: list[str], n: int = 18) -> list[int]:
    """Popular titles from the chosen genres, one per franchise."""
    wanted = set(genres)
    picks: list[int] = []
    seen: set[int] = set()
    for i in np.argsort(-model.recent_counts, kind="stable"):
        i = int(i)
        f = int(model.franchise[i])
        if f in seen or (wanted and not wanted & set(model.genres[i])):
            continue
        seen.add(f)
        picks.append(i)
        if len(picks) == n:
            break
    return picks


LOADER_SECONDS = float(os.environ.get("ANIMEREC_LOADER_SECONDS", "6"))
STORAGE_KEY = "animerec_prefs"
MAX_PICKS = 30


def one_at_a_time(key: str) -> None:
    """Keep only the first new title, so 'select all matches' (or Enter) cannot add dozens."""
    ss = st.session_state
    before = ss.get(f"_prev_{key}", [])
    added = [t for t in ss.get(key, []) if t not in before]
    if len(added) > 1:
        ss[key] = [t for t in before if t in ss[key]] + added[:1]


def remember(key: str, value: list[str]) -> list[str]:
    st.session_state[f"_prev_{key}"] = list(value)
    return value


def encode_prefs(liked: list[str], disliked: list[str]) -> str:
    ss = st.session_state
    payload = {
        "l": liked, "d": disliked, "g": ss.get("pref_genres", []), "a": ss.get("pref_avoid", []),
        "f": ss.get("pref_format", "Both"), "e": ss.get("pref_era", "Any time"),
    }
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")


def restore_prefs(token: str, options: list[str]) -> bool:
    """Rebuild a returning visitor's answers. Anything malformed just means onboarding again."""
    try:
        d = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
        known = set(options)
        ss = st.session_state
        ss["liked"] = [t for t in d.get("l", []) if t in known][:MAX_PICKS]
        ss["disliked"] = [t for t in d.get("d", []) if t in known][:MAX_PICKS]
        ss["pref_liked"] = list(ss["liked"])
        ss["pref_genres"] = [g for g in d.get("g", []) if g in GENRES]
        ss["pref_avoid"] = [g for g in d.get("a", []) if g in GENRES]
        ss["pref_format"] = d["f"] if d.get("f") in FORMATS else "Both"
        ss["pref_era"] = d["e"] if d.get("e") in ERAS else "Any time"
        ss["onboarded"] = True
        return True
    except (ValueError, TypeError, AttributeError, KeyError):
        return False


def browser_script(js: str) -> None:
    """Run a small script against this site's local storage. Sits at the end of the page."""
    components.html(f"<script>try{{{js}}}catch(e){{}}</script>", height=0)


def logo_img() -> str:
    """The AnimeRec mark as an embedded image, so it needs no extra request."""
    data = base64.b64encode((ROOT / "app" / "assets" / "logo_mark.svg").read_bytes()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" alt="AnimeRec">'


def background_url() -> str:
    path = ROOT / "app" / "assets" / "onboarding_bg.png"
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def render_loading(model: DemoRecommender) -> None:
    """Six-second interstitial built from the visitor's own picks where possible."""
    ss = st.session_state
    index = {t: i for i, t in enumerate(model.titles)}
    chosen = [index[t] for t in ss.get("pref_liked", []) if t in index]
    pool = chosen + [i for i in title_pool(model, ss.get("pref_genres", []), n=6) if i not in chosen]
    cards = ""
    for cls, i in zip("abc", pool):
        raw = model.covers[i] if SHOW_COVERS else None
        art = f"background-image: url('{html.escape(large(raw), quote=True)}');" if raw and raw.startswith(COVER_PREFIX) else ""
        cards += f'<div class="card {cls}" style="{art}"></div>'
    bg = background_url()
    bg_html = f'<div class="bg" style="background-image: url({bg});"></div>' if bg else ""
    st.markdown(
        f'<div class="ob-loader">{bg_html}<div class="stack">{cards}</div><div class="ring"></div><div class="msgs">'
        f"<span>Reading your taste</span><span>Searching the library</span><span>Lining up your top 10</span></div>"
        '<div class="bar"><i></i></div></div>',
        unsafe_allow_html=True,
    )


def ob_save(step: int) -> None:
    """Copy this step's widgets into plain state, which survives when the widgets disappear."""
    ss = st.session_state
    if step == 0:
        ss["pref_genres"] = ss.get("ob_genres") or []
    elif step == 1:
        merged = list(dict.fromkeys([*ss.get("pref_liked", []), *(ss.get("ob_search") or [])]))
        ss["pref_liked"] = merged
    elif step == 2:
        ss["pref_avoid"] = ss.get("ob_avoid") or []
    elif step == 3:
        ss["pref_format"] = ss.get("ob_format") or "Both"
        ss["pref_era"] = ss.get("ob_era") or "Any time"


def ob_go(step: int, delta: int) -> None:
    ob_save(step)
    st.session_state["ob_step"] = step + delta


def ob_finish(step: int | None = None) -> None:
    if step is not None:
        ob_save(step)
    ss = st.session_state
    ss["liked"] = list(ss.get("pref_liked", []))[:MAX_PICKS]
    ss["onboarded"] = True
    ss["hero_intro"] = True  # plays the entrance animation once


def ob_start_loading(step: int) -> None:
    ob_save(step)
    st.session_state["ob_loading"] = True


def ob_toggle(title: str) -> None:
    current = list(st.session_state.get("pref_liked", []))
    st.session_state["pref_liked"] = [t for t in current if t != title] if title in current else [*current, title]


def ob_retake() -> None:
    ss = st.session_state
    ss["onboarded"] = False
    ss["ob_step"] = 0
    ss["ob_forget"] = True
    st.query_params.pop("p", None)


def onboarding_background() -> None:
    """Slowly drifting collage behind onboarding, darkened so the text stays readable."""
    url = background_url()
    if url:
        st.markdown(f"<style>:root {{ --ob-bg: url('{url}'); }}</style>", unsafe_allow_html=True)


def render_onboarding(model: DemoRecommender, options: list[str]) -> None:
    ss = st.session_state
    onboarding_background()
    step = int(ss.get("ob_step", 0))
    heads = [
        ("What do you like to watch?", "Pick as many genres as you like."),
        ("Which of these have you loved?", "Choose a few. The more you pick, the better your recommendations."),
        ("Anything you'd rather skip?", "We'll keep these genres out of your recommendations."),
        ("Last thing: what are you in the mood for?", "You can change this any time."),
    ]
    title, sub = heads[step]
    dots = "".join(f'<span class="{"on" if i <= step else ""}"></span>' for i in range(OB_STEPS))
    st.markdown(
        f'<div class="ob-head"><div class="ob-brand">{logo_img()}AnimeRec</div><div class="ob-dots">{dots}</div>'
        f"<h1>{html.escape(title)}</h1><p>{html.escape(sub)}</p></div>",
        unsafe_allow_html=True,
    )
    with st.container(key="ob_body"):
        if step == 0:
            st.pills("Genres", GENRES, selection_mode="multi", key="ob_genres", default=ss.get("pref_genres", []), label_visibility="collapsed")
        elif step == 1:
            picked = ss.get("pref_liked", [])
            ss.setdefault("ob_search", [])
            pool = title_pool(model, ss.get("pref_genres", []))
            with st.container(key="ob_grid"):
                for start in range(0, len(pool), 6):
                    cols = st.columns(6)
                    for col, i in zip(cols, pool[start : start + 6]):
                        name = model.titles[i]
                        on = name in picked
                        url = model.covers[i] if SHOW_COVERS and str(model.covers[i] or "").startswith(COVER_PREFIX) else None
                        art = f'<img src="{html.escape(large(url), quote=True)}" alt="" loading="lazy" referrerpolicy="no-referrer">' if url else ""
                        check = '<div class="ob-check">&#10003;</div>' if on else ""
                        col.markdown(f'<div class="ob-poster{" on" if on else ""}">{art}{check}</div>', unsafe_allow_html=True)
                        col.button(name, key=f"ob_pick_{i}", on_click=ob_toggle, args=(name,), use_container_width=True)
            remember("ob_search", st.multiselect("Not here? Search for it", options, key="ob_search", placeholder="Search for a title", max_selections=MAX_PICKS, on_change=one_at_a_time, args=("ob_search",)))
        elif step == 2:
            choices = [g for g in GENRES if g not in ss.get("pref_genres", [])]
            st.pills("Genres to skip", choices, selection_mode="multi", key="ob_avoid", default=[g for g in ss.get("pref_avoid", []) if g in choices], label_visibility="collapsed")
        else:
            st.markdown('<div class="ob-q">Series or movies?</div>', unsafe_allow_html=True)
            st.pills("Format", list(FORMATS), selection_mode="single", key="ob_format", default=ss.get("pref_format", "Both"), label_visibility="collapsed")
            st.markdown('<div class="ob-q">How recent?</div>', unsafe_allow_html=True)
            st.pills("Era", ERAS, selection_mode="single", key="ob_era", default=ss.get("pref_era", "Any time"), label_visibility="collapsed")

    with st.container(key="ob_actions"):
        cols = st.columns(3 if step > 0 else 2)
        if step > 0:
            cols[0].button("Back", key="ob_back", on_click=ob_go, args=(step, -1))
        cols[-2].button("Skip", key="ob_skip", on_click=ob_finish, args=(step,))
        if step < OB_STEPS - 1:
            cols[-1].button("Next", key="ob_next", on_click=ob_go, args=(step, 1))
        else:
            cols[-1].button("Show my picks", key="ob_next", on_click=ob_start_loading, args=(step,))

    # A returning visitor on the bare URL is sent back to their saved answers.
    if ss.pop("ob_forget", False):
        browser_script(f"parent.localStorage.removeItem('{STORAGE_KEY}');")
    elif step == 0:
        browser_script(
            # The sandboxed frame cannot navigate, so the redirect runs in the page itself.
            "const d=parent.document,e=d.createElement('script');"
            f"e.textContent=\"(()=>{{const s=localStorage.getItem('{STORAGE_KEY}'),q=new URLSearchParams(location.search);"
            "if(s&&!q.get('p')){q.set('p',s);location.search=q.toString();}})()\";d.head.appendChild(e);"
        )


if not (BUNDLE / "metadata.json").exists():
    st.error("The model bundle was not found. Run `make all` to build it, then reload this page.")
    st.stop()

model = get_model(str(BUNDLE))
options = sorted(set(model.titles), key=str.lower)
years = [int(y) for y in model.years if y is not None]
y_min, y_max = min(years), max(years)
snapshot = html.escape(str(model.meta.get("snapshot", "the data snapshot")))

saved = st.query_params.get("p")
if saved and not st.session_state.get("onboarded") and not st.session_state.get("ob_loading"):
    restore_prefs(saved, options)

if st.session_state.get("ob_loading"):
    render_loading(model)
    time.sleep(LOADER_SECONDS)
    st.session_state["ob_loading"] = False
    ob_finish()
    st.rerun()

if not st.session_state.get("onboarded"):
    render_onboarding(model, options)
    st.stop()

pref_genres = st.session_state.get("pref_genres", [])
pref_avoid = st.session_state.get("pref_avoid", [])
pref_nav = FORMATS.get(st.session_state.get("pref_format", "Both"), "All")
pref_era = st.session_state.get("pref_era", "Any time")
pref_years = (max(y_min, y_max - 10), y_max) if pref_era == "Last 10 years" else (y_min, y_max)

with st.container(key="siderail"):
    st.markdown(f'<div class="brandmark">{logo_img()}</div>', unsafe_allow_html=True)
    with st.container(key="pop_search"), st.popover(":material/search:"):
        liked = remember("liked", st.multiselect("Titles you loved", options, key="liked", placeholder="Search for a title", max_selections=MAX_PICKS, on_change=one_at_a_time, args=("liked",)))
        disliked = remember("disliked", st.multiselect("Titles you disliked (optional)", options, key="disliked", placeholder="Search for a title", max_selections=MAX_PICKS, on_change=one_at_a_time, args=("disliked",)))
        if not liked and not disliked:
            st.caption("Quick start")
            for i, title in enumerate(starter_titles(model)):
                st.button(title, key=f"starter_{i}", on_click=add_favorite, args=(title,), use_container_width=True)
        else:
            st.button("Start over", key="reset", on_click=clear_all, use_container_width=True)
    nav = st.pills("Type", NAV, default=pref_nav, selection_mode="single", key="nav", label_visibility="collapsed", format_func=lambda n: f":material/{NAV_ICONS[n]}:")
    with st.container(key="pop_refine"), st.popover(":material/add:"):
        chip = st.pills("Genre", GENRES, selection_mode="single", key="genre_chip")
        year_range = st.slider("Release year", y_min, y_max, pref_years, key="years")
        new_only = st.toggle("New titles only", value=pref_era == "Recent releases", key="new_only", help=f"Released in the year before {snapshot}, or with no viewer history yet.")
        same_franchise = st.toggle("Include sequels of titles I picked", key="same_franchise")
        st.button("Retake preferences", key="retake", on_click=ob_retake, use_container_width=True)

filters = {
    # Picked titles drive the model; chosen genres only filter when there are no titles to learn from.
    "genres": [chip] if chip else (pref_genres or None if not liked and not disliked else None),
    "exclude_genres": pref_avoid or None,
    "year_range": None if year_range == (y_min, y_max) else year_range,
    "types": [nav] if nav and nav != "All" else None,
    "include_same_franchise": same_franchise,
}
result = model.recommend(liked=liked, disliked=disliked, new_only=new_only, **filters)

for name in result.unknown:
    st.warning(f"Not in the catalog, so it was ignored: {name}")

label = "Your top 10" if result.personalized else "Popular right now"
if not result.items:
    st.markdown('<div class="empty"></div>', unsafe_allow_html=True)
    st.info(result.message)
else:
    top = result.items[0]
    st.markdown(f'<div class="herorail">{hero(top, f"{label} · #1", intro=st.session_state.pop("hero_intro", False))}</div>', unsafe_allow_html=True)
    with st.container(key="love"):
        st.button("♡ Love this", key="love_top", on_click=add_favorite, args=(top.title,), use_container_width=True)
    with st.container(key="rows"):
        if len(result.items) > 1:
            if result.personalized:
                row_title(f"Top picks in {chip}" if chip else "Top picks for you")
            else:
                row_title(f"Trending in {chip}" if chip else "Trending Now")
            rail(result.items[1:])
        fresh_items = []
        if result.personalized and not new_only:
            fresh_items = model.recommend(liked=liked, disliked=disliked, new_only=True, **filters).items
            if fresh_items:
                row_title("New titles you might like")
                rail(fresh_items)
    # Popups sit outside the animated rows so they stay fixed to the window.
    popups = {r.item_idx: r for r in [*result.items, *fresh_items]}
    st.markdown("".join(modal(r) for r in popups.values()), unsafe_allow_html=True)

art_note = (
    "Cover and banner art load from MyAnimeList and AniList and belong to their rights holders."
    if SHOW_COVERS else "Poster art is generated."
)
st.markdown(
    f'<div class="foot">Built on MyAnimeList activity up to {snapshot}, so newer shows are not included. Your answers are saved only in this browser. '
    f"Star ratings are MyAnimeList community scores, shown for reference and not used by the recommender. "
    f"Data: MyAnimeList Dataset by azathoth42 (ODbL database, DbCL contents) and MyAnimeList Dataset by svanoo (CC0). {art_note}</div>",
    unsafe_allow_html=True,
)

token = encode_prefs(liked, disliked)
if st.query_params.get("p") != token:
    st.query_params["p"] = token
browser_script(f"parent.localStorage.setItem('{STORAGE_KEY}','{token}');")
