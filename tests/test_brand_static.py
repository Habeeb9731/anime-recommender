"""The boot page must not show Streamlit branding."""
from __future__ import annotations

import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "deploy" / "brand_static.py"
spec = importlib.util.spec_from_file_location("brand_static", path)
brand_static = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brand_static)

PAGE = "<html><head><title>Streamlit</title></head><body><div id='root'></div></body></html>"


def test_title_and_splash_are_replaced():
    out = brand_static.brand_html(PAGE, b"<svg/>")
    assert "<title>AnimeRec</title>" in out and "Streamlit" not in out
    assert 'id="brand-splash"' in out
    assert out.index('id="brand-splash"') < out.index("id='root'")


def test_branding_can_run_twice():
    once = brand_static.brand_html(PAGE, b"<svg/>")
    assert brand_static.brand_html(once, b"<svg/>") == once


def test_chat_widget_is_inlined_once():
    out = brand_static.brand_html(PAGE, b"<svg/>", (".ask{}", "console.log(1)"))
    assert out.count('<style id="ask-widget">.ask{}</style>') == 1
    assert out.index("<script>console.log(1)</script>") > out.index("id='root'")
    assert brand_static.brand_html(out, b"<svg/>", (".ask{}", "console.log(1)")) == out


def test_widget_files_exist_and_parse():
    root = Path(__file__).resolve().parents[1] / "app" / "assets" / "ask"
    js = (root / "ask.js").read_text()
    assert "/api/ask" in js and "</script" not in js
    assert "</style" not in (root / "ask.css").read_text()
