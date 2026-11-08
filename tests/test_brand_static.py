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
