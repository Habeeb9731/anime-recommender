"""Replace Streamlit's boot page branding, so the first frame a visitor sees is ours.

Runs while the Docker image is built. Safe to run more than once.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
from pathlib import Path

MARKER = "brand-boot"
TITLE = "AnimeRec"
DESCRIPTION = "Tell it what you loved and get ten anime picks, each with a reason."
BOOT = """<style id="brand-boot">
html, body, #root { background: #000 !important; }
[data-testid="stAppSkeleton"], [data-testid="stStatusWidget"], [data-testid="stDecoration"], [data-testid="stToolbar"] { display: none !important; }
#brand-splash { position: fixed; inset: 0; z-index: 2147483000; background: #000; display: grid; place-items: center; transition: opacity .25s ease, visibility .25s ease; animation: brand-giveup 0s 15s forwards; }
#brand-splash img { width: 64px; height: 64px; animation: brand-pulse 1.1s ease-in-out infinite alternate; }
body:has(.hero, .ob-head, .ob-loader, [data-testid="stAlert"]) #brand-splash { opacity: 0; visibility: hidden; }
@keyframes brand-pulse { from { opacity: .55; transform: scale(.94); } to { opacity: 1; transform: scale(1.04); } }
@keyframes brand-giveup { to { opacity: 0; visibility: hidden; } }
@media (prefers-reduced-motion: reduce) { #brand-splash img { animation: none; } }
</style>"""


def brand_html(html: str, logo_svg: bytes) -> str:
    """Return the boot page with our title, background and splash screen."""
    if MARKER in html:
        return html
    tag = f"<title>{TITLE}</title>"
    html = re.sub(r"<title>.*?</title>", tag, html, count=1, flags=re.S) if "<title>" in html else html.replace("</head>", f"{tag}</head>", 1)
    meta = f'<meta name="description" content="{DESCRIPTION}" /><meta name="theme-color" content="#000000" />'
    html = html.replace("</head>", f"{meta}{BOOT}</head>", 1)
    data = base64.b64encode(logo_svg).decode()
    splash = f'<div id="brand-splash"><img src="data:image/svg+xml;base64,{data}" alt="AnimeRec"></div>'
    return re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + splash, html, count=1)


def main() -> None:
    import streamlit

    root = Path(__file__).resolve().parents[1]
    static = Path(streamlit.__file__).parent / "static"
    index = static / "index.html"
    index.write_text(brand_html(index.read_text(), (root / "app" / "assets" / "logo_mark.svg").read_bytes()))
    shutil.copy(root / "app" / "assets" / "favicon.png", static / "favicon.png")
    manifest = static / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        data.update(name=TITLE, short_name=TITLE)
        manifest.write_text(json.dumps(data))
    print("Branded", index)


if __name__ == "__main__":
    main()
