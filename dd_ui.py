"""
DERMADISC look and feel: theme CSS, the Skin-layers logo, line icons and small HTML components.
Pages live in app.py; nothing here touches the database or the AI.
"""
BUILD = "2026-10-02h"  # must match app.py; tells the app this file is up to date
import io
import math
import base64
import html
from functools import lru_cache

from PIL import Image, ImageDraw

E = html.escape

# ─────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────
C = {
    "bg": "#0D1424", "bar": "#0F1729", "surface": "#141D31", "surface2": "#1B2640", "line": "#26324D",
    "line2": "#2D3A58", "text": "#E9EEF7", "soft": "#C9D3E6", "muted": "#A7B3CB", "faint": "#93A1BC",
    "teal": "#2CC7B4", "teal_ink": "#06221F", "peach": "#F2B79A", "rose": "#E0937E", "plum": "#B7616A",
    "green": "#4FD69C", "amber": "#F5B544", "red": "#FF6B6B", "violet": "#9DB0FF",
}
URG = {"low": C["green"], "medium": C["amber"], "high": C["red"], "critical": C["red"], "unsure": C["violet"]}
CHART_COLORS = [C["teal"], C["peach"], C["violet"], C["amber"], C["rose"], C["green"], C["red"]]

FONTS_URL = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800"
             "&family=Figtree:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@400;600"
             "&family=Noto+Sans+Bengali:wght@400;600&family=Noto+Sans+Tamil:wght@400;600"
             "&family=Noto+Sans+Telugu:wght@400;600&display=swap")
BODY_FONT = "Figtree, 'Noto Sans Devanagari', 'Noto Sans Bengali', 'Noto Sans Tamil', 'Noto Sans Telugu', sans-serif"
HEAD_FONT = "'Bricolage Grotesque', 'Noto Sans Devanagari', 'Noto Sans Bengali', 'Noto Sans Tamil', 'Noto Sans Telugu', sans-serif"


# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
CSS = f"""
<style>
@import url('{FONTS_URL}');
html, body, .stApp, [class*="css"], button, input, textarea, select {{ font-family: {BODY_FONT}; }}
.stApp {{ background: {C['bg']}; color: {C['text']}; }}
h1, h2, h3, h4 {{ font-family: {HEAD_FONT} !important; letter-spacing: -0.2px; color: {C['text']}; }}

/* Streamlit chrome off: this is an app, not a notebook */
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer, [data-testid="stDecoration"] {{ display: none !important; }}
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{ display: none !important; }}
[data-testid="stMainBlockContainer"], .block-container {{ max-width: 1240px; padding: 0 2rem 4rem; }}

/* buttons */
button[data-testid="stBaseButton-primary"], [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button[kind="primary"] {{
  background: {C['teal']} !important; color: {C['teal_ink']} !important; border: 0 !important; border-radius: 12px !important;
  font-weight: 700 !important; padding: 0.65rem 1.25rem !important; }}
button[data-testid="stBaseButton-primary"]:hover, [data-testid="stFormSubmitButton"] button:hover {{ background: #4FD8C7 !important; }}
button[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button, a[data-testid="stBaseLinkButton-secondary"] {{
  background: transparent !important; color: {C['text']} !important; border: 1px solid {C['line2']} !important;
  border-radius: 12px !important; font-weight: 600 !important; padding: 0.65rem 1.1rem !important; }}
button[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover, a[data-testid="stBaseLinkButton-secondary"]:hover {{
  border-color: {C['teal']} !important; color: {C['text']} !important; }}
button[data-testid="stBaseButton-tertiary"] {{ color: {C['muted']} !important; font-weight: 500 !important; }}
button[data-testid="stBaseButton-tertiary"]:hover {{ color: {C['text']} !important; }}

/* inputs */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], [data-testid="stNumberInputContainer"],
[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"] {{
  background: {C['bar']} !important; border-color: {C['line2']} !important; border-radius: 10px !important; }}
label, [data-testid="stWidgetLabel"] p {{ color: {C['soft']} !important; font-weight: 600 !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: {C['bar']}; border: 1.5px dashed {C['line2']}; border-radius: 16px; padding: 2rem 1rem; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: {C['teal']}; }}

/* tabs */
[data-baseweb="tab-list"] {{ gap: 4px; background: {C['bar']}; border-radius: 12px; padding: 4px; }}
[data-baseweb="tab"] {{ border-radius: 9px !important; padding: 8px 16px !important; color: {C['faint']} !important; font-weight: 600 !important; }}
[data-baseweb="tab"][aria-selected="true"] {{ background: {C['surface2']} !important; color: {C['text']} !important; }}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display: none !important; }}

/* metrics, expanders, alerts, forms, dataframes */
[data-testid="stMetric"] {{ background: {C['surface']}; border: 1px solid {C['line']}; border-radius: 16px; padding: 16px 20px; }}
[data-testid="stMetricLabel"] p {{ color: {C['faint']} !important; font-weight: 500 !important; }}
[data-testid="stMetricValue"] {{ font-family: {HEAD_FONT}; font-weight: 700; }}
[data-testid="stExpander"] details {{ background: {C['surface']}; border: 1px solid {C['line']} !important; border-radius: 14px; }}
[data-testid="stForm"] {{ background: {C['surface']}; border: 1px solid {C['line']}; border-radius: 18px; padding: 1.4rem; }}
[data-testid="stAlert"] {{ border-radius: 14px; }}
[data-testid="stDataFrame"] {{ border: 1px solid {C['line']}; border-radius: 14px; overflow: hidden; }}
[data-testid="stPopoverBody"] {{ background: {C['surface']} !important; border: 1px solid {C['line2']} !important; border-radius: 16px !important; }}
hr {{ border-color: {C['line']} !important; }}

/* ── navbar ── */
div.st-key-navbar {{ position: sticky; top: 10px; z-index: 990; background: rgba(15,23,41,0.92); backdrop-filter: blur(10px);
  border: 1px solid #1E2A44; border-radius: 16px; padding: 8px 14px; margin: 12px 0 1.6rem; box-shadow: 0 10px 30px rgba(3,7,18,0.35); }}
div.st-key-navbar button[data-testid="stBaseButton-tertiary"] {{ padding: 8px 14px !important; border-radius: 9px !important; }}
div.st-key-navbar button[data-testid="stBaseButton-tertiary"]:hover {{ background: {C['surface2']} !important; }}
div.st-key-navbar [data-testid="stPopover"] button {{ background: transparent !important; border: 0 !important; color: {C['muted']} !important; font-weight: 500 !important; }}
div.st-key-navbar [data-testid="stPopover"] button:hover {{ color: {C['text']} !important; background: {C['surface2']} !important; }}
div.st-key-profile [data-testid="stPopover"] button {{ background: {C['surface']} !important; border: 1px solid {C['line']} !important;
  border-radius: 999px !important; color: {C['text']} !important; font-weight: 600 !important; padding: 6px 16px !important; }}
div.st-key-profile [data-testid="stPopover"] button:hover {{ border-color: {C['teal']} !important; }}
.brand {{ display: flex; align-items: center; gap: 10px; }}
.brand-word {{ font-family: {HEAD_FONT}; font-weight: 700; font-size: 21px; color: {C['text']}; }}
.menu-head {{ padding: 4px 4px 12px; border-bottom: 1px solid {C['line']}; margin-bottom: 6px; }}
.menu-head b {{ display: block; }}
.menu-head span {{ color: {C['faint']}; font-size: 13px; }}
div[class*="st-key-menu_"] button {{ justify-content: flex-start !important; color: {C['text']} !important; }}
div.st-key-menu_logout button {{ color: #FF8A8A !important; }}

/* ── cards and type ── */
.card {{ background: {C['surface']}; border: 1px solid {C['line']}; border-radius: 18px; padding: 22px 24px; margin: 0 0 14px; }}
.muted {{ color: {C['muted']}; }}
.faint {{ color: {C['faint']}; }}
.small {{ font-size: 14px; }}
.page-head {{ display: flex; gap: 16px; align-items: center; margin: 4px 0 18px; }}
.page-head .ico {{ width: 48px; height: 48px; border-radius: 14px; display: flex; align-items: center; justify-content: center; flex: 0 0 auto; }}
.page-head h1 {{ margin: 0 !important; padding: 0 !important; font-size: 30px !important; line-height: 1.15 !important; }}
.page-head p {{ margin: 4px 0 0; color: {C['muted']}; font-size: 15px; }}
.section-title {{ font-family: {HEAD_FONT}; font-weight: 600; font-size: 21px; margin: 26px 0 12px; color: {C['text']}; }}
.pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px; font-weight: 600; font-size: 13px; white-space: nowrap; }}
.pill i {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
.chip {{ display: inline-block; background: rgba(44,199,180,0.10); border: 1px solid rgba(44,199,180,0.28); color: #9FEDE3;
  border-radius: 999px; padding: 2px 10px; font-size: 13px; margin: 4px 6px 0 0; }}
.disclaimer {{ color: #7F8CA6; font-size: 13px; margin-top: 18px; line-height: 1.5; }}

/* hero + login */
div.st-key-hero {{ background: linear-gradient(120deg, #15233E 0%, #121C33 60%); border: 1px solid {C['line']}; border-radius: 24px; padding: 28px 32px; }}
.hero-title {{ font-family: {HEAD_FONT}; font-weight: 700; font-size: 38px; line-height: 1.1; margin: 0 0 10px; color: {C['text']}; }}
.hero-text {{ color: {C['muted']}; font-size: 16px; line-height: 1.55; max-width: 460px; margin: 0 0 16px; }}
.login-hero h1 {{ font-size: 50px !important; line-height: 1.05 !important; margin: 0 0 16px !important; letter-spacing: -1px; }}
.trust {{ display: flex; gap: 12px; align-items: center; color: {C['soft']}; font-size: 15px; margin: 12px 0; }}
div.st-key-authcard {{ background: {C['surface']}; border: 1px solid {C['line']}; border-radius: 20px; padding: 28px 30px; }}
div.st-key-uvcard, div.st-key-recent, div[class*="st-key-reportphoto"], div[class*="st-key-aibox"] {{ background: {C['surface']}; border: 1px solid {C['line']}; border-radius: 20px; padding: 20px 22px; }}

/* stat strip */
.stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); background: {C['surface']}; border: 1px solid {C['line']}; border-radius: 18px; }}
.stats > div {{ padding: 18px 22px; border-right: 1px solid #222E49; }}
.stats > div:last-child {{ border-right: 0; }}
.stats .lbl {{ font-size: 14px; color: {C['faint']}; }}
.stats .val {{ font-family: {HEAD_FONT}; font-weight: 700; font-size: 28px; margin-top: 2px; }}
@media (max-width: 720px) {{ .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .stats > div:nth-child(2) {{ border-right: 0; }} }}

/* clickable tiles and rows: the whole card is one invisible button */
div[class*="st-key-tile_"] {{ position: relative; background: {C['surface']}; border: 1px solid #222E49; border-radius: 16px; padding: 20px; min-height: 168px;
  transition: border-color .15s ease, background .15s ease; }}
div[class*="st-key-tile_"]:hover {{ border-color: rgba(44,199,180,0.6); background: #17213A; }}
div[class*="st-key-row_"] {{ position: relative; border-top: 1px solid #222E49; padding: 10px 6px; transition: background .15s ease; }}
div[class*="st-key-row_"]:hover {{ background: rgba(44,199,180,0.05); }}
div[class*="st-key-rowsel_"] {{ background: rgba(44,199,180,0.08); }}
div[class*="st-key-tilebtn_"], div[class*="st-key-rowbtn_"] {{ position: absolute !important; inset: 0; z-index: 5; margin: 0 !important; }}
div[class*="st-key-tilebtn_"] *, div[class*="st-key-rowbtn_"] * {{ width: 100% !important; height: 100% !important; }}
div[class*="st-key-tilebtn_"] button, div[class*="st-key-tilebtn_"] a, div[class*="st-key-rowbtn_"] button {{ opacity: 0 !important; cursor: pointer; }}
.tile-ico {{ width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; }}
.tile-title {{ font-weight: 600; font-size: 16px; color: {C['text']}; margin-top: 12px; }}
.tile-desc {{ color: {C['faint']}; font-size: 14px; line-height: 1.45; margin-top: 6px; }}
.scanrow {{ display: grid; grid-template-columns: minmax(0, 2.4fr) minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.2fr); gap: 12px; align-items: center; font-size: 15px; }}
.scanrow .who {{ display: flex; gap: 14px; align-items: center; }}
.scanrow img, .scanrow .noimg {{ width: 42px; height: 42px; border-radius: 50%; object-fit: cover; flex: 0 0 auto; background: {C['surface2']}; }}
.scanrow .sub {{ font-size: 13px; color: {C['faint']}; }}
.scanhead {{ color: {C['faint']}; font-size: 13px; font-weight: 600; padding: 4px 6px 8px; }}

/* report */
.banner {{ display: flex; gap: 14px; align-items: center; border-radius: 16px; padding: 16px 20px; margin-bottom: 14px; }}
.banner b {{ font-size: 17px; display: block; }}
.diag {{ display: flex; gap: 22px; align-items: center; }}
.diag h2 {{ margin: 2px 0 6px !important; padding: 0 !important; font-size: 30px !important; line-height: 1.12 !important; }}
.bar {{ height: 8px; border-radius: 4px; background: #222E49; margin: 6px 0 12px; }}
.bar > div {{ height: 8px; border-radius: 4px; }}
.steps {{ list-style: none; margin: 0 0 22px; padding: 0; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
.steps li {{ display: flex; gap: 10px; align-items: center; font-weight: 600; color: {C['faint']}; }}
.steps .dot {{ width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; }}
.steps .line {{ width: 48px; height: 2px; background: {C['line2']}; }}
.steps .line.done {{ background: {C['teal']}; }}

/* scanning animation (shown while the AI works) */
.scanning {{ display: flex; flex-direction: column; align-items: center; gap: 14px; padding: 28px; }}
.scanning .lens {{ position: relative; width: 220px; height: 220px; border-radius: 50%; overflow: hidden; border: 2px solid {C['teal']};
  box-shadow: 0 0 0 10px rgba(44,199,180,0.08); }}
.scanning .lens img {{ width: 100%; height: 100%; object-fit: cover; }}
.scanning .beam {{ position: absolute; left: 0; right: 0; height: 36px; top: -36px;
  background: linear-gradient(180deg, rgba(44,199,180,0) 0%, rgba(44,199,180,0.55) 85%, {C['teal']} 100%); animation: sweep 1.8s ease-in-out infinite; }}
@keyframes sweep {{ 0% {{ top: -36px; }} 100% {{ top: 220px; }} }}
@media (prefers-reduced-motion: reduce) {{ .scanning .beam {{ animation: none; top: 92px; }} }}
</style>
"""


def nav_active_css(page_key):
    return (f"<style>div.st-key-nav_{page_key} button {{ background: {C['surface2']} !important; "
            f"color: {C['text']} !important; font-weight: 600 !important; }}</style>")


# ─────────────────────────────────────────────
# LOGO (C. Skin layers)
# ─────────────────────────────────────────────
def logo_svg(size=34, uid="lg"):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 96 96" aria-hidden="true"><defs><clipPath id="{uid}">'
            f'<circle cx="48" cy="48" r="44"></circle></clipPath></defs><g clip-path="url(#{uid})">'
            '<rect width="96" height="96" fill="#1B2D4A"></rect>'
            '<path d="M0 26c16-8 32 8 48 0s32-8 48 0v-30H0z" fill="#F2B79A"></path>'
            '<path d="M0 26c16-8 32 8 48 0s32-8 48 0v20c-16-8-32 8-48 0S16 38 0 46z" fill="#E0937E"></path>'
            '<path d="M0 46c16-8 32 8 48 0s32-8 48 0v22c-16-8-32 8-48 0S16 60 0 68z" fill="#B7616A"></path>'
            '<path d="M0 68c16-8 32 8 48 0s32-8 48 0" fill="none" stroke="#2CC7B4" stroke-width="4"></path></g></svg>')


def brand_html(size=34, uid="lg_nav"):
    return f'<div class="brand">{logo_svg(size, uid)}<span class="brand-word">Dermadisc</span></div>'


@lru_cache(maxsize=8)
def logo_png(size=256):
    """The same logo as a PNG (favicon, PDF header). Drawn at 4x and scaled down for smooth edges."""
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    k = s / 96.0

    def wave(y0, amp=4.0):
        return [(x * k, (y0 + amp * math.sin(-2 * math.pi * x / 48.0)) * k) for x in [i * 0.5 for i in range(0, 193)]]

    layer = Image.new("RGBA", (s, s), "#1B2D4A")
    d = ImageDraw.Draw(layer)
    w1, w2, w3 = wave(26), wave(46), wave(68)
    d.polygon([(0, 0), (s, 0)] + w1[::-1], fill="#F2B79A")
    d.polygon(w1 + w2[::-1], fill="#E0937E")
    d.polygon(w2 + w3[::-1], fill="#B7616A")
    d.line(w3, fill="#2CC7B4", width=int(4 * k))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).ellipse((4 * k, 4 * k, 92 * k, 92 * k), fill=255)
    img.paste(layer, (0, 0), mask)
    return img.resize((size, size), Image.LANCZOS)


def logo_png_bytes(size=256):
    buf = io.BytesIO()
    logo_png(size).save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────
# ICONS (line icons, 24 x 24)
# ─────────────────────────────────────────────
ICON_PATHS = {
    "scan": '<path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2"></path><circle cx="12" cy="12" r="3.5"></circle>',
    "trend": '<path d="M3 17l6-6 4 4 8-8"></path><path d="M15 7h6v6"></path>',
    "target": '<circle cx="12" cy="12" r="9"></circle><circle cx="12" cy="12" r="5"></circle><circle cx="12" cy="12" r="1"></circle>',
    "chat": '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"></path>',
    "shield": '<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z"></path><path d="M12 8v5M12 16h.01"></path>',
    "book": '<path d="M4 5a2 2 0 0 1 2-2h12v16H6a2 2 0 0 0-2 2z"></path><path d="M4 19a2 2 0 0 1 2-2h12"></path>',
    "drop": '<path d="M12 3c3 4 6 7 6 11a6 6 0 0 1-12 0c0-4 3-7 6-11z"></path>',
    "chip": '<rect x="5" y="5" width="14" height="14" rx="2"></rect><path d="M9 9h6v6H9zM9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"></path>',
    "heart": '<path d="M12 21s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 11c0 5.6-7 10-7 10z"></path>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>',
    "user": '<circle cx="12" cy="8" r="4"></circle><path d="M4 21a8 8 0 0 1 16 0"></path>',
    "admin": '<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z"></path><path d="m9 12 2 2 4-4"></path>',
    "sun": '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path>',
    "bulb": '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.8.7 1 1.5 1 2.5h6c0-1 .2-1.8 1-2.5A6 6 0 0 0 12 3z"></path>',
    "pin": '<path d="M12 2a7 7 0 0 1 7 7c0 5-7 13-7 13S5 14 5 9a7 7 0 0 1 7-7z"></path><circle cx="12" cy="9" r="2.5"></circle>',
    "globe": '<circle cx="12" cy="12" r="9"></circle><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"></path>',
    "lock": '<rect x="4" y="10" width="16" height="11" rx="2"></rect><path d="M8 10V7a4 4 0 0 1 8 0v3"></path>',
    "check": '<circle cx="12" cy="12" r="9"></circle><path d="m8 12 3 3 5-6"></path>',
    "clock": '<circle cx="12" cy="12" r="9"></circle><path d="M12 7v6l3 2"></path>',
    "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"></path><path d="M12 9v4M12 17h.01"></path>',
    "info": '<circle cx="12" cy="12" r="9"></circle><path d="M12 11v5M12 8h.01"></path>',
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"></rect><circle cx="9" cy="10" r="2"></circle><path d="m21 16-5-5-9 9"></path>',
    "flask": '<path d="M9 3h6M10 3v6L4.5 18.5A2 2 0 0 0 6.2 21h11.6a2 2 0 0 0 1.7-2.5L14 9V3"></path><path d="M7 15h10"></path>',
    "gauge": '<path d="M12 14l4-4"></path><path d="M3.3 17a9 9 0 1 1 17.4 0"></path>',
}

# tinted icon squares: (icon colour, background tint)
TINTS = {
    "teal": (C["teal"], "rgba(44,199,180,0.14)"), "peach": (C["peach"], "rgba(242,183,154,0.14)"),
    "violet": (C["violet"], "rgba(157,176,255,0.14)"), "amber": (C["amber"], "rgba(245,181,68,0.14)"),
    "red": ("#FF8A8A", "rgba(255,138,138,0.14)"), "green": (C["green"], "rgba(79,214,156,0.14)"),
}


def icon(name, color=None, size=22, width=1.8):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color or C["teal"]}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICON_PATHS[name]}</svg>')


def icon_square(name, tint="teal", size=42, radius=12, icon_size=22):
    fg, bg = TINTS[tint]
    return (f'<span class="tile-ico" style="width:{size}px;height:{size}px;border-radius:{radius}px;background:{bg}">'
            f'{icon(name, fg, icon_size)}</span>')


# ─────────────────────────────────────────────
# COMPONENTS
# ─────────────────────────────────────────────
def page_header(icon_name, title, subtitle="", tint="teal"):
    fg, bg = TINTS[tint]
    sub = f"<p>{E(subtitle)}</p>" if subtitle else ""
    return (f'<div class="page-head"><span class="ico" style="background:{bg}">{icon(icon_name, fg, 24)}</span>'
            f'<div><h1>{E(title)}</h1>{sub}</div></div>')


def section_title(text):
    return f'<div class="section-title">{E(text)}</div>'


def pill(level, text):
    color = URG.get(str(level).lower(), C["faint"])
    return (f'<span class="pill" style="background:{color}1F;color:{color}"><i style="background:{color}"></i>{E(text)}</span>')


def ring(pct, caption, color=None, size=112):
    pct = max(0, min(100, float(pct or 0)))
    circ = 2 * math.pi * 50
    on = circ * pct / 100
    color = color or C["teal"]
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 120 120" role="img" aria-label="{pct:.0f}%" style="flex:0 0 auto">'
            f'<circle cx="60" cy="60" r="50" fill="none" stroke="#222E49" stroke-width="10"></circle>'
            f'<circle cx="60" cy="60" r="50" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round" '
            f'stroke-dasharray="{on:.1f} {circ:.1f}" transform="rotate(-90 60 60)"></circle>'
            f'<text x="60" y="60" text-anchor="middle" fill="{C["text"]}" font-family="Bricolage Grotesque, sans-serif" font-weight="700" font-size="26">{pct:.0f}%</text>'
            f'<text x="60" y="80" text-anchor="middle" fill="{C["faint"]}" font-family="Figtree, sans-serif" font-size="12">{E(caption)}</text></svg>')


def bar_row(label, pct, color):
    pct = max(0, min(100, float(pct or 0)))
    return (f'<div style="display:flex;justify-content:space-between;font-size:14px"><span>{E(label)}</span>'
            f'<span class="faint">{pct:.0f}%</span></div><div class="bar"><div style="width:{pct:.0f}%;background:{color}"></div></div>')


def stepper(current, labels):
    """current = 1, 2 or 3."""
    parts = []
    for i, label in enumerate(labels, start=1):
        if i < current:
            tick = (f'<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="{C["teal"]}" stroke-width="3" '
                    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m5 12 5 5 9-10"></path></svg>')
            dot = f'<span class="dot" style="background:rgba(44,199,180,0.16)">{tick}</span>'
            style = ""
        elif i == current:
            dot = f'<span class="dot" style="background:{C["teal"]};color:{C["teal_ink"]}">{i}</span>'
            style = f' style="color:{C["text"]}"'
        else:
            dot = f'<span class="dot" style="border:1.5px solid {C["line2"]}">{i}</span>'
            style = ""
        cur = ' aria-current="step"' if i == current else ""
        parts.append(f"<li{cur}{style}>{dot}{E(label)}</li>")
        if i < len(labels):
            parts.append(f'<li aria-hidden="true" class="line{" done" if i < current else ""}"></li>')
    return '<ol class="steps">' + "".join(parts) + "</ol>"


def stats_strip(items):
    """items = [(label, value, colour or None)]"""
    cells = ""
    for lbl, val, col in items:
        style = f' style="color:{col}"' if col else ""
        cells += f'<div><div class="lbl">{E(lbl)}</div><div class="val"{style}>{E(str(val))}</div></div>'

    return f'<div class="stats">{cells}</div>'


def skin_layers_svg(width=420, uid="skin", labels=None):
    """Cross-section of skin (epidermis, dermis, subcutis) in the logo's colours, with a teal scan line
    gliding over the surface. labels = (epidermis, dermis, subcutis) text, or None for the compact version."""
    E_ = html.escape
    vb_w = 470 if labels else 330
    height = round(width * 320 / vb_w)
    # wavy layer boundaries
    top = "M10 104 C60 94 110 112 165 102 S270 92 320 104"
    mid = "M320 150 C270 160 220 142 165 152 S60 162 10 148"
    low = "M10 228 C70 218 120 238 175 226 S280 216 320 230"
    fat = "".join(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="#C97A80" stroke="#9E5059" stroke-width="1.5"></ellipse>'
                  for x, y, rx, ry in [(40, 262, 26, 20), (92, 270, 24, 22), (146, 258, 26, 19), (204, 272, 28, 22),
                                       (262, 260, 26, 20), (306, 278, 22, 20), (64, 302, 24, 16), (120, 306, 26, 16),
                                       (178, 304, 24, 15), (236, 306, 26, 15), (290, 308, 22, 14)])
    lab = ""
    if labels:
        rows = [(118, "#F2B79A", labels[0]), (190, "#E0937E", labels[1]), (268, "#B7616A", labels[2])]
        for y, col, text in rows:
            lab += (f'<path d="M300 {y}H344" stroke="#93A1BC" stroke-width="1.2" stroke-dasharray="3 3"></path>'
                    f'<circle cx="300" cy="{y}" r="4" fill="#E9EEF7"></circle>'
                    f'<rect x="352" y="{y - 7}" width="14" height="14" rx="4" fill="{col}"></rect>'
                    f'<text x="374" y="{y + 5}" fill="#C9D3E6" font-family="Figtree, Noto Sans, sans-serif" font-size="15" font-weight="600">{E_(text)}</text>')
    return f'''<svg width="{width}" height="{height}" viewBox="0 0 {vb_w} 320" role="img" aria-label="Cross-section of skin layers" style="max-width:100%;height:auto">
<style>
@keyframes {uid}_sweep {{ 0% {{ transform: translateX(0); }} 50% {{ transform: translateX(270px); }} 100% {{ transform: translateX(0); }} }}
.{uid}_beam {{ animation: {uid}_sweep 7s ease-in-out infinite; }}
@media (prefers-reduced-motion: reduce) {{ .{uid}_beam {{ animation: none; }} }}
</style>
<defs>
<clipPath id="{uid}_c"><rect x="10" y="20" width="310" height="290" rx="26"></rect></clipPath>
<linearGradient id="{uid}_b" x1="0" x2="1"><stop offset="0" stop-color="#2CC7B4" stop-opacity="0"></stop><stop offset="0.5" stop-color="#2CC7B4" stop-opacity="0.35"></stop><stop offset="1" stop-color="#2CC7B4" stop-opacity="0"></stop></linearGradient>
</defs>
<g clip-path="url(#{uid}_c)">
<rect x="10" y="20" width="310" height="290" fill="#16243F"></rect>
<rect x="10" y="20" width="310" height="290" fill="#B7616A"></rect>
{fat}
<path d="{low} V0 H10 Z" fill="#E0937E"></path>
<path d="M60 205 c10 -8 20 8 30 0 s20 8 30 0 M64 214 c10 -8 20 8 30 0 s20 8 30 0" fill="none" stroke="#B85F63" stroke-width="2.4" stroke-linecap="round"></path>
<path d="M88 220 V150" stroke="#B85F63" stroke-width="2.4"></path>
<path d="M205 166 q10 22 20 0 q10 22 20 0 q10 22 20 0" fill="none" stroke="#C4444F" stroke-width="2.2" stroke-linecap="round"></path>
<path d="M210 186 q10 22 20 0 q10 22 20 0 q10 22 20 0" fill="none" stroke="#7C8FD6" stroke-width="2.2" stroke-linecap="round"></path>
<path d="M145 112 L176 244 L194 244 L162 108 Z" fill="#C8796C"></path>
<ellipse cx="185" cy="246" rx="16" ry="11" fill="#8E4B4F"></ellipse>
<ellipse cx="190" cy="176" rx="11" ry="15" fill="#F2C6A4" stroke="#C8796C" stroke-width="1.5"></ellipse>
<path d="{mid} V0 H320 Z" fill="#F2B79A" transform="translate(0,0)"></path>
<path d="{mid}" fill="none" stroke="#D99A80" stroke-width="1.5"></path>
<path d="M10 0 H320 V104 C270 92 220 112 165 102 S60 94 10 104 Z" fill="#16243F"></path>
<path d="{top}" fill="none" stroke="#F9D9C6" stroke-width="5" stroke-linecap="round"></path>
<path d="M178 240 L148 106 L118 30" fill="none" stroke="#3B2B2E" stroke-width="4" stroke-linecap="round"></path>
<g class="{uid}_beam"><rect x="10" y="20" width="40" height="290" fill="url(#{uid}_b)"></rect><rect x="29" y="20" width="2" height="290" fill="#2CC7B4" fill-opacity="0.9"></rect></g>
</g>
<rect x="10" y="20" width="310" height="290" rx="26" fill="none" stroke="#26324D" stroke-width="2"></rect>
{lab}
</svg>'''


def img_data_uri(jpeg_bytes):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


def scanning_html(jpeg_bytes, text):
    return (f'<div class="card scanning"><div class="lens"><img src="{img_data_uri(jpeg_bytes)}" alt=""><div class="beam"></div></div>'
            f'<div style="font-weight:600">{E(text)}</div></div>')


def banner(level, title, text):
    color = URG.get(level, C["amber"])
    ic = {"low": "check", "medium": "clock", "high": "alert"}.get(level, "info")
    return (f'<div class="banner" role="status" style="background:{color}1A;border:1px solid {color}73">{icon(ic, color, 26, 2)}'
            f'<div><b style="color:{color}">{E(title)}</b><span class="small" style="color:{C["soft"]}">{E(text)}</span></div></div>')
