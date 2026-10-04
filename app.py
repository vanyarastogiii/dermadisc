"""
DERMADISC - AI-powered skin disease detection.

Run with:   python -m streamlit run app.py

Files next to this one:
  dd_ui.py     look and feel: theme, logo, icons, components
  dd_core.py   image analysis, AI calls, trained model, lesion measurement, risk model, reference content
  dd_db.py     SQLite database: users, scans, translation cache  (creates dermadisc.db automatically)
  dd_i18n.py   six languages: English, Hindi, Bengali, Marathi, Tamil, Telugu
  dd_pdf.py    PDF reports (needs the fonts/ folder)
"""
import io
import os
import secrets
import zipfile
from datetime import datetime, timedelta

import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image

# ── Make sure the helper files are the matching version ──
# Streamlit keeps imported files in memory, and on OneDrive it often misses that they changed.
# Reload any that are stale; if a file on disk is still old, say exactly which one to replace.
import importlib
import sys as _sys
BUILD = "2026-10-02h"
_stale = []
for _name in ("dd_i18n", "dd_ui", "dd_db", "dd_core", "dd_pdf"):
    try:
        _mod = importlib.import_module(_name)
        if getattr(_mod, "BUILD", None) != BUILD:
            _mod = importlib.reload(_mod)
        if getattr(_mod, "BUILD", None) != BUILD:
            _stale.append(_name + ".py")
    except ImportError as _e:
        if _name != "dd_pdf":
            _stale.append(f"{_name}.py ({_e})")
if _stale:
    st.set_page_config(page_title="Dermadisc", layout="centered")
    st.error("These files in your skin_detection folder are from an older version: **" + ", ".join(_stale) +
             "**. Copy them again from the latest download, then stop the app (Ctrl+C) and start it again.")
    st.stop()

import dd_ui as ui
from dd_ui import C, E

st.set_page_config(page_title="Dermadisc", page_icon=ui.logo_png(64), layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

import dd_db as db
from dd_core import (
    SKIN_TYPE_POINTS, UV_LEVELS, DISEASES, SKIN_TYPES, GENERAL_TIPS, INGREDIENTS,
    image_hash, enhance_image, assess_quality, color_profile, fetch_free_vision_models, get_models,
    analyze_skin_image, ask_ai_text, build_context_text, load_local_model, predict_local, occlusion_heatmap, agreement,
    segment_lesion, lesion_metrics, outline_overlay, compute_risk, risk_tips, style_chart, uv_level, fetch_uv,
    load_uploaded,
)
from dd_i18n import (
    LANGUAGES, LANG_ORDER, AI_LANGUAGE_NAME, SEEN, t, tr, tr_list, t_class, t_urgency, cur_lang,
    flush_translations, pretranslate_all,
)

try:
    import dd_pdf
    PDF_OK = True
except Exception:                  # fpdf2 not installed
    PDF_OK = False


@st.cache_resource(show_spinner=False)
def _init_database():
    db.init_db()
    return True


_init_database()

for _k, _v in {"user": None, "lang": "en", "preferred_model": "Auto (try all)", "chat": [], "det": {}}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

MAPS_URL = "https://www.google.com/maps/search/dermatologist+near+me"


# ─────────────────────────────────────────────
# SMALL HELPERS
# ─────────────────────────────────────────────
def md(html_text):
    st.markdown(html_text, unsafe_allow_html=True)


def fmt(options, fn=None):
    """Option labels computed now (not lazily), so widgets keep working when the language changes."""
    fn = fn or tr
    labels = {o: fn(o) for o in options}
    return lambda o: labels.get(o, o)


def user():
    return st.session_state.user


def shared_api_key():
    """The key the original app used: .streamlit/secrets.toml or the OPENROUTER_API_KEY environment variable."""
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return str(st.secrets["OPENROUTER_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def api_key():
    """The user's own key if they saved one, otherwise the shared key from secrets.toml / environment."""
    return (user() or {}).get("api_key") or shared_api_key()


def to_jpeg(image, max_side=800, quality=85):
    img = image.convert("RGB").copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def open_image(blob):
    try:
        return Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception:
        return None


def thumb_uri(scan_id):
    """Small round thumbnail for scan lists, cached for the session."""
    cache = st.session_state.setdefault("thumbs", {})
    if scan_id not in cache:
        blob = db.scan_image(scan_id)
        img = open_image(blob) if blob else None
        if img is None:
            cache[scan_id] = None
        else:
            w, h = img.size
            side = min(w, h)
            img = img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2)).resize((84, 84))
            cache[scan_id] = ui.img_data_uri(to_jpeg(img, 84, 80))
    return cache[scan_id]


def mask_key(k):
    return (k[:8] + "…" + k[-4:]) if k and len(k) > 14 else ("•" * len(k or ""))


def disease_label(pred):
    """AI prediction name in the user's language, with the English name when it differs."""
    local = pred.get("disease_local") or pred.get("disease", "")
    eng = pred.get("disease", "")
    return local, (eng if eng and eng != local else "")


def scan_condition(row):
    if row.get("top_condition_local") or row.get("top_condition"):
        return row.get("top_condition_local") or row.get("top_condition")
    if row.get("cnn_prediction"):
        if (row.get("cnn_confidence") or 0) < UNSURE_BELOW:
            return tr("Not sure") + f' ({t_class(row["cnn_prediction"])}?)'
        return t_class(row["cnn_prediction"])
    return "-"


def urgency_text(level):
    return {"low": t_urgency("low"), "medium": tr("See a doctor soon"), "high": tr("See a doctor urgently")}.get(level, "-")


def scan_title(row):
    return f'#{row["id"]} · {row["created_at"][:16]} · {scan_condition(row)}'


def nice_date(ts):
    try:
        return datetime.strptime(ts[:10], "%Y-%m-%d").strftime("%d %b %Y")
    except Exception:
        return ts[:10] if ts else "-"


def pdf_bytes_for(scan_id, owner_id=None):
    cache = st.session_state.setdefault("pdf_cache", {})
    key = (scan_id, owner_id)
    if key not in cache:
        scan = db.get_scan(scan_id, owner_id)
        if not scan:
            return None, None
        cache[key] = (dd_pdf.build_pdf(scan), dd_pdf.report_filename(scan))
    return cache[key]


def forget_pdf(scan_id):
    for k in list(st.session_state.get("pdf_cache", {})):
        if k[0] == scan_id:
            st.session_state.pdf_cache.pop(k, None)


def pdf_download_button(scan_id, owner_id=None, key=None, primary=True):
    if not PDF_OK:
        st.caption("PDF needs: python -m pip install fpdf2 uharfbuzz")
        return
    try:
        data, name = pdf_bytes_for(scan_id, owner_id)
    except Exception as e:
        st.error(f"PDF error: {e}")
        return
    if data:
        st.download_button(t("download_pdf"), data=data, file_name=name, mime="application/pdf", icon=":material/download:",
                           key=key or f"pdf_{scan_id}", type="primary" if primary else "secondary")


def zip_of_reports(rows, owner_id=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            scan = db.get_scan(r["id"], owner_id)
            if scan:
                folder = f'{scan["username"]}/' if owner_id is None else ""
                z.writestr(folder + dd_pdf.report_filename(scan), dd_pdf.build_pdf(scan))
    return buf.getvalue()


def scans_dataframe(rows, with_user=False):
    cols = (["username"] if with_user else []) + ["id", "created_at", "language", "body_area", "top_condition",
                                                   "confidence", "urgency", "red_flags", "cnn_prediction", "cnn_confidence", "ai_model"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def go(page, **state):
    """Button callback: switch page (runs before the rerun, so the new page draws immediately)."""
    st.session_state.nav = page
    for k, v in state.items():
        st.session_state[k] = v


def new_scan():
    st.session_state.det = {}
    st.session_state.nav = "nav_detection"


def logout():
    for k in list(st.session_state.keys()):
        if k not in ("lang",):
            del st.session_state[k]
    st.rerun()


def _language_changed(key):
    choice = st.session_state[key]
    st.session_state.lang = choice
    if user():
        db.update_user(user()["id"], language=choice)
        st.session_state.user["language"] = choice


def language_picker(key, label=True):
    st.session_state[key] = cur_lang()
    st.selectbox(t("language"), LANG_ORDER, format_func=lambda c: LANGUAGES[c]["native"], key=key,
                 on_change=_language_changed, args=(key,), label_visibility="visible" if label else "collapsed")


# ─────────────────────────────────────────────
# LOGIN / SIGN UP
# ─────────────────────────────────────────────
def auth_page():
    left, right = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with left:
        md(f'<div style="margin-top:28px">{ui.brand_html(40, "lg_auth")}</div>')
        md(f'''<div class="login-hero" style="margin-top:34px"><h1>{E(t("login_headline"))}</h1>
            <p class="hero-text" style="font-size:18px;max-width:520px">{E(t("login_sub"))}</p>
            <div class="trust">{ui.icon("pin")}<span>{E(t("trust_data"))}</span></div>
            <div class="trust">{ui.icon("globe")}<span>English, हिन्दी, বাংলা, मराठी, தமிழ், తెలుగు</span></div>
            <div class="trust">{ui.icon("lock")}<span>{E(t("trust_private"))}</span></div></div>''')
        md(f'<div style="display:flex;justify-content:center;margin-top:18px">'
           f'{ui.skin_layers_svg(470, "skin_auth", (t("skin_epidermis"), t("skin_dermis"), t("skin_subcutis")))}</div>')
    with right:
        c1, c2 = st.columns([1, 1])
        with c2:
            language_picker("lang_auth", label=False)
        with st.container(key="authcard"):
            md(f'<h2 style="margin:0;font-size:30px">{E(t("welcome_back"))}</h2><p class="muted" style="margin:4px 0 14px">{E(t("signin_sub"))}</p>')
            tab_login, tab_signup = st.tabs([t("login"), t("signup")])
            with tab_login:
                with st.form("login_form", border=False):
                    username = st.text_input(t("username"))
                    password = st.text_input(t("password"), type="password")
                    ok = st.form_submit_button(t("login"), type="primary", width="stretch")
                if ok:
                    u, err = db.authenticate(username, password)
                    if err:
                        st.error(t(err))
                    else:
                        st.session_state.user = u
                        st.session_state.lang = u.get("language") or cur_lang()
                        st.session_state.nav = "nav_dashboard"
                        st.rerun()
            with tab_signup:
                with st.form("signup_form", border=False):
                    full_name = st.text_input(t("full_name"))
                    new_user = st.text_input(t("username"), help=t("err_username"))
                    email = st.text_input(t("email_optional"))
                    c1, c2 = st.columns(2)
                    pw1 = c1.text_input(t("password"), type="password", help=t("err_password_short"))
                    pw2 = c2.text_input(t("confirm_password"), type="password")
                    lang = st.selectbox(t("language"), LANG_ORDER, index=LANG_ORDER.index(cur_lang()),
                                        format_func=lambda c: LANGUAGES[c]["native"])
                    key = st.text_input(t("api_key_optional"), type="password", placeholder="sk-or-v1-...", help=t("api_key_help"))
                    consent = st.checkbox(t("consent"))
                    ok = st.form_submit_button(t("create_account"), type="primary", width="stretch")
                if ok:
                    if pw1 != pw2:
                        st.error(t("err_password_match"))
                    elif not consent:
                        st.error(t("err_consent"))
                    else:
                        uid, err = db.create_user(new_user, pw1, full_name, email, lang, key)
                        if err:
                            st.error(t(err))
                        else:
                            st.session_state.user = db.get_user(uid)
                            st.session_state.lang = lang
                            st.session_state.nav = "nav_dashboard"
                            st.rerun()
        md(f'<div class="disclaimer">{E(t("disclaimer"))}</div>')


# ─────────────────────────────────────────────
# NAVBAR + PROFILE MENU
# ─────────────────────────────────────────────
LEARN = [("nav_library", ":material/menu_book:"), ("nav_skincare", ":material/spa:"), ("nav_model", ":material/memory:")]
TOOLS = [("nav_progress", ":material/trending_up:"), ("nav_abcde", ":material/track_changes:"),
         ("nav_dermabot", ":material/forum:"), ("nav_risk", ":material/health_and_safety:")]


def navbar(u, page):
    md(ui.nav_active_css({"nav_dashboard": "nav_dashboard", "nav_detection": "nav_detection", "nav_scans": "nav_scans"}.get(page, "none")))
    with st.container(key="navbar"):
        c_logo, c_links, c_user = st.columns([1.5, 5.2, 1.9], vertical_alignment="center")
        with c_logo:
            md(ui.brand_html(32, "lg_nav"))
        with c_links:
            with st.container(horizontal=True, gap="small", vertical_alignment="center"):
                st.button(t("nav_home"), key="nav_nav_dashboard", type="tertiary", icon=":material/home:", on_click=go, args=("nav_dashboard",))
                st.button(t("nav_new_scan"), key="nav_nav_detection", type="tertiary", icon=":material/center_focus_strong:", on_click=new_scan)
                st.button(t("nav_scans"), key="nav_nav_scans", type="tertiary", icon=":material/folder_open:", on_click=go, args=("nav_scans",))
                with st.popover(t("nav_learn"), icon=":material/school:", key="pop_learn"):
                    for k, ic in LEARN:
                        st.button(t(k), key=f"menu_{k}_l", icon=ic, type="tertiary", on_click=go, args=(k,), width="stretch")
                with st.popover(t("nav_tools"), icon=":material/handyman:", key="pop_tools"):
                    for k, ic in TOOLS:
                        st.button(t(k), key=f"menu_{k}_t", icon=ic, type="tertiary", on_click=go, args=(k,), width="stretch")
                    st.link_button(t("find_derm"), MAPS_URL, icon=":material/location_on:", type="tertiary", width="stretch")
        with c_user:
            with st.container(key="profile"):
                with st.popover(u.get("full_name") or u["username"], icon=":material/account_circle:", key="pop_profile", width="stretch"):
                    profile_menu(u)


def profile_menu(u):
    role = t("role_admin") if u["role"] == "admin" else t("role_user")
    md(f'<div class="menu-head"><b>{E(u.get("full_name") or u["username"])}</b><span>{E(u.get("email") or u["username"])} · {E(role)}</span></div>')
    st.button(t("nav_account"), key="menu_account", icon=":material/person:", type="tertiary", on_click=go, args=("nav_account",), width="stretch")
    st.button(t("nav_scans"), key="menu_scans", icon=":material/folder:", type="tertiary", on_click=go, args=("nav_scans",), width="stretch")
    if u["role"] == "admin":
        st.button(t("nav_admin"), key="menu_admin", icon=":material/admin_panel_settings:", type="tertiary",
                  on_click=go, args=("nav_admin",), width="stretch")
    st.divider()
    language_picker("lang_menu")
    options = ["Auto (try all)"] + get_models()
    st.session_state.preferred_model = st.selectbox(
        t("ai_model"), options, format_func=fmt(options, lambda o: t("auto_models") if o == "Auto (try all)" else o),
        index=options.index(st.session_state.preferred_model) if st.session_state.preferred_model in options else 0, key="menu_model")
    live_n = len(fetch_free_vision_models())
    st.caption(t("models_live", n=live_n) if live_n else t("models_offline"))
    if st.button(t("refresh_models"), key="menu_refresh", icon=":material/refresh:", type="tertiary", width="stretch"):
        fetch_free_vision_models.clear()
        st.rerun()
    st.caption(t("model_loaded") if load_local_model()[0] is not None else t("model_missing"))
    st.divider()
    if st.button(t("logout"), key="menu_logout", icon=":material/logout:", type="tertiary", width="stretch"):
        logout()


# ─────────────────────────────────────────────
# SCAN LISTS (dashboard + My Scans)
# ─────────────────────────────────────────────
def row_pill(r):
    if not r.get("top_condition") and r.get("cnn_prediction") and (r.get("cnn_confidence") or 0) < UNSURE_BELOW:
        return ui.pill("unsure", tr("Not sure"))
    return ui.pill(r["urgency"], urgency_text(r["urgency"]))


def scan_rows(rows, selected=None, on_open="nav_scans"):
    md(f'<div class="scanrow scanhead"><span>{E(t("scan"))}</span><span>{E(t("body_area"))}</span>'
       f'<span>{E(t("date"))}</span><span>{E(t("urgency"))}</span></div>')
    for r in rows:
        if r["id"] == selected:
            md(f'<style>div.st-key-row_{r["id"]} {{ background: rgba(44,199,180,0.08); }}</style>')
        with st.container(key=f"row_{r['id']}"):
            uri = thumb_uri(r["id"])
            img = f'<img src="{uri}" alt="">' if uri else '<span class="noimg"></span>'
            if r.get("top_condition") and r.get("cnn_prediction"):
                sub = tr("AI and trained model")
            elif r.get("top_condition"):
                sub = tr("AI analysis")
            else:
                sub = tr("Trained model only")
            md(f'<div class="scanrow"><div class="who">{img}<div><div style="font-weight:600">{E(scan_condition(r))}</div>'
               f'<div class="sub">{E(sub)}</div></div></div><span class="muted">{E(tr(r.get("body_area") or "-"))}</span>'
               f'<span class="muted">{E(nice_date(r["created_at"]))}</span><span>{row_pill(r)}</span></div>')
            st.button(scan_title(r), key=f"rowbtn_{r['id']}", on_click=go, args=(on_open,), kwargs={"my_pick": r["id"]})


# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
TILES = [
    ("nav_progress", "trend", "teal", "See if a spot has changed between two photos."),
    ("nav_abcde", "target", "peach", "Seven questions doctors ask about a mole."),
    ("nav_dermabot", "chat", "violet", "Questions about your result, in your language."),
    ("nav_risk", "shield", "amber", "Your skin risk score and today's UV index."),
    ("nav_library", "book", "teal", "40 skin conditions, treatments and urgency."),
    ("nav_skincare", "drop", "peach", "Routines for your skin type, with a quiz."),
    ("nav_model", "chip", "violet", "Accuracy, training and the confusion matrix."),
]
TIPS = [
    "Check your whole body once a month in good light, and photograph any mole you want to keep an eye on.",
    "Sunscreen works best when you use enough: about a teaspoon for the face and neck, reapplied every 2 hours outdoors.",
    "A mole that looks different from all your others - the 'ugly duckling' - is worth showing to a dermatologist.",
    "Moisturise within a minute of washing, while your skin is still damp, to lock the water in.",
    "Wash pillowcases weekly and clean your phone screen - both touch your face every day.",
    "UV rays pass through clouds and windows; daily sunscreen matters even on grey days and indoors near windows.",
    "Patch-test new skin products on your inner arm for 48 hours before using them on your face.",
]


def tile(key, ic, tint, desc):
    with st.container(key=f"tile_{key}"):
        md(f'{ui.icon_square(ic, tint)}<div class="tile-title">{E(t(key))}</div><div class="tile-desc">{E(tr(desc))}</div>')
        st.button(t(key), key=f"tilebtn_{key}", on_click=go, args=(key,))


def uv_card():
    with st.container(key="uvcard"):
        city = st.session_state.get("uv_city_saved", "")
        if not city:
            uv_hint = tr("Enter your city to see today's UV level.")
            md(f'<div style="display:flex;justify-content:space-between;align-items:center"><b>{E(tr("UV index"))}</b>{ui.icon("sun", C["amber"])}</div>'
               f'<p class="muted small" style="margin:8px 0">{E(uv_hint)}</p>')
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            name = c1.text_input("City", placeholder="Mumbai", label_visibility="collapsed", key="uv_dash_city")
            if c2.button("", icon=":material/arrow_forward:", key="uv_dash_go") and name.strip():
                st.session_state.uv_city_saved = name.strip()
                st.rerun()
            return
        data, err = fetch_uv(city)
        if err or not data:
            md(f'<b>{E(tr("UV index"))}</b><p class="muted small">{E(err or "")}</p>')
        else:
            level_name, color, advice = uv_level(data["current"])
            vals = data["values"][7:19] or data["values"]
            peak = max(vals) if vals else 1
            bars = "".join(f'<span style="flex:1;height:{max(6, v / max(peak, 0.1) * 100):.0f}%;'
                           f'background:{uv_level(v)[1] if v >= 3 else C["line2"]};border-radius:3px"></span>' for v in vals)
            md(f'''<div style="display:flex;justify-content:space-between;align-items:center"><b>{E(tr("UV index in {city}").format(city=data["name"].split(",")[0]))}</b>{ui.icon("sun", C["amber"])}</div>
                <div style="display:flex;align-items:baseline;gap:10px;margin-top:6px"><span style="font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:44px;color:{color};line-height:1">{data["current"]:.1f}</span>
                <span style="font-weight:600;color:{color}">{E(tr(level_name))}</span></div>
                <div aria-hidden="true" style="display:flex;align-items:flex-end;gap:4px;height:52px;margin:12px 0">{bars}</div>
                <p class="muted small" style="margin:0">{E(tr(advice))}</p>''')
        if st.button(tr("Change city"), key="uv_change", type="tertiary"):
            st.session_state.uv_city_saved = ""
            st.rerun()


def page_dashboard():
    u = user()
    rows = db.list_scans(u["id"], limit=500)
    hour = datetime.now().hour
    greet = tr("Good morning, {name}") if hour < 12 else tr("Good afternoon, {name}") if hour < 17 else tr("Good evening, {name}")
    first = (u.get("full_name") or u["username"]).split()[0]
    if rows:
        last = rows[0]
        intro = tr("Your last scan on {date} looked like {condition}. Urgency: {urgency}.").format(
            date=nice_date(last["created_at"]), condition=scan_condition(last), urgency=urgency_text(last["urgency"]))
    else:
        intro = tr("You haven't scanned anything yet. It takes about a minute: take a clear photo, answer a few questions, get your result.")

    hero_col, uv_col = st.columns([2, 1], gap="medium")
    with hero_col:
        with st.container(key="hero"):
            c1, c2 = st.columns([3, 1.25], vertical_alignment="center")
            with c1:
                md(f'<div class="hero-title">{E(greet.format(name=first))}</div><p class="hero-text">{E(intro)}</p>')
                with st.container(horizontal=True, gap="small"):
                    st.button(t("start_scan"), type="primary", icon=":material/center_focus_strong:", on_click=new_scan, key="hero_scan")
                    st.button(tr("Compare two photos"), icon=":material/compare:", on_click=go, args=("nav_progress",), key="hero_compare")
            with c2:
                md(ui.skin_layers_svg(200, "skin_dash"))
    with uv_col:
        uv_card()

    needs = sum(1 for r in rows if r["urgency"] in ("medium", "high"))
    confs = [r["confidence"] for r in rows if r.get("confidence") is not None]
    md('<div style="height:18px"></div>' + ui.stats_strip([
        (t("total_scans"), len(rows), None),
        (tr("Need a doctor"), needs, C["amber"] if needs else None),
        (t("avg_conf"), f"{round(sum(confs) / len(confs))}%" if confs else "-", None),
        (t("last_scan"), nice_date(rows[0]["created_at"]) if rows else "-", None),
    ]))

    md(ui.section_title(tr("Tools and guides")))
    cells = TILES + [None]
    for i in range(0, len(cells), 4):
        for col, item in zip(st.columns(4, gap="small"), cells[i:i + 4]):
            with col:
                if item is None:
                    with st.container(key="tile_maps"):
                        md(f'{ui.icon_square("heart", "red")}<div class="tile-title">{E(t("find_derm"))}</div>'
                           f'<div class="tile-desc">{E(tr("Clinics near you on the map."))}</div>')
                        st.link_button(t("find_derm"), MAPS_URL, key="tilebtn_maps")
                else:
                    tile(*item)

    md('<div style="height:18px"></div>')
    with st.container(key="recent"):
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.markdown(f'<div class="section-title" style="margin:0">{E(t("recent_scans"))}</div>', unsafe_allow_html=True)
        if rows:
            c2.button(f'{t("see_all")} ({len(rows)})', type="tertiary", on_click=go, args=("nav_scans",), key="see_all")
            scan_rows(rows[:5])
        else:
            md(f'<p class="muted" style="margin:10px 0 4px">{E(t("no_scans_yet"))}</p>')

    tip = TIPS[datetime.now().timetuple().tm_yday % len(TIPS)]
    md(f'<div class="card" style="display:flex;gap:16px;align-items:flex-start;border-style:dashed;margin-top:18px">{ui.icon("bulb", C["peach"], 26)}'
       f'<div><b>{E(tr("Skin tip of the day"))}</b><p class="muted" style="margin:4px 0 0;line-height:1.55">{E(tr(tip))}</p></div></div>')


# ─────────────────────────────────────────────
# SCAN FLOW: upload -> details -> result
# ─────────────────────────────────────────────
BODY_AREAS = ["Face", "Scalp", "Neck", "Chest", "Back", "Arm", "Hand", "Leg", "Foot", "Groin", "Nails", "Other"]
SEXES = ["Prefer not to say", "Female", "Male", "Other"]
DURATIONS = ["Less than a week", "1-4 weeks", "1-6 months", "More than 6 months", "Since birth"]
SYMPTOMS = ["Itching", "Pain", "Burning", "Bleeding", "Oozing / pus", "Spreading", "Changing size or colour", "Fever",
            "Dryness / scaling"]


def page_detection():
    det = st.session_state.det
    step = det.get("step", 1)
    md(ui.page_header("scan", t("nav_new_scan"), tr("Three quick steps: a photo, a few details, then your result.")))
    md(ui.stepper(step, [t("step_upload"), t("step_details"), t("results")]))
    if step == 1:
        scan_step_upload(det)
    elif step == 2:
        scan_step_details(det)
    else:
        scan_step_result(det)


def scan_step_upload(det):
    tab_up, tab_cam = st.tabs([t("upload_tab"), t("camera_tab")])
    with tab_up:
        uploaded = st.file_uploader(t("upload_label"), type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    with tab_cam:
        captured = st.camera_input(t("camera_label"))
    source = uploaded or captured
    if source:
        img = load_uploaded(source)
        h = image_hash(img)
        if h != det.get("image_hash"):
            det.clear()
            det.update(step=1, image=img, image_hash=h)
    if not det.get("image"):
        md(f'<div class="card" style="display:flex;gap:16px;align-items:flex-start">{ui.icon_square("image", "teal")}<div>'
           f'<b>{E(tr("Tips for a good photo"))}</b><p class="muted" style="margin:4px 0 0;line-height:1.6">'
           f'{E(tr("Use daylight, hold the camera 10-15 cm away, keep the spot in the centre and in focus, and avoid flash and filters."))}</p></div></div>')
        return

    img = det["image"]
    quality = assess_quality(img)
    colors = color_profile(img)
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        enhance = st.toggle(t("auto_enhance"), value=det.get("enhance", False), key="enhance_toggle")
        det["enhance"] = enhance
        st.image(enhance_image(img) if enhance else img, width="stretch")
    with c2:
        ok = not quality["issues"]
        items = "".join(f'<div class="trust" style="color:{C["amber"]}">{ui.icon("alert", C["amber"], 18)}<span>{E(tr(i))}</span></div>'
                        for i in quality["issues"])
        if not items:
            items = f'<div class="trust">{ui.icon("check", C["green"], 18)}<span>{E(t("quality_ok"))}</span></div>'
        md(f'''<div class="card"><div style="display:flex;justify-content:space-between;align-items:center">
            <b>{E(t("image_quality"))}</b>{ui.pill("low" if ok else "medium", str(quality["score"]) + "/100")}</div>
            {items}
            <div class="small faint" style="margin-top:10px">{E(t("resolution"))} {quality["resolution"]} · {E(t("brightness"))} {quality["brightness"]} · {E(t("sharpness"))} {quality["sharpness"]}</div>
            <div style="display:flex;gap:10px;align-items:center;margin-top:12px"><span style="width:26px;height:26px;border-radius:8px;background:{colors["hex"]};border:1px solid {C["line2"]}"></span>
            <span class="small muted">{E(t("redness_index"))}: {colors["redness_index"]}</span></div></div>''')
        st.button(t("continue"), type="primary", icon=":material/arrow_forward:", width="stretch", key="to_details",
                  on_click=lambda: det.update(step=2))


def run_analysis(det, ctx, use_ai):
    img = det["image"]
    analysis_image = enhance_image(img) if det.get("enhance") else img
    holder = st.empty()
    holder.markdown(ui.scanning_html(to_jpeg(analysis_image, 300, 80), t("analyzing") if use_ai else tr("Running our trained model...")),
                    unsafe_allow_html=True)
    local = None
    sess, meta, _ = load_local_model()
    if sess is not None:
        preds, ms = predict_local(analysis_image, sess, meta)
        local = {"preds": preds, "ms": ms}
    result = None
    if use_ai:
        try:
            result = analyze_skin_image(analysis_image, api_key(), build_context_text(ctx),
                                        st.session_state.preferred_model, AI_LANGUAGE_NAME[cur_lang()])
        except Exception as e:
            det["error"] = str(e)
    holder.empty()
    if use_ai and result is None:
        return False
    if local and result and result.get("predictions"):
        local["agreement"] = agreement(local["preds"][0]["disease"], result["predictions"])[0]
    det["scan_id"] = db.save_scan(user()["id"], cur_lang(), ctx, result, local, assess_quality(img), color_profile(img),
                                  to_jpeg(analysis_image))
    det["step"] = 3
    det.pop("error", None)
    st.session_state.chat = []
    return True


def scan_step_details(det):
    if not det.get("image"):
        det["step"] = 1
        st.rerun()
    c1, c2 = st.columns([1, 2.2], gap="large")
    with c1:
        st.image(det["image"], width="stretch")
        st.button(t("back"), icon=":material/arrow_back:", key="to_upload", on_click=lambda: det.update(step=1))
    with c2:
        prev = det.get("ctx") or {}
        has_key = bool(api_key())
        has_model = load_local_model()[0] is not None
        with st.form("details_form"):
            md(f'<b style="font-size:18px">{E(t("patient_details"))}</b><p class="muted small" style="margin:2px 0 10px">'
               f'{E(tr("These details help the AI give a more accurate answer."))}</p>')
            a, b = st.columns(2)
            age = a.number_input(t("age"), 1, 110, int(prev.get("age", 25)))
            sex = b.selectbox(t("sex"), SEXES, index=SEXES.index(prev.get("sex", SEXES[0])), format_func=fmt(SEXES))
            area = a.selectbox(t("body_area"), BODY_AREAS, index=BODY_AREAS.index(prev.get("area", "Arm")), format_func=fmt(BODY_AREAS))
            duration = b.selectbox(t("duration"), DURATIONS, index=DURATIONS.index(prev.get("duration", DURATIONS[1])), format_func=fmt(DURATIONS))
            symptoms = st.multiselect(t("symptoms"), SYMPTOMS, default=prev.get("symptoms", []), format_func=fmt(SYMPTOMS))
            notes = st.text_area(t("notes"), value=prev.get("notes", ""), height=80)
            new_key = ""
            if not has_key:
                new_key = st.text_input(t("api_key_optional"), type="password", placeholder="sk-or-v1-...", help=t("api_key_help"))
                st.caption(tr("The AI analysis needs a free OpenRouter key (openrouter.ai/keys). Paste it once and it is saved to your account. "
                              "Without it, only our trained model looks at the photo, and it only knows 7 types of moles and spots."))
            go_ai = st.form_submit_button(t("analyze"), type="primary", icon=":material/auto_awesome:", width="stretch")
            go_cnn = False
            if not has_key:
                go_cnn = st.form_submit_button(t("save_cnn_only"), icon=":material/memory:", width="stretch", disabled=not has_model)
        ctx = {"age": age, "sex": sex, "area": area, "duration": duration, "symptoms": symptoms, "notes": notes}
        if go_ai and not has_key and not save_api_key(new_key):
            st.error(tr("Paste your OpenRouter API key first, or use the trained model only."))
            go_ai = False
        if go_ai or go_cnn:
            det["ctx"] = ctx
            if run_analysis(det, ctx, use_ai=bool(go_ai)):
                st.rerun()
        if det.get("error"):
            st.error(t("analysis_failed") + "\n\n" + tr("Try again in a moment, or pick another model in the profile menu. "
                                                        "Free models have daily limits.") + "\n\n" + det["error"])
            if has_model and st.button(t("save_cnn_only"), icon=":material/memory:", key="fallback_cnn"):
                det["ctx"] = ctx
                if run_analysis(det, ctx, use_ai=False):
                    st.rerun()


def scan_step_result(det):
    scan = db.get_scan(det.get("scan_id"), user()["id"]) if det.get("scan_id") else None
    if not scan:
        det.clear()
        st.rerun()
    md(f'<div class="small faint" style="margin:-8px 0 14px">{E(t("scan_saved", id=scan["id"]))}</div>')
    render_report(scan, owner_id=user()["id"], actions=True)
    st.button(t("start_scan"), icon=":material/add_a_photo:", on_click=new_scan, key="again")


# ─────────────────────────────────────────────
# RESULT REPORT (scan flow, My Scans, Admin)
# ─────────────────────────────────────────────
UNSURE_BELOW = 55   # trained-model confidence (%) below which we say "not sure" instead of naming a condition


def cnn_unsure(scan):
    """True when only the trained model looked and it could not match the photo to one of its 7 lesion types."""
    result, local = scan.get("result"), scan.get("local") or {}
    return not result and bool(local.get("preds")) and local["preds"][0].get("confidence", 0) < UNSURE_BELOW


def report_level(scan):
    result, local = scan.get("result"), scan.get("local") or {}
    if cnn_unsure(scan):
        return "unsure"
    if result and result.get("urgency"):
        return result["urgency"]
    if local.get("preds"):
        return local["preds"][0].get("urgency", "medium")
    return scan.get("urgency") or "medium"


BANNER_TEXT = {
    "low": ("Low urgency", "Keep an eye on it and check it again in a month."),
    "medium": ("See a dermatologist within 1 to 2 weeks", "Not an emergency, but it should be examined in person."),
    "high": ("See a doctor as soon as possible", "Some features need a prompt in-person check."),
    "unsure": ("Not sure what this is", "Our trained model could not match this photo to the skin spots it knows. "
                                        "Run the AI analysis below, or show it to a dermatologist."),
}


def save_api_key(key):
    key = (key or "").strip()
    if key:
        db.update_user(user()["id"], api_key=key)
        st.session_state.user["api_key"] = key
    return key


def run_ai_on_scan(scan):
    """Run the vision AI on a scan that was saved with the trained model only, and store the result."""
    img = open_image(scan["image"]) if scan.get("image") else None
    if img is None:
        st.error(tr("The photo for this scan could not be read."))
        return
    with st.spinner(t("analyzing")):
        try:
            result = analyze_skin_image(img, api_key(), build_context_text(scan.get("context") or {}),
                                        st.session_state.preferred_model, AI_LANGUAGE_NAME[cur_lang()])
        except Exception as e:
            st.error(t("analysis_failed") + "\n\n" + str(e))
            return
    local = scan.get("local") or None
    if local and local.get("preds") and result.get("predictions"):
        local = dict(local)
        local["agreement"] = agreement(local["preds"][0]["disease"], result["predictions"])[0]
    db.update_scan_result(scan["id"], result, local)
    forget_pdf(scan["id"])
    st.rerun()


def ai_box(scan):
    """Second-opinion card for a trained-model-only scan: add a key (if needed) and run the AI on the same photo."""
    sid = scan["id"]
    with st.container(key=f"aibox_{sid}"):
        md(f'<b>{E(t("second_opinion"))}</b><p class="small" style="color:{C["soft"]};line-height:1.5;margin:8px 0 10px">'
           f'{E(tr("Only our trained model looked at this scan. The AI analysis recognises about 40 skin conditions, including vitiligo, eczema and acne."))}</p>')
        key_in = ""
        if not api_key():
            key_in = st.text_input(t("api_key_optional"), type="password", placeholder="sk-or-v1-...", key=f"aikey_{sid}",
                                   help=t("api_key_help"))
            st.caption(tr("Free key: openrouter.ai/keys. It is saved to your account."))
        if st.button(tr("Run AI analysis on this photo"), type="primary", icon=":material/auto_awesome:", key=f"runai_{sid}",
                     width="stretch"):
            if not api_key() and not save_api_key(key_in):
                st.error(tr("Paste your OpenRouter API key first."))
            else:
                run_ai_on_scan(scan)


def render_report(scan, owner_id=None, actions=True, admin=False):
    result, local, ctx = scan.get("result"), scan.get("local") or {}, scan.get("context") or {}
    quality = scan.get("quality") or {}
    sid = scan["id"]
    level = report_level(scan)
    left, right = st.columns([5, 7], gap="large")

    with left:
        with st.container(key=f"reportphoto_{sid}"):
            views = ["photo", "heat"]
            view = st.segmented_control(t("photo"), views, default="photo", key=f"view_{sid}", label_visibility="collapsed",
                                        format_func=fmt(views, lambda v: t("photo") if v == "photo" else t("where_looked")))
            heat = scan.get("heatmap")
            if view == "heat":
                if not heat:
                    sess, meta, _ = load_local_model()
                    img = open_image(scan["image"]) if scan.get("image") else None
                    if sess is not None and img is not None:
                        with st.spinner(tr("Hiding parts of the image one by one to see what matters...")):
                            overlay = occlusion_heatmap(img, sess, meta)[0]
                        heat = to_jpeg(overlay)
                        db.update_scan_heatmap(sid, heat)
                        forget_pdf(sid)
                if heat:
                    st.image(heat, width="stretch")
                    md(f'<div style="display:flex;gap:8px;align-items:center" class="small faint"><span>{E(t_urgency("low"))}</span>'
                       '<span style="flex:1;height:8px;border-radius:4px;background:linear-gradient(90deg,#00008b,#0080ff,#00ffff,#ffff00,#ff8000,#ff0000)"></span>'
                       f'<span>{E(t_urgency("high"))}</span></div>')
                else:
                    st.caption(t("model_missing"))
            elif scan.get("image") and open_image(scan["image"]) is not None:
                st.image(scan["image"], width="stretch")
            else:
                st.caption(t("photo") + ": -")
        if quality:
            ok = quality.get("score", 0) >= 70
            md(f'<div class="card" style="display:flex;gap:12px;align-items:center;padding:14px 18px;margin-top:14px">'
               f'{ui.icon("check" if ok else "alert", C["green"] if ok else C["amber"], 20, 2)}'
               f'<span class="small" style="color:{C["soft"]}">{E(t("quality_score"))}: {quality.get("score", "-")}/100 · {E(str(quality.get("resolution", "")))}</span></div>')
        details = [(t("age"), ctx.get("age", "-")), (t("sex"), tr(ctx.get("sex", "-"))), (t("body_area"), tr(ctx.get("area", "-"))),
                   (t("duration"), tr(ctx.get("duration", "-"))),
                   (t("symptoms"), ", ".join(tr_list(ctx.get("symptoms") or [])) or t("none_reported"))]
        md('<div class="card" style="padding:14px 18px">' + "".join(
            f'<div class="small" style="margin:3px 0"><span class="faint">{E(k)}:</span> {E(str(v))}</div>' for k, v in details) + "</div>")

    with right:
        title, text = BANNER_TEXT.get(level, BANNER_TEXT["medium"])
        md(ui.banner(level, tr(title), tr(text)))

        if result and result.get("predictions") and result.get("is_skin_image", True):
            top = result["predictions"][0]
            name, eng = disease_label(top)
            desc, conf = top.get("description", ""), top.get("confidence", 0)
            others = [(disease_label(p)[0], p["confidence"]) for p in result["predictions"][1:3]]
        elif result and not result.get("is_skin_image", True):
            name, eng, desc, conf, others = tr("No skin found"), "", t("not_skin"), 0, []
        elif local.get("preds") and cnn_unsure(scan):
            top = local["preds"][0]
            name, eng, conf = tr("No confident match"), "", top["confidence"]
            desc = (tr("Our trained model only knows 7 types of moles and spots (from the HAM10000 dataset). "
                       "Conditions such as vitiligo, eczema, acne, psoriasis or fungal infections are outside what it learned, "
                       "so its guesses here are not reliable.") + " " + tr("Closest guess") + f": {t_class(top['disease'])} ({top['confidence']}%).")
            others = [(t_class(p["disease"]), p["confidence"]) for p in local["preds"][1:3]]
        elif local.get("preds"):
            top = local["preds"][0]
            name, eng, desc, conf = t_class(top["disease"]), "", tr(top.get("desc", "")), top["confidence"]
            others = [(t_class(p["disease"]), p["confidence"]) for p in local["preds"][1:3]]
        else:
            name, eng, desc, conf, others = "-", "", "", 0, []

        eng_html = f'<span class="faint" style="font-size:15px"> ({E(eng)})</span>' if eng else ""
        ring_col = C["violet"] if level == "unsure" else None
        md(f'''<div class="card diag">{ui.ring(conf, t("confidence").lower(), ring_col)}
            <div><div class="small faint">{E(tr("Trained model result") if level == "unsure" else t("most_likely"))}</div><h2>{E(name)}{eng_html}</h2>
            <p class="muted" style="margin:0;line-height:1.55">{E(desc)}</p></div></div>''')

        c1, c2 = st.columns(2, gap="small")
        with c1:
            bars = "".join(ui.bar_row(n, p, col) for (n, p), col in zip(others, [C["red"], C["violet"]]))
            if not bars:
                bars = f'<p class="faint small">{E(t("pdf_none"))}</p>'
            md(f'<div class="card" style="min-height:150px"><b>{E(t("other_possibilities"))}</b><div style="margin-top:12px">{bars}</div></div>')
        with c2:
            if result and local.get("preds"):
                status = local.get("agreement") or agreement(local["preds"][0]["disease"], result.get("predictions", []))[0]
                col = {"agree": C["green"], "partial": C["amber"]}.get(status, C["red"])
                ic = {"agree": "check", "partial": "info"}.get(status, "alert")
                lp = local["preds"][0]
                body = (f'{E(t(status + "_text"))}<br><span class="faint small">{E(t("cnn_title"))}: '
                        f'{E(t_class(lp["disease"]))} ({lp["confidence"]}%)</span>')
            elif local.get("preds") and actions and not admin:
                ai_box(scan)
                body = None
            elif local.get("preds"):
                col, ic = C["violet"], "info"
                body = E(tr("Only our trained model looked at this scan. Add an API key in My Account for a second opinion from the vision AI."))
            else:
                col, ic = C["faint"], "info"
                body = E(tr("Our trained model was not available for this scan."))
            if body is not None:
                md(f'<div class="card" style="min-height:150px"><b>{E(t("second_opinion"))}</b><div style="display:flex;gap:10px;align-items:flex-start;margin-top:12px">'
                   f'{ui.icon(ic, col, 20, 2)}<span style="line-height:1.5;color:{C["soft"]}">{body}</span></div></div>')

        if result and (result.get("red_flags") or result.get("next_steps")):
            flags = "".join(f"<li>{E(f)}</li>" for f in result.get("red_flags", [])) or f"<li>{E(t('pdf_none'))}</li>"
            steps = "".join(f"<li>{E(s)}</li>" for s in result.get("next_steps", [])) or "<li>-</li>"
            feats = ""
            if result.get("predictions"):
                feats = "".join(f'<span class="chip">{E(f)}</span>' for f in result["predictions"][0].get("key_features", []))
            feats_html = f'<div style="margin-top:8px">{feats}</div>' if feats else ""
            md(f'''<div class="card" style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px">
                <div><b style="color:#FF8A8A">{E(t("red_flags"))}</b><ul style="margin:8px 0 0;padding-left:18px;color:{C["soft"]};line-height:1.7">{flags}</ul>{feats_html}</div>
                <div><b style="color:{C["green"]}">{E(t("next_steps"))}</b><ul style="margin:8px 0 0;padding-left:18px;color:{C["soft"]};line-height:1.7">{steps}</ul></div></div>''')

        if actions:
            with st.container(horizontal=True, gap="small"):
                pdf_download_button(sid, owner_id, key=f"pdf_report_{sid}_{'a' if admin else 'u'}")
                st.link_button(t("find_derm"), MAPS_URL, icon=":material/location_on:")
                if not admin:
                    st.button(t("ask_dermabot"), icon=":material/forum:", key=f"ask_{sid}", on_click=go, args=("nav_dermabot",))

        if local.get("preds"):
            with st.expander(t("cnn_title")):
                lp = local["preds"][0]
                if lp.get("views"):
                    st.caption(f'{lp.get("agree_views", "-")}/{lp["views"]} · {t("uncertain") if lp.get("uncertain") else t("confident")}')
                md("".join(ui.bar_row(t_class(p["disease"]), p["confidence"], C["teal"]) for p in local["preds"][:5]))
        if result and result.get("_model"):
            st.caption(f'{t("ai_model")}: {result["_model"]}')
        md(f'<div class="disclaimer">{E(t("disclaimer"))}</div>')


# ─────────────────────────────────────────────
# MY SCANS
# ─────────────────────────────────────────────
def scan_filters(prefix, with_user=False):
    c1, c2, c3, c4 = st.columns([2.2, 1.6, 1.1, 1.1], vertical_alignment="bottom")
    search = c1.text_input(t("search"), key=f"{prefix}_search", placeholder=tr("Condition or body area"))
    urg = c2.multiselect(t("urgency"), ["low", "medium", "high"], format_func=fmt(["low", "medium", "high"], t_urgency), key=f"{prefix}_urg")
    d_from = c3.date_input(t("from_date"), value=None, key=f"{prefix}_from")
    d_to = c4.date_input(t("to_date"), value=None, key=f"{prefix}_to")
    uid = None
    if with_user:
        users = db.list_users()
        opts = [None] + [x["id"] for x in users]
        names = {x["id"]: x["username"] for x in users}
        all_label = t("all")
        uid = st.selectbox(t("user"), opts, format_func=lambda i: all_label if i is None else names.get(i, i), key=f"{prefix}_user")
    return search.strip() or None, urg or None, d_from, d_to, uid


def page_scans():
    u = user()
    md(ui.page_header("folder", t("nav_scans"), tr("Every scan you have saved, with its full report.")))
    total = db.stats(u["id"])["scans"] or 0
    if not total:
        md(f'<div class="card" style="text-align:center;padding:40px"><div style="display:flex;justify-content:center">'
           f'{ui.icon_square("scan", "teal", 56, 16, 28)}</div><p class="muted" style="margin:14px 0 0">{E(t("no_scans_yet"))}</p></div>')
        st.button(t("start_scan"), type="primary", icon=":material/center_focus_strong:", on_click=new_scan, key="empty_scan")
        return

    search, urg, d_from, d_to, _ = scan_filters("my")
    rows = db.list_scans(u["id"], urgency=urg, search=search, date_from=d_from, date_to=d_to)
    if not rows:
        st.info(t("no_results"))
        return
    ids = [r["id"] for r in rows]
    if st.session_state.get("my_pick") not in ids:
        st.session_state.my_pick = ids[0]
    with st.container(key="recent"):
        scan_rows(rows[:50], selected=st.session_state.my_pick)

    md(ui.section_title(f'{t("scan")} #{st.session_state.my_pick}'))
    scan = db.get_scan(st.session_state.my_pick, u["id"])
    if scan:
        render_report(scan, owner_id=u["id"])
        confirm_key = f"confirm_del_{scan['id']}"
        if not st.session_state.get(confirm_key):
            st.button(t("delete"), icon=":material/delete:", key=f"del_{scan['id']}", type="tertiary",
                      on_click=lambda k=confirm_key: st.session_state.update({k: True}))
        elif st.button(t("confirm_delete"), icon=":material/warning:", key=f"del2_{scan['id']}"):
            db.delete_scan(scan["id"], u["id"])
            st.session_state.pop(confirm_key, None)
            st.session_state.pop("my_pick", None)
            forget_pdf(scan["id"])
            st.session_state.get("thumbs", {}).pop(scan["id"], None)
            st.toast(t("scan_deleted"))
            st.rerun()

    md(ui.section_title(t("my_data")))
    with st.container(horizontal=True, gap="small"):
        st.download_button(t("export_csv"), scans_dataframe(rows).to_csv(index=False).encode("utf-8-sig"),
                           f"dermadisc_{u['username']}_scans.csv", "text/csv", icon=":material/table_view:")
        if PDF_OK:
            if st.button(t("prepare_zip"), key="my_zip_btn", icon=":material/folder_zip:"):
                with st.spinner("..."):
                    st.session_state.my_zip = zip_of_reports(rows, u["id"])
            if st.session_state.get("my_zip"):
                st.download_button(t("download_all_zip"), st.session_state.my_zip, f"dermadisc_{u['username']}_reports.zip",
                                   "application/zip", icon=":material/download:", type="primary")


# ─────────────────────────────────────────────
# ACCOUNT
# ─────────────────────────────────────────────
def page_account():
    u = user()
    role = t("role_admin") if u["role"] == "admin" else t("role_user")
    md(ui.page_header("user", t("nav_account"), f'{u["username"]} · {role}'))
    if u.get("must_change_pw"):
        st.warning(t("change_default_pw") if u["username"] == db.DEFAULT_ADMIN_USER
                   else tr("Please choose a new password - you are using a temporary one."))
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.form("profile_form"):
            md(f'<b style="font-size:18px">{E(t("profile"))}</b>')
            full_name = st.text_input(t("full_name"), value=u.get("full_name") or "")
            email = st.text_input(t("email_optional"), value=u.get("email") or "")
            lang = st.selectbox(t("language"), LANG_ORDER, index=LANG_ORDER.index(u.get("language") or "en"),
                                format_func=lambda c: LANGUAGES[c]["native"])
            if st.form_submit_button(t("save"), type="primary"):
                if email and not db.EMAIL_RE.match(email):
                    st.error(t("err_email"))
                else:
                    db.update_user(u["id"], full_name=full_name.strip(), email=email.strip(), language=lang)
                    st.session_state.lang = lang
                    st.session_state.user = db.get_user(u["id"])
                    st.toast(t("saved"))
                    st.rerun()
        with st.form("pw_form"):
            md(f'<b style="font-size:18px">{E(t("change_password"))}</b>')
            old = st.text_input(t("current_password"), type="password")
            new1 = st.text_input(t("new_password"), type="password")
            new2 = st.text_input(t("confirm_password"), type="password")
            if st.form_submit_button(t("change_password"), type="primary"):
                if new1 != new2:
                    st.error(t("err_password_match"))
                else:
                    err = db.change_password(u["id"], old, new1)
                    if err:
                        st.error(t(err))
                    else:
                        st.session_state.user = db.get_user(u["id"])
                        st.success(t("password_changed"))
    with c2:
        with st.form("key_form"):
            md(f'<b style="font-size:18px">{E(t("api_key"))}</b>')
            st.caption(t("key_saved_masked", key=mask_key(u.get("api_key"))) if u.get("api_key")
                       else (tr("Using the shared key from secrets.toml") + f" ({mask_key(shared_api_key())})") if shared_api_key()
                       else t("no_key_saved"))
            new_key = st.text_input(t("api_key"), type="password", placeholder="sk-or-v1-...", help=t("api_key_help"),
                                    label_visibility="collapsed")
            a, b = st.columns(2)
            save_key = a.form_submit_button(t("save"), type="primary", width="stretch")
            clear_key = b.form_submit_button(t("delete"), width="stretch")
            if save_key and new_key.strip():
                if not new_key.strip().startswith("sk-or-"):
                    st.error(tr("Invalid key format. OpenRouter keys start with sk-or-"))
                else:
                    db.update_user(u["id"], api_key=new_key.strip())
                    st.session_state.user = db.get_user(u["id"])
                    st.toast(t("saved"))
                    st.rerun()
            if clear_key:
                db.update_user(u["id"], api_key="")
                st.session_state.user = db.get_user(u["id"])
                st.rerun()
        md(f'<p class="small muted">{E(t("api_key_help"))}</p>')
        with st.expander(t("delete_account_title"), icon=":material/warning:"):
            md(f'<p class="small muted">{E(t("delete_account_help"))}</p>')
            confirm = st.text_input(t("username"), key="del_acc_confirm")
            if st.button(t("delete_account"), key="del_acc_btn", icon=":material/delete_forever:"):
                if confirm.strip().lower() != u["username"].lower():
                    st.error(t("err_login"))
                elif u["role"] == "admin" and db.count_admins() <= 1:
                    st.error(t("last_admin"))
                else:
                    db.delete_user(u["id"])
                    logout()


# ─────────────────────────────────────────────
# ADMIN PANEL
# ─────────────────────────────────────────────
def content_strings():
    texts = set(SEEN)
    for name, d in DISEASES.items():
        texts.update([name, d["desc"], d["tx"], d["type"]])
    for name, d in SKIN_TYPES.items():
        texts.update([name, d["desc"], d["tip"], *d["signs"], *d["am"], *d["pm"], *d["use"], *d["avoid"]])
    for _, title, text in GENERAL_TIPS:
        texts.update([title, text])
    for row in INGREDIENTS:
        texts.update(row)
    texts.update(BODY_AREAS + SEXES + DURATIONS + SYMPTOMS + TIPS + [d for _, _, _, d in TILES])
    return sorted(x for x in texts if x)


def bar_chart(df, x, y, color=None, height=220, title=None, horizontal=True):
    color = color or C["teal"]
    if horizontal:
        ch = alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=color).encode(
            x=alt.X(f"{x}:Q", title=None), y=alt.Y(f"{y}:N", sort="-x", title=None), tooltip=[y, x])
    else:
        ch = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=color).encode(
            x=alt.X(f"{x}:N", title=None), y=alt.Y(f"{y}:Q", title=None), tooltip=[x, y])
    return style_chart(ch.properties(height=height, title=title or ""))


def scans_charts(rows):
    df = pd.DataFrame(rows)
    df["condition"] = [scan_condition(r) for r in rows]
    c1, c2 = st.columns(2)
    counts = df["condition"].value_counts().reset_index()
    counts.columns = ["condition", "count"]
    c1.altair_chart(bar_chart(counts.head(8), "count", "condition", C["teal"], title=t("condition")), width="stretch")
    urg = df["urgency"].fillna("").value_counts().reset_index()
    urg.columns = ["urgency", "count"]
    urg["label"] = [t_urgency(x) if x else "-" for x in urg["urgency"]]
    c2.altair_chart(style_chart(alt.Chart(urg).mark_arc(innerRadius=55).encode(
        theta="count:Q", color=alt.Color("label:N", scale=alt.Scale(
            domain=[t_urgency("low"), t_urgency("medium"), t_urgency("high")], range=[C["green"], C["amber"], C["red"]]),
            legend=alt.Legend(title=None)), tooltip=["label", "count"]).properties(height=220, title=t("urgency"))),
        width="stretch")


def page_admin():
    me = user()
    md(ui.page_header("admin", t("nav_admin"), tr("Users, every scan, exports and translations."), "amber"))
    tab_over, tab_users, tab_scans, tab_tr = st.tabs([t("admin_overview"), t("users"), t("all_scans"), t("translations")])
    with tab_over:
        s = db.stats()
        md(ui.stats_strip([(t("total_users"), s["users"], None), (t("total_scans"), s["scans"] or 0, None),
                           (t("high_urgency"), s["high"], C["red"] if s["high"] else None), (t("active_7d"), s["active_7d"], None)]))
        rows = db.list_scans(limit=5000)
        if rows:
            df = pd.DataFrame(rows)
            df["day"] = pd.to_datetime(df["created_at"]).dt.date.astype(str)
            since = (datetime.now() - timedelta(days=30)).date().isoformat()
            per_day = df[df["day"] >= since].groupby("day").size().reset_index(name="scans")
            st.altair_chart(bar_chart(per_day, "day", "scans", C["teal"], 200, tr("Scans per day (last 30 days)"), horizontal=False),
                            width="stretch")
            scans_charts(rows)
            lang_counts = df["language"].map(lambda c: LANGUAGES.get(c, {}).get("native", c)).value_counts().reset_index()
            lang_counts.columns = ["language", "scans"]
            st.altair_chart(bar_chart(lang_counts, "scans", "language", C["peach"], 180, t("language")), width="stretch")
        else:
            st.info(t("no_scans_yet"))

    with tab_users:
        users = db.list_users()
        udf = pd.DataFrame(users)[["id", "username", "full_name", "email", "role", "language", "active", "scans",
                                   "high_urgency", "created_at", "last_login"]]
        st.dataframe(udf, width="stretch", hide_index=True)
        st.download_button(t("export_csv"), udf.to_csv(index=False).encode("utf-8-sig"), "dermadisc_users.csv", "text/csv",
                           icon=":material/table_view:")
        user_titles = {x["id"]: f'{x["username"]} ({x["full_name"] or "-"})' for x in users}
        pick = st.selectbox(t("select_user"), list(user_titles), format_func=user_titles.get, key="adm_user_pick")
        target = db.get_user(pick)
        if target:
            is_self = target["id"] == me["id"]
            active_pill = ui.pill("low" if target["active"] else "high", t("active") if target["active"] else t("inactive"))
            target_role = t("role_admin") if target["role"] == "admin" else t("role_user")
            md(f'<div class="card" style="padding:14px 18px"><b>{E(target["username"])}</b> &nbsp; {active_pill} &nbsp; '
               f'<span class="faint">{E(target_role)} · {E(LANGUAGES.get(target["language"], {}).get("native", ""))}</span></div>')
            with st.container(horizontal=True, gap="small"):
                if st.button(t("deactivate") if target["active"] else t("activate"), key="adm_act", icon=":material/block:"):
                    if is_self:
                        st.error(t("cannot_self"))
                    elif target["active"] and target["role"] == "admin" and db.count_admins() <= 1:
                        st.error(t("last_admin"))
                    else:
                        db.update_user(target["id"], active=0 if target["active"] else 1)
                        st.rerun()
                if st.button(t("make_user") if target["role"] == "admin" else t("make_admin"), key="adm_role", icon=":material/badge:"):
                    if is_self:
                        st.error(t("cannot_self"))
                    elif target["role"] == "admin" and db.count_admins() <= 1:
                        st.error(t("last_admin"))
                    else:
                        db.update_user(target["id"], role="user" if target["role"] == "admin" else "admin")
                        st.rerun()
                if st.button(t("reset_password"), key="adm_reset", icon=":material/key:"):
                    temp = secrets.token_urlsafe(6)
                    db.set_password(target["id"], temp, must_change=True)
                    st.session_state.adm_temp = (target["username"], temp)
                if st.button(t("delete"), key="adm_del", icon=":material/delete:"):
                    if is_self:
                        st.error(t("cannot_self"))
                    elif target["role"] == "admin" and db.count_admins() <= 1:
                        st.error(t("last_admin"))
                    else:
                        st.session_state.adm_confirm_del = target["id"]
            if st.session_state.get("adm_temp") and st.session_state.adm_temp[0] == target["username"]:
                st.success(t("temp_password", user=target["username"], pw=st.session_state.adm_temp[1]))
            if st.session_state.get("adm_confirm_del") == target["id"]:
                if st.button(t("confirm_delete") + f" ({target['username']})", key="adm_del2", icon=":material/warning:"):
                    db.delete_user(target["id"])
                    st.session_state.pop("adm_confirm_del", None)
                    st.rerun()

    with tab_scans:
        search, urg, d_from, d_to, uid = scan_filters("adm", with_user=True)
        rows = db.list_scans(user_id=uid, urgency=urg, search=search, date_from=d_from, date_to=d_to)
        if not rows:
            st.info(t("no_results"))
        else:
            sdf = scans_dataframe(rows, with_user=True)
            st.dataframe(sdf, width="stretch", hide_index=True)
            with st.container(horizontal=True, gap="small"):
                st.download_button(t("export_csv"), sdf.to_csv(index=False).encode("utf-8-sig"), "dermadisc_all_scans.csv",
                                   "text/csv", icon=":material/table_view:")
                if PDF_OK and st.button(t("prepare_zip"), key="adm_zip_btn", icon=":material/folder_zip:"):
                    with st.spinner("..."):
                        st.session_state.adm_zip = zip_of_reports(rows[:300])
                if st.session_state.get("adm_zip"):
                    st.download_button(t("download_all_zip"), st.session_state.adm_zip, "dermadisc_reports.zip",
                                       "application/zip", icon=":material/download:", type="primary")
            titles = {r["id"]: f'{scan_title(r)} · {r["username"]}' for r in rows}
            pick = st.selectbox(t("select_scan"), list(titles), format_func=titles.get, key="adm_pick")
            scan = db.get_scan(pick) if pick else None
            if scan:
                render_report(scan, owner_id=None, admin=True)
                confirm_key = f"adm_confirm_scan_{scan['id']}"
                if not st.session_state.get(confirm_key):
                    st.button(t("delete"), icon=":material/delete:", key=f"adm_scan_del_{scan['id']}", type="tertiary",
                              on_click=lambda k=confirm_key: st.session_state.update({k: True}))
                elif st.button(t("confirm_delete"), icon=":material/warning:", key=f"adm_scan_del2_{scan['id']}"):
                    db.delete_scan(scan["id"])
                    st.session_state.pop(confirm_key, None)
                    forget_pdf(scan["id"])
                    st.rerun()

    with tab_tr:
        counts = db.translation_counts()
        texts = content_strings()
        md(ui.stats_strip([(LANGUAGES[l]["native"], counts.get(l, 0), None) for l in LANG_ORDER if l != "en"][:4]))
        st.caption(tr("Core menus and buttons are built in. Longer content is translated by the AI once, stored here, and shared by every user.")
                   + f" ({len(texts)}; {LANGUAGES['te']['native']}: {counts.get('te', 0)})")
        if not api_key():
            st.info(t("no_api_key"))
        elif st.button(t("pretranslate"), type="primary", icon=":material/translate:"):
            bar = st.progress(0.0)
            n = pretranslate_all(texts, api_key(), st.session_state.preferred_model,
                                 progress=lambda frac, name: bar.progress(frac, text=name))
            st.success(f"{n}")
        if st.button(t("clear_cache"), icon=":material/delete_sweep:"):
            db.clear_translations()
            for k in ("tr_cache", "tr_loaded", "tr_attempted"):
                st.session_state.pop(k, None)
            st.rerun()


# ─────────────────────────────────────────────
# ABCDE CHECKER
# ─────────────────────────────────────────────
def page_abcde():
    md(ui.page_header("target", t("nav_abcde"),
                      tr("Dermatologists use the ABCDE rule to spot moles that may need attention. Look at the mole carefully and answer honestly."),
                      "peach"))
    questions = [
        ("A - Asymmetry", "If you draw a line through the middle, do the two halves look different?"),
        ("B - Border", "Are the edges irregular, ragged, notched or blurred?"),
        ("C - Colour", "Does it have more than one colour (shades of brown, black, red, white or blue)?"),
        ("D - Diameter", "Is it larger than 6 mm (about the size of a pencil eraser)?"),
        ("E - Evolving", "Has it changed in size, shape, colour or height recently?"),
        ("Bleeding / itching", "Does it bleed, itch, or crust without injury?"),
        ("Ugly duckling", "Does it look different from your other moles?"),
    ]
    with st.form("abcde"):
        answers = {}
        for title, q in questions:
            answers[title] = st.radio(f"**{tr(title)}** - {tr(q)}", ["No", "Not sure", "Yes"], horizontal=True,
                                      key=f"abcde_{title}", format_func=fmt(["No", "Not sure", "Yes"]))
        submitted = st.form_submit_button(tr("Evaluate"), type="primary")
    if submitted:
        pts = {"No": 0, "Not sure": 0.5, "Yes": 1}
        score = sum(pts[v] for v in answers.values())
        if score >= 3 or answers["E - Evolving"] == "Yes":
            level, title, msg = ("high", "See a dermatologist soon",
                                 "Several warning signs are present. Please book a dermatologist appointment within the next 1-2 weeks.")
        elif score >= 1.5:
            level, title, msg = ("medium", "Monitor closely",
                                 "Some features are worth watching. Take a dated photo now and compare it in 4 weeks, or get it checked.")
        else:
            level, title, msg = "low", "Low concern", "No major warning signs. Keep doing monthly self-checks."
        md(ui.banner(level, f"{tr(title)} ({score:g} / 7)", tr(msg)))
        df = pd.DataFrame({"Sign": [tr(k) for k in answers], "Score": [pts[v] for v in answers.values()],
                           "Answer": [tr(v) for v in answers.values()]})
        st.altair_chart(style_chart(alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), title=None), y=alt.Y("Sign:N", sort=None, title=None),
            color=alt.Color("Answer:N", scale=alt.Scale(domain=[tr("No"), tr("Not sure"), tr("Yes")],
                                                        range=[C["green"], C["amber"], C["red"]]), legend=alt.Legend(title=None)),
            tooltip=["Sign", "Answer"]).properties(height=230)), width="stretch")
        md(f'<div class="disclaimer">{E(tr("This checklist is a screening aid, not a diagnosis. Only a dermatologist (often with dermoscopy or a biopsy) can confirm melanoma."))}</div>')


# ─────────────────────────────────────────────
# PROGRESS TRACKER
# ─────────────────────────────────────────────
def page_progress():
    md(ui.page_header("trend", t("nav_progress"),
                      tr("A change over time is the most important warning sign of melanoma. Upload an earlier and a recent photo of the same spot - Dermadisc finds the spot and measures what changed.")))
    c1, c2 = st.columns(2, gap="large")
    with c1:
        md(f'<b>{E(tr("Earlier photo"))}</b>')
        f1 = st.file_uploader(tr("Earlier photo"), type=["jpg", "jpeg", "png"], key="prog_old", label_visibility="collapsed")
        d1 = st.date_input(t("date"), key="prog_d1", value=None)
    with c2:
        md(f'<b>{E(tr("Recent photo"))}</b>')
        f2 = st.file_uploader(tr("Recent photo"), type=["jpg", "jpeg", "png"], key="prog_new", label_visibility="collapsed")
        d2 = st.date_input(t("date"), key="prog_d2", value=None)
    md(f'<div class="card" style="display:flex;gap:14px;align-items:flex-start">{ui.icon("info", C["violet"], 22)}<span class="muted small" style="line-height:1.6">'
       f'{E(tr("Take both photos from the same distance and angle, in similar light, with the spot in the centre. Sizes are measured relative to the photo, so a different distance changes the size reading."))}</span></div>')
    if not (f1 and f2):
        return

    imgs = [load_uploaded(f1), load_uploaded(f2)]
    masks = [segment_lesion(im) for im in imgs]
    sess, meta, _ = load_local_model()
    results = []
    for col, label, im, mask in zip(st.columns(2, gap="large"), ["Earlier photo", "Recent photo"], imgs, masks):
        with col:
            if mask is None:
                st.image(im, caption=f"{tr(label)} - {tr('no clear lesion found')}", width="stretch")
                results.append(None)
                continue
            st.image(outline_overlay(im, mask), caption=f"{tr(label)} - {tr('detected lesion outlined')}", width="stretch")
            met = lesion_metrics(im, mask)
            if sess is not None:
                preds, _ = predict_local(im, sess, meta)
                met.update(cnn=preds[0]["disease"], cnn_conf=preds[0]["confidence"], cnn_urgency=preds[0]["urgency"])
            results.append(met)
    if None in results:
        st.warning(tr("Couldn't find a clear lesion in one of the photos. Use a close-up photo where the spot is darker than the surrounding skin and roughly centred."))
        return

    old, new = results
    days = (d2 - d1).days if (d1 and d2) else None
    rows = [("Size (% of photo)", "area_pct", 20, "D - Diameter / size"),
            ("Asymmetry score", "asymmetry", 25, "A - Asymmetry (0 = symmetric)"),
            ("Border irregularity", "border", 20, "B - Border (lower = smoother edge)"),
            ("Colour variation", "colour_variation", 25, "C - Colour (higher = more shades)")]
    flags, table_rows = [], ""
    for label, key, limit, help_txt in rows:
        a, b = old[key], new[key]
        pct = (b - a) / a * 100 if a else 0
        flagged = pct > limit
        if flagged:
            flags.append(f"{tr(label)}: +{pct:.0f}%")
        color = C["red"] if flagged else (C["amber"] if pct > limit / 2 else C["green"])
        arrow = "▲" if pct > 1 else ("▼" if pct < -1 else "●")
        table_rows += (f'<tr style="border-top:1px solid #222E49"><td style="padding:10px 0;color:{C["text"]}">{E(tr(label))}'
                       f'<div class="small faint">{E(tr(help_txt))}</div></td><td>{a}</td><td>{b}</td>'
                       f'<td style="color:{color};font-weight:700">{arrow} {pct:+.0f}%</td></tr>')
    if "cnn" in old and "cnn" in new:
        changed = old["cnn"] != new["cnn"]
        worse = new["cnn_urgency"] == "high" and old["cnn_urgency"] != "high"
        if worse:
            flags.append(f"{t('cnn_title')}: {t_class(new['cnn'])}")
        color = C["red"] if worse else (C["amber"] if changed else C["green"])
        table_rows += (f'<tr style="border-top:1px solid #222E49"><td style="padding:10px 0;color:{C["text"]}">{E(t("cnn_title"))}</td>'
                       f'<td>{E(t_class(old["cnn"]))} ({old["cnn_conf"]}%)</td><td>{E(t_class(new["cnn"]))} ({new["cnn_conf"]}%)</td>'
                       f'<td style="color:{color};font-weight:700">{"≠" if changed else "="}</td></tr>')
    days_txt = f' · {days} {E(tr("days apart"))}' if days is not None else ""
    md(f'''<div class="card" style="overflow-x:auto"><b>{E(tr("Automated ABCD comparison"))}</b><span class="faint small">{days_txt}</span>
        <table style="width:100%;border-collapse:collapse;font-size:15px;color:{C["soft"]};margin-top:10px">
        <tr class="small faint"><td>{E(tr("Measure"))}</td><td>{E(tr("Earlier photo"))}</td><td>{E(tr("Recent photo"))}</td><td>{E(tr("Change"))}</td></tr>
        {table_rows}</table></div>''')
    if flags:
        md(ui.banner("high", tr("Changes detected - book a dermatologist check"), " · ".join(flags)))
    else:
        md(ui.banner("low", tr("No significant change detected."), tr("Keep taking a photo every 1-3 months to keep tracking this spot.")))
    md(f'<div class="disclaimer">{E(tr("Measurements are automatic image estimates and depend on photo distance, angle and lighting. Any spot that is growing, changing colour, bleeding or itching should be checked by a dermatologist."))}</div>')


# ─────────────────────────────────────────────
# DERMABOT
# ─────────────────────────────────────────────
def page_dermabot():
    u = user()
    md(ui.page_header("chat", t("nav_dermabot"),
                      tr("Ask about skin conditions, skincare, or your latest scan result. DermaBot gives general information only - not a diagnosis."),
                      "violet"))
    if not api_key():
        st.info(t("no_api_key"))
        st.button(t("nav_account"), icon=":material/person:", on_click=go, args=("nav_account",), key="bot_account")
        return
    system = ("You are DermaBot, a friendly dermatology education assistant inside the Dermadisc app. "
              "Give clear, concise, general information in simple language. Never claim to diagnose. "
              "Recommend seeing a dermatologist for anything worrying, and tell the user to seek urgent care "
              "for signs like fast-spreading redness, fever, or a rapidly changing mole. "
              f"Always reply in {AI_LANGUAGE_NAME[cur_lang()]}.")
    latest = db.list_scans(u["id"], limit=1)
    scan = db.get_scan(latest[0]["id"], u["id"]) if latest else None
    if scan:
        parts = []
        if scan.get("result") and scan["result"].get("predictions"):
            r = scan["result"]
            parts.append("Vision AI: " + ", ".join(f"{p['disease']} ({p['confidence']}%)" for p in r["predictions"])
                         + f". Urgency: {r['urgency']}. Assessment: {r['overall_assessment']}")
        if scan.get("local") and scan["local"].get("preds"):
            p = scan["local"]["preds"][0]
            parts.append(f"Trained CNN: {p['disease']} ({p['confidence']}%)")
        if scan.get("context"):
            parts.append("Context: " + build_context_text(scan["context"]))
        system += f"\n\nThe user's latest saved scan (#{scan['id']}, {scan['created_at']}): " + " | ".join(parts)
        md(f'<div class="card" style="display:flex;gap:12px;align-items:center;padding:12px 18px">{ui.icon("check", C["green"], 20, 2)}'
           f'<span class="small">{E(tr("DermaBot can see your latest scan"))}: #{scan["id"]} · {E(scan_condition(scan))}</span></div>')

    if not st.session_state.chat:
        suggestions = (["What does my result mean?", "How can I prevent this from getting worse?", "When should I see a doctor?"]
                       if scan else ["What is the difference between eczema and psoriasis?", "How do I choose a sunscreen?",
                                     "What are early signs of skin cancer?"])
        with st.container(horizontal=True, gap="small"):
            for s in suggestions:
                if st.button(tr(s), key=f"sug_{s}", icon=":material/chat_bubble:"):
                    st.session_state.chat.append({"role": "user", "content": tr(s)})
                    st.rerun()
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/blur_circular:"):
            st.markdown(msg["content"])
    prompt = st.chat_input(tr("Ask DermaBot..."))
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        st.rerun()
    if st.session_state.chat and st.session_state.chat[-1]["role"] == "user":
        with st.chat_message("assistant", avatar=":material/blur_circular:"):
            with st.spinner("..."):
                try:
                    reply = ask_ai_text([{"role": "system", "content": system}] + st.session_state.chat[-10:],
                                        api_key(), st.session_state.preferred_model)
                except Exception:
                    reply = t("analysis_failed")
            st.markdown(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})
    if st.session_state.chat and st.button(t("delete"), key="clear_chat", icon=":material/delete_sweep:", type="tertiary"):
        st.session_state.chat = []
        st.rerun()


# ─────────────────────────────────────────────
# RISK ANALYSIS + UV
# ─────────────────────────────────────────────
def uv_section():
    md(ui.section_title(tr("Today's UV index")))
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    city = c1.text_input("City", placeholder="Mumbai, Delhi, Chennai...", label_visibility="collapsed", key="uv_city")
    go_btn = c2.button(tr("Check UV"), icon=":material/wb_sunny:", width="stretch")
    if not (city and (go_btn or st.session_state.get("uv_last") == city)):
        st.caption(tr("Enter a city to see today's live UV level and sun-protection advice."))
        return
    st.session_state.uv_last = city
    data, err = fetch_uv(city.strip())
    if err:
        st.warning(err)
        return
    cur_name, cur_color, advice = uv_level(data["current"])
    max_name, max_color, _ = uv_level(data["max"])
    md(f'''<div class="card"><div class="small faint">{E(data["name"])}</div>
        <div style="display:flex;gap:2rem;align-items:flex-end;flex-wrap:wrap;margin-top:6px">
        <div><div style="font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:44px;color:{cur_color};line-height:1">{data["current"]:.1f}</div>
        <div style="color:{cur_color};font-weight:600">{E(tr("Now"))} - {E(tr(cur_name))}</div></div>
        <div><div style="font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:26px;color:{max_color};line-height:1">{data["max"]:.1f}</div>
        <div style="color:{max_color};font-weight:600;font-size:14px">{E(tr("Today's peak"))} - {E(tr(max_name))}</div></div></div>
        <p class="muted" style="margin:12px 0 0">{E(tr(advice))}</p></div>''')
    if data["values"]:
        df = pd.DataFrame({"Hour": data["hours"], "UV": data["values"]})
        df["Level"] = [tr(uv_level(v)[0]) for v in df["UV"]]
        st.altair_chart(style_chart(alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Hour:N", title=None, axis=alt.Axis(labelAngle=0, values=data["hours"][::3])), y=alt.Y("UV:Q", title=None),
            color=alt.Color("Level:N", scale=alt.Scale(domain=[tr(l[1]) for l in UV_LEVELS], range=[l[2] for l in UV_LEVELS]),
                            legend=alt.Legend(orient="bottom", title=None)), tooltip=["Hour", "UV", "Level"]).properties(height=190)),
            width="stretch")
        risky = [hh for hh, v in zip(data["hours"], data["values"]) if v >= 3]
        if risky:
            st.caption(tr("Sun protection needed today from {start} to {end} (UV 3 or higher).").format(start=risky[0], end=risky[-1]))


def page_risk():
    md(ui.page_header("shield", t("nav_risk"), tr("Your personal skin risk score and today's UV level."), "amber"))
    uv_section()
    md(ui.section_title(tr("Your skin risk score")))
    lifestyles, exposures = ["Healthy", "Moderate", "Unhealthy"], ["Low", "Medium", "High"]
    burns, families, mole_opts = ["Never", "1-2", "3-5", "More than 5"], ["None", "Mild", "Significant"], ["Fewer than 20", "20-50", "More than 50"]
    skin_types = list(SKIN_TYPE_POINTS.keys())
    with st.form("risk_form"):
        age = st.slider(t("age"), 1, 100, 25)
        c1, c2 = st.columns(2)
        lifestyle = c1.selectbox(tr("Lifestyle"), lifestyles, format_func=fmt(lifestyles))
        exposure = c1.selectbox(tr("Sun Exposure"), exposures, format_func=fmt(exposures))
        sunburns = c1.selectbox(tr("Blistering sunburns in your life"), burns, format_func=fmt(burns))
        skin_type = c2.selectbox(tr("Skin Type (Fitzpatrick)"), skin_types, format_func=fmt(skin_types))
        family_history = c2.selectbox(tr("Family History of skin disease"), families, format_func=fmt(families))
        moles = c2.selectbox(tr("Number of moles on body"), mole_opts, format_func=fmt(mole_opts))
        submitted = st.form_submit_button(tr("Calculate Risk"), type="primary")
    if submitted:
        risk, breakdown = compute_risk(age, lifestyle, exposure, skin_type, family_history, sunburns, moles)
        if risk > 70:
            level, title, advice = "high", "High Risk", "Dermatologist check-ups every 6 months recommended."
        elif risk > 40:
            level, title, advice = "medium", "Moderate Risk", "Annual skin check-up recommended. Use SPF 30+ daily."
        else:
            level, title, advice = "low", "Low Risk", "Maintain healthy habits. Annual self-checks advised."
        c1, c2 = st.columns([1, 2.2], gap="large", vertical_alignment="center")
        with c1:
            md(f'<div class="card" style="display:flex;justify-content:center">{ui.ring(risk, tr("risk"), ui.URG[level], 150)}</div>')
        with c2:
            md(ui.banner(level, tr(title), tr(advice)))
            for tip_text in risk_tips(exposure, skin_type, lifestyle, sunburns, moles):
                md(f'<div class="trust" style="margin:8px 0">{ui.icon("check", C["teal"], 18)}<span>{E(tr(tip_text))}</span></div>')
        df = pd.DataFrame({"Factor": [tr(k) for k in breakdown], "Points": list(breakdown.values())})
        st.altair_chart(bar_chart(df, "Points", "Factor", C["amber"], 240, tr("What drives your risk")), width="stretch")


# ─────────────────────────────────────────────
# DISEASE LIBRARY
# ─────────────────────────────────────────────
def page_library():
    md(ui.page_header("book", t("nav_library"), tr("40 skin conditions with what they look like, how they are treated, and how urgent they are.")))
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    search = c1.text_input(t("search"), placeholder=tr("Name or symptom, e.g. itchy"))
    urg_opts = ["All", "Low", "Medium", "High", "Critical"]
    urg_filter = c2.selectbox(t("urgency"), urg_opts, format_func=fmt(urg_opts, lambda x: t("all") if x == "All" else t_urgency(x)))
    types = sorted({d["type"] for d in DISEASES.values()})
    type_filter = c3.selectbox(tr("Category"), ["All"] + types, format_func=fmt(["All"] + types, lambda x: t("all") if x == "All" else tr(x)))
    q = search.lower().strip()
    filtered = {k: v for k, v in DISEASES.items()
                if (not q or any(q in s.lower() for s in (k, v["desc"], tr(k), tr(v["desc"]))))
                and (urg_filter == "All" or v["urgency"] == urg_filter)
                and (type_filter == "All" or v["type"] == type_filter)}
    st.caption(f"{len(filtered)} / {len(DISEASES)}")
    items = list(filtered.items())
    for i in range(0, len(items), 2):
        for col, (name, info) in zip(st.columns(2, gap="small"), items[i:i + 2]):
            local_name = tr(name)
            eng = f' <span class="faint small">({E(name)})</span>' if local_name != name else ""
            col.markdown(f'''<div class="card" style="min-height:170px">
                <div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start">
                <div><b style="font-size:17px">{E(local_name)}</b>{eng}<div><span class="chip">{E(tr(info["type"]))}</span></div></div>
                {ui.pill(info["urgency"].lower(), t_urgency(info["urgency"]))}</div>
                <p class="muted" style="margin:10px 0 6px;line-height:1.5">{E(tr(info["desc"]))}</p>
                <div class="small" style="color:{C["soft"]}"><span class="faint">{E(tr("Treatment"))}:</span> {E(tr(info["tx"]))}</div></div>''',
                         unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SKIN CARE GUIDE
# ─────────────────────────────────────────────
def chips_tr(items, color=None):
    style = f' style="color:{color};border-color:{color}55;background:{color}14"' if color else ""
    return "".join(f'<span class="chip"{style}>{E(tr(i))}</span>' for i in items)


def bullets_tr(items):
    return "".join(f'<div class="trust" style="margin:6px 0">{ui.icon("check", C["teal"], 16)}<span class="small">{E(tr(i))}</span></div>' for i in items)


def page_skincare():
    md(ui.page_header("drop", t("nav_skincare"), tr("Routines for every skin type. Not sure of your type? Take the quick quiz first."), "peach"))
    with st.expander(tr("Find your skin type (4-question quiz)"), icon=":material/quiz:"):
        with st.form("skin_quiz", border=False):
            q1_opts = ["Tight or flaky", "Comfortable", "Shiny all over", "Shiny only on forehead/nose/chin"]
            q2_opts = ["Barely visible", "Visible on nose only", "Large and visible on most of the face"]
            q3_opts = ["Rarely reacts", "Sometimes stings or turns red", "Often red, itchy or burning"]
            q4_opts = ["Rarely", "Sometimes (e.g. before periods/stress)", "Very often"]
            q1 = st.radio(tr("1. How does your face feel 1 hour after washing (no products)?"), q1_opts, format_func=fmt(q1_opts))
            q2 = st.radio(tr("2. How visible are your pores?"), q2_opts, format_func=fmt(q2_opts))
            q3 = st.radio(tr("3. How does your skin react to new products?"), q3_opts, format_func=fmt(q3_opts))
            q4 = st.radio(tr("4. How often do you get breakouts?"), q4_opts, format_func=fmt(q4_opts))
            go_quiz = st.form_submit_button(tr("Show my skin type"), type="primary")
        if go_quiz:
            base = {"Tight or flaky": "Dry", "Comfortable": "Normal", "Shiny all over": "Oily",
                    "Shiny only on forehead/nose/chin": "Combination"}[q1]
            if base == "Normal" and q2.startswith("Large"):
                base = "Oily"
            extra = (["Sensitive"] if q3 == "Often red, itchy or burning" else []) + (["Acne-prone"] if q4 == "Very often" else [])
            extra_txt = (" + " + " & ".join(tr(x) for x in extra)) if extra else ""
            md(ui.banner("low", f"{tr(base)}{extra_txt}", tr("See the matching tab below for your routine.")))

    tabs = st.tabs([tr(k) for k in SKIN_TYPES])
    for tab, (name, info) in zip(tabs, SKIN_TYPES.items()):
        with tab:
            md(f'''<div class="card" style="border-left:4px solid {info["color"]}"><b style="font-size:20px;color:{info["color"]}">{E(tr(name))}</b>
                <p class="muted" style="margin:6px 0 4px;line-height:1.55">{E(tr(info["desc"]))}</p><div>{chips_tr(info["signs"])}</div></div>''')
            c1, c2 = st.columns(2, gap="small")
            c1.markdown(f'<div class="card" style="min-height:220px"><b>{E(tr("Morning routine"))}</b>{bullets_tr(info["am"])}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="card" style="min-height:220px"><b>{E(tr("Night routine"))}</b>{bullets_tr(info["pm"])}</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="small")
            c1.markdown(f'<div class="card"><b style="color:{C["green"]}">{E(tr("Look for"))}</b><div>{chips_tr(info["use"], C["green"])}</div></div>',
                        unsafe_allow_html=True)
            c2.markdown(f'<div class="card"><b style="color:#FF8A8A">{E(tr("Avoid"))}</b><div>{chips_tr(info["avoid"], "#FF8A8A")}</div></div>',
                        unsafe_allow_html=True)
            md(f'<div class="card" style="display:flex;gap:14px;align-items:flex-start;border-style:dashed">{ui.icon("bulb", C["peach"], 22)}'
               f'<span class="muted">{E(tr(info["tip"]))}</span></div>')

    md(ui.section_title(tr("Healthy skin habits for everyone")))
    tip_icons = ["sun", "drop", "clock", "check", "flask", "heart"]
    cards = list(zip(GENERAL_TIPS, tip_icons))
    for i in range(0, len(cards), 3):
        for col, ((_, title, text), ic) in zip(st.columns(3, gap="small"), cards[i:i + 3]):
            col.markdown(f'<div class="card" style="min-height:210px">{ui.icon_square(ic, "teal")}<div class="tile-title">{E(tr(title))}</div>'
                         f'<div class="tile-desc">{E(tr(text))}</div></div>', unsafe_allow_html=True)

    md(ui.section_title(tr("Ingredient cheat sheet")))
    rows = "".join(f'<tr style="border-top:1px solid #222E49"><td style="padding:10px 10px 10px 0;color:{C["text"]};font-weight:600">{E(tr(n))}</td>'
                   f'<td>{E(tr(w))}</td><td style="color:#9FEDE3">{E(tr(b))}</td></tr>' for n, w, b in INGREDIENTS)
    md(f'''<div class="card" style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:15px;color:{C["muted"]}">
        <tr class="small faint"><td style="padding-bottom:8px">{E(tr("Ingredient"))}</td><td>{E(tr("What it does"))}</td><td>{E(tr("Best for"))}</td></tr>
        {rows}</table></div>''')
    md(f'<div class="disclaimer">{E(tr("These are general skin care guidelines. For prescription treatments or persistent problems, consult a dermatologist."))}</div>')


# ─────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────
def page_model():
    md(ui.page_header("chip", t("nav_model"), "DERMADISC-Net: EfficientNet-B0 trained on HAM10000", "violet"))
    sess, meta, err = load_local_model()
    if sess is None:
        st.warning(f"{t('model_missing')}: {err}")
        md("""<div class="card"><b>How to add the trained model</b><ol class="muted" style="line-height:1.8;margin:8px 0 0">
            <li>Open <code>train_dermadisc.ipynb</code> in Google Colab</li>
            <li>Runtime → Change runtime type → T4 GPU, then Runtime → Run all</li>
            <li>Put <code>dermadisc_model.onnx</code> and <code>dermadisc_model_meta.json</code> next to <code>app.py</code></li>
            <li><code>python -m pip install onnxruntime</code> and restart the app</li></ol></div>""")
        return
    m = meta.get("metrics", {})
    if meta.get("dry_run"):
        st.error("This model came from a pipeline test run on random data - its predictions are meaningless. Retrain in Colab.")
    md(ui.stats_strip([(tr("Test accuracy"), f"{m.get('test_accuracy', 0) * 100:.1f}%", None),
                       (tr("Balanced accuracy"), f"{m.get('test_balanced_accuracy', 0) * 100:.1f}%", None),
                       ("Macro F1", f"{m.get('test_macro_f1', 0):.3f}", None),
                       (tr("Model size"), f"{meta.get('model_size_mb', '?')} MB", None)]))
    sizes = meta.get("split_sizes", {})
    md(f'''<div class="card" style="margin-top:16px"><b>Model card</b><div class="muted small" style="line-height:1.9;margin-top:6px">
        <b>Architecture:</b> {E(meta.get("architecture", ""))}<br>
        <b>Dataset:</b> {E(meta.get("dataset", ""))} · train {sizes.get("train", "?")} / validation {sizes.get("validation", "?")} / test {sizes.get("test", "?")}<br>
        <b>Classes:</b> {E(", ".join(t_class(c) for c in meta.get("classes", [])))}<br>
        <b>Training:</b> {meta.get("epochs")} epochs · batch {meta.get("batch_size")} · {E(meta.get("optimizer", ""))} · lr {meta.get("learning_rate")}<br>
        <b>Loss:</b> {E(meta.get("loss", ""))}<br><b>Augmentation:</b> {E(meta.get("augmentation", ""))}<br>
        <b>Deployment:</b> {E(meta.get("framework", ""))} · CPU, offline · {E(meta.get("trained_at", ""))}</div></div>''')
    steps = ["HAM10000", "Resize + augment", "EfficientNet-B0 (ImageNet)", "Fine-tune (weighted loss)", "Test set evaluation",
             "ONNX export", "4-view TTA", "Explainable AI heatmap"]
    arrow = f' <span style="color:{C["teal"]}">→</span> '
    md('<div class="card" style="line-height:2.3">' + arrow.join(f'<span class="chip">{s}</span>' for s in steps) + '</div>')
    hist = pd.DataFrame(meta.get("history", []))
    if not hist.empty:
        c1, c2 = st.columns(2)
        loss_df = hist.melt("epoch", ["train_loss", "val_loss"], var_name="split", value_name="loss")
        c1.altair_chart(style_chart(alt.Chart(loss_df).mark_line(point=True).encode(
            x="epoch:O", y="loss:Q", color=alt.Color("split:N", scale=alt.Scale(range=[C["teal"], C["peach"]])),
        ).properties(height=230, title="Loss")), width="stretch")
        acc_df = hist.melt("epoch", ["train_acc", "val_acc", "val_balanced_acc"], var_name="metric", value_name="score")
        c2.altair_chart(style_chart(alt.Chart(acc_df).mark_line(point=True).encode(
            x="epoch:O", y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N", scale=alt.Scale(range=[C["teal"], C["green"], C["amber"]])),
        ).properties(height=230, title="Accuracy")), width="stretch")
    per = meta.get("per_class", {})
    if per:
        md(ui.section_title(tr("Per-class performance (test set)")))
        pdf_df = pd.DataFrame(per).T.reset_index().rename(columns={"index": "Class", "f1-score": "f1"})
        pdf_df["Class"] = pdf_df["Class"].map(t_class)
        st.dataframe(pdf_df, width="stretch", hide_index=True)
    cm = meta.get("confusion_matrix")
    if cm:
        names = [t_class(n) for n in meta["classes"]]
        rows = [{"Actual": names[i], "Predicted": names[j], "count": v, "pct": round(v / (sum(r) or 1) * 100, 1)}
                for i, r in enumerate(cm) for j, v in enumerate(r)]
        cdf = pd.DataFrame(rows)
        base = alt.Chart(cdf).encode(x=alt.X("Predicted:N", sort=names), y=alt.Y("Actual:N", sort=names))
        heat = base.mark_rect().encode(color=alt.Color("pct:Q", scale=alt.Scale(range=[C["surface2"], C["teal"]]), title="%"),
                                       tooltip=["Actual", "Predicted", "count", "pct"])
        text = base.mark_text(fontSize=11).encode(text="count:Q",
                                                  color=alt.condition(alt.datum.pct > 50, alt.value(C["teal_ink"]), alt.value(C["soft"])))
        md(ui.section_title(tr("Confusion matrix (test set)")))
        st.altair_chart(style_chart((heat + text).properties(height=380)), width="stretch")
    md(f'<div class="disclaimer">{E(tr("Hybrid design: DERMADISC-Net is a specialist for 7 skin-lesion types and runs offline. The vision-language AI covers a wider range of conditions. The app cross-checks both, and flags disagreement for dermatologist review."))}</div>')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
PAGES = {
    "nav_dashboard": page_dashboard, "nav_detection": page_detection, "nav_scans": page_scans,
    "nav_progress": page_progress, "nav_abcde": page_abcde, "nav_dermabot": page_dermabot, "nav_risk": page_risk,
    "nav_library": page_library, "nav_skincare": page_skincare, "nav_model": page_model, "nav_account": page_account,
}


def main_app():
    fresh = db.get_user(user()["id"])           # role / active changes take effect immediately
    if not fresh or not fresh["active"]:
        logout()
    st.session_state.user = fresh
    u = fresh
    pages = dict(PAGES)
    if u["role"] == "admin":
        pages["nav_admin"] = page_admin
    page = st.session_state.get("nav", "nav_dashboard")
    if page not in pages:
        page = st.session_state.nav = "nav_dashboard"

    navbar(u, page)
    if u.get("must_change_pw") and page != "nav_account":
        st.warning(t("change_default_pw") if u["username"] == db.DEFAULT_ADMIN_USER
                   else tr("Please choose a new password in My Account - you are using a temporary one."))
    pages[page]()
    flush_translations(api_key(), st.session_state.preferred_model)


if user():
    main_app()
else:
    auth_page()
