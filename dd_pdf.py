"""
DERMADISC PDF reports - works in all six app languages (needs the fonts/ folder next to app.py).
"""
BUILD = "2026-10-02h"  # must match app.py; tells the app this file is up to date
import io
import os
from datetime import datetime

from fpdf import FPDF
from fpdf.fonts import FontFace

import dd_db as db
from dd_i18n import t_lang, t_class, t_urgency

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(APP_DIR, "fonts")
# One font per language. The Indian-script fonts are Noto Sans + Noto Sans <script> merged into a single
# file, so a report never has to switch fonts mid-line (switching breaks text shaping in the PDF library).
FONT_FILES = {
    "latin": "NotoSans",
    "deva": "DermadiscSansDevanagari",
    "beng": "DermadiscSansBengali",
    "taml": "DermadiscSansTamil",
    "telu": "DermadiscSansTelugu",
}
LANG_FONT = {"en": "latin", "hi": "deva", "mr": "deva", "bn": "beng", "ta": "taml", "te": "telu"}
# (ISO 15924 script code, BCP-47 language code) for the text-shaping engine
LANG_SHAPING = {"en": ("latn", "en"), "hi": ("deva", "hi"), "mr": ("deva", "mr"), "bn": ("beng", "bn"),
                "ta": ("taml", "ta"), "te": ("telu", "te")}

NAVY = (22, 36, 63)        # header band and table heads (matches the app's dark theme)
TEAL = (44, 199, 180)      # brand accent
TEAL_INK = (14, 122, 110)  # teal dark enough to read on white paper
DARK = (24, 32, 50)
GREY = (105, 115, 135)
LIGHT = (240, 246, 248)
URGENCY_RGB = {"low": (0, 160, 90), "medium": (210, 150, 0), "high": (220, 50, 50)}


def fonts_available():
    return all(os.path.exists(os.path.join(FONT_DIR, f"{f}-{w}.ttf")) for f in FONT_FILES.values() for w in ("Regular", "Bold"))


class ReportPDF(FPDF):
    def __init__(self, lang, footer_note=""):
        super().__init__(format="A4")
        self.lang = lang
        self.footer_note = footer_note
        self.unicode_ok = fonts_available()
        if self.unicode_ok:
            for fam, file in FONT_FILES.items():
                self.add_font(fam, "", os.path.join(FONT_DIR, f"{file}-Regular.ttf"))
                self.add_font(fam, "B", os.path.join(FONT_DIR, f"{file}-Bold.ttf"))
            self.base = LANG_FONT.get(lang, "latin")
            # only used if text from a *different* script appears (e.g. a Tamil name in a Hindi report)
            self.set_fallback_fonts([f for f in ("deva", "beng", "taml", "telu") if f != self.base])
            # Text shaping joins Indian-script letters correctly (needs the uharfbuzz package).
            # The script is set explicitly: otherwise a line that starts with English ("DERMADISC ...")
            # is shaped as Latin and the Indian-script words on that line come out garbled.
            script, language = LANG_SHAPING.get(lang, (None, None))
            try:
                self.set_text_shaping(True, script=script, language=language)
            except Exception:
                try:
                    self.set_text_shaping(True)
                except Exception:
                    pass
        else:
            self.base = "helvetica"
        self.set_auto_page_break(True, margin=18)
        self.set_margins(14, 14, 14)

    def txt(self, s):
        s = "" if s is None else str(s)
        if self.unicode_ok:
            return s
        return s.encode("latin-1", "replace").decode("latin-1")   # last-resort fallback without fonts

    def font(self, size=10, bold=False, color=DARK):
        self.set_font(self.base, "B" if bold else "", size)
        self.set_text_color(*color)

    def footer(self):
        self.set_y(-12)
        self.font(8, color=GREY)
        left = "Dermadisc" + (f"  ·  {self.footer_note}" if self.footer_note else "")
        self.cell(0, 6, self.txt(left), align="L")
        self.set_x(-40)
        self.cell(26, 6, self.txt(t_lang("pdf_page", self.lang, n=self.page_no())), align="R")

    # ── building blocks ──
    def section(self, title):
        if self.get_y() > 255:
            self.add_page()
        self.ln(3)
        self.font(12, True, TEAL_INK)
        self.cell(0, 8, self.txt(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*TEAL)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def para(self, text, size=10, color=DARK, bold=False, h=5.5):
        self.font(size, bold, color)
        self.multi_cell(0, h, self.txt(text), new_x="LMARGIN", new_y="NEXT", align="L")

    def bullets(self, items, empty_text):
        items = [i for i in (items or []) if str(i).strip()]
        if not items:
            self.para(empty_text, color=GREY)
            return
        for it in items:
            self.font(10)
            self.cell(5, 5.5, "•")
            self.multi_cell(0, 5.5, self.txt(it), new_x="LMARGIN", new_y="NEXT", align="L")

    def kv_table(self, pairs, col1=42):
        # NOTE: bold is only used for single-language labels. Bold text that mixes scripts and wraps
        # onto a second line is rendered incorrectly by the PDF library, so values stay regular weight.
        self.font(9.5)
        self.set_draw_color(225, 225, 235)
        self.set_line_width(0.2)
        label = FontFace(emphasis="BOLD", color=GREY, fill_color=LIGHT)
        value = FontFace(color=DARK, fill_color=(255, 255, 255))
        with self.table(col_widths=(col1, self.epw - col1), text_align="LEFT", line_height=6,
                        first_row_as_headings=False, borders_layout="HORIZONTAL_LINES") as table:
            for k, v in pairs:
                row = table.row()
                row.cell(self.txt(k), style=label)
                row.cell(self.txt(v), style=value)


def _local_values(lang, values):
    """Translate stored English option values (e.g. body area 'Arm') using the shared translation cache."""
    values = [v for v in values if v]
    if lang == "en" or not values:
        return {}
    try:
        return db.get_translations(lang, values)
    except Exception:
        return {}


def build_pdf(scan, lang=None):
    """scan = dict from dd_db.get_scan(). Returns PDF bytes."""
    lang = lang or scan.get("language") or "en"
    T = lambda key, **kw: t_lang(key, lang, **kw)
    result = scan.get("result") or None
    ctx = scan.get("context") or {}
    local = scan.get("local") or {}
    quality = scan.get("quality") or {}
    colors = scan.get("colors") or {}

    pdf = ReportPDF(lang, footer_note=T("pdf_footer_model", model=scan.get("ai_model") or "-"))
    pdf.add_page()

    # ── header band ──
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pdf.w, 30, style="F")
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 30, pdf.w, 1.6, style="F")
    text_x = 14
    try:
        import dd_ui
        pdf.image(io.BytesIO(dd_ui.logo_png_bytes(256)), x=14, y=6, w=18, h=18)
        text_x = 36
    except Exception:
        pass
    pdf.set_xy(text_x, 7)
    pdf.font(22, True, (255, 255, 255))
    pdf.cell(0, 10, "Dermadisc")
    pdf.set_xy(text_x, 17)
    pdf.font(10, False, (201, 211, 230))
    pdf.cell(0, 6, pdf.txt(T("app_tagline")))
    pdf.set_xy(100, 9)
    pdf.font(14, True, (255, 255, 255))
    pdf.cell(96, 8, pdf.txt(T("pdf_title")), align="R")
    pdf.set_xy(100, 17)
    pdf.font(9, False, (201, 211, 230))
    pdf.cell(96, 6, pdf.txt(f'{T("pdf_report_id")}: #{scan.get("id", "-")}'), align="R")
    pdf.set_y(36)

    # ── report meta ──
    name = scan.get("full_name") or scan.get("username") or "-"
    pdf.kv_table([
        (T("pdf_name"), name),
        (T("pdf_scan_date"), scan.get("created_at", "-")),
        (T("pdf_generated"), datetime.now().strftime("%Y-%m-%d %H:%M")),
    ])

    # ── patient details ──
    pdf.section(T("patient_details"))
    loc = _local_values(lang, [ctx.get("sex"), ctx.get("area"), ctx.get("duration"), *(ctx.get("symptoms") or [])])
    L = lambda v: loc.get(v, v)
    symptoms = ", ".join(L(s) for s in (ctx.get("symptoms") or [])) or T("none_reported")
    pdf.kv_table([
        (T("age"), ctx.get("age", "-")),
        (T("sex"), L(ctx.get("sex", "-"))),
        (T("body_area"), L(ctx.get("area", "-"))),
        (T("duration"), L(ctx.get("duration", "-"))),
        (T("symptoms"), symptoms),
        (T("notes"), ctx.get("notes") or "-"),
    ], col1=60)

    # ── images ──
    def readable(blob):
        try:
            from PIL import Image as _I
            _I.open(io.BytesIO(blob)).verify()
            return True
        except Exception:
            return False

    imgs = [(T("photo"), scan.get("image")), (T("heatmap"), scan.get("heatmap"))]
    imgs = [(cap, b) for cap, b in imgs if b and readable(b)]
    if imgs:
        pdf.section(T("pdf_images"))
        y0 = pdf.get_y()
        box_w, box_h = 86, 72
        if y0 + box_h + 12 > pdf.h - 18:
            pdf.add_page()
            y0 = pdf.get_y()
        for n, (cap, blob) in enumerate(imgs):
            x = pdf.l_margin + n * (box_w + 10)
            pdf.image(io.BytesIO(blob), x=x, y=y0, w=box_w, h=box_h, keep_aspect_ratio=True)
            pdf.set_xy(x, y0 + box_h + 1)
            pdf.font(8.5, False, GREY)
            pdf.cell(box_w, 5, pdf.txt(cap), align="C")
        pdf.set_y(y0 + box_h + 8)
        if scan.get("heatmap"):
            pdf.para(T("pdf_heatmap_note"), size=8.5, color=GREY, h=4.5)

    # ── AI analysis ──
    pdf.section(T("pdf_ai_section"))
    if not result:
        pdf.para(T("pdf_no_ai"), color=GREY)
    elif not result.get("is_skin_image", True):
        pdf.para(T("not_skin"), color=URGENCY_RGB["medium"])
    else:
        urg = result.get("urgency", "medium")
        pdf.set_fill_color(*LIGHT)
        pdf.font(9, True, GREY)
        pdf.cell(0, 6, pdf.txt(T("overall_assessment").upper()), new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_fill_color(*LIGHT)
        pdf.font(10.5)
        pdf.multi_cell(0, 6, pdf.txt(result.get("overall_assessment", "")), new_x="LMARGIN", new_y="NEXT", fill=True, align="L")
        pdf.font(10.5, True, URGENCY_RGB.get(urg, DARK))
        pdf.cell(0, 7, pdf.txt(f'{T("urgency")}: {T("urgency_" + urg) if urg in ("low", "medium", "high") else urg}'),
                 new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(3)

        preds = result.get("predictions") or []
        if preds:
            pdf.font(9)
            head = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=NAVY)
            pdf.set_draw_color(225, 225, 235)
            pdf.set_fill_color(*LIGHT)
            with pdf.table(col_widths=(46, 14, 68, 54), text_align="LEFT", line_height=5.2,
                           headings_style=head, borders_layout="HORIZONTAL_LINES") as table:
                h = table.row()
                for c in (T("condition"), "%", T("pdf_observation"), T("advice")):
                    h.cell(pdf.txt(c))
                for p in preds:
                    r = table.row()
                    name_local = p.get("disease_local") or p.get("disease", "")
                    name = name_local if name_local == p.get("disease") else f'{name_local} ({p.get("disease", "")})'
                    obs = p.get("description", "")
                    if p.get("key_features"):
                        obs += f'\n{T("pdf_key_features")}: ' + ", ".join(p["key_features"])
                    r.cell(pdf.txt(name))          # regular weight on purpose (mixed scripts, see kv_table)
                    r.cell(f'{p.get("confidence", 0)}%')
                    r.cell(pdf.txt(obs))
                    r.cell(pdf.txt(p.get("recommendation", "")))
            pdf.ln(2)

        pdf.font(11, True, (200, 40, 40))
        pdf.cell(0, 7, pdf.txt(T("red_flags")), new_x="LMARGIN", new_y="NEXT")
        pdf.bullets(result.get("red_flags"), T("pdf_none"))
        pdf.ln(1)
        pdf.font(11, True, (0, 140, 80))
        pdf.cell(0, 7, pdf.txt(T("next_steps")), new_x="LMARGIN", new_y="NEXT")
        pdf.bullets(result.get("next_steps"), "-")

    # ── trained model ──
    lpreds = local.get("preds") or []
    if lpreds:
        pdf.section(T("cnn_title"))
        pdf.font(9.5)
        head = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=NAVY)
        pdf.set_draw_color(225, 225, 235)
        pdf.set_fill_color(*LIGHT)
        with pdf.table(col_widths=(110, 36, 36), text_align="LEFT", line_height=6,
                       headings_style=head, borders_layout="HORIZONTAL_LINES") as table:
            h = table.row()
            for c in (T("condition"), T("pdf_probability"), T("urgency")):
                h.cell(pdf.txt(c))
            for p in lpreds[:3]:
                r = table.row()
                r.cell(pdf.txt(t_class(p.get("disease", ""), lang)))
                r.cell(f'{p.get("confidence", 0)}%')
                r.cell(pdf.txt(t_urgency(p.get("urgency", "medium"), lang)))
        if not result and lpreds[0].get("confidence", 0) < 55:
            pdf.ln(2)
            pdf.para(T("pdf_cnn_unsure"), color=(110, 80, 200))
        status = local.get("agreement")
        if status in ("agree", "partial", "disagree"):
            pdf.ln(2)
            pdf.font(10, True)
            pdf.cell(0, 6, pdf.txt(T("pdf_agreement")), new_x="LMARGIN", new_y="NEXT")
            pdf.para(T(f"{status}_text"), color={"agree": (0, 140, 80), "partial": (190, 130, 0)}.get(status, (200, 40, 40)))

    # ── image analysis ──
    if quality:
        pdf.section(T("image_quality"))
        pdf.kv_table([
            (T("quality_score"), f'{quality.get("score", "-")}/100'),
            (T("resolution"), quality.get("resolution", "-")),
            (T("brightness"), quality.get("brightness", "-")),
            (T("sharpness"), quality.get("sharpness", "-")),
            (T("avg_colour"), colors.get("hex", "-")),
            (T("redness_index"), colors.get("redness_index", "-")),
        ], col1=60)

    # ── disclaimer ──
    pdf.ln(5)
    if pdf.get_y() > 250:
        pdf.add_page()
    pdf.set_fill_color(255, 246, 224)
    pdf.set_draw_color(240, 192, 96)
    pdf.font(9, False, (120, 80, 0))
    pdf.multi_cell(0, 5, pdf.txt(T("disclaimer")),
                   border=1, fill=True, padding=3, new_x="LMARGIN", new_y="NEXT", align="L")

    return bytes(pdf.output())


def report_filename(scan):
    date = (scan.get("created_at") or "")[:10] or datetime.now().strftime("%Y-%m-%d")
    return f"dermadisc_report_{scan.get('id', 'scan')}_{date}.pdf"
