"""
DERMADISC core logic - image analysis, AI calls, trained model, lesion measurement,
risk model, UV data and reference content. No page layout code lives here.
"""
BUILD = "2026-10-02h"  # must match app.py; tells the app this file is up to date
import os
import io
import json
import time
import base64
import hashlib
import html

import numpy as np
import requests
import streamlit as st
import altair as alt
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageStat

# Free vision models on OpenRouter. They are tried in order with automatic fallback.
MODELS = [
    "qwen/qwen2.5-vl-72b-instruct:free",
    "qwen/qwen2.5-vl-32b-instruct:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "google/gemma-4-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "openrouter/free",
    "openrouter/auto",
]

PROMPT_TEMPLATE = """You are a dermatology AI assistant that supports (never replaces) a clinician.
Analyze the attached skin image together with the patient context below.

PATIENT CONTEXT:
<<CONTEXT>>

Tasks:
1. Decide whether the image actually shows human skin. If it does not, set "is_skin_image" to false and return an empty predictions list.
2. Give the TOP 3 most likely skin conditions, most likely first, with a confidence 0-100.
3. List any red-flag features you can see (e.g. irregular border, multiple colours, bleeding, rapid growth, signs of spreading infection). Use an empty list if none.
4. Suggest practical next steps.

Reply ONLY with valid JSON - no markdown, no extra text, nothing outside the JSON.

{
  "is_skin_image": true,
  "image_quality": "good",
  "predictions": [
    {
      "disease": "condition name in English",
      "disease_local": "condition name in <<LANG>>",
      "confidence": 85,
      "description": "what you see in 1-2 sentences",
      "key_features": ["feature 1", "feature 2"],
      "recommendation": "brief advice"
    }
  ],
  "red_flags": ["..."],
  "overall_assessment": "one sentence summary",
  "urgency": "low",
  "next_steps": ["step 1", "step 2"]
}

image_quality must be exactly: good, fair, or poor.
urgency must be exactly: low, medium, or high.
LANGUAGE: keep every JSON key and the "disease" value in English. Write "disease_local", "description",
"key_features", "recommendation", "red_flags", "overall_assessment" and "next_steps" in <<LANG>>."""



# ─────────────────────────────────────────────
# IMAGE PROCESSING HELPERS
# ─────────────────────────────────────────────
def image_to_base64(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()


def enhance_image(image):
    """Auto-contrast + mild sharpening to make lesion features clearer."""
    img = ImageOps.autocontrast(image, cutoff=1)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    return img


def assess_quality(image):
    """Heuristic image-quality check: brightness, sharpness (edge variance), resolution."""
    gray = image.convert("L")
    brightness = ImageStat.Stat(gray).mean[0]
    w, h = image.size
    # crop 2px border - FIND_EDGES creates artificial edges at the image frame
    edges = gray.filter(ImageFilter.FIND_EDGES).crop((2, 2, max(w - 2, 3), max(h - 2, 3)))
    sharpness = ImageStat.Stat(edges).var[0]

    issues = []
    if brightness < 60:
        issues.append("Image is too dark - use better lighting.")
    elif brightness > 215:
        issues.append("Image is overexposed - avoid direct flash or glare.")
    if sharpness < 60:
        issues.append("Image looks blurry - hold the camera steady and focus on the affected area.")
    if min(w, h) < 224:
        issues.append("Low resolution - move closer or use a higher-quality photo.")

    score = max(100 - 30 * len(issues), 10)
    return {
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "resolution": f"{w}x{h}",
        "issues": issues,
        "score": score,
    }


def color_profile(image):
    """Average colour + a simple redness (erythema) index - heuristic only."""
    small = image.resize((128, 128))
    r, g, b = ImageStat.Stat(small).mean[:3]
    redness = (r - (g + b) / 2) / 255 * 100
    return {
        "hex": "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b)),
        "rgb": (int(r), int(g), int(b)),
        "redness_index": round(redness, 1),
    }


# ─────────────────────────────────────────────
# OPENROUTER HELPERS
# ─────────────────────────────────────────────
# Preferred model families (tried first when several free vision models are available)
PREFERRED_FAMILIES = ["qwen", "gemma", "llama", "mistral", "nvidia", "moonshot", "kimi", "glm"]
MAX_MODELS_TO_TRY = 8


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_free_vision_models():
    """Ask OpenRouter which models are free right now AND accept images. Cached for 30 min."""
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        data = r.json().get("data", [])
    except Exception:
        return []

    found = []
    for m in data:
        mid = str(m.get("id", ""))
        arch = m.get("architecture") or {}
        inputs = [str(x).lower() for x in (arch.get("input_modalities") or [])]
        modality = str(arch.get("modality", "")).lower().split("->")[0]
        accepts_image = "image" in inputs or "image" in modality
        pricing = m.get("pricing") or {}

        def is_zero(v):
            try:
                return float(v) == 0
            except (TypeError, ValueError):
                return False

        is_free = mid.endswith(":free") or (is_zero(pricing.get("prompt")) and is_zero(pricing.get("completion")))
        if accepts_image and is_free and not mid.startswith("openrouter/"):
            fam = next((i for i, f in enumerate(PREFERRED_FAMILIES) if f in mid.lower()), len(PREFERRED_FAMILIES))
            found.append((fam, -(m.get("context_length") or 0), mid))
    found.sort()
    return [mid for _, _, mid in found]


def get_models():
    """Live free vision models first, then the static list as a backup."""
    live = fetch_free_vision_models()
    return live + [m for m in MODELS if m not in live]


def model_order(preferred):
    models = get_models()
    if preferred in models:
        models = [preferred] + [m for m in models if m != preferred]
    return models[:MAX_MODELS_TO_TRY]


def call_openrouter(messages, api_key, model, max_tokens=1200):
    resp = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://dermadisc.app",
            "X-Title": "DERMADISC",
        },
        json={"model": model, "max_tokens": max_tokens, "messages": messages},
        timeout=60,
    )
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"HTTP {resp.status_code} (non-JSON response)")

    if "error" in data:
        err = data["error"]
        raise RuntimeError(err.get("message", str(err)) if isinstance(err, dict) else str(err))

    content = data["choices"][0]["message"].get("content")
    if not content:
        raise RuntimeError("empty response")
    return content.strip(), data.get("model", model)


def parse_json(raw):
    raw = raw.replace("```json", "").replace("```", "").strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(raw[start:end])


def normalize_result(result):
    """Clean up / validate whatever the model returned."""
    preds = []
    for p in result.get("predictions", []) or []:
        try:
            conf = int(float(p.get("confidence", 0)))
        except (TypeError, ValueError):
            conf = 0
        feats = p.get("key_features", [])
        preds.append({
            "disease": str(p.get("disease", "Unknown")),
            "disease_local": str(p.get("disease_local") or p.get("disease", "Unknown")),
            "confidence": max(0, min(conf, 100)),
            "description": str(p.get("description", "")),
            "key_features": [str(f) for f in feats] if isinstance(feats, list) else [],
            "recommendation": str(p.get("recommendation", "")),
        })
    preds.sort(key=lambda x: x["confidence"], reverse=True)

    urgency = str(result.get("urgency", "low")).lower().strip()
    if urgency not in ("low", "medium", "high"):
        urgency = "medium"

    def as_list(x):
        return [str(i) for i in x] if isinstance(x, list) else []

    return {
        "is_skin_image": bool(result.get("is_skin_image", True)),
        "image_quality": str(result.get("image_quality", "fair")).lower(),
        "predictions": preds[:3],
        "red_flags": as_list(result.get("red_flags")),
        "overall_assessment": str(result.get("overall_assessment", "")),
        "urgency": urgency,
        "next_steps": as_list(result.get("next_steps")),
    }


def analyze_skin_image(image, api_key, context_text, preferred, language="English"):
    img_b64 = image_to_base64(image)
    prompt = PROMPT_TEMPLATE.replace("<<CONTEXT>>", context_text).replace("<<LANG>>", language)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ],
    }]

    errors = []
    for model in model_order(preferred):
        try:
            raw, used = call_openrouter(messages, api_key, model)
            result = normalize_result(parse_json(raw))
            result["_model"] = used
            return result
        except requests.exceptions.Timeout:
            errors.append(f"{model}: timed out")
        except Exception as e:
            # Bad JSON or API error -> fall back to the next model instead of giving up
            errors.append(f"{model}: {e}")
    raise RuntimeError("\n".join(f"• {e}" for e in errors))


def ask_ai_text(messages, api_key, preferred="Auto (try all)", max_tokens=700):
    """Text-only chat call with the same automatic model fallback. Returns the reply text or raises."""
    errors = []
    for model in model_order(preferred):
        try:
            reply, _ = call_openrouter(messages, api_key, model, max_tokens=max_tokens)
            return reply
        except Exception as e:
            errors.append(f"{model}: {e}")
    raise RuntimeError("\n".join(errors))


def build_context_text(ctx):
    ctx = {"age": "-", "sex": "-", "area": "-", "duration": "-", "symptoms": [], "notes": "", **(ctx or {})}
    return (
        f"Age: {ctx['age']}\n"
        f"Sex: {ctx['sex']}\n"
        f"Body area: {ctx['area']}\n"
        f"Duration: {ctx['duration']}\n"
        f"Symptoms: {', '.join(ctx['symptoms']) if ctx['symptoms'] else 'none reported'}\n"
        f"Notes: {ctx['notes'] or 'none'}"
    )




# ─────────────────────────────────────────────
# LOCAL TRAINED MODEL (DERMADISC-Net, ONNX)
# ─────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(APP_DIR, "dermadisc_model.onnx")
LOCAL_META_PATH = os.path.join(APP_DIR, "dermadisc_model_meta.json")


@st.cache_resource(show_spinner=False)
def load_local_model():
    """Returns (session, meta, error). Works without the model - the app just hides those features."""
    if not os.path.exists(LOCAL_MODEL_PATH) or not os.path.exists(LOCAL_META_PATH):
        return None, None, "Model files not found next to app.py"
    try:
        import onnxruntime as ort
    except ImportError:
        return None, None, "onnxruntime is not installed (pip install onnxruntime)"
    try:
        sess = ort.InferenceSession(LOCAL_MODEL_PATH, providers=["CPUExecutionProvider"])
        with open(LOCAL_META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        return sess, meta, None
    except Exception as e:
        return None, None, f"Could not load model: {e}"


def preprocess_local(image, meta):
    """Same pre-processing as training: 256 bicubic -> 224 bilinear -> ImageNet normalisation."""
    size = int(meta.get("img_size", 224))
    img = image.convert("RGB").resize((256, 256), Image.BICUBIC).resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(meta["mean"], dtype=np.float32)) / np.array(meta["std"], dtype=np.float32)
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def jet_colormap(h):
    """h in [0,1] -> RGB uint8 (blue = low importance, red = high importance)."""
    r = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def occlusion_heatmap(image, sess, meta, patch=48, stride=16, batch=32):
    """
    Explainable AI - occlusion sensitivity.
    Slide a grey patch over the image and measure how much the top-class probability drops.
    Regions where hiding the pixels hurts the prediction most are the ones the model relied on.
    Returns (overlay image, heat-only image, class name, base probability, ms).
    """
    t0 = time.perf_counter()
    x = preprocess_local(image, meta)
    name = sess.get_inputs()[0].name
    base_probs = softmax(sess.run(None, {name: x})[0][0])
    cls = int(np.argmax(base_probs))
    base = float(base_probs[cls])

    size = x.shape[-1]
    positions = [(r, c) for r in range(0, size - patch + 1, stride) for c in range(0, size - patch + 1, stride)]
    heat = np.zeros((size, size), dtype=np.float32)
    count = np.zeros((size, size), dtype=np.float32)

    for i in range(0, len(positions), batch):
        chunk = positions[i:i + batch]
        xb = np.repeat(x, len(chunk), axis=0)
        for j, (r, c) in enumerate(chunk):
            xb[j, :, r:r + patch, c:c + patch] = 0.0          # 0 = dataset-mean colour after normalisation
        probs = softmax(sess.run(None, {name: xb})[0])[:, cls]
        for j, (r, c) in enumerate(chunk):
            heat[r:r + patch, c:c + patch] += base - probs[j]
            count[r:r + patch, c:c + patch] += 1

    heat = np.clip(heat / np.maximum(count, 1), 0, None)
    if heat.max() > 0:
        heat = heat / heat.max()

    # map back onto the original photo (the model sees the full image squashed to 224x224)
    disp = image.convert("RGB").copy()
    disp.thumbnail((640, 640))
    w, h = disp.size
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    hm = np.asarray(heat_img, dtype=np.float32) / 255.0
    colours = jet_colormap(hm).astype(np.float32)
    alpha = (0.15 + 0.55 * hm)[..., None]
    overlay = (np.asarray(disp, dtype=np.float32) * (1 - alpha) + colours * alpha).astype(np.uint8)

    infos = meta.get("class_info") or [{"name": n} for n in meta["classes"]]
    ms = (time.perf_counter() - t0) * 1000
    return Image.fromarray(overlay), Image.fromarray(jet_colormap(hm)), infos[cls]["name"], base * 100, ms


UNCERTAIN_TOP = 50.0      # top probability below this -> uncertain
UNCERTAIN_MARGIN = 15.0   # or top-1 minus top-2 below this -> uncertain


def predict_local(image, sess, meta, tta=True):
    """
    Test-time augmentation (TTA): the model sees the original + horizontally flipped + vertically flipped
    + rotated image, and the 4 probability vectors are averaged. Skin lesions have no fixed orientation,
    so this gives steadier, more reliable predictions.
    """
    arr = preprocess_local(image, meta)
    if tta:
        arr = np.concatenate([arr, arr[..., ::-1], arr[..., ::-1, :], np.rot90(arr, 1, axes=(2, 3))]).copy()

    t0 = time.perf_counter()
    logits = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
    ms = (time.perf_counter() - t0) * 1000

    view_probs = softmax(logits)
    probs = view_probs.mean(axis=0)
    order = np.argsort(-probs)
    # how many of the views voted for the same top class -> stability
    agree_views = int((view_probs.argmax(axis=1) == order[0]).sum())
    infos = meta.get("class_info") or [{"name": n, "urgency": "medium", "desc": ""} for n in meta["classes"]]
    preds = [{
        "disease": infos[i]["name"],
        "confidence": round(float(probs[i]) * 100, 1),
        "urgency": infos[i].get("urgency", "medium"),
        "desc": infos[i].get("desc", ""),
    } for i in order]
    top1, top2 = preds[0]["confidence"], (preds[1]["confidence"] if len(preds) > 1 else 0)
    preds[0]["uncertain"] = top1 < UNCERTAIN_TOP or (top1 - top2) < UNCERTAIN_MARGIN
    preds[0]["margin"] = round(top1 - top2, 1)
    preds[0]["views"] = len(view_probs)
    preds[0]["agree_views"] = agree_views
    return preds, ms


# keyword groups used to check whether the trained model and the vision AI agree
CANON = [
    ("akiec", ["actinic"]),
    ("bcc", ["basal cell", "bcc"]),
    ("mel", ["melanoma"]),
    ("nv", ["nevus", "nevi", "naevus", "mole"]),
    ("df", ["dermatofibroma"]),
    ("vasc", ["vascular", "angioma", "hemangioma", "haemangioma", "pyogenic"]),
    ("bkl", ["seborrheic keratosis", "seborrhoeic keratosis", "benign keratosis", "lentigo", "keratosis"]),
]


def canonical(name):
    n = name.lower()
    for key, words in CANON:
        if any(w in n for w in words):
            return key
    return None


def agreement(local_top, ai_preds):
    c = canonical(local_top)
    ai_c = [canonical(p["disease"]) for p in ai_preds]
    if c and ai_c and ai_c[0] == c:
        return "agree", "✅ Both models agree on the top diagnosis - higher confidence in this result."
    if c and c in ai_c:
        return "partial", "🟡 Partial agreement - the trained model's prediction appears in the AI's top 3."
    return "disagree", ("⚠️ The models disagree. This can happen when the image is not a close-up lesion photo "
                        "(our model is trained on dermatoscopic lesion images). A dermatologist review is recommended.")


# ─────────────────────────────────────────────
# AUTOMATED LESION ANALYSIS (segmentation + ABCD metrics)
# ─────────────────────────────────────────────
SEG_SIZE = 192


def otsu_threshold(gray):
    """Classic Otsu: pick the threshold that best separates dark lesion pixels from lighter skin."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    cum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))
    mean_all = cum_mean[-1] / total
    w0 = cum / total
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = cum_mean / np.maximum(cum, 1)
        mu1 = (cum_mean[-1] - cum_mean) / np.maximum(total - cum, 1)
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[np.isnan(between)] = 0
    return int(np.argmax(between)), mean_all


def largest_component(mask):
    """Keep only the largest connected blob (4-connectivity BFS) - removes hair, noise and shadows."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    best_label, best_size, cur = 0, 0, 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if labels[y0, x0]:
            continue
        cur += 1
        stack, size = [(y0, x0)], 0
        labels[y0, x0] = cur
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = cur
                    stack.append((ny, nx))
        if size > best_size:
            best_size, best_label = size, cur
    return labels == best_label if best_label else np.zeros_like(mask, dtype=bool)


def remove_lighting(gray):
    """Fit and subtract a smooth lighting plane (a*x + b*y + c) so uneven light isn't mistaken for a lesion.
    Two passes: the second ignores dark pixels so the lesion itself doesn't bend the plane."""
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w)], axis=1).astype(np.float64)
    z = gray.ravel().astype(np.float64)
    keep = np.ones_like(z, dtype=bool)
    for _ in range(2):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        resid = z - A @ coef
        keep = resid > -1.5 * resid[keep].std()
    return (z - A @ coef).reshape(h, w)


MIN_LESION_CONTRAST = 12.0   # lesion must be at least this much darker than the surrounding skin (0-255 scale)


def segment_lesion(image):
    """Returns a boolean mask (SEG_SIZE x SEG_SIZE) of the main dark lesion, or None if nothing clear is found."""
    small = image.convert("RGB").resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2))
    resid = remove_lighting(np.asarray(small.convert("L"), dtype=np.float32))
    lo, hi = np.percentile(resid, 0.5), np.percentile(resid, 99.5)
    scaled = np.clip((resid - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    thr, _ = otsu_threshold(scaled)
    mask = scaled < thr
    # ignore a thin border so dark photo corners / vignetting aren't picked up
    b = int(SEG_SIZE * 0.04)
    mask[:b, :] = mask[-b:, :] = False
    mask[:, :b] = mask[:, -b:] = False
    # smooth the mask (close small gaps, remove specks)
    m_img = Image.fromarray((mask * 255).astype(np.uint8))
    m_img = m_img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))   # closing
    m_img = m_img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))   # opening
    mask = largest_component(np.asarray(m_img) > 127)
    frac = mask.mean()
    if frac < 0.002 or frac > 0.85:      # nothing found, or the whole photo is "lesion"
        return None
    # the blob must be clearly darker than the skin around it - otherwise it's just skin texture
    if resid[~mask].mean() - resid[mask].mean() < MIN_LESION_CONTRAST:
        return None
    return mask


def lesion_metrics(image, mask):
    """Automated ABCD: Asymmetry, Border irregularity, Colour variation, Diameter (relative size)."""
    arr = np.asarray(image.convert("RGB").resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR), dtype=np.float32)
    area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()

    # A - asymmetry: overlap of the lesion with its own mirror image around the centroid
    def mirror_score(flip_axis):
        coords_y, coords_x = ys.copy(), xs.copy()
        if flip_axis == 0:
            coords_y = np.round(2 * cy - ys).astype(int)
        else:
            coords_x = np.round(2 * cx - xs).astype(int)
        ok = (coords_y >= 0) & (coords_y < SEG_SIZE) & (coords_x >= 0) & (coords_x < SEG_SIZE)
        overlap = mask[coords_y[ok], coords_x[ok]].sum()
        return 1 - overlap / area
    asymmetry = round(float((mirror_score(0) + mirror_score(1)) / 2) * 100, 1)

    # B - border irregularity: compactness = perimeter^2 / (4*pi*area); a perfect circle = 1
    m = mask.astype(np.uint8)
    edge = m - np.asarray(Image.fromarray(m * 255).filter(ImageFilter.MinFilter(3))) // 255
    perimeter = max(int(edge.sum()), 1)
    border = round(float(perimeter ** 2 / (4 * np.pi * area)), 2)

    # C - colour variation inside the lesion + mean lesion colour
    pix = arr[mask]
    colour_std = round(float(pix.std(axis=0).mean()), 1)
    mean_rgb = pix.mean(axis=0)
    lesion_hex = "#{:02x}{:02x}{:02x}".format(*[int(v) for v in mean_rgb])
    skin = arr[~mask].mean(axis=0) if (~mask).any() else mean_rgb
    contrast = round(float(np.abs(skin - mean_rgb).mean()), 1)

    # D - relative size (% of the photo) and equivalent diameter as % of photo width
    area_pct = round(area / mask.size * 100, 2)
    diam_pct = round(float(2 * np.sqrt(area / np.pi) / SEG_SIZE * 100), 1)

    return {"area_pct": area_pct, "diameter_pct": diam_pct, "asymmetry": asymmetry, "border": border,
            "colour_variation": colour_std, "contrast": contrast, "lesion_hex": lesion_hex}


def outline_overlay(image, mask, color=(255, 60, 60)):
    disp = image.convert("RGB").copy()
    disp.thumbnail((480, 480))
    w, h = disp.size
    m = Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    edge = np.asarray(m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))) > 0
    out = np.asarray(disp).copy()
    out[edge] = color
    return Image.fromarray(out)


# ─────────────────────────────────────────────
# RISK MODEL
# ─────────────────────────────────────────────
SKIN_TYPE_POINTS = {
    "Very Fair (Type I)": 15, "Fair (Type II)": 10, "Medium (Type III)": 5,
    "Olive (Type IV)": 3, "Brown (Type V)": 2, "Dark (Type VI)": 1,
}


def compute_risk(age, lifestyle, exposure, skin_type, family_history, sunburns, moles):
    breakdown = {
        "Age": round(age * 0.3, 1),
        "Lifestyle": {"Healthy": 0, "Moderate": 6, "Unhealthy": 14}[lifestyle],
        "Sun exposure": {"Low": 0, "Medium": 10, "High": 18}[exposure],
        "Skin type": SKIN_TYPE_POINTS[skin_type],
        "Family history": {"None": 0, "Mild": 6, "Significant": 14}[family_history],
        "Past sunburns": {"Never": 0, "1-2": 4, "3-5": 8, "More than 5": 12}[sunburns],
        "Number of moles": {"Fewer than 20": 0, "20-50": 5, "More than 50": 10}[moles],
    }
    score = min(int(sum(breakdown.values())), 100)
    return score, breakdown


def risk_tips(exposure, skin_type, lifestyle, sunburns, moles):
    tips = []
    if exposure != "Low" or SKIN_TYPE_POINTS[skin_type] >= 10:
        tips.append("Use broad-spectrum SPF 30+ sunscreen daily and reapply every 2 hours outdoors.")
    if exposure == "High":
        tips.append("Avoid direct sun between 10 AM and 4 PM; wear a hat and UV-protective clothing.")
    if lifestyle != "Healthy":
        tips.append("Improve sleep, hydration and diet; avoid smoking - all affect skin healing.")
    if sunburns in ("3-5", "More than 5"):
        tips.append("Frequent past sunburns raise skin-cancer risk - do a monthly self skin exam.")
    if moles != "Fewer than 20":
        tips.append("Track your moles with photos and use the ABCDE checker for any changes.")
    if not tips:
        tips.append("Keep up your healthy habits and do a yearly self skin check.")
    return tips


# ─────────────────────────────────────────────
# CHART HELPER (dark-theme friendly altair)
# ─────────────────────────────────────────────
def style_chart(chart):
    return chart.configure(background="transparent").configure_axis(
        labelColor="#aabbcc", titleColor="#aabbcc", gridColor="#2a2f45", domainColor="#3a3f55"
    ).configure_view(strokeWidth=0).configure_legend(labelColor="#aabbcc", titleColor="#aabbcc")


URGENCY_COLOR = {"low": "#00ff8c", "medium": "#ffc800", "high": "#ff4444"}
URGENCY_LABEL = {
    "low": "LOW URGENCY",
    "medium": "MODERATE - SEE A DOCTOR SOON",
    "high": "HIGH - CONSULT A DOCTOR URGENTLY",
}


# ─────────────────────────────────────────────
# LIVE UV INDEX (Open-Meteo - free, no API key)
# ─────────────────────────────────────────────
UV_LEVELS = [
    (3, "Low", "#00ff8c", "Minimal risk. Sunglasses on bright days; SPF if you burn easily or are near snow/water."),
    (6, "Moderate", "#ffc800", "Use SPF 30+, wear a hat and sunglasses, and seek shade around midday."),
    (8, "High", "#ff8c00", "Use SPF 30-50+, reapply every 2 hours, cover up, and reduce sun time between 10 AM and 4 PM."),
    (11, "Very High", "#ff4444", "Extra protection needed - SPF 50+, protective clothing, avoid midday sun. Unprotected skin can burn in under 15 minutes."),
    (99, "Extreme", "#c04cff", "Avoid sun exposure between 10 AM and 4 PM. Skin can burn in minutes - full cover, SPF 50+, shade."),
]


def uv_level(uv):
    for limit, name, color, advice in UV_LEVELS:
        if uv < limit:
            return name, color, advice
    return UV_LEVELS[-1][1:]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_uv(city):
    """Returns (data, error). Uses Open-Meteo geocoding + forecast APIs."""
    try:
        g = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": city, "count": 1}, timeout=10).json()
        if not g.get("results"):
            return None, f"Couldn't find a place called '{city}'."
        place = g["results"][0]
        f = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "uv_index", "hourly": "uv_index", "daily": "uv_index_max",
            "timezone": "auto", "forecast_days": 1,
        }, timeout=10).json()
        hourly = f.get("hourly", {})
        return {
            "name": ", ".join(x for x in [place.get("name"), place.get("admin1"), place.get("country")] if x),
            "current": float((f.get("current") or {}).get("uv_index") or 0),
            "max": float(((f.get("daily") or {}).get("uv_index_max") or [0])[0] or 0),
            "hours": [t[-5:] for t in hourly.get("time", [])],
            "values": [float(v or 0) for v in hourly.get("uv_index", [])],
        }, None
    except Exception as e:
        return None, f"Couldn't fetch UV data right now ({e.__class__.__name__}). Check your internet connection."



DISEASES = {
    "Acne":                  {"desc": "Clogged hair follicles causing pimples and inflammation.", "tx": "Benzoyl peroxide, retinoids, antibiotics", "urgency": "Low", "type": "Inflammatory"},
    "Eczema":                {"desc": "Chronic itchy dry skin inflammation.", "tx": "Moisturizers, corticosteroid creams", "urgency": "Low", "type": "Inflammatory"},
    "Psoriasis":             {"desc": "Autoimmune condition causing scaly red patches.", "tx": "Topical steroids, phototherapy, biologics", "urgency": "Medium", "type": "Autoimmune"},
    "Vitiligo":              {"desc": "White patches due to loss of skin pigment cells.", "tx": "Phototherapy, topical steroids", "urgency": "Low", "type": "Autoimmune"},
    "Rosacea":               {"desc": "Chronic facial redness and visible blood vessels.", "tx": "Metronidazole, azelaic acid, laser therapy", "urgency": "Low", "type": "Inflammatory"},
    "Fungal Infection":      {"desc": "Ring-shaped itchy scaly lesions from fungus.", "tx": "Antifungal creams, oral antifungals", "urgency": "Low", "type": "Infection"},
    "Melanoma":              {"desc": "Serious skin cancer from irregular changing moles.", "tx": "Surgical excision, immunotherapy", "urgency": "Critical", "type": "Cancer"},
    "Basal Cell Carcinoma":  {"desc": "Most common skin cancer; pearly bump or non-healing sore, usually on sun-exposed skin.", "tx": "Surgical excision, Mohs surgery, topical therapy", "urgency": "High", "type": "Cancer"},
    "Actinic Keratosis":     {"desc": "Rough, scaly sun-damage patch that can become cancerous.", "tx": "Cryotherapy, 5-FU cream, photodynamic therapy", "urgency": "Medium", "type": "Pre-cancer"},
    "Hives":                 {"desc": "Raised itchy welts from allergic reactions.", "tx": "Antihistamines, avoid triggers", "urgency": "Low", "type": "Allergic"},
    "Warts":                 {"desc": "Benign HPV growths on skin.", "tx": "Salicylic acid, cryotherapy", "urgency": "Low", "type": "Infection"},
    "Cellulitis":            {"desc": "Bacterial infection causing redness and swelling.", "tx": "Antibiotics (penicillin, cephalexin)", "urgency": "High", "type": "Infection"},
    "Chickenpox":            {"desc": "Viral rash with itchy blister-like spots.", "tx": "Antivirals, calamine lotion", "urgency": "Medium", "type": "Infection"},
    "Contact Dermatitis":    {"desc": "Inflammation from allergen or irritant contact.", "tx": "Identify trigger, corticosteroid cream", "urgency": "Low", "type": "Allergic"},
    "Impetigo":              {"desc": "Contagious bacterial infection with crusted sores.", "tx": "Topical or oral antibiotics", "urgency": "Medium", "type": "Infection"},
    "Ringworm":              {"desc": "Circular itchy fungal patches.", "tx": "Antifungal creams", "urgency": "Low", "type": "Infection"},
    "Sunburn":               {"desc": "UV radiation damage causing redness and pain.", "tx": "Aloe vera, cool compresses, NSAIDs", "urgency": "Low", "type": "Environmental"},
    "Lupus":                 {"desc": "Autoimmune butterfly rash on face.", "tx": "Hydroxychloroquine, sunscreen", "urgency": "High", "type": "Autoimmune"},
    "Scabies":               {"desc": "Mite infestation causing intense night itching.", "tx": "Permethrin cream, ivermectin", "urgency": "Medium", "type": "Infection"},
    "Boils":                 {"desc": "Pus-filled lumps from Staph infection.", "tx": "Warm compresses, drainage, antibiotics", "urgency": "Medium", "type": "Infection"},
    "Seborrheic Dermatitis": {"desc": "Scaly dandruff patches on oily skin areas.", "tx": "Antifungal shampoos, ketoconazole", "urgency": "Low", "type": "Inflammatory"},
    "Herpes Simplex":        {"desc": "Viral blisters on lips or skin.", "tx": "Acyclovir, valacyclovir", "urgency": "Medium", "type": "Infection"},
    "Squamous Cell Carcinoma": {"desc": "Skin cancer appearing as a firm red bump or scaly sore that may bleed, usually on sun-exposed skin.", "tx": "Surgical excision, Mohs surgery, radiation", "urgency": "High", "type": "Cancer"},
    "Seborrheic Keratosis":  {"desc": "Harmless waxy, 'stuck-on' brown growths common with age.", "tx": "No treatment needed; cryotherapy or curettage for cosmetic removal", "urgency": "Low", "type": "Benign growth"},
    "Dermatofibroma":        {"desc": "Small firm brownish bump, often on the legs, that dimples when pinched.", "tx": "Usually none; surgical removal if bothersome", "urgency": "Low", "type": "Benign growth"},
    "Keloid":                {"desc": "Raised, thick scar that grows beyond the original wound.", "tx": "Steroid injections, silicone sheets, cryotherapy", "urgency": "Low", "type": "Benign growth"},
    "Melasma":               {"desc": "Brown-grey patches on the face, often triggered by sun and hormones.", "tx": "Strict sunscreen, hydroquinone, azelaic acid, tranexamic acid", "urgency": "Low", "type": "Pigmentation"},
    "Post-inflammatory Hyperpigmentation": {"desc": "Dark spots left behind after acne, injury or inflammation.", "tx": "Sunscreen, niacinamide, vitamin C, retinoids", "urgency": "Low", "type": "Pigmentation"},
    "Tinea Versicolor":      {"desc": "Yeast overgrowth causing light or dark scaly patches on chest and back.", "tx": "Ketoconazole or selenium sulfide wash, oral antifungals", "urgency": "Low", "type": "Infection"},
    "Athlete's Foot":        {"desc": "Itchy, peeling fungal infection between the toes.", "tx": "Antifungal creams/powders, keep feet dry", "urgency": "Low", "type": "Infection"},
    "Shingles":              {"desc": "Painful blistering rash in a band on one side of the body, from reactivated chickenpox virus.", "tx": "Antivirals (valacyclovir) within 72 hours, pain relief", "urgency": "High", "type": "Infection"},
    "Molluscum Contagiosum": {"desc": "Small, pearly, dimpled bumps caused by a pox virus; common in children.", "tx": "Often resolves alone; cryotherapy, curettage", "urgency": "Low", "type": "Infection"},
    "Folliculitis":          {"desc": "Inflamed hair follicles forming small red or pus-filled bumps.", "tx": "Antibacterial wash, topical antibiotics, avoid shaving irritation", "urgency": "Low", "type": "Infection"},
    "Pityriasis Rosea":      {"desc": "Oval scaly patches starting with one larger 'herald patch', often in a Christmas-tree pattern on the back.", "tx": "Usually clears in 6-8 weeks; antihistamines for itch", "urgency": "Low", "type": "Inflammatory"},
    "Lichen Planus":         {"desc": "Itchy, flat, purple-ish bumps on wrists, ankles or inside the mouth.", "tx": "Topical steroids, antihistamines", "urgency": "Medium", "type": "Autoimmune"},
    "Alopecia Areata":       {"desc": "Autoimmune patchy hair loss, often in round coin-sized patches.", "tx": "Steroid injections, minoxidil, JAK inhibitors", "urgency": "Low", "type": "Autoimmune"},
    "Urticarial Angioedema": {"desc": "Deep swelling of lips, eyes or throat, sometimes with hives.", "tx": "Emergency care if breathing affected; antihistamines, epinephrine", "urgency": "Critical", "type": "Allergic"},
    "Diaper Rash":           {"desc": "Red, irritated skin in the nappy area from moisture and friction.", "tx": "Frequent changes, zinc oxide barrier cream", "urgency": "Low", "type": "Environmental"},
    "Miliaria (Heat Rash)":  {"desc": "Tiny itchy red bumps from blocked sweat ducts in hot, humid weather.", "tx": "Cool environment, loose cotton clothing, calamine lotion", "urgency": "Low", "type": "Environmental"},
    "Hyperhidrosis":         {"desc": "Excessive sweating of hands, feet or underarms beyond what's needed for cooling.", "tx": "Aluminium chloride antiperspirant, iontophoresis, botox", "urgency": "Low", "type": "Other"},
}
LIB_COLORS = {"Low": "#00ff8c", "Medium": "#ffc800", "High": "#ff4444", "Critical": "#ff0000"}


# ─────────────────────────────────────────────
# SKIN CARE GUIDE
# ─────────────────────────────────────────────
SKIN_TYPES = {
    "Normal": {
        "icon": "😊", "color": "#00ff8c",
        "desc": "Well balanced - not too oily or dry, few imperfections and barely visible pores.",
        "signs": ["Even texture", "Small pores", "Rare breakouts", "Comfortable after washing"],
        "am": ["Gentle cleanser", "Vitamin C serum", "Light moisturiser", "SPF 30+ sunscreen"],
        "pm": ["Gentle cleanser", "Moisturiser", "Retinol 2-3x a week (optional)"],
        "use": ["Vitamin C", "Hyaluronic acid", "Niacinamide", "Ceramides"],
        "avoid": ["Over-exfoliating", "Harsh scrubs", "Skipping sunscreen"],
        "tip": "Don't fix what isn't broken - a simple, consistent routine keeps normal skin healthy.",
    },
    "Dry": {
        "icon": "🏜️", "color": "#ffc800",
        "desc": "Produces less oil, so it can feel tight, rough or flaky and shows fine lines more easily.",
        "signs": ["Tightness after washing", "Flaky or rough patches", "Dull look", "Itchiness"],
        "am": ["Cream or milk cleanser", "Hyaluronic acid on damp skin", "Rich ceramide moisturiser", "SPF 30+ sunscreen"],
        "pm": ["Cream cleanser", "Hydrating serum", "Thick moisturiser or facial oil", "Lip balm"],
        "use": ["Ceramides", "Hyaluronic acid", "Glycerin", "Shea butter", "Squalane"],
        "avoid": ["Hot showers", "Foaming / alcohol-based products", "Strong fragrances"],
        "tip": "Apply moisturiser within 60 seconds of washing to lock in water.",
    },
    "Oily": {
        "icon": "✨", "color": "#00d4ff",
        "desc": "Overactive oil glands cause shine, enlarged pores and a tendency to breakouts.",
        "signs": ["Shiny all over by midday", "Large visible pores", "Frequent blackheads", "Makeup slides off"],
        "am": ["Gel or foaming cleanser", "Niacinamide serum", "Oil-free gel moisturiser", "Matte SPF 30+ sunscreen"],
        "pm": ["Double cleanse", "Salicylic acid (BHA) 2-3x a week", "Light gel moisturiser"],
        "use": ["Salicylic acid", "Niacinamide", "Clay masks", "Retinoids", "Oil-free gels"],
        "avoid": ["Skipping moisturiser", "Heavy oils & butters", "Over-washing (triggers more oil)"],
        "tip": "Oily skin still needs moisture - dehydrated skin produces even more oil.",
    },
    "Combination": {
        "icon": "🌗", "color": "#7b2ff7",
        "desc": "Oily T-zone (forehead, nose, chin) with normal or dry cheeks.",
        "signs": ["Shiny T-zone", "Dry or normal cheeks", "Larger pores on nose", "Occasional breakouts"],
        "am": ["Gentle gel cleanser", "Niacinamide serum", "Light lotion moisturiser", "SPF 30+ sunscreen"],
        "pm": ["Gentle cleanser", "BHA on T-zone only", "Richer cream on cheeks"],
        "use": ["Niacinamide", "Hyaluronic acid", "Gentle BHA", "Lightweight lotions"],
        "avoid": ["One harsh product for the whole face", "Heavy creams on T-zone"],
        "tip": "Try 'multi-masking' - clay mask on the T-zone, hydrating mask on the cheeks.",
    },
    "Sensitive": {
        "icon": "🌸", "color": "#ff6bcb",
        "desc": "Reacts easily with redness, stinging or itching to products, weather or stress.",
        "signs": ["Redness or flushing", "Stinging from products", "Itchy or burning", "Reacts to fragrance"],
        "am": ["Fragrance-free gentle cleanser", "Soothing serum (centella/panthenol)", "Barrier moisturiser", "Mineral SPF (zinc oxide)"],
        "pm": ["Micellar water or gentle cleanser", "Calming moisturiser", "Keep the routine minimal"],
        "use": ["Centella asiatica", "Panthenol", "Ceramides", "Oat extract", "Zinc oxide"],
        "avoid": ["Fragrance & essential oils", "Alcohol", "Physical scrubs", "Too many new products at once"],
        "tip": "Patch-test new products on your inner arm for 48 hours before using on the face.",
    },
    "Acne-prone": {
        "icon": "🎯", "color": "#ff6464",
        "desc": "Prone to clogged pores, pimples and post-acne marks; often oily or combination.",
        "signs": ["Recurring pimples", "Blackheads/whiteheads", "Dark marks after acne", "Tender bumps"],
        "am": ["Salicylic acid cleanser", "Niacinamide serum", "Non-comedogenic moisturiser", "Oil-free SPF 30+"],
        "pm": ["Gentle cleanser", "Benzoyl peroxide or adapalene (alternate nights)", "Light moisturiser"],
        "use": ["Salicylic acid", "Benzoyl peroxide", "Adapalene", "Niacinamide", "Azelaic acid"],
        "avoid": ["Picking or popping pimples", "Comedogenic oils (coconut oil)", "Dirty phone screens & pillowcases"],
        "tip": "Give any acne treatment 8-12 weeks before judging - and see a dermatologist for painful cystic acne.",
    },
    "Mature": {
        "icon": "🌿", "color": "#aabbcc",
        "desc": "With age the skin loses collagen and moisture, showing fine lines, laxity and uneven tone.",
        "signs": ["Fine lines & wrinkles", "Loss of firmness", "Age spots", "Dryness"],
        "am": ["Hydrating cleanser", "Vitamin C serum", "Peptide moisturiser", "SPF 50 sunscreen"],
        "pm": ["Cream cleanser", "Retinoid (start slowly)", "Rich moisturiser", "Eye cream"],
        "use": ["Retinoids", "Peptides", "Vitamin C", "Hyaluronic acid", "Ceramides"],
        "avoid": ["Sun exposure without SPF", "Harsh exfoliation", "Smoking"],
        "tip": "Daily sunscreen is the most proven anti-ageing product there is.",
    },
}

GENERAL_TIPS = [
    ("☀️", "Sun protection", "Use broad-spectrum SPF 30+ every day - even indoors near windows and on cloudy days. Reapply every 2 hours outdoors."),
    ("💧", "Hydration & diet", "Drink enough water and eat fruits, vegetables, nuts and omega-3 rich foods. Limit sugar and highly processed food."),
    ("😴", "Sleep & stress", "7-9 hours of sleep helps skin repair. Stress can trigger acne, eczema and psoriasis flares."),
    ("🧼", "Hygiene", "Wash pillowcases weekly, clean your phone screen, and never sleep with makeup on."),
    ("🧪", "Patch testing", "Introduce one new product at a time and patch-test for 48 hours to spot reactions."),
    ("🩺", "Know when to see a doctor", "Changing moles, wounds that don't heal in 3 weeks, spreading redness with fever, or severe acne need a dermatologist."),
]

INGREDIENTS = [
    ("Niacinamide", "Oil control, pores, redness, dark spots", "All skin types"),
    ("Hyaluronic acid", "Deep hydration, plumping", "All, especially dry"),
    ("Salicylic acid (BHA)", "Unclogs pores, blackheads, acne", "Oily, acne-prone"),
    ("Glycolic acid (AHA)", "Exfoliation, dullness, texture", "Normal, dry, mature"),
    ("Retinol / Retinoids", "Anti-ageing, acne, cell turnover", "Most types (start slow)"),
    ("Vitamin C", "Brightening, antioxidant, sun-damage repair", "All skin types"),
    ("Ceramides", "Repairs the skin barrier", "Dry, sensitive"),
    ("Benzoyl peroxide", "Kills acne bacteria", "Acne-prone"),
    ("Azelaic acid", "Acne, rosacea, pigmentation", "Sensitive, acne-prone"),
    ("Zinc oxide", "Gentle mineral sun protection", "Sensitive"),
]


def chips(items, color="#9fe9ff"):
    return "".join(f'<span class="chip" style="color:{color}">{html.escape(i)}</span>' for i in items)


def bullet_list(items):
    return "".join(f'<div class="desc-text" style="margin-top:0.2rem">• {html.escape(i)}</div>' for i in items)



def load_uploaded(f):
    img = Image.open(f).convert("RGB")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.LANCZOS)
    return img


