"""
DERMADISC database layer (SQLite - a single file, nothing to install).

Tables
  users         accounts (passwords stored as salted PBKDF2-SHA256 hashes, never in plain text)
  scans         every saved analysis, including the photo, heatmap and full results
  translations  cache of AI-translated text, shared by all users
"""
BUILD = "2026-10-02h"  # must match app.py; tells the app this file is up to date
import os
import re
import json
import hmac
import sqlite3
import hashlib
import secrets
from contextlib import contextmanager
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DERMADISC_DB", os.path.join(APP_DIR, "dermadisc.db"))
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASSWORD = os.environ.get("DERMADISC_ADMIN_PASSWORD", "admin123")
PBKDF2_ROUNDS = 200_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    full_name       TEXT DEFAULT '',
    email           TEXT DEFAULT '',
    pw_hash         TEXT NOT NULL,
    salt            TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'user',
    language        TEXT NOT NULL DEFAULT 'en',
    api_key         TEXT DEFAULT '',
    must_change_pw  INTEGER NOT NULL DEFAULT 0,
    active          INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    last_login      TEXT
);
CREATE TABLE IF NOT EXISTS scans (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at          TEXT NOT NULL,
    language            TEXT DEFAULT 'en',
    body_area           TEXT DEFAULT '',
    top_condition       TEXT DEFAULT '',
    top_condition_local TEXT DEFAULT '',
    confidence          INTEGER,
    urgency             TEXT DEFAULT '',
    red_flags           INTEGER DEFAULT 0,
    ai_model            TEXT DEFAULT '',
    cnn_prediction      TEXT DEFAULT '',
    cnn_confidence      REAL,
    result_json         TEXT,
    context_json        TEXT,
    quality_json        TEXT,
    colors_json         TEXT,
    local_json          TEXT,
    image               BLOB,
    heatmap             BLOB
);
CREATE INDEX IF NOT EXISTS idx_scans_user ON scans(user_id, created_at);
CREATE TABLE IF NOT EXISTS translations (
    lang      TEXT NOT NULL,
    src_hash  TEXT NOT NULL,
    src       TEXT NOT NULL,
    text      TEXT NOT NULL,
    PRIMARY KEY (lang, src_hash)
);
"""

USER_FIELDS = {"full_name", "email", "role", "language", "api_key", "must_change_pw", "active", "last_login"}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect():
    """Open a connection, commit on success, roll back on error, always close."""
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────
# PASSWORDS
# ─────────────────────────────────────────────
def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ROUNDS)
    return digest.hex(), salt


def verify_password(password, pw_hash, salt):
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, pw_hash)


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)
        has_admin = conn.execute("SELECT 1 FROM users WHERE role = 'admin' LIMIT 1").fetchone()
        if not has_admin:
            pw_hash, salt = hash_password(DEFAULT_ADMIN_PASSWORD)
            conn.execute(
                "INSERT INTO users (username, full_name, pw_hash, salt, role, must_change_pw, created_at) "
                "VALUES (?, ?, ?, ?, 'admin', 1, ?)",
                (DEFAULT_ADMIN_USER, "Administrator", pw_hash, salt, now()),
            )


# ─────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_new_user(username, password, email=""):
    """Returns an error code (for translation) or None."""
    if not USERNAME_RE.match(username or ""):
        return "err_username"
    if len(password or "") < 6:
        return "err_password_short"
    if email and not EMAIL_RE.match(email):
        return "err_email"
    return None


def create_user(username, password, full_name="", email="", language="en", api_key="", role="user"):
    """Returns (user_id, None) or (None, error_code)."""
    err = validate_new_user(username, password, email)
    if err:
        return None, err
    pw_hash, salt = hash_password(password)
    try:
        with connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, full_name, email, pw_hash, salt, role, language, api_key, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (username.strip(), full_name.strip(), email.strip(), pw_hash, salt, role, language,
                 (api_key or "").strip(), now()),
            )
            return cur.lastrowid, None
    except sqlite3.IntegrityError:
        return None, "err_username_taken"


def authenticate(username, password):
    """Returns (user_dict, None) or (None, error_code)."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", ((username or "").strip(),)).fetchone()
        if not row or not verify_password(password or "", row["pw_hash"], row["salt"]):
            return None, "err_login"
        if not row["active"]:
            return None, "err_inactive"
        conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now(), row["id"]))
    return get_user(row["id"]), None


def get_user(user_id):
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        return None
    user = dict(row)
    user.pop("pw_hash", None)
    user.pop("salt", None)
    return user


def update_user(user_id, **fields):
    fields = {k: v for k, v in fields.items() if k in USER_FIELDS}
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE users SET {sets} WHERE id = ?", (*fields.values(), user_id))


def set_password(user_id, new_password, must_change=False):
    if len(new_password or "") < 6:
        return "err_password_short"
    pw_hash, salt = hash_password(new_password)
    with connect() as conn:
        conn.execute("UPDATE users SET pw_hash = ?, salt = ?, must_change_pw = ? WHERE id = ?",
                     (pw_hash, salt, 1 if must_change else 0, user_id))
    return None


def change_password(user_id, old_password, new_password):
    with connect() as conn:
        row = conn.execute("SELECT pw_hash, salt FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row or not verify_password(old_password or "", row["pw_hash"], row["salt"]):
        return "err_old_password"
    return set_password(user_id, new_password)


def list_users():
    with connect() as conn:
        rows = conn.execute("""
            SELECT u.id, u.username, u.full_name, u.email, u.role, u.language, u.active,
                   u.created_at, u.last_login, COUNT(s.id) AS scans,
                   SUM(CASE WHEN s.urgency = 'high' THEN 1 ELSE 0 END) AS high_urgency
            FROM users u LEFT JOIN scans s ON s.user_id = u.id
            GROUP BY u.id ORDER BY u.created_at DESC""").fetchall()
    return [dict(r) for r in rows]


def count_admins():
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND active = 1").fetchone()[0]


def delete_user(user_id):
    with connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


# ─────────────────────────────────────────────
# SCANS
# ─────────────────────────────────────────────
def _dump(x):
    return json.dumps(x, ensure_ascii=False) if x is not None else None


def save_scan(user_id, language, ctx, result, local, quality, colors, image_jpeg, heatmap_jpeg=None):
    top = (result or {}).get("predictions", [{}])
    top = top[0] if top else {}
    cnn_top = (local or {}).get("preds", [{}])[0] if local else {}
    local_small = None
    if local:
        local_small = {"preds": local.get("preds", [])[:5], "agreement_text": local.get("agreement_text", ""),
                       "agreement": local.get("agreement", "")}
    with connect() as conn:
        cur = conn.execute("""
            INSERT INTO scans (user_id, created_at, language, body_area, top_condition, top_condition_local,
                confidence, urgency, red_flags, ai_model, cnn_prediction, cnn_confidence,
                result_json, context_json, quality_json, colors_json, local_json, image, heatmap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            user_id, now(), language, (ctx or {}).get("area", ""),
            top.get("disease", "") if result else "", top.get("disease_local", "") if result else "",
            top.get("confidence") if result else None,
            (result or {}).get("urgency", "") or (cnn_top.get("urgency", "") if cnn_top else ""),
            len((result or {}).get("red_flags", [])), (result or {}).get("_model", ""),
            cnn_top.get("disease", ""), cnn_top.get("confidence"),
            _dump(result), _dump(ctx), _dump(quality), _dump(colors), _dump(local_small),
            image_jpeg, heatmap_jpeg,
        ))
        return cur.lastrowid


def update_scan_result(scan_id, result, local=None):
    """Attach an AI result to a scan that was first saved with the trained model only."""
    top = (result or {}).get("predictions", [{}])
    top = top[0] if top else {}
    local_small = None
    if local:
        local_small = {"preds": local.get("preds", [])[:5], "agreement_text": local.get("agreement_text", ""),
                       "agreement": local.get("agreement", "")}
    with connect() as conn:
        conn.execute("""UPDATE scans SET result_json = ?, top_condition = ?, top_condition_local = ?, confidence = ?,
                        urgency = ?, red_flags = ?, ai_model = ?, local_json = COALESCE(?, local_json) WHERE id = ?""", (
            _dump(result), top.get("disease", ""), top.get("disease_local", ""), top.get("confidence"),
            (result or {}).get("urgency", "") or "medium", len((result or {}).get("red_flags", [])),
            (result or {}).get("_model", ""), _dump(local_small), scan_id))


def update_scan_heatmap(scan_id, heatmap_jpeg):
    with connect() as conn:
        conn.execute("UPDATE scans SET heatmap = ? WHERE id = ?", (heatmap_jpeg, scan_id))


def _scan_row(row, full=True):
    d = dict(row)
    for k in ("result_json", "context_json", "quality_json", "colors_json", "local_json"):
        if k in d:
            d[k[:-5]] = json.loads(d[k]) if d[k] else None
            del d[k]
    if not full:
        d.pop("image", None)
        d.pop("heatmap", None)
    return d


LIST_COLUMNS = ("s.id, s.user_id, u.username, s.created_at, s.language, s.body_area, s.top_condition, "
                "s.top_condition_local, s.confidence, s.urgency, s.red_flags, s.ai_model, s.cnn_prediction, "
                "s.cnn_confidence, (s.heatmap IS NOT NULL) AS has_heatmap")


def list_scans(user_id=None, urgency=None, search=None, date_from=None, date_to=None, limit=1000):
    q = f"SELECT {LIST_COLUMNS} FROM scans s JOIN users u ON u.id = s.user_id WHERE 1=1"
    args = []
    if user_id is not None:
        q += " AND s.user_id = ?"
        args.append(user_id)
    if urgency:
        q += f" AND s.urgency IN ({','.join('?' * len(urgency))})"
        args.extend(urgency)
    if search:
        q += " AND (s.top_condition LIKE ? OR s.cnn_prediction LIKE ? OR s.body_area LIKE ? OR u.username LIKE ?)"
        args.extend([f"%{search}%"] * 4)
    if date_from:
        q += " AND s.created_at >= ?"
        args.append(f"{date_from} 00:00:00")
    if date_to:
        q += " AND s.created_at <= ?"
        args.append(f"{date_to} 23:59:59")
    q += " ORDER BY s.created_at DESC, s.id DESC LIMIT ?"
    args.append(limit)
    with connect() as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


def get_scan(scan_id, user_id=None):
    """user_id given -> only returns the scan if it belongs to that user (privacy)."""
    q = "SELECT s.*, u.username, u.full_name FROM scans s JOIN users u ON u.id = s.user_id WHERE s.id = ?"
    args = [scan_id]
    if user_id is not None:
        q += " AND s.user_id = ?"
        args.append(user_id)
    with connect() as conn:
        row = conn.execute(q, args).fetchone()
    return _scan_row(row) if row else None


def scan_image(scan_id):
    """Just the stored photo (for thumbnails), without loading the rest of the scan."""
    with connect() as conn:
        row = conn.execute("SELECT image FROM scans WHERE id = ?", (scan_id,)).fetchone()
    return row["image"] if row else None


def delete_scan(scan_id, user_id=None):
    q, args = "DELETE FROM scans WHERE id = ?", [scan_id]
    if user_id is not None:
        q += " AND user_id = ?"
        args.append(user_id)
    with connect() as conn:
        return conn.execute(q, args).rowcount


def stats(user_id=None):
    where, args = ("WHERE user_id = ?", [user_id]) if user_id is not None else ("", [])
    with connect() as conn:
        r = conn.execute(f"""SELECT COUNT(*) AS scans,
                SUM(CASE WHEN urgency = 'high' THEN 1 ELSE 0 END) AS high,
                AVG(confidence) AS avg_conf, MAX(created_at) AS last_scan
                FROM scans {where}""", args).fetchone()
        out = dict(r)
        if user_id is None:
            out["users"] = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            out["active_7d"] = conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM scans WHERE created_at >= datetime('now', '-7 days', 'localtime')"
            ).fetchone()[0]
    out["high"] = out["high"] or 0
    return out


# ─────────────────────────────────────────────
# TRANSLATION CACHE
# ─────────────────────────────────────────────
def _h(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def get_translations(lang, texts):
    texts = list(dict.fromkeys(texts))
    if not texts:
        return {}
    out = {}
    with connect() as conn:
        for i in range(0, len(texts), 400):
            chunk = texts[i:i + 400]
            hashes = {_h(t): t for t in chunk}
            rows = conn.execute(
                f"SELECT src_hash, text FROM translations WHERE lang = ? AND src_hash IN ({','.join('?' * len(hashes))})",
                (lang, *hashes.keys())).fetchall()
            for r in rows:
                out[hashes[r["src_hash"]]] = r["text"]
    return out


def put_translations(lang, mapping):
    with connect() as conn:
        conn.executemany("INSERT OR REPLACE INTO translations (lang, src_hash, src, text) VALUES (?, ?, ?, ?)",
                         [(lang, _h(src), src, txt) for src, txt in mapping.items()])


def translation_counts():
    with connect() as conn:
        return {r["lang"]: r["n"] for r in conn.execute("SELECT lang, COUNT(*) AS n FROM translations GROUP BY lang")}


def clear_translations(lang=None):
    with connect() as conn:
        if lang:
            conn.execute("DELETE FROM translations WHERE lang = ?", (lang,))
        else:
            conn.execute("DELETE FROM translations")
