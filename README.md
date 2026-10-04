# Dermadisc - AI Skin Disease Detection

## Setup (one time)

```
cd skin_detection
python -m pip install -r requirements.txt
```

The new design needs **Streamlit 1.55 or newer**. Check with `python -m streamlit version`;
if it's older, run `python -m pip install -U streamlit`.

Keep your trained model files (`dermadisc_model.onnx` and `dermadisc_model_meta.json`) in this folder.

## Run

```
python -m streamlit run app.py
```

## First login

| | |
|---|---|
| Admin username | `admin` |
| Admin password | `admin123` |

**Change it straight away** in *My Account → Change password* (the app reminds you until you do).
Everyone else creates their own account with **Sign up** and adds their own OpenRouter API key
(free at https://openrouter.ai/keys) - either while signing up or later in *My Account*.

## What's in the folder

| File | What it does |
|---|---|
| `app.py` | The app's pages (run this) |
| `dd_ui.py` | The look: dark theme, colours, logo, icons, cards, navigation bar styling |
| `.streamlit/config.toml` | Dark theme colours for Streamlit's own widgets - keep this folder (it's hidden on Mac/Linux) |
| `dd_core.py` | Image analysis, AI calls, trained model, lesion measurement, risk model, reference content |
| `dd_db.py` | Database: user accounts, saved scans, translation cache |
| `dd_i18n.py` | Languages: English, हिन्दी, বাংলা, मराठी, தமிழ், తెలుగు |
| `dd_pdf.py` | PDF reports in all six languages |
| `fonts/` | Fonts for the PDF reports (Noto Sans + Indian scripts) - keep this folder |
| `train_dermadisc.ipynb` | Training notebook for the CNN (run in Google Colab) |
| `dermadisc.db` | Created automatically on first run - **all accounts and scans live here** |

## AI key (for the vision AI analysis)

The app looks for an OpenRouter key in this order:
1. The key the signed-in user saved in *My Account* (or pasted into the scan form).
2. A shared key in `.streamlit/secrets.toml`:
   ```
   OPENROUTER_API_KEY = "sk-or-v1-..."
   ```
3. The `OPENROUTER_API_KEY` environment variable.

With a shared key in `secrets.toml`, every account gets the AI analysis without entering anything.

## Accounts and data

* Every scan is saved to the signed-in user's account: photo, heatmap, AI results, trained-model results.
* Users see **only their own scans**, can download each as a **PDF**, export all as **CSV**, or download **all reports as a ZIP**.
* Admins get the **Admin Panel**: statistics, every user and every scan, filters, PDF/CSV/ZIP export,
  activate/deactivate accounts, reset passwords (temporary password), promote to admin.
* Passwords are stored as salted PBKDF2-SHA256 hashes - never in plain text.
* Users' OpenRouter keys and photos are stored in `dermadisc.db` on this computer.
  Back that file up, and don't share or upload it.

## Languages

* Menus, buttons, messages and PDF labels are built in for all six languages.
* AI results and DermaBot reply in the user's language.
* Longer content (disease library, skin care guide, tips) is translated by the AI the first time a page is
  opened in a language, saved in the database, and then instant for everyone.
  Admins can translate everything at once: *Admin Panel → Translations → Translate all content now*.

## Getting around

* **Top bar**: Home, New scan, My scans, plus **Learn** (disease library, skin care, about the model)
  and **Tools** (progress tracker, ABCDE check, DermaBot, risk and UV) menus.
* **Your name (top right)**: account, admin panel, language, AI model, sign out.
* **New scan** is three steps: photo → details → result. On the result, switch between the photo and the
  heatmap to see where the trained model looked.

## Troubleshooting

* **App looks plain / white, or the top bar is missing** → Streamlit is older than 1.55
  (`python -m pip install -U streamlit`), or the `.streamlit` folder wasn't copied next to `app.py`.

* **PDF button missing** → `python -m pip install fpdf2 uharfbuzz`
* **Indian-language letters look broken in PDFs** → `uharfbuzz` is not installed (it joins the letters correctly).
* **Trained model: not found** → put the two model files next to `app.py` and `python -m pip install onnxruntime`.
* **Forgot the admin password** → delete `dermadisc.db` *(this deletes all accounts and scans)*, or ask another admin to reset it.
