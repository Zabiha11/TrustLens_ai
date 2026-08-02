# TrustLens AI — Codebase and Execution Flow

This document explains the purpose of every application source file, how files depend on one another, and the order in which code runs.

## 1. Architecture overview

```text
Browser
  │
  ├─ HTML templates (templates/) + browser JavaScript (static/js/)
  │     │ fetch JSON or submit CSV
  ▼
Flask application (app.py)
  │ registers blueprints from routes/
  ▼
Route handlers (routes/)
  │ validate input and call services
  ▼
Analysis services (utils/)
  │ preprocessing → vectorizer → XGBoost model → scores/explanations
  ▼
JSON response → JavaScript renders current result
```

The primary single-review path is:

```text
templates/analyze.html
  → static/js/analyze.js
  → POST /api/predict
  → routes/api.py: predict_review()
  → utils/predictor.py: ReviewFraudDetector.predict()
  → utils/preprocessing.py
  → models/vectorizer.pkl + models/model.pkl
  → routes/api.py builds trust/XAI/debug response
  → static/js/analyze.js renders the response
```

## 2. Application startup

### `app.py`

This is the executable Flask entry point.

1. Imports configuration and all Flask blueprints.
2. `create_app()` creates the `Flask` object, configures its secret key and upload limit, and enables application logging.
3. Creates required upload/data directories.
4. Registers every route blueprint.
5. When run directly, starts Flask on `0.0.0.0:5000`.

Run it with:

```powershell
.\.venv\Scripts\python.exe app.py
```

Only one running `app.py` server should own port 5000. A server holds model artifacts in memory after their first request, so restart it after retraining.

### `config.py`

Central configuration module. It loads `.env`, then defines:

- Flask settings and debug switch
- optional Groq API configuration
- paths for data, uploads, model/vectorizer artifacts, and prediction statistics
- CSV limits and default rating
- label meanings: `0 = Genuine`, `1 = Fake`

Every module imports this file rather than hard-coding project paths.

### `routes/__init__.py`

Imports blueprints from each route module and exposes them for `app.py` to register.

## 3. Pages, templates, and browser scripts

### Templates (`templates/`)

| File | Purpose | Triggered by |
|---|---|---|
| `base.html` | Shared navigation, page shell, global CSS/JS blocks. | Extended by every page template. |
| `index.html` | Landing page. | `GET /` in `routes/main.py`. |
| `analyze.html` | Single-review form, product URL form, CSV upload, result area, batch table. | `GET /analyze`; loads `static/js/analyze.js`. |
| `dashboard.html` | Dashboard cards/charts. | `GET /dashboard`; loads `dashboard.js`. |
| `network.html` | Reviewer/product network view. | `GET /network`; loads `network.js`. |
| `risk_analysis.html` | Reviewer risk table/view. | `GET /risk-analysis`; loads `risk.js`. |
| `temporal_analysis.html` | Time-series/burst view. | `GET /temporal-analysis`; loads `temporal.js`. |
| `investigation.html` | Investigation workspace UI. | `GET /investigation`; loads `investigation.js`. |
| `about.html` | Static project/about page. | `GET /about`. |

### Browser scripts (`static/js/`)

| File | What it does |
|---|---|
| `main.js` | Shared browser behaviour used by the base layout. |
| `analyze.js` | Reads the current textarea/rating values, posts them to `/api/predict`, handles uploads/URL analysis, and renders model output plus the Developer Debug Panel. |
| `dashboard.js` | Requests dashboard summary data and renders dashboard components. |
| `network.js` | Requests `/api/network/data` and renders network information. |
| `risk.js` | Requests `/api/risk/data` and renders reviewer-risk information. |
| `temporal.js` | Requests `/api/temporal/data` and renders temporal/burst information. |
| `investigation.js` | Handles the investigation page interactions. |

### `static/css/style.css`

Contains the visual styling shared by the templates: layout, cards, result badges, input controls, charts, and responsive presentation.

## 4. Flask route files

### `routes/main.py`

Contains page-only routes. Each route calls `render_template(...)`; it does not perform model inference.

| URL | Template |
|---|---|
| `/` | `index.html` |
| `/analyze` | `analyze.html` |
| `/investigation` | `investigation.html` |
| `/about` | `about.html` |

### `routes/api.py`

This is the main prediction API.

| Endpoint | Flow |
|---|---|
| `GET /api/health` | Reports whether model and vectorizer files exist. |
| `POST /api/predict` | Validates JSON → logs review/request ID → invokes model → builds XAI/trust/debug payload → stores prediction history → returns JSON. |
| `POST /api/upload` | Validates uploaded CSV → parses rows → batch-predicts → enriches results → stores them → returns JSON. |
| `POST /api/product-url-analyze` | Fetches product reviews through `product_url_analyzer.py` → predicts each valid review. |
| `POST /api/batch-predict` | Accepts a JSON list of reviews and runs batch inference. |

`_build_phase3_payload()` is called after model inference. It adds SHAP, LIME, AI-detection, and trust-score information to the original model result.

`/api/predict` also adds the `debug` object. This object is always based on the current request and contains submitted review, processed review, raw probability, model name, vector shape, timings, timestamp, and request ID.

### `routes/dashboard.py`

Serves the dashboard page and `/api/dashboard/stats`. The stats endpoint combines stored prediction history with global model feature importance, reviewer risk, temporal detection, and network analysis summaries.

### `routes/explainability.py`

Exposes model explanation APIs:

- `GET /api/explainability/summary`: global feature importance
- `POST /api/explainability/explain-review`: SHAP, LIME, AI-detection, and trust-score details for a supplied review

### `routes/network.py`

Serves the network page and `/api/network/data`. It loads the active dataset, builds a `ReviewNetworkAnalyzer`, and returns network summary plus interactive PyVis HTML.

### `routes/risk.py`

Serves the reviewer-risk page and `/api/risk/data`. It loads the active dataset, creates `ReviewerRiskScorer`, and returns reviewer risk profiles.

### `routes/temporal.py`

Serves the temporal page and `/api/temporal/data`. It loads the active dataset, creates `TemporalFraudDetector`, and returns trend/burst information.

## 5. Prediction pipeline

### `utils/preprocessing.py`

Contains the reusable transformations used by training and inference.

- `clean_review_text()`: lowercases text, removes URLs/unsupported punctuation, and normalizes spaces.
- `normalize_rating()`: converts input to a valid rating from 1 to 5.
- `build_feature_matrix()`: calls `vectorizer.transform(cleaned_text)` and appends rating as one sparse numeric feature.
- `preprocess_review_for_inference()`: returns the cleaned text and normalized rating for one request.

### `utils/predictor.py`

Owns real model inference.

1. `get_detector()` returns one lazily created `ReviewFraudDetector` per Flask process.
2. `_ensure_loaded()` loads `models/model.pkl` and `models/vectorizer.pkl` with Joblib on first use.
3. `_validate_artifacts()` blocks models trained from known placeholder YelpChi text.
4. `predict()` preprocesses the submitted review, builds the vector, calls the model's real `predict_proba()`, and returns a `PredictionResult`.
5. `predict_batch()` performs the equivalent work for multiple reviews.

`PredictionResult.to_dict()` exposes the label, confidence, raw class probabilities, and model diagnostics. No frontend fake/demo predictions are generated here.

### `utils/trust_score.py`

Calculates the 0–100 Trust Score. It combines fake probability with optional reviewer, temporal, community, and AI-writing risk signals, chooses a risk level/badge, and writes a natural-language summary.

### `utils/explainability.py`

Uses the same loaded detector/model/vectorizer as prediction. It creates:

- SHAP feature contributions for one review
- LIME word-level contributions for one review
- global XGBoost feature importance for dashboard/API use

### `utils/ai_detector.py`

Detects likely AI-generated writing. If `GROQ_API_KEY` is configured, it calls Groq for structured JSON. If not, or if the call fails, it applies offline phrase/structure heuristics. This signal does not replace the XGBoost Fake/Genuine prediction.

### `utils/product_url_analyzer.py`

Fetches/retrieves review content from a submitted public product URL for the product URL analysis path. `routes/api.py` passes retrieved texts to the same predictor used for manually submitted reviews.

## 6. Dataset intelligence and persistence

### `utils/helpers.py`

Shared input/data functions.

- Validates uploaded CSV files.
- Accepts common column aliases such as `text`, `review`, or `stars`.
- Normalizes rows to `review_text`, `user_id`, `product_id`, `rating`, and `timestamp`.
- Loads `data/sample_fraud_reviews.csv` as the active data source for dashboard/risk/network/temporal pages.
- Ensures the upload directory exists.

### `utils/stats_store.py`

Maintains `data/prediction_stats.json`, protected by a thread lock. Each prediction route calls it after a successful response to update totals, recent items, AI-detection aggregate statistics, and trust-score history.

### `utils/risk_scoring.py`

Creates a per-reviewer risk profile from dataset records. Scores use review volume, rating extremity, product concentration, duplicate-text similarity, and review velocity.

### `utils/temporal_analysis.py`

Parses timestamps, builds daily/hourly/weekly counts, and detects bursts with Isolation Forest plus DBSCAN and count thresholds.

### `utils/network_analysis.py`

Builds a user-to-product bipartite NetworkX graph, projects it into a user graph, detects communities with Louvain (or connected-components fallback), produces cluster suspicion scores, and generates PyVis network HTML.

### `utils/__init__.py`

Marks `utils` as a Python package.

## 7. Training and dataset preparation

### `models/train_model.py`

Trains the production artifacts.

```text
CSV input
 → validate required columns and text diversity
 → clean text + normalize ratings
 → stratified train/test split
 → fit TF-IDF on training text only
 → append rating feature
 → train XGBClassifier
 → evaluate accuracy/precision/recall/F1/ROC-AUC
 → save model.pkl and vectorizer.pkl
```

The script rejects data that becomes too repetitive after replacing numeric IDs. This protects against a degenerate model that returns one probability for unrelated text.

### `models/prepare_deceptive_opinion_spam.py`

Converts the downloaded Deceptive Opinion Spam corpus to the required training schema. It maps source-provided gold labels and source polarity; it does not invent text or labels.

### `models/prepare_yelpchi.py`

Attempts to convert YelpChi data. The available project YelpChi release lacks raw review content, so this script now raises an error rather than producing placeholder text for a TF-IDF classifier.

### `models/model.pkl` and `models/vectorizer.pkl`

Generated artifacts, not source code. `model.pkl` is the trained XGBoost classifier. `vectorizer.pkl` is the fitted TF-IDF vocabulary and weighting model. They must be generated as a matched pair by `train_model.py`.

## 8. Tests and support files

### `tests/test_prediction_pipeline.py`

Tests that a small real fitted test model gives different raw probabilities for different text, and that an artifact using the known placeholder vocabulary is rejected.

### `requirements.txt`

Lists the Python dependencies: Flask, pandas, NumPy, scikit-learn, SciPy, XGBoost, SHAP, LIME, NetworkX, PyVis, Plotly, Groq, and dotenv support.

### `.env.example`

Template for environment variables. Copy it to `.env`; do not commit a real `.env` file containing credentials.

### `.gitignore` and `models/.gitignore`

Prevent committing secrets, environments, upload output, stats files, Python cache, and generated model artifacts.

## 9. Exact single-review sequence

```text
1. User types review text and rating in analyze.html.
2. analyze.js catches the form submit event.
3. analyze.js reads reviewText.value and reviewRating.value at that moment.
4. analyze.js POSTs JSON to /api/predict.
5. api.py validates review_text and makes a request ID.
6. predictor.py loads artifacts if this is the first request in this Flask process.
7. preprocessing.py cleans the submitted text and normalizes the rating.
8. predictor.py transforms this text with the saved TF-IDF vectorizer.
9. predictor.py appends the rating feature and calls XGBoost predict_proba().
10. api.py enriches the real prediction with XAI, AI-detection, trust score, and debug values.
11. stats_store.py records the completed prediction.
12. Flask returns JSON.
13. analyze.js receives that JSON and replaces the result HTML with current values.
14. The expandable Developer Debug Panel makes the request data visible for verification.
```

## 10. Important operational note

The current process keeps the detector singleton and its loaded model artifacts in memory. Training new `.pkl` files does not change predictions until the Flask server is restarted. If the UI appears to show an old response, verify that only one Flask process owns port 5000, restart it, and use `Ctrl+F5` in the browser.
