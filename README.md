# TrustLens AI

TrustLens AI is a Flask application for review-deception analysis. It runs a TF-IDF vectorizer and XGBoost classifier, with trust scoring and explainability results in the browser.

## Features

- Single-review analysis at `/analyze`
- CSV batch predictions
- Raw prediction probabilities, SHAP/LIME information, and AI-writing detection
- Developer Debug Panel showing the exact request and inference diagnostics
- Model training from real labeled review data

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set a secure `SECRET_KEY` in `.env`. `GROQ_API_KEY` is optional; without it, AI-writing detection uses its offline fallback.

## Run

```powershell
.\.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) and navigate to `/analyze`.

Use one Flask server process only. Restart it after changing model files or application code so it reloads those files.

## Train a model

Your dataset must contain these columns:

```text
review_text,rating,label
```

- `review_text`: real review content
- `rating`: value from 1 to 5
- `label`: `0` for Genuine; `1` for Fake/Deceptive

```powershell
.\.venv\Scripts\python.exe models\train_model.py --data-path data\your_labeled_reviews.csv
```

Training checks content diversity and refuses placeholder/template data. This prevents a model from returning the same probability for every review.

### Deceptive Opinion Spam helper

`models/prepare_deceptive_opinion_spam.py` converts a source CSV with `deceptive`, `polarity`, and `text` columns into the required schema.

```powershell
.\.venv\Scripts\python.exe models\prepare_deceptive_opinion_spam.py --source data\deceptive_opinion_spam_source.csv --output data\deceptive_opinion_spam.csv
.\.venv\Scripts\python.exe models\train_model.py --data-path data\deceptive_opinion_spam.csv
```

The helper maps the corpus's source-provided labels (`truthful` / `deceptive`) to `0` / `1`; it does not generate review text or labels.

## API

`POST /api/predict`

```json
{
  "review_text": "This product is amazing. Excellent quality and fast delivery.",
  "rating": 5
}
```

The response contains `prediction`, `confidence`, `probabilities`, `trust_score`, and `debug`. The Debug Panel is populated from the current request and includes submitted/processed text, model name, vector shape, probability, timing, timestamp, and request ID.

```powershell
$payload = @{ review_text = 'Terrible product. Completely useless. Waste of money.'; rating = 1 } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/predict' -Method Post -ContentType 'application/json' -Body $payload
```

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Troubleshooting

### The page shows an old result

Stop duplicate Flask processes, start one server, then hard-refresh the browser with `Ctrl+F5`. A process that already loaded an old model keeps it in memory until restarted.

### Training rejects the dataset

The dataset has insufficient real text variation. Do not remove the validation check; replace it with real labeled review content. IDs appended to a template sentence are not valid distinct reviews.

### The rating does not change

The single-review form submits the current Rating field. It defaults to `3`; change that field to submit a different rating.

## Project layout

- `app.py` — Flask entry point
- `routes/` — HTML routes and REST API
- `utils/predictor.py` — TF-IDF + XGBoost inference
- `models/train_model.py` — training and evaluation
- `templates/` and `static/` — web interface
- `tests/` — prediction-pipeline tests

`models/model.pkl` and `models/vectorizer.pkl` are generated locally and intentionally ignored by Git.
