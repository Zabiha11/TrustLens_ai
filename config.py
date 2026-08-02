"""Application configuration — values loaded from environment with sensible defaults."""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

# Flask
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-trustlens-change-in-production")
DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")

# Groq API Configuration for AI Review Detection
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()


# Paths
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
STATS_FILE = BASE_DIR / "data" / "prediction_stats.json"
SAMPLE_DATA_FILE = BASE_DIR / "data" / "sample_fraud_reviews.csv"
ACTIVE_DATA_FILE = BASE_DIR / "data" / "active_reviews.csv"

MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

# Upload limits
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16 MB
ALLOWED_EXTENSIONS = {"csv"}
MAX_CSV_ROWS = int(os.environ.get("MAX_CSV_ROWS", 5000))

# ML defaults (used when rating is omitted at inference)
DEFAULT_RATING = float(os.environ.get("DEFAULT_RATING", 3.0))

# Training defaults (overridable via CLI in train_model.py)
TFIDF_MAX_FEATURES = int(os.environ.get("TFIDF_MAX_FEATURES", 10000))
TFIDF_NGRAM_RANGE = (1, 2)
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

# Label mapping
LABEL_GENUINE = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_GENUINE: "Genuine", LABEL_FAKE: "Fake"}
