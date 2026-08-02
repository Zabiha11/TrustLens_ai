"""Shared helpers for route handlers."""

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

import config


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def parse_csv_upload(file: FileStorage) -> list[dict[str, Any]]:
    """Parse uploaded CSV and extract review_text (+ optional rating, user_id, product_id, timestamp)."""
    filename = secure_filename(file.filename or "upload.csv")
    if not allowed_file(filename):
        raise ValueError("Only CSV files are allowed")

    raw = file.read()
    if not raw:
        raise ValueError("Uploaded file is empty")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"Invalid CSV format: {exc}") from exc

    return parse_dataframe_reviews(df)


def parse_dataframe_reviews(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas DataFrame of reviews into normalized dict records."""
    text_col = _resolve_column(df, ["review_text", "text", "review", "content"])
    if text_col is None:
        raise ValueError("CSV must contain a 'review_text' column")

    rating_col = _resolve_column(df, ["rating", "stars", "score"])
    user_col = _resolve_column(df, ["user_id", "user", "reviewer_id", "author", "username"])
    prod_col = _resolve_column(df, ["product_id", "product", "item_id", "asin", "item"])
    time_col = _resolve_column(df, ["timestamp", "date", "created_at", "time", "datetime"])

    if len(df) > config.MAX_CSV_ROWS:
        raise ValueError(f"CSV exceeds maximum of {config.MAX_CSV_ROWS} rows")

    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text or text.lower() == "nan":
            continue

        entry: dict[str, Any] = {
            "review_text": text,
            "user_id": str(row[user_col]).strip() if user_col and pd.notna(row[user_col]) else f"USER_{idx+1:03d}",
            "product_id": str(row[prod_col]).strip() if prod_col and pd.notna(row[prod_col]) else "PROD_DEFAULT",
            "rating": float(row[rating_col]) if rating_col and pd.notna(row[rating_col]) else config.DEFAULT_RATING,
            "timestamp": _normalize_timestamp(row[time_col]) if time_col else _normalize_timestamp(None),
        }
        rows.append(entry)

    if not rows:
        raise ValueError("No valid reviews found in CSV")

    return rows


def _normalize_timestamp(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return datetime.now(timezone.utc).isoformat()

    try:
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            raise ValueError
        return timestamp.isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def save_dataset_rows(rows: list[dict[str, Any]]) -> None:
    """Persist normalized review rows to the active dataset for analytics."""
    if not rows:
        return

    records = []
    for row in rows:
        user_id = str(row.get("user_id") or "UNKNOWN_USER").strip() or "UNKNOWN_USER"
        product_id = str(row.get("product_id") or "UNKNOWN_PRODUCT").strip() or "UNKNOWN_PRODUCT"
        review_text = str(row.get("review_text") or "").strip()
        rating = float(row.get("rating", config.DEFAULT_RATING) or config.DEFAULT_RATING)
        timestamp = _normalize_timestamp(row.get("timestamp"))

        if not review_text:
            continue

        records.append(
            {
                "user_id": user_id,
                "product_id": product_id,
                "review_text": review_text,
                "rating": rating,
                "timestamp": timestamp,
            }
        )

    if not records:
        return

    df = pd.DataFrame(records)
    df = df[["user_id", "product_id", "review_text", "rating", "timestamp"]]

    if config.ACTIVE_DATA_FILE.exists():
        existing = pd.read_csv(config.ACTIVE_DATA_FILE)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(config.ACTIVE_DATA_FILE, index=False)
    else:
        df.to_csv(config.ACTIVE_DATA_FILE, index=False)


def load_active_dataset() -> pd.DataFrame:
    """Load reviews dataset from the persisted active dataset or the sample dataset."""
    target_file = config.ACTIVE_DATA_FILE if config.ACTIVE_DATA_FILE.exists() else config.SAMPLE_DATA_FILE
    if not target_file.exists():
        return pd.DataFrame(columns=["user_id", "product_id", "review_text", "rating", "timestamp"])

    try:
        df = pd.read_csv(target_file)
        rows = parse_dataframe_reviews(df)
        return pd.DataFrame(rows)
    except ValueError:
        return pd.DataFrame(columns=["user_id", "product_id", "review_text", "rating", "timestamp"])



def _resolve_column(df: pd.DataFrame, candidates: list[str]):
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    return None


def ensure_upload_dir() -> Path:
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return config.UPLOAD_DIR
