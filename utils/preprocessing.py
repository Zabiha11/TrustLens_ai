"""Text and feature preprocessing for review fraud detection."""

import re
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix, hstack


def clean_review_text(text: str) -> str:
    """Normalize review text for TF-IDF vectorization."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_rating(rating: Union[int, float, str, None], default: float = 3.0) -> float:
    """Clamp rating to Yelp-style 1–5 scale."""
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return default
    return float(np.clip(value, 1.0, 5.0))


def build_feature_matrix(vectorizer, texts: list[str], ratings: list[float]) -> csr_matrix:
    """Combine TF-IDF text features with normalized rating column."""
    cleaned = [clean_review_text(t) for t in texts]
    text_features = vectorizer.transform(cleaned)
    rating_features = csr_matrix(np.array(ratings, dtype=np.float32).reshape(-1, 1))
    return hstack([text_features, rating_features], format="csr")


def preprocess_review_for_inference(review_text: str, rating: float) -> tuple[list[str], list[float], list[str]]:
    """Return cleaned texts, normalized ratings, and processed review values for logging and inference."""
    cleaned_text = clean_review_text(review_text)
    normalized_rating = normalize_rating(rating, 3.0)
    return [cleaned_text], [normalized_rating], [cleaned_text]
