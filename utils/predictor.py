"""Prediction engine and extensible fraud-detector interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import logging
from time import perf_counter

import joblib
import numpy as np

import config
from utils.preprocessing import (
    build_feature_matrix,
    clean_review_text,
    normalize_rating,
    preprocess_review_for_inference,
)

logger = logging.getLogger(__name__)


class ModelIntegrityError(ValueError):
    """Raised when a serialized artifact is not safe to use for inference."""


@dataclass
class PredictionResult:
    """Standard prediction output consumed by API and UI layers."""

    review_text: str
    rating: float
    label: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    explanation: str = ""
    processed_review: str = ""
    vector_shape: tuple[int, int] = (0, 0)
    model_name: str = ""
    inference_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_text": self.review_text,
            "rating": self.rating,
            "prediction": self.label_name,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "explanation": self.explanation,
            "debug": {
                "processed_review": self.processed_review,
                "prediction_probability": round(self.confidence, 6),
                "model_name": self.model_name,
                "vector_shape": list(self.vector_shape),
                "inference_time_ms": round(self.inference_time_ms, 3),
            },
        }


class BaseFraudDetector(ABC):
    """Abstract base for pluggable fraud detection modules (future phases)."""

    @abstractmethod
    def predict(self, review_text: str, rating: Optional[float] = None) -> PredictionResult:
        pass

    @abstractmethod
    def predict_batch(
        self, reviews: list[str], ratings: Optional[list[float]] = None
    ) -> list[PredictionResult]:
        pass


class ReviewFraudDetector(BaseFraudDetector):
    """Phase 1 detector: TF-IDF + XGBoost on review text and rating."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
    ):
        self.model_path = model_path or config.MODEL_PATH
        self.vectorizer_path = vectorizer_path or config.VECTORIZER_PATH
        self._model = None
        self._vectorizer = None

    def _ensure_loaded(self) -> None:
        if self._model is None or self._vectorizer is None:
            if not self.model_path.exists() or not self.vectorizer_path.exists():
                raise FileNotFoundError(
                    "Model artifacts not found. Run: python models/train_model.py"
                )
            logger.info("Loading model artifact from %s", self.model_path)
            logger.info("Loading vectorizer artifact from %s", self.vectorizer_path)
            self._model = joblib.load(self.model_path)
            self._vectorizer = joblib.load(self.vectorizer_path)
            self._validate_artifacts()

    def _validate_artifacts(self) -> None:
        """Reject artifacts trained from YelpChi metadata placeholders, not review text."""
        vocabulary = set(getattr(self._vectorizer, "vocabulary_", {}))
        placeholder_terms = {"review", "sample", "generic", "restaurant", "feedback", "text"}
        if placeholder_terms.issubset(vocabulary):
            raise ModelIntegrityError(
                "The loaded model was trained on generated placeholder review text and cannot "
                "produce valid text predictions. Supply a labeled dataset with real review_text "
                "values and retrain with models/train_model.py."
            )

    def _format_result(
        self, review_text: str, rating: float, proba: np.ndarray, *, processed_review: str,
        vector_shape: tuple[int, int], inference_time_ms: float, explanation: str = ""
    ) -> PredictionResult:
        label = int(np.argmax(proba))
        confidence = float(proba[label])
        return PredictionResult(
            review_text=review_text,
            rating=rating,
            label=label,
            label_name=config.LABEL_NAMES[label],
            confidence=confidence,
            probabilities={
                config.LABEL_NAMES[i]: float(proba[i]) for i in range(len(proba))
            },
            explanation=explanation,
            processed_review=processed_review,
            vector_shape=vector_shape,
            model_name=type(self._model).__name__,
            inference_time_ms=inference_time_ms,
        )

    def predict(self, review_text: str, rating: Optional[float] = None) -> PredictionResult:
        self._ensure_loaded()
        rating = normalize_rating(rating, config.DEFAULT_RATING)
        cleaned_texts, normalized_ratings, processed_reviews = preprocess_review_for_inference(review_text, rating)
        logger.info("Input review: %s", review_text)
        logger.info("Processed review: %s", processed_reviews[0])

        started_at = perf_counter()
        features = build_feature_matrix(self._vectorizer, cleaned_texts, normalized_ratings)
        logger.info("Vectorizer transform shape: %s", features.shape)
        proba = self._model.predict_proba(features)[0]
        inference_time_ms = (perf_counter() - started_at) * 1000
        result = self._format_result(
            review_text, rating, proba, processed_review=processed_reviews[0],
            vector_shape=features.shape, inference_time_ms=inference_time_ms,
            explanation="TF-IDF + rating classification using the loaded model's raw probabilities.",
        )
        logger.info("Prediction: %s", result.label_name)
        logger.info("Confidence score: %.6f", result.confidence)
        return result

    def predict_batch(
        self, reviews: list[str], ratings: Optional[list[float]] = None
    ) -> list[PredictionResult]:
        self._ensure_loaded()
        if not reviews:
            return []

        if ratings is None:
            ratings = [config.DEFAULT_RATING] * len(reviews)
        elif len(ratings) != len(reviews):
            raise ValueError("reviews and ratings must have the same length")

        ratings = [normalize_rating(r, config.DEFAULT_RATING) for r in ratings]
        cleaned_texts = [clean_review_text(text) for text in reviews]
        logger.info("Batch input count: %d", len(reviews))
        started_at = perf_counter()
        features = build_feature_matrix(self._vectorizer, cleaned_texts, ratings)
        logger.info("Batch vectorizer transform shape: %s", features.shape)
        probas = self._model.predict_proba(features)
        inference_time_ms = ((perf_counter() - started_at) * 1000) / len(reviews)

        return [
            self._format_result(
                text,
                rating,
                proba,
                processed_review=cleaned_text,
                vector_shape=(1, features.shape[1]),
                inference_time_ms=inference_time_ms,
                explanation="TF-IDF + rating classification using the loaded model's raw probabilities.",
            )
            for text, cleaned_text, rating, proba in zip(reviews, cleaned_texts, ratings, probas)
        ]


# Singleton used by routes — swap implementation for future modules
_detector: Optional[ReviewFraudDetector] = None


def get_detector() -> ReviewFraudDetector:
    global _detector
    if _detector is None:
        _detector = ReviewFraudDetector()
    return _detector
