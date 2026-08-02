"""Explainable AI (XAI) Module — SHAP & LIME Explanations.

Provides model-level SHAP feature attributions and review-level LIME word explanations
for transparent fraud detection predictions.
"""

from typing import Any, Optional
import numpy as np
import scipy.sparse as sp
from lime.lime_text import LimeTextExplainer
import shap

import config
from utils.predictor import get_detector
from utils.preprocessing import build_feature_matrix, clean_review_text, normalize_rating


class ExplainabilityEngine:
    """Computes SHAP and LIME explanations for TF-IDF + XGBoost fraud detection model."""

    def __init__(self):
        self.detector = get_detector()
        self.detector._ensure_loaded()
        self.model = self.detector._model
        self.vectorizer = self.detector._vectorizer
        self._feature_names: Optional[list[str]] = None
        self._tree_explainer: Optional[shap.TreeExplainer] = None

    def _get_feature_names(self) -> list[str]:
        """Extract vocabulary feature names plus rating feature name."""
        if self._feature_names is None:
            try:
                vocab_names = list(self.vectorizer.get_feature_names_out())
            except AttributeError:
                vocab_names = [f"word_{i}" for i in range(self.vectorizer.vocabulary_)]
            vocab_names.append("__rating_feature__")
            self._feature_names = vocab_names
        return self._feature_names

    def _get_tree_explainer(self) -> shap.TreeExplainer:
        """Lazy initializer for SHAP TreeExplainer."""
        if self._tree_explainer is None:
            try:
                self._tree_explainer = shap.TreeExplainer(self.model)
            except Exception:
                self._tree_explainer = shap.Explainer(self.model)
        return self._tree_explainer

    def explain_shap_single(self, review_text: str, rating: float = 3.0, top_k: int = 8) -> dict[str, Any]:
        """Compute SHAP feature attributions for a single review."""
        feature_names = self._get_feature_names()
        rating = normalize_rating(rating, config.DEFAULT_RATING)
        features = build_feature_matrix(self.vectorizer, [review_text], [rating])

        try:
            explainer = self._get_tree_explainer()
            shap_values = explainer.shap_values(features)

            # Handle multi-class / 2D / 3D shap value shapes across XGBoost versions
            if isinstance(shap_values, list):
                vals = shap_values[1][0]  # Fake class SHAP values
            elif len(shap_values.shape) == 3:
                vals = shap_values[0, :, 1]
            elif len(shap_values.shape) == 2:
                vals = shap_values[0]
            else:
                vals = np.zeros(features.shape[1])

            # Convert sparse array to dense array for indexing
            dense_features = features.toarray()[0]
            non_zero_indices = np.where(dense_features > 0)[0]

            contributions = []
            for idx in non_zero_indices:
                if idx < len(feature_names):
                    fname = feature_names[idx]
                    weight = float(vals[idx])
                    val = float(dense_features[idx])
                    contributions.append({
                        "feature": "Rating Score" if fname == "__rating_feature__" else fname,
                        "weight": round(weight, 4),
                        "value": round(val, 4),
                        "impact": "Fake" if weight > 0 else "Genuine"
                    })

            # Sort by absolute weight magnitude
            contributions.sort(key=lambda c: abs(c["weight"]), reverse=True)

            positive_contributions = [c for c in contributions if c["weight"] > 0][:top_k]
            negative_contributions = [c for c in contributions if c["weight"] < 0][:top_k]

            return {
                "top_features": contributions[:top_k],
                "positive_contributions": positive_contributions,  # Increases Fake probability
                "negative_contributions": negative_contributions,  # Increases Genuine probability
                "total_features_evaluated": len(contributions),
            }
        except Exception as exc:
            return {
                "error": f"SHAP calculation failed: {exc}",
                "top_features": [],
                "positive_contributions": [],
                "negative_contributions": [],
            }

    def explain_lime_single(self, review_text: str, rating: float = 3.0, num_features: int = 10) -> dict[str, Any]:
        """Compute LIME word importance explanation for an individual review text."""
        rating = normalize_rating(rating, config.DEFAULT_RATING)

        def lime_predict_proba(texts: list[str]) -> np.ndarray:
            """Prediction callback function required by LIME explainer."""
            ratings = [rating] * len(texts)
            matrix = build_feature_matrix(self.vectorizer, texts, ratings)
            return self.model.predict_proba(matrix)

        cleaned_text = clean_review_text(review_text)
        if not cleaned_text:
            cleaned_text = review_text

        try:
            explainer = LimeTextExplainer(class_names=["Genuine", "Fake"], random_state=42)
            exp = explainer.explain_instance(
                cleaned_text,
                lime_predict_proba,
                num_features=num_features,
                labels=(1,)  # Explain class 1 (Fake)
            )

            list_exp = exp.as_list(label=1)
            words_increasing_fake = []
            words_supporting_genuine = []

            for word, weight in list_exp:
                rounded_weight = round(float(weight), 4)
                item = {"word": word, "weight": rounded_weight}
                if rounded_weight > 0:
                    words_increasing_fake.append(item)
                else:
                    words_supporting_genuine.append(item)

            return {
                "lime_features": list_exp,
                "words_increasing_fake": words_increasing_fake,
                "words_supporting_genuine": words_supporting_genuine,
            }
        except Exception as exc:
            return {
                "error": f"LIME explanation failed: {exc}",
                "lime_features": [],
                "words_increasing_fake": [],
                "words_supporting_genuine": [],
            }

    def get_global_model_explanation(self, top_n: int = 15) -> dict[str, Any]:
        """Extract top global feature importance from trained XGBoost model."""
        feature_names = self._get_feature_names()
        try:
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:top_n]

            top_global = []
            for idx in top_indices:
                if idx < len(feature_names):
                    fname = feature_names[idx]
                    score = float(importances[idx])
                    top_global.append({
                        "feature": "Rating Score" if fname == "__rating_feature__" else fname,
                        "importance": round(score, 4),
                    })

            return {"global_feature_importances": top_global}
        except Exception as exc:
            return {"error": f"Global feature extraction failed: {exc}", "global_feature_importances": []}


# Global singleton instance
_explainer_engine: Optional[ExplainabilityEngine] = None


def get_explainability_engine() -> ExplainabilityEngine:
    global _explainer_engine
    if _explainer_engine is None:
        _explainer_engine = ExplainabilityEngine()
    return _explainer_engine
