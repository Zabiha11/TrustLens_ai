"""Explainability API Blueprint."""

from flask import Blueprint, jsonify, request

import config
from utils.ai_detector import get_ai_detector
from utils.explainability import get_explainability_engine
from utils.trust_score import TrustScoreCalculator

explainability_bp = Blueprint("explainability", __name__, url_prefix="/api/explainability")


@explainability_bp.route("/summary", methods=["GET"])
def model_summary():
    engine = get_explainability_engine()
    return jsonify(engine.get_global_model_explanation())


@explainability_bp.route("/explain-review", methods=["POST"])
def explain_review():
    data = request.get_json(silent=True) or {}
    review_text = (data.get("review_text") or "").strip()
    if not review_text:
        return jsonify({"error": "review_text is required"}), 400

    rating = data.get("rating", config.DEFAULT_RATING)

    try:
        engine = get_explainability_engine()
        shap_res = engine.explain_shap_single(review_text, rating)
        lime_res = engine.explain_lime_single(review_text, rating)

        # AI Detection
        ai_detector = get_ai_detector()
        ai_res = ai_detector.detect(review_text)

        # Trust Score calculation
        fake_prob = float(data.get("fake_probability", 0.5))
        trust_payload = TrustScoreCalculator.calculate(
            fake_probability=fake_prob,
            reviewer_risk_score=float(data.get("reviewer_risk_score", 0.0)),
            temporal_anomaly_score=float(data.get("temporal_anomaly_score", 0.0)),
            community_suspicion_score=float(data.get("community_suspicion_score", 0.0)),
            ai_generated_probability=ai_res.ai_probability,
        )

        return jsonify({
            "shap": shap_res,
            "lime": lime_res,
            "ai_detection": ai_res.to_dict(),
            "trust_score": trust_payload,
        })
    except Exception as exc:
        return jsonify({"error": f"Explanation failed: {exc}"}), 500
