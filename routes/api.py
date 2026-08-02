"""REST API endpoints for review fraud prediction."""

import logging
from datetime import datetime, timezone
from time import perf_counter
from urllib.parse import urlparse
from uuid import uuid4

from flask import Blueprint, jsonify, request

import config
from utils.ai_detector import get_ai_detector
from utils.explainability import get_explainability_engine
from utils.helpers import load_active_dataset, parse_csv_upload, save_dataset_rows
from utils.network_analysis import ReviewNetworkAnalyzer
from utils.predictor import ModelIntegrityError, get_detector
from utils.product_url_analyzer import fetch_reviews_from_url
from utils.risk_scoring import ReviewerRiskScorer
from utils.stats_store import get_stats_store
from utils.temporal_analysis import TemporalFraudDetector
from utils.trust_score import TrustScoreCalculator

api_bp = Blueprint("api", __name__, url_prefix="/api")
logger = logging.getLogger(__name__)


@api_bp.route("/health", methods=["GET"])
def health():
    model_ready = config.MODEL_PATH.exists() and config.VECTORIZER_PATH.exists()
    return jsonify({"status": "ok", "model_ready": model_ready})


def _build_phase3_payload(result, review_text: str, rating: float, metadata: dict | None = None) -> dict:
    """Compose full prediction payload with XAI, AI detection and Trust Score.

    Trust Score metadata (reviewer_risk_score, temporal_anomaly_score,
    community_suspicion_score) is resolved as follows:
    1. Use values explicitly present in `metadata` (e.g. from a row in a CSV upload).
    2. Otherwise, attempt to look up the actual calculated risk/community scores
       from the current active dataset analyzers by user_id.
    3. If user is unknown, fall back to GLOBAL AVERAGE scores computed from the
       active dataset rather than hardcoded 0.0, so the Trust Score budget
       (100 points) is exercised dynamically even for ad-hoc reviews.
    """
    payload = result.to_dict()
    metadata = metadata or {}
    logger.info("[PAYLOAD] Start composing for review len=%s rating=%s", len(review_text), rating)

    engine = get_explainability_engine()
    t0 = perf_counter()
    shap_res = engine.explain_shap_single(review_text, rating)
    logger.info("[PAYLOAD] SHAP computed in %.2f ms top=%s", (perf_counter()-t0)*1000, len(shap_res.get("top_features", [])))

    t1 = perf_counter()
    lime_res = engine.explain_lime_single(review_text, rating)
    logger.info("[PAYLOAD] LIME computed in %.2f ms fake_words=%s gen_words=%s", (perf_counter()-t1)*1000,
                len(lime_res.get("words_increasing_fake", [])), len(lime_res.get("words_supporting_genuine", [])))

    t2 = perf_counter()
    ai_detector = get_ai_detector()
    ai_res = ai_detector.detect(review_text)
    logger.info("[PAYLOAD] AI detection in %.2f ms prob=%.3f status=%s", (perf_counter()-t2)*1000,
                ai_res.ai_probability, ai_res.status)

    fake_prob = float(result.probabilities.get("Fake", 0.5))
    logger.info("[PAYLOAD] Model fake_prob=%.5f label=%s conf=%.5f", fake_prob, result.label_name, result.confidence)

    user_id = metadata.get("user_id") or metadata.get("reviewer_id")
    timestamp = metadata.get("timestamp")

    global_avg_risk = 0.0
    global_avg_temporal = 0.0
    global_avg_community = 0.0
    risk_by_user: dict[str, float] = {}
    community_by_user: dict[str, float] = {}
    temporal_score = 0.0
    df = None
    active_dataset_len = 0

    try:
        df = load_active_dataset()
        active_dataset_len = len(df)
        if len(df) > 0:
            scorer = ReviewerRiskScorer(df)
            risk_summary = scorer.get_summary()
            global_avg_risk = float(risk_summary.get("avg_risk_score", 0.0))
            for profile in risk_summary.get("reviewers", []):
                risk_by_user[str(profile["user_id"])] = float(profile["risk_score"])

            temporal_detector = TemporalFraudDetector(df)
            bursts = temporal_detector.get_summary().get("detected_bursts", [])
            burst_scores = [float(b.get("anomaly_score", 0.0)) for b in bursts]
            global_avg_temporal = (sum(burst_scores) / len(burst_scores)) if burst_scores else 0.0
            if timestamp:
                try:
                    import pandas as pd
                    ts_dt = pd.to_datetime(timestamp, errors="coerce")
                    if pd.notna(ts_dt):
                        ts_str = ts_dt.strftime("%Y-%m-%d %H:00")
                        for b in bursts:
                            if b.get("timestamp") == ts_str:
                                temporal_score = float(b.get("anomaly_score", 0.0))
                                break
                except Exception:
                    pass

            net_analyzer = ReviewNetworkAnalyzer(df)
            clusters = net_analyzer.get_summary().get("clusters", [])
            cluster_avg_suspicions = [float(c.get("suspicion_score", 0.0)) for c in clusters]
            global_avg_community = (sum(cluster_avg_suspicions) / len(cluster_avg_suspicions)) if cluster_avg_suspicions else 0.0
            for cluster in clusters:
                for uid in cluster.get("members", []):
                    community_by_user[str(uid)] = float(cluster.get("suspicion_score", 0.0))

            logger.info("[PAYLOAD] Dataset baselines: avg_risk=%.1f avg_temporal=%.1f avg_community=%.1f users=%d",
                        global_avg_risk, global_avg_temporal, global_avg_community, len(risk_by_user))
    except Exception as exc:
        logger.warning("[PAYLOAD] Could not load dataset baselines: %s", exc)

    reviewer_risk_score = float(metadata.get("reviewer_risk_score")) if metadata.get("reviewer_risk_score") is not None else None
    if reviewer_risk_score is None and user_id and str(user_id) in risk_by_user:
        reviewer_risk_score = risk_by_user[str(user_id)]
        logger.info("[PAYLOAD] Lookup user=%s => reviewer_risk=%.1f", user_id, reviewer_risk_score)
    if reviewer_risk_score is None:
        reviewer_risk_score = global_avg_risk

    temporal_anomaly_score = float(metadata.get("temporal_anomaly_score")) if metadata.get("temporal_anomaly_score") is not None else None
    if temporal_anomaly_score is None and temporal_score > 0:
        temporal_anomaly_score = temporal_score
    if temporal_anomaly_score is None:
        temporal_anomaly_score = global_avg_temporal

    community_suspicion_score = float(metadata.get("community_suspicion_score")) if metadata.get("community_suspicion_score") is not None else None
    if community_suspicion_score is None and user_id and str(user_id) in community_by_user:
        community_suspicion_score = community_by_user[str(user_id)]
        logger.info("[PAYLOAD] Lookup user=%s => community_suspicion=%.1f", user_id, community_suspicion_score)
    if community_suspicion_score is None:
        community_suspicion_score = global_avg_community

    trust_payload = TrustScoreCalculator.calculate(
        fake_probability=fake_prob,
        reviewer_risk_score=reviewer_risk_score,
        temporal_anomaly_score=temporal_anomaly_score,
        community_suspicion_score=community_suspicion_score,
        ai_generated_probability=ai_res.ai_probability,
    )
    logger.info("[PAYLOAD] Trust=%.1f level=%s | f_prob=%.1f r_risk=%.1f temp=%.1f comm=%.1f ai=%.1f",
                trust_payload["trust_score"], trust_payload["risk_level"],
                fake_prob*100, reviewer_risk_score, temporal_anomaly_score,
                community_suspicion_score, ai_res.ai_probability*100)

    payload["shap"] = shap_res
    payload["lime"] = lime_res
    payload["ai_detection"] = ai_res.to_dict()
    payload["trust_score"] = trust_payload
    payload["trust_score"]["metadata_sources"] = {
        "reviewer_risk_score": round(reviewer_risk_score, 1),
        "temporal_anomaly_score": round(temporal_anomaly_score, 1),
        "community_suspicion_score": round(community_suspicion_score, 1),
        "user_id": str(user_id) if user_id else None,
        "active_dataset_rows": active_dataset_len,
    }
    return payload


@api_bp.route("/predict", methods=["POST"])
def predict_review():
    request_id = str(uuid4())
    started_at = perf_counter()
    data = request.get_json(silent=True) or {}
    review_text = (data.get("review_text") or "").strip()

    if not review_text:
        return jsonify({"error": "review_text is required"}), 400

    rating = data.get("rating", config.DEFAULT_RATING)

    try:
        logger.info("[%s] Received review_text: %s", request_id, review_text)
        detector = get_detector()
        result = detector.predict(review_text, rating)
        logger.info("Returning prediction: %s | confidence: %.6f", result.label_name, result.confidence)
        payload = _build_phase3_payload(result, review_text, result.rating, data)
        payload["debug"].update({
            "submitted_review": review_text,
            "prediction": result.label_name,
            "trust_score": payload["trust_score"]["trust_score"],
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_time_ms": round((perf_counter() - started_at) * 1000, 3),
        })

        get_stats_store().record_prediction(
            review_text=review_text,
            label_name=result.label_name,
            confidence=result.confidence,
            rating=result.rating,
            ai_detection=payload.get("ai_detection"),
            trust_score=payload.get("trust_score"),
        )
        save_dataset_rows([
            {
                "user_id": f"MANUAL_{uuid4().hex[:8]}",
                "product_id": "MANUAL_REVIEW",
                "review_text": review_text,
                "rating": result.rating,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ])
        logger.info("[%s] Returning response (request time: %.3f ms)", request_id, payload["debug"]["request_time_ms"])
        return jsonify(payload)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ModelIntegrityError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500



@api_bp.route("/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        rows = parse_csv_upload(file)
        save_dataset_rows(rows)
        detector = get_detector()
        reviews = [r["review_text"] for r in rows]
        ratings = [r.get("rating") for r in rows]
        results = detector.predict_batch(reviews, ratings)
        payload = []
        for result, row in zip(results, rows):
            payload.append(_build_phase3_payload(result, row["review_text"], result.rating, row))
        get_stats_store().record_batch(payload)
        return jsonify({"count": len(payload), "predictions": payload})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Upload processing failed: {exc}"}), 500


@api_bp.route("/product-url-analyze", methods=["POST"])
def product_url_analyze():
    step_start = perf_counter()
    data = request.get_json(silent=True) or {}
    product_url = (data.get("product_url") or "").strip()
    logger.info("[URL_ANALYZE] Step 1 — Request received: product_url=%s remote_addr=%s", product_url, request.remote_addr)
    if not product_url:
        return jsonify({"error": "product_url is required"}), 400

    try:
        logger.info("[URL_ANALYZE] Step 2 — Calling fetch_reviews_from_url (max=20)")
        reviews = fetch_reviews_from_url(product_url, max_reviews=20)
        logger.info("[URL_ANALYZE] Step 3 — Extracted %s reviews", len(reviews))
        if not reviews:
            raise ValueError("No review content could be extracted from the URL.")

        detector = get_detector()
        result_items = []
        dataset_rows = []
        parsed_url = urlparse(product_url)
        product_id = f"URL_{parsed_url.netloc}"
        logger.info("[URL_ANALYZE] Step 4 — Running ML predictions for %s reviews", len(reviews))

        for idx, item in enumerate(reviews, start=1):
            review_text = str(item.get("review_text") or "").strip()
            if not review_text:
                continue

            rating = item.get("rating", config.DEFAULT_RATING)
            result = detector.predict(review_text, rating)
            payload = _build_phase3_payload(result, review_text, result.rating, {})
            result_items.append(payload)

            dataset_rows.append(
                {
                    "user_id": f"URL_{parsed_url.netloc}_{idx:03d}",
                    "product_id": product_id,
                    "review_text": review_text,
                    "rating": result.rating,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        logger.info("[URL_ANALYZE] Step 5 — Persisting %s rows to active dataset + stats store", len(dataset_rows))
        save_dataset_rows(dataset_rows)
        get_stats_store().record_batch(result_items)
        elapsed_ms = (perf_counter() - step_start) * 1000
        logger.info("[URL_ANALYZE] Step 6 — Done in %.1fms: count=%s source=%s", elapsed_ms, len(result_items), product_url)
        return jsonify({"count": len(result_items), "predictions": result_items, "source_url": product_url})
    except ValueError as exc:
        logger.warning("[URL_ANALYZE] Validation/extraction error: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("[URL_ANALYZE] Unexpected failure during URL processing")
        return jsonify({"error": f"URL analysis failed: {exc}"}), 500


@api_bp.route("/batch-predict", methods=["POST"])
def batch_predict():
    data = request.get_json(silent=True) or {}
    reviews = data.get("reviews")

    if not reviews or not isinstance(reviews, list):
        return jsonify({"error": "reviews must be a non-empty list"}), 400

    texts = []
    ratings = []
    for item in reviews:
        if isinstance(item, str):
            text = item.strip()
            if text:
                texts.append(text)
                ratings.append(config.DEFAULT_RATING)
        elif isinstance(item, dict):
            text = (item.get("review_text") or item.get("text") or "").strip()
            if text:
                texts.append(text)
                ratings.append(item.get("rating", config.DEFAULT_RATING))

    if not texts:
        return jsonify({"error": "No valid reviews in request"}), 400

    if len(texts) > config.MAX_CSV_ROWS:
        return jsonify({"error": f"Maximum {config.MAX_CSV_ROWS} reviews per batch"}), 400

    try:
        detector = get_detector()
        results = detector.predict_batch(texts, ratings)
        payload = []
        for result, text, rating in zip(results, texts, ratings):
            payload.append(_build_phase3_payload(result, text, rating, {}))
        get_stats_store().record_batch(payload)
        return jsonify({"count": len(payload), "predictions": payload})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Batch prediction failed: {exc}"}), 500
