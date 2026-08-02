"""Dashboard routes and summary API."""

from flask import Blueprint, jsonify, render_template

from utils.explainability import get_explainability_engine
from utils.helpers import load_active_dataset
from utils.network_analysis import ReviewNetworkAnalyzer
from utils.risk_scoring import ReviewerRiskScorer
from utils.stats_store import get_stats_store
from utils.temporal_analysis import TemporalFraudDetector

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@dashboard_bp.route("/api/dashboard/stats")
def dashboard_stats():
    base_stats = get_stats_store().get_summary()
    base_stats["explainability_summary"] = get_explainability_engine().get_global_model_explanation()

    # Load Phase 2 Intelligence metrics
    try:
        df = load_active_dataset()
        net_analyzer = ReviewNetworkAnalyzer(df)
        risk_scorer = ReviewerRiskScorer(df)
        temp_detector = TemporalFraudDetector(df)

        risk_summary = risk_scorer.get_summary()
        avg_risk = risk_summary.get("avg_risk_score", 0.0)
        avg_trust = round(max(0.0, 100.0 - avg_risk), 1)

        phase2_stats = {
            "suspicious_users": risk_summary.get("high_risk_count", 0),
            "detected_bursts": temp_detector.get_summary().get("total_bursts_found", 0),
            "reviewer_communities": net_analyzer.get_summary().get("community_count", 0),
            "average_trust_score": avg_trust,
            "network_summary": net_analyzer.get_summary(),
            "risk_summary": risk_summary,
            "temporal_summary": temp_detector.get_summary(),
        }
        base_stats.update(phase2_stats)
    except Exception as exc:
        base_stats["phase2_error"] = str(exc)

    return jsonify(base_stats)

