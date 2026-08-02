"""Reviewer Risk Analysis Flask Blueprint."""

from flask import Blueprint, jsonify, render_template

from utils.helpers import load_active_dataset
from utils.risk_scoring import ReviewerRiskScorer

risk_bp = Blueprint("risk", __name__)


@risk_bp.route("/risk-analysis")
def risk_page():
    return render_template("risk_analysis.html")


@risk_bp.route("/api/risk/data")
def risk_data():
    df = load_active_dataset()
    scorer = ReviewerRiskScorer(df)
    return jsonify(scorer.get_summary())
