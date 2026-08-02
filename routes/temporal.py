"""Temporal Fraud Analysis Flask Blueprint."""

from flask import Blueprint, jsonify, render_template

from utils.helpers import load_active_dataset
from utils.temporal_analysis import TemporalFraudDetector

temporal_bp = Blueprint("temporal", __name__)


@temporal_bp.route("/temporal-analysis")
def temporal_page():
    return render_template("temporal_analysis.html")


@temporal_bp.route("/api/temporal/data")
def temporal_data():
    df = load_active_dataset()
    detector = TemporalFraudDetector(df)
    return jsonify(detector.get_summary())
