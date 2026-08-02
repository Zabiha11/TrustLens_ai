"""Reviewer Network Analysis Flask Blueprint."""

from flask import Blueprint, jsonify, render_template

from utils.helpers import load_active_dataset
from utils.network_analysis import ReviewNetworkAnalyzer

network_bp = Blueprint("network", __name__)


@network_bp.route("/network")
def network_page():
    return render_template("network.html")


@network_bp.route("/api/network/data")
def network_data():
    df = load_active_dataset()
    analyzer = ReviewNetworkAnalyzer(df)
    summary = analyzer.get_summary()
    graph_html = analyzer.generate_pyvis_html()
    summary["graph_html"] = graph_html
    return jsonify(summary)
