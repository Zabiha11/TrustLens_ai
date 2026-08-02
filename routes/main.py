"""Main page routes — Home, Analyze, About."""

from flask import Blueprint, render_template

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    return render_template("index.html")


@main_bp.route("/analyze")
def analyze():
    return render_template("analyze.html")


@main_bp.route("/investigation")
def investigation():
    return render_template("investigation.html")


@main_bp.route("/about")
def about():
    return render_template("about.html")
