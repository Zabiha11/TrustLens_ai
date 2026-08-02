"""Flask blueprints for TrustLens AI."""

from routes.main import main_bp
from routes.api import api_bp
from routes.dashboard import dashboard_bp
from routes.network import network_bp
from routes.risk import risk_bp
from routes.temporal import temporal_bp
from routes.explainability import explainability_bp

__all__ = [
    "main_bp",
    "api_bp",
    "dashboard_bp",
    "network_bp",
    "risk_bp",
    "temporal_bp",
    "explainability_bp",
]
