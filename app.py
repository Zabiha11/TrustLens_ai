"""TrustLens AI — Flask application entry point."""

import logging

from flask import Flask

import config
from routes import (
    api_bp,
    dashboard_bp,
    explainability_bp,
    main_bp,
    network_bp,
    risk_bp,
    temporal_bp,
)
from utils.helpers import ensure_upload_dir


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app.logger.setLevel(logging.INFO)

    ensure_upload_dir()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(network_bp)
    app.register_blueprint(risk_bp)
    app.register_blueprint(temporal_bp)
    app.register_blueprint(explainability_bp)

    @app.context_processor
    def inject_globals():
        return {"app_name": "TrustLens AI"}

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=config.DEBUG, host="0.0.0.0", port=5000)
