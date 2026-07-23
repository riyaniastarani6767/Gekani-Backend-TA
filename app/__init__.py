"""
app/__init__.py
Application factory: bikin instance Flask, load config, init MongoDB,
register blueprint.
"""

from flask import Flask
from flask_cors import CORS

from app.config import Config
from app.extensions import init_mongo
from app.utils.logger import setup_app_logger


def create_app():
    Config.validate()

    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app)  # izinkan request dari Vue (beda port)

    logger = setup_app_logger()

    # Koneksi MongoDB
    init_mongo(app)

    # Registrasi blueprint
    from app.blueprints.auth import auth_bp
    from app.blueprints.analysis import analysis_bp

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(analysis_bp, url_prefix='/api')

    logger.info("Aplikasi Flask siap.")

    return app
