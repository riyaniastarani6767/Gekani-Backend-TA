"""
app/__init__.py
Application factory: bikin instance Flask, load config, init MongoDB,
register blueprint.
"""

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

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

    # File melebihi MAX_CONTENT_LENGTH (50MB): kembalikan JSON supaya
    # frontend bisa menampilkan pesan yang jelas.
    @app.errorhandler(RequestEntityTooLarge)
    def file_terlalu_besar(e):
        return jsonify({"error": "Ukuran file melebihi batas maksimal 50MB."}), 413

    logger.info("Aplikasi Flask siap.")

    return app