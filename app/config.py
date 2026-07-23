"""
config.py
Konfigurasi aplikasi Flask, dibaca dari environment variable (.env).
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    MONGO_URI = os.getenv('MONGO_URI')
    SECRET_KEY = os.getenv('SECRET_KEY')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB, sesuai batas upload di UI (KF1)

    @staticmethod
    def validate():
        """Cek env variable wajib sudah terisi, biar error jelas di awal bukan pas runtime."""
        missing = []
        if not Config.MONGO_URI:
            missing.append('MONGO_URI')
        if not Config.SECRET_KEY:
            missing.append('SECRET_KEY')
        if missing:
            raise RuntimeError(
                f"Environment variable belum diisi: {', '.join(missing)}. Cek file .env."
            )
