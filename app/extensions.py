"""
extensions.py
Inisialisasi koneksi MongoDB. Dipisah dari __init__.py supaya bisa
di-import oleh service/blueprint lain tanpa circular import.

PENTING: db diakses lewat fungsi get_db(), BUKAN diimpor langsung
sebagai variabel (from app.extensions import db). Kalau diimpor sebagai
variabel, modul yang meng-import akan tetap memegang nilai None selamanya
kalau proses import-nya terjadi sebelum init_mongo() dipanggil -- karena
Python meng-copy nilai saat itu, bukan membuat referensi hidup ke module.
Fungsi get_db() selalu mengambil nilai TERKINI dari modul ini.
"""

import logging
import certifi
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

logger = logging.getLogger("extensions")

mongo_client = None
_db = None


def init_mongo(app):
    """
    Inisialisasi koneksi MongoDB dari app.config['MONGO_URI'].
    - Kalau URI Atlas (mongodb+srv://) -> pakai TLS dengan certifi.
    - Kalau URI lokal (mongodb://)     -> tanpa TLS.
    Dipanggil sekali di app factory (app/__init__.py).
    """
    global mongo_client, _db

    uri = app.config['MONGO_URI']

    if uri.startswith('mongodb+srv://'):
        mongo_client = MongoClient(uri, tlsCAFile=certifi.where())
    else:
        mongo_client = MongoClient(uri)

    try:
        mongo_client.admin.command('ping')
        _db = mongo_client.get_default_database()
        logger.info(f"MongoDB terhubung. Database: '{_db.name}'")
    except ConnectionFailure as e:
        logger.error(f"Gagal konek ke MongoDB: {e}")
        raise

    return _db


def get_db():
    """
    Ambil instance database MongoDB TERKINI. Selalu panggil ini
    di dalam fungsi (saat request masuk), JANGAN di top-level import.
    """
    if _db is None:
        raise RuntimeError("Database belum diinisialisasi. Pastikan create_app() sudah dipanggil.")
    return _db
