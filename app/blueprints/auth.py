"""
blueprints/auth.py
Autentikasi sederhana (login/logout), sesuai KF/KNF: sistem single-tenant,
akun dibuat oleh administrator (bukan self-register), sesuai teks UI Login:
"Gunakan akun yang telah didaftarkan oleh administrator."

Password di-hash pakai werkzeug.security (sudah bawaan dependency Flask,
tidak perlu install library tambahan). Autentikasi bersifat stateless
sederhana -- tidak pakai JWT/session cookie kompleks, karena skala sistem
single-user/admin. Frontend menyimpan status login di Pinia store
setelah /login sukses.
"""

import logging
from flask import Blueprint, request, jsonify
from werkzeug.security import check_password_hash

from app.extensions import get_db

logger = logging.getLogger("auth")

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Body JSON: { "username": "...", "password": "..." }
    Return: { "status": "ok", "user": { "id": "...", "username": "..." } }
    """
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"error": "Username dan password wajib diisi."}), 400

    user = get_db().users.find_one({"username": username})

    if user is None or not check_password_hash(user['password'], password):
        logger.warning(f"Login gagal untuk username: '{username}'")
        return jsonify({"error": "Username atau password salah."}), 401

    logger.info(f"Login berhasil: '{username}'")

    return jsonify({
        "status": "ok",
        "user": {
            "id": str(user['_id']),
            "username": user['username'],
        }
    }), 200


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """
    Stateless -- tidak ada session server-side yang perlu dihapus.
    Endpoint ini ada supaya frontend punya call yang konsisten,
    penghapusan status login dilakukan di sisi client (Pinia store).
    """
    return jsonify({"status": "ok"}), 200


@auth_bp.route('/ping', methods=['GET'])
def ping():
    return {"status": "auth blueprint aktif"}
