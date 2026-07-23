"""
utils/logger.py
Setup logging: tampil di terminal DAN disimpan ke file .log.
1 file log per proses analisis, nama file pakai timestamp supaya
tidak tertimpa kalau user menjalankan beberapa kali analisis.

PENTING: setiap modul di services/ (preprocessing, kmeans, labeling,
abc_analysis, decision_rules) punya logger sendiri lewat
logging.getLogger(nama_modul). Supaya semua pesan dari modul-modul itu
ikut tercatat di file log sesi analisis yang sedang berjalan, handler
yang sama di-attach ke SEMUA logger tersebut, bukan cuma logger utama.
"""

import logging
import os
from datetime import datetime

SERVICE_LOGGER_NAMES = [
    "preprocessing", "kmeans", "labeling", "abc_analysis", "decision_rules"
]


def setup_analysis_logger(logs_dir="logs"):
    """
    Buat logger baru untuk 1 sesi analisis, dan sambungkan juga ke semua
    logger modul services/ supaya pesan mereka ikut tercatat.

    Return: (logger, log_filepath)
    """
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(logs_dir, f"analysis_{timestamp}.log")

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setFormatter(formatter)

    main_logger = logging.getLogger(f"analysis_{timestamp}")

    all_logger_names = [f"analysis_{timestamp}"] + SERVICE_LOGGER_NAMES
    for name in all_logger_names:
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        lg.propagate = False
        lg.addHandler(console_handler)
        lg.addHandler(file_handler)

    return main_logger, log_filepath


def setup_app_logger():
    """Logger umum untuk aplikasi (bukan sesi analisis spesifik), tampil di terminal saja."""
    logger = logging.getLogger("app")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger
