"""
tests/conftest.py
Pengaturan bersama untuk seluruh pengujian (pytest).

- Menambahkan folder backend ke sys.path supaya modul `app` bisa di-import.
- Menyediakan data hasil sistem (dataset Yudi Motor) yang dihitung SEKALI
  lalu dipakai ulang oleh banyak test, supaya pengujian tidak lambat.
"""

import os
import sys
import logging

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

# Matikan log INFO dari services selama testing supaya output pytest bersih.
logging.disable(logging.CRITICAL)

DATASET = os.path.join(BACKEND_DIR, 'sample_data', 'dataPenjualanYudiMotor_2021-2025_FINAL.csv')


@pytest.fixture(scope='session')
def dataset_path():
    if not os.path.exists(DATASET):
        pytest.skip(f"Dataset tidak ditemukan: {DATASET}")
    return DATASET


@pytest.fixture(scope='session')
def df_agregat(dataset_path):
    """Hasil preprocessing lengkap (agregasi -> winsorize -> log -> z-score -> PCA)."""
    from app.services import preprocessing
    return preprocessing.aggregate_data(dataset_path, tahun_awal=2021, tahun_akhir=2025)


@pytest.fixture(scope='session')
def data_pca(df_agregat):
    """Matriks 2 komponen utama (input K-Means), shape (126, 2)."""
    return df_agregat[['pc1_scaled', 'pc2_scaled']].values.astype(float)


@pytest.fixture(scope='session')
def hasil_optimal_k(df_agregat):
    from app.services import kmeans
    return kmeans.find_optimal_k(df_agregat, ['pc1', 'pc2'])


@pytest.fixture(scope='session')
def hasil_kmeans_k4(df_agregat):
    from app.services import kmeans
    return kmeans.run_kmeans(df_agregat, ['pc1', 'pc2'], 4)


@pytest.fixture(scope='session')
def hasil_labeling(df_agregat, hasil_kmeans_k4):
    from app.services import labeling
    df_kmeans, centroids = hasil_kmeans_k4
    return labeling.label_clusters(df_kmeans, df_agregat.attrs['cluster_features'], centroids)


@pytest.fixture(scope='session')
def hasil_abc(df_agregat):
    from app.services import abc_analysis
    return abc_analysis.classify_abc(df_agregat)