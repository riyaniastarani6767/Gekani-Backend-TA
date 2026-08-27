"""
pca.py
Reduksi dimensi menggunakan Principal Component Analysis (PCA), diimplementasi
manual (matriks kovarians -> eigendecomposition -> proyeksi), tanpa
scikit-learn, sesuai ketentuan BAB III.

PCA dipakai untuk memadatkan 5 fitur K-Means terstandarisasi menjadi 2
komponen utama, sehingga informasi dari kelima fitur tetap terwakili namun
kompleksitas dimensi terkendali (mengatasi curse of dimensionality yang
menyebabkan K-Means kesulitan memisahkan klaster secara jelas bila kelima
fitur dipakai langsung tanpa reduksi).
"""

import numpy as np
import logging

logger = logging.getLogger("pca")


def pca_fit(data_scaled, n_components=2):
    """
    Hitung komponen utama dari data yang SUDAH distandarisasi (z-score).

    data_scaled: np.ndarray shape (n_samples, n_features)
    n_components: jumlah komponen utama yang diambil

    Return: dict {
        "data_reduced": np.ndarray shape (n_samples, n_components),
        "components": np.ndarray shape (n_features, n_components),
            -- dipakai lagi untuk proyeksi data baru (lihat pca_transform)
        "explained_variance_ratio": np.ndarray shape (n_features,),
            -- urutan menurun, dipakai untuk scree plot
    }
    """
    cov_matrix = np.cov(data_scaled, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    explained_variance_ratio = eigvals / eigvals.sum()
    components = eigvecs[:, :n_components]

    data_reduced = data_scaled @ components

    total_variance = explained_variance_ratio[:n_components].sum()
    logger.info(
        f"PCA: {n_components} komponen mencakup {total_variance * 100:.2f}% variansi "
        f"(per komponen: {np.round(explained_variance_ratio[:n_components] * 100, 2).tolist()})"
    )

    return {
        "data_reduced": data_reduced,
        "components": components,
        "explained_variance_ratio": explained_variance_ratio,
    }


def pca_transform(data_scaled, components):
    """
    Proyeksikan data (yang SUDAH distandarisasi dengan mean/std dari fit awal)
    ke ruang komponen utama yang sudah ada. Dipakai untuk memproyeksikan titik
    arketipe pada labeling.py ke ruang PCA yang sama dengan centroid hasil
    training, bukan untuk melatih ulang PCA.

    data_scaled: np.ndarray shape (n_samples, n_features)
    components: np.ndarray shape (n_features, n_components), dari pca_fit()

    Return: np.ndarray shape (n_samples, n_components)
    """
    return data_scaled @ components
