"""
labeling.py
Layer pemberian nama bisnis pada hasil klaster K-Means.

K-Means murni menghasilkan label numerik (0, 1, 2, ...) yang urutannya
tidak punya makna bisnis. Modul ini memetakan tiap klaster ke salah satu
dari 5 kondisi penjualan (Produk Laris, Produk Stabil, Produk Musiman,
Jarang Terjual, Produk Grosir) berdasarkan kedekatan karakteristik centroid
terhadap 5 titik arketipe yang didefinisikan dari distribusi persentil data.

Pendekatan: nearest-archetype + optimal bijective assignment (brute-force
permutasi, karena K selalu <= 5 sehingga jumlah kombinasi kecil/trivial).
Tidak menggunakan scipy -- murni numpy + itertools.
"""

import numpy as np
from itertools import permutations
import logging

logger = logging.getLogger("labeling")

ARCHETYPE_NAMES = [
    "Produk Laris",
    "Produk Stabil",
    "Produk Musiman",
    "Jarang Terjual",
    "Produk Grosir",
]

def _euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def derive_archetypes(df, feature_cols):
    """
    Definisikan 5 titik arketipe di ruang fitur ASLI (belum di-scale),
    berbasis persentil distribusi data.

    feature_cols urutan HARUS: [frekuensi_transaksi, avg_qty_per_transaksi, seasonal_coeff]

    Return: np.ndarray shape (5, n_features), urutan sesuai ARCHETYPE_NAMES
    """
    freq_col, qty_col, season_col = feature_cols

    p = df[feature_cols].quantile([0.1, 0.25, 0.5, 0.75, 0.9])

    archetypes_raw = np.array([
        [p.loc[0.9, freq_col], p.loc[0.5, qty_col], p.loc[0.1, season_col]],
        [p.loc[0.5, freq_col], p.loc[0.5, qty_col], p.loc[0.25, season_col]],
        [p.loc[0.25, freq_col], p.loc[0.5, qty_col], p.loc[0.9, season_col]],
        [p.loc[0.1, freq_col], p.loc[0.25, qty_col], p.loc[0.5, season_col]],
        [p.loc[0.25, freq_col], p.loc[0.9, qty_col], p.loc[0.5, season_col]],
    ])
    return archetypes_raw


def _scale_archetypes(archetypes_raw, scale_mean, scale_std):
    scale_mean = np.array(scale_mean)
    scale_std = np.array(scale_std)
    return (archetypes_raw - scale_mean) / scale_std


def _optimal_assignment(centroids, archetypes_scaled):
    """
    Cari pemetaan cluster -> archetype yang meminimalkan total jarak,
    dengan syarat bijective (1 cluster hanya dapat 1 archetype).
    Brute-force permutasi archetype index (K selalu <= 5, max 120 kombinasi).

    Return: (mapping dict {cluster_index: archetype_index}, dist_matrix)
    """
    k = len(centroids)
    n_archetype = len(archetypes_scaled)

    dist_matrix = np.zeros((k, n_archetype))
    for i, centroid in enumerate(centroids):
        for j, archetype in enumerate(archetypes_scaled):
            dist_matrix[i, j] = _euclidean(centroid, archetype)

    best_total_dist = None
    best_assignment = None

    for archetype_indices in permutations(range(n_archetype), k):
        total_dist = sum(dist_matrix[i, archetype_indices[i]] for i in range(k))
        if best_total_dist is None or total_dist < best_total_dist:
            best_total_dist = total_dist
            best_assignment = archetype_indices

    mapping = {i: best_assignment[i] for i in range(k)}
    return mapping, dist_matrix


def label_clusters(df, feature_cols, centroids):
    """
    Beri nama bisnis pada tiap klaster hasil run_kmeans().

    df: DataFrame hasil run_kmeans (sudah punya kolom 'cluster'), dan masih
        punya df.attrs dari preprocessing (scale_mean, scale_std)
    feature_cols: list nama fitur dasar, urutan HARUS sama dgn saat training
    centroids: np.ndarray hasil run_kmeans (posisi centroid di ruang scaled)

    Return: df dengan kolom baru 'kondisi_penjualan' (nama bisnis)
    """
    scale_mean = df.attrs['scale_mean']
    scale_std = df.attrs['scale_std']

    archetypes_raw = derive_archetypes(df, feature_cols)
    archetypes_scaled = _scale_archetypes(archetypes_raw, scale_mean, scale_std)

    mapping, dist_matrix = _optimal_assignment(centroids, archetypes_scaled)

    logger.info("Hasil pemetaan klaster -> kondisi penjualan (nearest-archetype):")
    for cluster_idx, archetype_idx in mapping.items():
        nama = ARCHETYPE_NAMES[archetype_idx]
        jarak = dist_matrix[cluster_idx, archetype_idx]
        logger.info(f"  Cluster {cluster_idx} -> '{nama}' (jarak={jarak:.4f})")

    df = df.copy()
    df['kondisi_penjualan'] = df['cluster'].map(
        lambda c: ARCHETYPE_NAMES[mapping[c]]
    )

    logger.info(f"Distribusi kondisi_penjualan:\n{df['kondisi_penjualan'].value_counts().to_string()}")

    return df
