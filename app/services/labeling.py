"""
labeling.py
Layer pemberian nama bisnis pada hasil klaster K-Means (revisi -- 5 fitur +
PCA, K optimal final = 4).

K-Means murni menghasilkan label numerik (0, 1, 2, ...) yang urutannya
tidak punya makna bisnis. Modul ini memetakan tiap klaster ke salah satu
dari 4 kondisi penjualan (Produk Harian, Produk Langka, Produk Andalan,
Produk Premium) berdasarkan kedekatan karakteristik centroid terhadap 4
titik arketipe yang didefinisikan dari distribusi persentil 5 fitur asli.

Karena clustering sekarang dijalankan di ruang PCA (2 komponen), arketipe
yang semula didefinisikan di ruang 5 fitur asli perlu diproyeksikan lebih
dulu ke ruang PCA yang sama (melalui pipeline transformasi yang identik
dengan data: log1p -> standarisasi -> PCA) sebelum bisa dibandingkan
dengan posisi centroid hasil training.

Pendekatan: nearest-archetype + optimal bijective assignment (brute-force
permutasi, karena K selalu kecil sehingga jumlah kombinasi trivial).
Tidak menggunakan scipy -- murni numpy + itertools.
"""

import numpy as np
from itertools import permutations
import logging

from app.services.pca import pca_transform

logger = logging.getLogger("labeling")

ARCHETYPE_NAMES = [
    "Produk Harian",
    "Produk Langka",
    "Produk Andalan",
    "Produk Premium",
]


def _euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def derive_archetypes(df, feature_cols):
    """
    Definisikan 4 titik arketipe di ruang 5 fitur ASLI (belum diwinsorize/
    di-log/distandarisasi), berbasis persentil distribusi data.

    feature_cols urutan HARUS sama dengan CLUSTER_FEATURES di preprocessing.py:
    [frekuensi_transaksi, avg_qty_per_transaksi, total_kuantitas, avg_harga,
     n_bulan_aktif]

    Return: np.ndarray shape (4, n_features), urutan sesuai ARCHETYPE_NAMES
    """
    freq_col, qty_col, total_qty_col, harga_col, bulan_col = feature_cols

    p = df[feature_cols].quantile([0.1, 0.25, 0.5, 0.75, 0.9])

    archetypes_raw = np.array([
        # Produk Harian: frekuensi tinggi, harga murah, selalu aktif tiap bulan
        [p.loc[0.9, freq_col], p.loc[0.5, qty_col], p.loc[0.9, total_qty_col],
         p.loc[0.1, harga_col], p.loc[0.9, bulan_col]],
        # Produk Langka: frekuensi rendah, harga mahal, kurang konsisten hadir
        [p.loc[0.1, freq_col], p.loc[0.25, qty_col], p.loc[0.1, total_qty_col],
         p.loc[0.9, harga_col], p.loc[0.1, bulan_col]],
        # Produk Andalan: menengah-tinggi di volume, penyumbang pendapatan besar
        [p.loc[0.75, freq_col], p.loc[0.75, qty_col], p.loc[0.75, total_qty_col],
         p.loc[0.5, harga_col], p.loc[0.9, bulan_col]],
        # Produk Premium: harga mahal (kedua), frekuensi menengah, tetap konsisten
        [p.loc[0.5, freq_col], p.loc[0.25, qty_col], p.loc[0.25, total_qty_col],
         p.loc[0.75, harga_col], p.loc[0.75, bulan_col]],
    ])
    return archetypes_raw


def _project_archetypes_to_pca(archetypes_raw, df, feature_cols):
    """
    Proyeksikan arketipe (ruang fitur asli) ke ruang PCA yang sama dengan
    centroid hasil training, melalui pipeline transformasi identik dengan
    data: log1p (pakai col_min dari data asli, supaya konsisten) ->
    standarisasi (pakai scale_mean/scale_std tersimpan) -> proyeksi PCA
    (pakai pca_components tersimpan).
    """
    col_mins = df[feature_cols].min().values
    archetypes_log = np.log1p(archetypes_raw - col_mins + 1)

    scale_mean = np.array(df.attrs['scale_mean'])
    scale_std = np.array(df.attrs['scale_std'])
    archetypes_scaled = (archetypes_log - scale_mean) / scale_std

    pca_components = np.array(df.attrs['pca_components'])
    archetypes_pca = pca_transform(archetypes_scaled, pca_components)

    return archetypes_pca


def _optimal_assignment(centroids, archetypes_pca):
    """
    Cari pemetaan cluster -> archetype yang meminimalkan total jarak,
    dengan syarat bijective (1 cluster hanya dapat 1 archetype).
    Brute-force permutasi archetype index.

    Jika jumlah cluster (k) melebihi jumlah archetype yang tersedia
    (mis. K-Means menghasilkan K > 4 pada dataset lain di masa depan),
    fallback ke nearest-archetype non-bijective supaya sistem tidak error
    (beberapa cluster boleh berbagi nama archetype yang sama).

    Return: (mapping dict {cluster_index: archetype_index}, dist_matrix)
    """
    k = len(centroids)
    n_archetype = len(archetypes_pca)

    dist_matrix = np.zeros((k, n_archetype))
    for i, centroid in enumerate(centroids):
        for j, archetype in enumerate(archetypes_pca):
            dist_matrix[i, j] = _euclidean(centroid, archetype)

    if k > n_archetype:
        logger.warning(
            f"Jumlah cluster ({k}) melebihi jumlah archetype yang didefinisikan "
            f"({n_archetype}). Menggunakan nearest-archetype non-bijective."
        )
        mapping = {i: int(np.argmin(dist_matrix[i])) for i in range(k)}
        return mapping, dist_matrix

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
        punya df.attrs dari preprocessing (scale_mean, scale_std,
        pca_components)
    feature_cols: list 5 nama fitur ASLI (df.attrs['cluster_features']),
        BUKAN feature_cols_pca -- dipakai untuk membangun arketipe di ruang
        fitur asli sebelum diproyeksikan ke ruang PCA
    centroids: np.ndarray hasil run_kmeans (posisi centroid di ruang PCA)

    Return: df dengan kolom baru 'kondisi_penjualan' (nama bisnis)
    """
    archetypes_raw = derive_archetypes(df, feature_cols)
    archetypes_pca = _project_archetypes_to_pca(archetypes_raw, df, feature_cols)

    mapping, dist_matrix = _optimal_assignment(centroids, archetypes_pca)

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
