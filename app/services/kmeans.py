"""
kmeans.py
Algoritma K-Means Clustering manual (tanpa scikit-learn), sesuai BAB III 7.6.2.
Meliputi: inisialisasi K-Means++, iterasi assign-update, evaluasi Elbow Method
(WCSS) dan Silhouette Score, pencarian K optimal pada rentang 2-5.
"""

import numpy as np
import logging

logger = logging.getLogger("kmeans")


def euclidean_distance(a, b):
    """Hitung jarak Euclidean antara dua titik (n-dimensi)."""
    return np.sqrt(np.sum((a - b) ** 2))


def initialize_centroids(data, k, seed=42):
    """Inisialisasi centroid dengan metode K-Means++."""
    np.random.seed(seed)
    centroids = [data[np.random.randint(0, len(data))]]
    for _ in range(k - 1):
        distances = np.array([
            min(euclidean_distance(point, c) ** 2 for c in centroids)
            for point in data
        ])
        probs = distances / distances.sum()
        cumulative = np.cumsum(probs)
        r = np.random.rand()
        for idx, prob in enumerate(cumulative):
            if r <= prob:
                centroids.append(data[idx])
                break
    return np.array(centroids)


def assign_clusters(data, centroids):
    """Assign setiap titik ke centroid terdekat."""
    labels = []
    for point in data:
        distances = [euclidean_distance(point, c) for c in centroids]
        labels.append(np.argmin(distances))
    return np.array(labels)


def update_centroids(data, labels, k):
    """Update centroid berdasarkan rata-rata anggota cluster."""
    centroids = []
    for i in range(k):
        members = data[labels == i]
        if len(members) == 0:
            centroids.append(data[np.random.randint(0, len(data))])
        else:
            centroids.append(members.mean(axis=0))
    return np.array(centroids)


def compute_wcss(data, labels, centroids):
    """Hitung WCSS (Within-Cluster Sum of Squares) untuk Elbow Method."""
    wcss = 0
    for i, point in enumerate(data):
        wcss += euclidean_distance(point, centroids[labels[i]]) ** 2
    return wcss


def compute_silhouette(data, labels):
    """Hitung Silhouette Score secara manual."""
    n = len(data)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    scores = []
    for i in range(n):
        same = data[labels == labels[i]]
        a = np.mean([
            euclidean_distance(data[i], p)
            for p in same if not np.array_equal(p, data[i])
        ]) if len(same) > 1 else 0
        b_vals = []
        for label in unique_labels:
            if label == labels[i]:
                continue
            other = data[labels == label]
            b_vals.append(np.mean([euclidean_distance(data[i], p) for p in other]))
        b = min(b_vals) if b_vals else 0
        scores.append((b - a) / max(a, b) if max(a, b) != 0 else 0)
    return float(np.mean(scores))


def _run_single_kmeans(data, k, max_iter=100, seed=42):
    """Jalankan 1x K-Means untuk K tertentu sampai konvergen. Return (labels, centroids, n_iter)."""
    centroids = initialize_centroids(data, k, seed=seed)
    labels = None
    n_iter = 0
    for iteration in range(max_iter):
        new_labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, new_labels, k)
        n_iter = iteration + 1
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centroids = new_centroids
    return labels, centroids, n_iter


def find_optimal_k(df, feature_cols, k_min=2, k_max=5, max_iter=100):
    """
    Mencari K optimal pada rentang [k_min, k_max] (default 2-5, sesuai BAB III 7.6.2)
    menggunakan Elbow Method (WCSS) + Silhouette Score manual.

    df: pandas.DataFrame, sudah berisi kolom '<feature>_scaled' untuk tiap feature_cols
    feature_cols: list nama fitur dasar

    Return: dict {
        "elbow_data": {"k": [...], "wcss": [...]},
        "silhouette_data": {"k": [...], "scores": [...]},
        "optimal_k": int
    }
    """
    scaled_cols = [f"{c}_scaled" for c in feature_cols]
    data = df[scaled_cols].values.astype(float)
    n_produk = len(data)

    logger.info(f"Mencari K optimal pada rentang {k_min}-{k_max} untuk {n_produk} produk...")

    k_max_effective = min(k_max, n_produk - 1)
    if k_max_effective < k_max:
        logger.warning(f"K maksimum diturunkan ke {k_max_effective} karena jumlah produk terbatas.")

    k_range = list(range(k_min, k_max_effective + 1))
    wcss_list = []
    silhouette_list = []

    for k in k_range:
        labels, centroids, n_iter = _run_single_kmeans(data, k, max_iter=max_iter)
        wcss = compute_wcss(data, labels, centroids)
        sil = compute_silhouette(data, labels)
        wcss_list.append(wcss)
        silhouette_list.append(sil)
        sizes = np.bincount(labels).tolist()
        logger.info(
            f"K={k} | konvergen di iterasi ke-{n_iter} | WCSS={wcss:.4f} | "
            f"Silhouette={sil:.4f} | ukuran klaster={sizes}"
        )

    optimal_idx = int(np.argmax(silhouette_list))
    optimal_k = k_range[optimal_idx]

    logger.info(
        f"K optimal terpilih: K={optimal_k} "
        f"(Silhouette Score tertinggi = {silhouette_list[optimal_idx]:.4f})"
    )

    return {
        "elbow_data": {"k": k_range, "wcss": wcss_list},
        "silhouette_data": {"k": k_range, "scores": silhouette_list},
        "optimal_k": optimal_k
    }


def run_kmeans(df, feature_cols, k, max_iter=100):
    """
    Fit final K-Means dengan K yang sudah ditentukan (hasil find_optimal_k).

    Return: (df_with_cluster, centroids_scaled)
        df_with_cluster: df asli + kolom 'cluster' (label numerik 0..k-1)
        centroids_scaled: np.ndarray shape (k, n_features), posisi centroid di ruang ternormalisasi
    """
    scaled_cols = [f"{c}_scaled" for c in feature_cols]
    data = df[scaled_cols].values.astype(float)

    logger.info(f"Menjalankan K-Means final dengan K={k}...")

    labels, centroids, n_iter = _run_single_kmeans(data, k, max_iter=max_iter)

    df = df.copy()
    df['cluster'] = labels

    logger.info(f"K-Means selesai. Konvergen di iterasi ke-{n_iter}. {len(np.unique(labels))} klaster terbentuk.")
    for c in range(k):
        logger.info(f"  Cluster {c}: {int(np.sum(labels == c))} produk, centroid(scaled)={centroids[c].round(3).tolist()}")

    return df, centroids
