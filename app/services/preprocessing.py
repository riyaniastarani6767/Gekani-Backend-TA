"""
preprocessing.py
Modul preprocessing data transaksi penjualan (BAB III, Gambar 3.14 -- revisi).
Alur: baca data -> missing value & duplikasi -> cleaning -> filter periode
      -> agregasi & feature engineering -> seleksi fitur -> winsorizing
      -> transformasi log -> standarisasi -> reduksi dimensi (PCA).
Semua proses manual (pandas/numpy), tanpa scikit-learn.
"""

import pandas as pd
import numpy as np
import logging

from app.services.pca import pca_fit

logger = logging.getLogger("preprocessing")

# 5 fitur final K-Means (lihat BAB III 7.6.1 revisi). seasonal_coeff
# dikeluarkan karena arah korelasinya berlawanan dengan frekuensi_transaksi
# dan total_kuantitas (lihat notebook eksperimen), yang secara struktural
# menurunkan kualitas pemisahan klaster.
CLUSTER_FEATURES = [
    'frekuensi_transaksi',
    'avg_qty_per_transaksi',
    'total_kuantitas',
    'avg_harga',
    'n_bulan_aktif',
]

# Fitur yang ditransformasi log1p (semua fitur final bersifat right-skewed).
LOG_FEATURES = list(CLUSTER_FEATURES)

# Batas persentil winsorizing, dipilih dari pengujian empiris (grid search
# persentil 5-30) yang menghasilkan Silhouette Score dan keseimbangan
# klaster terbaik secara bersamaan.
WINSORIZE_LOWER = 0.17
WINSORIZE_UPPER = 0.83

# Jumlah komponen utama PCA yang dipakai sebagai input K-Means.
PCA_N_COMPONENTS = 2


def standardize(data):
    """
    Manual StandardScaler: z = (x - mean) / std
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    return (data - mean) / std, mean, std


def profile_dataset(filepath):
    """Dipakai di halaman Preview Data untuk deteksi kolom otomatis."""
    try:
        df_header = pd.read_csv(filepath, nrows=0, encoding='utf-8', on_bad_lines='warn')
        return {"columns": df_header.columns.tolist()}
    except Exception as e:
        logger.error(f"Error saat profiling dataset: {e}")
        return {"columns": []}


def aggregate_data(filepath, tahun_awal=None, tahun_akhir=None):
    """
    Pipeline preprocessing (Gambar 3.14):
    baca data -> missing/duplikasi -> cleaning -> filter periode
    -> agregasi & feature engineering -> seleksi fitur -> winsorizing
    -> transformasi log -> standarisasi -> reduksi dimensi (PCA).
    """
    logger.info(f"Membaca file: {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='warn')
    n_awal = len(df)
    logger.info(f"Berhasil membaca {n_awal} baris.")

    kolom_wajib = ['no_invoice', 'tanggal', 'kategori', 'sub_kategori',
                   'kuantitas_terjual', 'total_sales']
    for col in kolom_wajib:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib '{col}' tidak ditemukan di file.")

    n_before_dedup = len(df)
    df.drop_duplicates(inplace=True)
    n_duplikat = n_before_dedup - len(df)

    df.dropna(subset=['sub_kategori', 'no_invoice', 'tanggal'], inplace=True)

    df['total_sales'] = pd.to_numeric(df['total_sales'], errors='coerce')
    df['kuantitas_terjual'] = pd.to_numeric(df['kuantitas_terjual'], errors='coerce')
    df.dropna(subset=['total_sales', 'kuantitas_terjual'], inplace=True)
    n_missing = n_before_dedup - n_duplikat - len(df)

    logger.info(f"Duplikat dihapus: {n_duplikat} baris.")
    logger.info(f"Missing value dihapus: {n_missing} baris.")

    n_before_clean = len(df)
    df = df[df['total_sales'] >= 0]
    n_negatif = n_before_clean - len(df)
    logger.info(f"Cleaning: {n_negatif} baris dihapus (total_sales negatif).")

    df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce', dayfirst=True)
    df.dropna(subset=['tanggal'], inplace=True)
    df['tahun'] = df['tanggal'].dt.year

    if tahun_awal is not None:
        df = df[df['tahun'] >= int(tahun_awal)]
    if tahun_akhir is not None:
        df = df[df['tahun'] <= int(tahun_akhir)]

    n_setelah_cleaning = len(df)
    logger.info(f"Setelah filter periode: {n_setelah_cleaning} baris.")

    if df.empty:
        raise ValueError("Tidak ada data pada rentang periode yang dipilih.")

    df['year_month'] = df['tanggal'].dt.to_period('M')

    logger.info("Melakukan agregasi & feature engineering...")

    freq = df.groupby('sub_kategori')['no_invoice'].nunique().rename('frekuensi_transaksi')
    qty_total = df.groupby('sub_kategori')['kuantitas_terjual'].sum().rename('total_kuantitas')
    revenue = df.groupby('sub_kategori')['total_sales'].sum().rename('total_pendapatan')
    kategori = df.groupby('sub_kategori')['kategori'].first().rename('kategori')
    avg_harga = df.groupby('sub_kategori')['harga_per_unit'].mean().rename('avg_harga')

    avg_qty = (qty_total / freq).rename('avg_qty_per_transaksi')

    # n_bulan_aktif: dari seluruh bulan pada rentang periode data, berapa
    # bulan produk ini benar-benar memiliki penjualan. Mengukur kontinuitas
    # kehadiran produk -- beda dari frekuensi_transaksi yang menghitung total
    # jumlah transaksi tanpa memandang sebarannya sepanjang waktu.
    monthly = df.groupby(['sub_kategori', 'year_month'])['total_sales'].sum().reset_index()
    n_bulan_aktif = monthly.groupby('sub_kategori')['year_month'].nunique().rename('n_bulan_aktif')

    agg_df = pd.concat(
        [kategori, freq, avg_qty, qty_total, avg_harga, n_bulan_aktif, revenue], axis=1
    ).reset_index().rename(columns={'sub_kategori': 'produk'})

    n_before_final_drop = len(agg_df)
    agg_df.dropna(inplace=True)
    if len(agg_df) < n_before_final_drop:
        logger.warning(f"{n_before_final_drop - len(agg_df)} produk dihapus karena fitur turunan NaN.")

    logger.info(f"Agregasi selesai: {len(agg_df)} produk unik.")

    cluster_features = CLUSTER_FEATURES

    # PENTING: nilai ASLI (frekuensi_transaksi, avg_harga, dst) di agg_df
    # TIDAK diubah -- kolom-kolom ini masih dipakai untuk ditampilkan ke
    # pengguna (Data Produk, Matrix Segmentasi, dsb). Winsorizing, log1p,
    # standarisasi, dan PCA hanya dijalankan di atas SALINAN terpisah
    # (features_for_model), khusus untuk kebutuhan clustering.
    features_for_model = agg_df[cluster_features].copy()

    # Winsorizing -- memangkas nilai ekstrem pada rentang persentil
    # WINSORIZE_LOWER - WINSORIZE_UPPER, tanpa menghapus produk dari data.
    # Diterapkan sebelum log1p karena outlier ekstrem (mis. produk dengan
    # std harga jauh di atas mayoritas) bisa "menarik" arah komponen utama
    # PCA apabila tidak ditangani lebih dulu.
    for col in cluster_features:
        lower_bound, upper_bound = features_for_model[col].quantile([WINSORIZE_LOWER, WINSORIZE_UPPER])
        features_for_model[col] = features_for_model[col].clip(lower=lower_bound, upper=upper_bound)

    # Transformasi log1p -- kelima fitur bersifat right-skewed (sebagian
    # kecil produk punya nilai jauh di atas mayoritas). K-Means berasumsi
    # cluster berbentuk spherical di ruang fitur; skewness melanggar asumsi
    # itu. log1p menekan rentang nilai ekstrem tanpa membuang data.
    for col in LOG_FEATURES:
        col_min = features_for_model[col].min()
        features_for_model[col] = np.log1p(features_for_model[col] - col_min + 1)

    # Standarisasi (z-score) -- menyamakan skala kelima fitur sebelum PCA,
    # supaya fitur dengan rentang nilai besar (mis. avg_harga dalam rupiah)
    # tidak mendominasi fitur dengan rentang nilai kecil (mis. n_bulan_aktif
    # yang hanya 0-60) hanya karena perbedaan skala.
    features_processed = features_for_model[cluster_features].values.astype(float)
    scaled, scale_mean, scale_std = standardize(features_processed)

    logger.info(f"Standarisasi selesai. Mean={scale_mean.round(2)}, Std={scale_std.round(2)}")

    # Reduksi dimensi PCA -- memadatkan 5 fitur terstandarisasi menjadi 2
    # komponen utama, supaya K-Means tidak kesulitan memisahkan klaster
    # akibat curse of dimensionality (lihat pca.py).
    pca_result = pca_fit(scaled, n_components=PCA_N_COMPONENTS)
    data_pca = pca_result["data_reduced"]
    pca_components = pca_result["components"]
    pca_explained_variance = pca_result["explained_variance_ratio"]

    # Kolom '<nama>_scaled' inilah yang dibaca oleh kmeans.py sebagai input
    # clustering (lihat find_optimal_k/run_kmeans: scaled_cols = f"{c}_scaled").
    # Dengan PCA, "fitur" yang dipakai K-Means bukan lagi 5 fitur asli,
    # melainkan 2 komponen utama -- feature_cols yang dikirim ke kmeans.py
    # sekarang harus ['pc1', 'pc2'], bukan cluster_features.
    for i in range(PCA_N_COMPONENTS):
        agg_df[f'pc{i + 1}_scaled'] = data_pca[:, i]

    agg_df.attrs['cluster_features'] = cluster_features
    agg_df.attrs['pca_feature_cols'] = [f'pc{i + 1}' for i in range(PCA_N_COMPONENTS)]
    agg_df.attrs['scale_mean'] = scale_mean.tolist()
    agg_df.attrs['scale_std'] = scale_std.tolist()
    agg_df.attrs['pca_components'] = pca_components.tolist()
    agg_df.attrs['pca_explained_variance'] = pca_explained_variance.tolist()
    agg_df.attrs['winsorize_lower'] = WINSORIZE_LOWER
    agg_df.attrs['winsorize_upper'] = WINSORIZE_UPPER
    agg_df.attrs['n_transaksi_awal'] = n_awal
    agg_df.attrs['n_transaksi_setelah_cleaning'] = n_setelah_cleaning
    agg_df.attrs['n_duplikat_dihapus'] = n_duplikat
    agg_df.attrs['n_missing_dihapus'] = n_missing
    agg_df.attrs['n_negatif_dihapus'] = n_negatif

    return agg_df


def preview_data(filepath, n=10):
    """
    Baca sebagian data untuk halaman Preview Data (KF2, KF3).
    Tidak melakukan agregasi -- hanya baca mentah + info rentang tahun tersedia.
    """
    df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='warn')

    tanggal_parsed = pd.to_datetime(df['tanggal'], errors='coerce', dayfirst=True)
    tahun_series = tanggal_parsed.dt.year.dropna()

    sample = df.head(n).copy()
    sample_rows = sample.astype(object).where(pd.notnull(sample), None).to_dict(orient='records')

    return {
        "columns": df.columns.tolist(),
        "sample_rows": sample_rows,
        "total_rows": len(df),
        "tahun_min": int(tahun_series.min()) if len(tahun_series) > 0 else None,
        "tahun_max": int(tahun_series.max()) if len(tahun_series) > 0 else None,
    }


def count_transaksi_in_range(filepath, tahun_awal, tahun_akhir):
    """
    Hitung cepat jumlah transaksi dalam rentang tahun tertentu, dipakai
    untuk update live counter di halaman Preview Data.
    """
    df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='warn', usecols=['tanggal'])
    tanggal_parsed = pd.to_datetime(df['tanggal'], errors='coerce', dayfirst=True)
    tahun = tanggal_parsed.dt.year
    mask = (tahun >= int(tahun_awal)) & (tahun <= int(tahun_akhir))
    return int(mask.sum())


def monthly_revenue(filepath, tahun_awal=None, tahun_akhir=None):
    """
    Agregasi total_sales per bulan (lintas semua produk), dipakai untuk
    grafik "Tren Penjualan Bulanan" di Dashboard.

    Return: list of dict, urut kronologis:
        [{"bulan": "2021-01", "label": "Jan 2021", "total": 12345000}, ...]
    """
    df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='warn')

    df['total_sales'] = pd.to_numeric(df['total_sales'], errors='coerce')
    df.dropna(subset=['total_sales'], inplace=True)
    df = df[df['total_sales'] >= 0]

    df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce', dayfirst=True)
    df.dropna(subset=['tanggal'], inplace=True)
    df['tahun'] = df['tanggal'].dt.year

    if tahun_awal is not None:
        df = df[df['tahun'] >= int(tahun_awal)]
    if tahun_akhir is not None:
        df = df[df['tahun'] <= int(tahun_akhir)]

    df['year_month'] = df['tanggal'].dt.to_period('M')

    monthly = df.groupby('year_month')['total_sales'].sum().sort_index()

    bulan_id = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']

    result = []
    for period, total in monthly.items():
        label = f"{bulan_id[period.month - 1]} {period.year}"
        result.append({"bulan": str(period), "label": label, "total": float(total)})

    return result
