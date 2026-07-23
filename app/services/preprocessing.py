"""
preprocessing.py
Modul preprocessing data transaksi penjualan (BAB III, Gambar 3.14).
Alur: baca data -> missing value & duplikasi -> cleaning -> filter periode
      -> agregasi & feature engineering -> seleksi fitur -> standarisasi.
Semua proses manual (pandas/numpy), tanpa scikit-learn.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("preprocessing")


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
    -> agregasi & feature engineering -> seleksi fitur -> standarisasi.
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
    qty_total = df.groupby('sub_kategori')['kuantitas_terjual'].sum().rename('total_qty')
    revenue = df.groupby('sub_kategori')['total_sales'].sum().rename('total_pendapatan')
    kategori = df.groupby('sub_kategori')['kategori'].first().rename('kategori')

    avg_qty = (qty_total / freq).rename('avg_qty_per_transaksi')

    monthly = df.groupby(['sub_kategori', 'year_month'])['total_sales'].sum().reset_index()
    season_stat = monthly.groupby('sub_kategori')['total_sales'].agg(['max', 'mean'])
    seasonal_coeff = (season_stat['max'] / season_stat['mean']).rename('seasonal_coeff')

    agg_df = pd.concat(
        [kategori, freq, avg_qty, seasonal_coeff, revenue], axis=1
    ).reset_index().rename(columns={'sub_kategori': 'produk'})

    n_before_final_drop = len(agg_df)
    agg_df.dropna(inplace=True)
    if len(agg_df) < n_before_final_drop:
        logger.warning(f"{n_before_final_drop - len(agg_df)} produk dihapus karena fitur turunan NaN.")

    logger.info(f"Agregasi selesai: {len(agg_df)} produk unik.")

    cluster_features = ['frekuensi_transaksi', 'avg_qty_per_transaksi', 'seasonal_coeff']

    features = agg_df[cluster_features].values.astype(float)
    scaled, scale_mean, scale_std = standardize(features)
    for i, col in enumerate(cluster_features):
        agg_df[f'{col}_scaled'] = scaled[:, i]

    logger.info(f"Standarisasi selesai. Mean={scale_mean.round(2)}, Std={scale_std.round(2)}")

    agg_df.attrs['cluster_features'] = cluster_features
    agg_df.attrs['scale_mean'] = scale_mean.tolist()
    agg_df.attrs['scale_std'] = scale_std.tolist()
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
