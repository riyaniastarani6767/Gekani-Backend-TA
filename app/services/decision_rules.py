"""
decision_rules.py
Decision rules berdasarkan Tabel 3.5 (Matriks Rekomendasi), BAB III 7.7.
Mengombinasikan kondisi_penjualan (hasil K-Means + labeling) dengan
prioritas_abc (hasil ABC Analysis) untuk menghasilkan rekomendasi
tindak lanjut pengelolaan persediaan.
"""

import logging

logger = logging.getLogger("decision_rules")

DECISION_MATRIX = {
    "Produk Laris": {
        "A": "Jaga stok dan lakukan promosi",
        "B": "Cek harga jual produk",
        "C": "Cek harga jual dan biaya pengadaan",
    },
    "Produk Stabil": {
        "A": "Promosikan produk dan jaga ketersediaan stok",
        "B": "Pertahankan stok",
        "C": "Sesuaikan harga jual",
    },
    "Produk Musiman": {
        "A": "Siapkan stok dalam jumlah besar sebelum musim penjualan",
        "B": "Sesuaikan jumlah stok dengan pola permintaan musiman",
        "C": "Lakukan pembelian hanya pada musim penjualan",
    },
    "Jarang Terjual": {
        "A": "Jaga stok dalam jumlah secukupnya",
        "B": "Evaluasi kembali keberadaan produk",
        "C": "Kurangi pembelian",
    },
    "Produk Grosir": {
        "A": "Siapkan stok ketika terdapat pesanan",
        "B": "Gabungkan penjualan dengan produk yang memiliki permintaan tinggi",
        "C": "Kurangi pembelian",
    },
}


def get_rekomendasi(kondisi_penjualan, prioritas_abc):
    """
    Lookup rekomendasi dari matriks Tabel 3.5.
    Return fallback message + log warning kalau kombinasi tidak ditemukan
    (harusnya tidak pernah terjadi kalau data konsisten).
    """
    row = DECISION_MATRIX.get(kondisi_penjualan)
    if row is None:
        logger.warning(f"Kondisi penjualan tidak dikenal: '{kondisi_penjualan}'")
        return "Rekomendasi tidak tersedia (kondisi penjualan tidak dikenal)"

    rekomendasi = row.get(prioritas_abc)
    if rekomendasi is None:
        logger.warning(f"Prioritas ABC tidak dikenal: '{prioritas_abc}'")
        return "Rekomendasi tidak tersedia (prioritas ABC tidak dikenal)"

    return rekomendasi


def apply_decision_rules(df):
    """
    Terapkan decision rules ke seluruh baris DataFrame hasil integrasi
    K-Means + ABC Analysis (df wajib punya kolom 'kondisi_penjualan'
    dan 'prioritas_abc').

    Return: df (copy) dengan kolom tambahan 'rekomendasi'
    """
    logger.info("Menerapkan decision rules (Tabel 3.5)...")

    df = df.copy()
    df['rekomendasi'] = df.apply(
        lambda row: get_rekomendasi(row['kondisi_penjualan'], row['prioritas_abc']),
        axis=1
    )

    logger.info("Distribusi kombinasi kondisi_penjualan x prioritas_abc:")
    combo_counts = df.groupby(['kondisi_penjualan', 'prioritas_abc']).size()
    logger.info(f"\n{combo_counts.to_string()}")

    n_fallback = df['rekomendasi'].str.startswith('Rekomendasi tidak tersedia').sum()
    if n_fallback > 0:
        logger.warning(f"{n_fallback} produk mendapat rekomendasi fallback -- cek konsistensi data.")

    return df
