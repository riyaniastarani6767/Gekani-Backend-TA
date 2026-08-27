"""
decision_rules.py
Decision rules berdasarkan Tabel 3.5 (Matriks Rekomendasi -- revisi 4x3),
BAB III 7.7. Mengombinasikan kondisi_penjualan (hasil K-Means + labeling)
dengan prioritas_abc (hasil ABC Analysis) untuk menghasilkan rekomendasi
tindak lanjut pengelolaan persediaan.
"""

import logging

logger = logging.getLogger("decision_rules")

DECISION_MATRIX = {
    "Produk Harian": {
        "A": "Stok harus selalu tersedia, karena produk sering dibeli.",
        "B": "Jaga stok tetap tersedia, sesuai kebutuhan pelanggan.",
        "C": "Tetap sediakan stok karena tetap sering dicari, tapi coba naikkan sedikit harganya biar untungnya lebih besar.",
    },
    "Produk Langka": {
        "A": "Tetap sediakan, meskipun jarang dibeli, karena harga barangnya tinggi.",
        "B": "Sediakan dalam jumlah kecil, karena pembeliannya tidak menentu.",
        "C": "Kurangi jumlah stok, karena jarang dibeli dan nilainya kecil.",
    },
    "Produk Andalan": {
        "A": "Utamakan stok, karena produk ini penting dan banyak dibutuhkan pelanggan.",
        "B": "Pertahankan stok, karena produk ini rutin menyumbang pendapatan toko.",
        "C": "Kurangi sedikit stoknya, meski tetap termasuk produk penting.",
    },
    "Produk Premium": {
        "A": "Tetap sediakan stok, tetapi tidak perlu dalam jumlah banyak.",
        "B": "Sediakan dalam jumlah terbatas, karena harganya cukup tinggi.",
        "C": "Kurangi sedikit stoknya, tapi tetap sediakan karena pembeliannya tetap rutin walau jumlahnya kecil.",
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
