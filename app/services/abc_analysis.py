"""
abc_analysis.py
ABC Analysis manual, sesuai BAB III 7.6.3 dan Gambar 3.16.
Alur: persiapan data -> perhitungan nilai total -> pengurutan berdasarkan nilai
      -> perhitungan persentase kumulatif -> klasifikasi A/B/C -> interpretasi.

Independen dari K-Means -- input hanya total_pendapatan per produk.
"""

import logging

logger = logging.getLogger("abc_analysis")

THRESHOLD_A = 0.80  # kumulatif <= 80% -> Kategori A
THRESHOLD_B = 0.95  # kumulatif <= 95% -> Kategori B, sisanya Kategori C


def classify_abc(df, revenue_col='total_pendapatan'):
    """
    Klasifikasi produk ke Kategori A/B/C berdasarkan kontribusi kumulatif
    terhadap total pendapatan (prinsip Pareto / 80-20).

    Penentuan kategori memakai nilai kumulatif SEBELUM produk ditambahkan,
    sehingga produk yang membuat nilai kumulatif melewati suatu threshold
    tetap dimasukkan ke kategori sebelumnya.

    df: pandas.DataFrame, wajib punya kolom 'produk' dan revenue_col
    revenue_col: nama kolom pendapatan (default 'total_pendapatan')

    Return: df (copy) dengan kolom tambahan:
        'persentase_kontribusi', 'persentase_kumulatif', 'prioritas_abc'
    """
    logger.info("Menjalankan ABC Analysis...")

    df = df.copy()
    total_revenue = df[revenue_col].sum()
    logger.info(f"Total pendapatan keseluruhan: Rp{total_revenue:,.0f}")

    if total_revenue <= 0:
        raise ValueError("Total pendapatan harus lebih besar dari 0 untuk ABC Analysis.")

    df = df.sort_values(by=revenue_col, ascending=False).reset_index(drop=True)

    df['persentase_kontribusi'] = df[revenue_col] / total_revenue
    df['persentase_kumulatif'] = df['persentase_kontribusi'].cumsum()

    kumulatif_sebelum = df['persentase_kumulatif'] - df['persentase_kontribusi']

    kategori = []
    for kum_sebelum in kumulatif_sebelum:
        if kum_sebelum < THRESHOLD_A:
            kategori.append('A')
        elif kum_sebelum < THRESHOLD_B:
            kategori.append('B')
        else:
            kategori.append('C')
    df['prioritas_abc'] = kategori

    counts = df['prioritas_abc'].value_counts().to_dict()
    logger.info(f"Distribusi kategori ABC: A={counts.get('A', 0)}, B={counts.get('B', 0)}, C={counts.get('C', 0)}")

    for kat in ['A', 'B', 'C']:
        subset = df[df['prioritas_abc'] == kat]
        if len(subset) > 0:
            kontribusi_kat = subset['persentase_kontribusi'].sum()
            logger.info(f"  Kategori {kat}: {len(subset)} produk, kontribusi {kontribusi_kat*100:.2f}% dari total pendapatan")

    if 'A' in counts and 'B' in counts:
        idx_transisi_ab = df[df['prioritas_abc'] == 'B'].index[0]
        logger.info(
            f"Transisi A->B terjadi pada produk ke-{idx_transisi_ab + 1} "
            f"('{df.loc[idx_transisi_ab, 'produk']}'), kumulatif={df.loc[idx_transisi_ab, 'persentase_kumulatif']*100:.2f}%"
        )
    if 'B' in counts and 'C' in counts:
        idx_transisi_bc = df[df['prioritas_abc'] == 'C'].index[0]
        logger.info(
            f"Transisi B->C terjadi pada produk ke-{idx_transisi_bc + 1} "
            f"('{df.loc[idx_transisi_bc, 'produk']}'), kumulatif={df.loc[idx_transisi_bc, 'persentase_kumulatif']*100:.2f}%"
        )

    return df
