"""
tests/test_3_hasil_penelitian.py
PENGUJIAN TAHAP 3 -- Konsistensi hasil sistem dengan angka yang dilaporkan
di BAB IV, serta kewajaran makna bisnis label klaster.

Jika suatu saat kode diubah dan salah satu angka di bawah ikut berubah,
test ini akan GAGAL -- sehingga angka di skripsi tidak diam-diam tertinggal.
"""

from collections import Counter

import numpy as np
import pytest

from app.services import decision_rules


def test_ringkasan_data(df_agregat):
    assert df_agregat.attrs['n_transaksi_setelah_cleaning'] == 66865  # baris item
    assert df_agregat.attrs['n_invoice'] == 24540                     # transaksi (nota)
    assert len(df_agregat) == 126                                     # produk
    assert df_agregat['total_pendapatan'].sum() == pytest.approx(41_198_092_500)


def test_variansi_pca(df_agregat):
    evr = np.array(df_agregat.attrs['pca_explained_variance'])
    assert evr[:2] * 100 == pytest.approx([71.17, 12.97], abs=0.005)
    assert evr[:2].sum() * 100 == pytest.approx(84.14, abs=0.005)


def test_wcss_per_k(hasil_optimal_k):
    assert hasil_optimal_k['elbow_data']['k'] == [2, 3, 4, 5]
    assert hasil_optimal_k['elbow_data']['wcss'] == pytest.approx([169.16, 73.33, 33.79, 19.53], abs=0.005)


def test_silhouette_per_k_dan_k_optimal(hasil_optimal_k):
    assert hasil_optimal_k['silhouette_data']['scores'] == pytest.approx([0.6060, 0.6546, 0.6608, 0.6268], abs=0.00005)
    assert hasil_optimal_k['optimal_k'] == 4


def test_distribusi_kondisi_penjualan(hasil_labeling):
    assert Counter(hasil_labeling['kondisi_penjualan']) == {
        'Produk Harian': 22, 'Produk Langka': 27, 'Produk Andalan': 56, 'Produk Premium': 21}


def test_distribusi_abc(hasil_abc):
    assert Counter(hasil_abc['prioritas_abc']) == {'A': 77, 'B': 29, 'C': 20}
    kontribusi = hasil_abc.groupby('prioritas_abc')['persentase_kontribusi'].sum()
    assert [kontribusi[k] * 100 for k in 'ABC'] == pytest.approx([80.08, 14.94, 4.98], abs=0.005)


def test_matriks_segmentasi_4x3(hasil_labeling, hasil_abc):
    gabung = hasil_labeling[['produk', 'kondisi_penjualan']].merge(hasil_abc[['produk', 'prioritas_abc']], on='produk')
    matriks = Counter(zip(gabung['kondisi_penjualan'], gabung['prioritas_abc']))
    assert matriks == {
        ('Produk Harian', 'A'): 3, ('Produk Harian', 'B'): 4, ('Produk Harian', 'C'): 15,
        ('Produk Langka', 'A'): 14, ('Produk Langka', 'B'): 10, ('Produk Langka', 'C'): 3,
        ('Produk Andalan', 'A'): 44, ('Produk Andalan', 'B'): 10, ('Produk Andalan', 'C'): 2,
        ('Produk Premium', 'A'): 16, ('Produk Premium', 'B'): 5,
    }


def test_semua_produk_mendapat_rekomendasi(hasil_labeling, hasil_abc):
    gabung = hasil_labeling[['produk', 'kondisi_penjualan']].merge(hasil_abc[['produk', 'prioritas_abc']], on='produk')
    hasil = decision_rules.apply_decision_rules(gabung)
    assert not hasil['rekomendasi'].str.startswith('Rekomendasi tidak tersedia').any()


# ---------------------------------------------------------------------------
# Kewajaran makna bisnis label (labeling.py)
# ---------------------------------------------------------------------------
def test_label_bijective_empat_nama_berbeda(hasil_labeling):
    assert hasil_labeling.groupby('cluster')['kondisi_penjualan'].nunique().eq(1).all()
    assert hasil_labeling['kondisi_penjualan'].nunique() == 4


def test_produk_harian_paling_sering_dibeli_dan_termurah(hasil_labeling):
    m = hasil_labeling.groupby('kondisi_penjualan')[['frekuensi_transaksi', 'avg_harga']].median()
    assert m['frekuensi_transaksi'].idxmax() == 'Produk Harian'
    assert m['avg_harga'].idxmin() == 'Produk Harian'


def test_produk_langka_paling_jarang_dibeli_dan_termahal(hasil_labeling):
    m = hasil_labeling.groupby('kondisi_penjualan')[['frekuensi_transaksi', 'avg_harga']].median()
    assert m['frekuensi_transaksi'].idxmin() == 'Produk Langka'
    assert m['avg_harga'].idxmax() == 'Produk Langka'


def test_premium_lebih_mahal_dari_andalan(hasil_labeling):
    m = hasil_labeling.groupby('kondisi_penjualan')['avg_harga'].median()
    assert m['Produk Premium'] > m['Produk Andalan'] > m['Produk Harian']