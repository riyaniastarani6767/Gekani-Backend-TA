"""
tests/test_1_perhitungan_manual.py
PENGUJIAN TAHAP 1 -- Contoh kecil yang bisa dihitung dengan tangan.

Setiap test memakai data kecil (2-5 titik/produk) yang jawabannya dapat
dihitung manual di kertas, lalu dibandingkan dengan output fungsi sistem.
Tujuannya membuktikan bahwa setiap RUMUS diimplementasikan dengan benar,
terlepas dari dataset penelitian.
"""

import numpy as np
import pandas as pd
import pytest

from app.services import kmeans, abc_analysis, decision_rules, preprocessing
from app.services.pca import pca_fit


# ---------------------------------------------------------------------------
# Jarak Euclidean (Persamaan 2.2)
# ---------------------------------------------------------------------------
def test_jarak_euclidean_segitiga_3_4_5():
    # sqrt((3-0)^2 + (4-0)^2) = sqrt(25) = 5
    assert kmeans.euclidean_distance(np.array([0, 0]), np.array([3, 4])) == pytest.approx(5.0)


def test_jarak_euclidean_titik_sama_bernilai_nol():
    assert kmeans.euclidean_distance(np.array([2, 7]), np.array([2, 7])) == 0


# ---------------------------------------------------------------------------
# Standarisasi z-score
# ---------------------------------------------------------------------------
def test_standarisasi_zscore():
    # data [1, 2, 3] -> mean 2, std populasi sqrt(2/3) = 0.8165
    # z = (x - 2) / 0.8165 -> [-1.2247, 0, 1.2247]
    z, mean, std = preprocessing.standardize(np.array([[1.0], [2.0], [3.0]]))
    assert mean[0] == pytest.approx(2.0)
    assert std[0] == pytest.approx(np.sqrt(2 / 3))
    assert z.ravel() == pytest.approx([-1.2247449, 0.0, 1.2247449])


def test_standarisasi_fitur_konstan_tidak_error():
    # std = 0 diganti 1 supaya tidak terjadi pembagian dengan nol
    z, _, std = preprocessing.standardize(np.array([[5.0], [5.0], [5.0]]))
    assert std[0] == 1
    assert np.all(z == 0)


# ---------------------------------------------------------------------------
# WCSS / Elbow (Persamaan 2.4)
# ---------------------------------------------------------------------------
def test_wcss_contoh_manual():
    # Dua klaster, centroid (0,1) dan (10,1). Setiap titik berjarak 1 dari
    # centroid-nya -> jarak^2 = 1 -> WCSS = 1 + 1 + 1 + 1 = 4
    data = np.array([[0, 0], [0, 2], [10, 0], [10, 2]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    centroids = np.array([[0, 1], [10, 1]], dtype=float)
    assert kmeans.compute_wcss(data, labels, centroids) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Silhouette Score (Persamaan 2.3)
# ---------------------------------------------------------------------------
def test_silhouette_contoh_manual():
    # Titik (0,0): a = jarak ke (0,2) = 2
    #              b = rata-rata jarak ke (10,0) & (10,2) = (10 + sqrt(104)) / 2 = 10.0990
    #              s = (b - a) / max(a, b) = (10.0990 - 2) / 10.0990 = 0.8020
    # Karena simetris, keempat titik memiliki s yang sama -> Silhouette = 0.8020
    data = np.array([[0, 0], [0, 2], [10, 0], [10, 2]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    b = (10 + np.sqrt(104)) / 2
    diharapkan = (b - 2) / b
    assert kmeans.compute_silhouette(data, labels) == pytest.approx(diharapkan)


def test_silhouette_titik_kembar_tetap_dihitung():
    # Kasus bug lama: dua produk dengan koordinat SAMA dalam satu klaster.
    # Titik (0,0) pertama: a = jarak ke (0,0) kedua = 0, b = 4 -> s = 1.
    # Kode lama ikut membuang titik kembar -> a tidak terdefinisi (NaN).
    data = np.array([[0, 0], [0, 0], [4, 0], [4, 0]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    assert kmeans.compute_silhouette(data, labels) == pytest.approx(1.0)


def test_silhouette_klaster_satu_anggota_bernilai_nol():
    # Definisi Rousseeuw (1987): s(i) = 0 bila klaster hanya berisi 1 anggota.
    # Titik (0,0): a = 1, b = jarak ke (5,5) = 7.0711 -> s = 0.8586
    # Titik (0,1): a = 1, b = jarak ke (5,5) = 6.4031 -> s = 0.8438
    # Titik (5,5): klaster tunggal -> s = 0
    # Silhouette = (0.8586 + 0.8438 + 0) / 3 = 0.5675
    data = np.array([[0, 0], [0, 1], [5, 5]], dtype=float)
    labels = np.array([0, 0, 1])
    s1 = (np.sqrt(50) - 1) / np.sqrt(50)
    s2 = (np.sqrt(41) - 1) / np.sqrt(41)
    assert kmeans.compute_silhouette(data, labels) == pytest.approx((s1 + s2 + 0) / 3)


def test_silhouette_satu_klaster_saja_bernilai_nol():
    data = np.array([[0, 0], [1, 1]], dtype=float)
    assert kmeans.compute_silhouette(data, np.array([0, 0])) == 0.0


# ---------------------------------------------------------------------------
# K-Means (inisialisasi K-Means++, iterasi, konvergensi)
# ---------------------------------------------------------------------------
def test_kmeans_memisahkan_dua_kelompok_jelas():
    # Kelompok kiri di sekitar (0,0), kelompok kanan di sekitar (10,10)
    kiri = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    kanan = kiri + 10
    data = np.vstack([kiri, kanan])
    labels, centroids, n_iter = kmeans._run_single_kmeans(data, 2)
    assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1
    assert labels[0] != labels[4]
    # centroid = rata-rata anggota: (0.5, 0.5) dan (10.5, 10.5)
    assert sorted(map(tuple, np.round(centroids, 6))) == [(0.5, 0.5), (10.5, 10.5)]
    assert n_iter <= 100


def test_centroid_adalah_rata_rata_anggota():
    # Persamaan 2.1: v_ij = (1/N_i) * sum(x_kj)
    data = np.array([[0, 0], [2, 4], [10, 10], [12, 14]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    c = kmeans.update_centroids(data, labels, 2)
    assert c[0] == pytest.approx([1, 2])
    assert c[1] == pytest.approx([11, 12])


def test_kmeans_plus_plus_selalu_menghasilkan_k_centroid():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(60, 2))
    for seed in range(100):
        for k in range(2, 6):
            c = kmeans.initialize_centroids(data, k, seed=seed)
            assert len(c) == k, f"seed={seed}, k={k}"


def test_kmeans_reproducible_dengan_seed_sama():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(40, 2))
    l1, _, _ = kmeans._run_single_kmeans(data, 3, seed=42)
    l2, _, _ = kmeans._run_single_kmeans(data, 3, seed=42)
    assert np.array_equal(l1, l2)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def test_pca_titik_segaris_satu_komponen_100_persen():
    # Semua titik berada di garis y = x -> seluruh variansi ada di 1 komponen
    x = np.array([-2, -1, 0, 1, 2], dtype=float)
    data = np.column_stack([x, x])
    hasil = pca_fit(data, n_components=2)
    assert hasil['explained_variance_ratio'][0] == pytest.approx(1.0)
    assert hasil['explained_variance_ratio'][1] == pytest.approx(0.0, abs=1e-12)


def test_pca_variansi_berurutan_menurun_dan_total_100_persen():
    rng = np.random.default_rng(2)
    hasil = pca_fit(rng.normal(size=(50, 5)), n_components=2)
    evr = hasil['explained_variance_ratio']
    assert np.all(np.diff(evr) <= 1e-12)
    assert evr.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ABC Analysis (Pareto 80/15/5)
# ---------------------------------------------------------------------------
def test_abc_contoh_manual_lima_produk():
    # Pendapatan: P1=500, P2=300, P3=100, P4=60, P5=40 -> total 1000
    # Kontribusi : 50%, 30%, 10%, 6%, 4%
    # Kumulatif SEBELUM produk: 0%, 50%, 80%, 90%, 96%
    #   P1: 0%  < 80% -> A      P2: 50% < 80% -> A
    #   P3: 80% (tidak < 80%, < 95%) -> B
    #   P4: 90% < 95% -> B      P5: 96% >= 95% -> C
    df = pd.DataFrame({'produk': ['P3', 'P1', 'P5', 'P2', 'P4'],
                       'total_pendapatan': [100, 500, 40, 300, 60]})
    hasil = abc_analysis.classify_abc(df).set_index('produk')['prioritas_abc'].to_dict()
    assert hasil == {'P1': 'A', 'P2': 'A', 'P3': 'B', 'P4': 'B', 'P5': 'C'}


def test_abc_kontribusi_per_kategori():
    df = pd.DataFrame({'produk': list('abcde'), 'total_pendapatan': [500, 300, 100, 60, 40]})
    hasil = abc_analysis.classify_abc(df)
    kontribusi = hasil.groupby('prioritas_abc')['persentase_kontribusi'].sum()
    assert kontribusi['A'] == pytest.approx(0.80)
    assert kontribusi['B'] == pytest.approx(0.16)
    assert kontribusi['C'] == pytest.approx(0.04)


def test_abc_total_pendapatan_nol_ditolak():
    df = pd.DataFrame({'produk': ['a', 'b'], 'total_pendapatan': [0, 0]})
    with pytest.raises(ValueError):
        abc_analysis.classify_abc(df)


# ---------------------------------------------------------------------------
# Decision rules (Tabel 3.5)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kondisi', ['Produk Harian', 'Produk Langka', 'Produk Andalan', 'Produk Premium'])
@pytest.mark.parametrize('abc', ['A', 'B', 'C'])
def test_decision_rules_12_kombinasi_terdefinisi(kondisi, abc):
    rekomendasi = decision_rules.get_rekomendasi(kondisi, abc)
    assert rekomendasi == decision_rules.DECISION_MATRIX[kondisi][abc]
    assert not rekomendasi.startswith('Rekomendasi tidak tersedia')


def test_decision_rules_input_tidak_dikenal_memberi_fallback():
    assert decision_rules.get_rekomendasi('Produk X', 'A').startswith('Rekomendasi tidak tersedia')
    assert decision_rules.get_rekomendasi('Produk Harian', 'Z').startswith('Rekomendasi tidak tersedia')


# ---------------------------------------------------------------------------
# Agregasi transaksi -> produk (feature engineering)
# ---------------------------------------------------------------------------
def test_agregasi_fitur_dari_csv_kecil(tmp_path):
    # 3 invoice, 5 baris. Produk "Oli" muncul di INV1 (2 baris) dan INV2
    # -> frekuensi_transaksi = 2 invoice unik (BUKAN 3 baris)
    csv = tmp_path / 'mini.csv'
    csv.write_text(
        'no_invoice,tanggal,bulan,tahun,kategori,sub_kategori,kuantitas_terjual,harga_per_unit,total_sales\n'
        'INV1,01-01-2021,Januari,2021,Oli,Oli,2,40000,80000\n'
        'INV1,01-01-2021,Januari,2021,Oli,Oli,1,40000,40000\n'
        'INV1,01-01-2021,Januari,2021,Busi,Busi,1,20000,20000\n'
        'INV2,15-02-2021,Februari,2021,Oli,Oli,3,42000,126000\n'
        'INV3,20-02-2021,Februari,2021,Ban,Ban,1,300000,300000\n'
    )
    df = preprocessing.aggregate_data(str(csv)).set_index('produk')
    oli = df.loc['Oli']
    assert oli['frekuensi_transaksi'] == 2           # INV1, INV2
    assert oli['total_kuantitas'] == 6               # 2 + 1 + 3
    assert oli['avg_qty_per_transaksi'] == pytest.approx(3.0)   # 6 / 2
    assert oli['total_pendapatan'] == 246000         # 80000 + 40000 + 126000
    assert oli['avg_harga'] == pytest.approx((40000 + 40000 + 42000) / 3)
    assert oli['n_bulan_aktif'] == 2                 # Januari & Februari
    assert df.attrs['n_invoice'] == 3
    assert df.attrs['n_transaksi_setelah_cleaning'] == 5