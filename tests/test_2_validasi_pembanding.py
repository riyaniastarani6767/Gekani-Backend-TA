"""
tests/test_2_validasi_pembanding.py
PENGUJIAN TAHAP 2 -- Validasi implementasi manual terhadap pembanding.

Seluruh algoritma di sistem ditulis MANUAL (numpy/pandas, tanpa library
machine learning). Untuk membuktikan kebenarannya, hasil sistem dibandingkan
dengan:
  (a) perhitungan independen menggunakan pandas langsung dari data mentah, dan
  (b) pustaka pembanding scikit-learn.

PENTING: scikit-learn HANYA dipakai di file pengujian ini sebagai "kunci
jawaban", TIDAK dipakai oleh sistem. Sistem tetap 100% implementasi manual.
"""

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip('sklearn')
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from app.services import kmeans, preprocessing


@pytest.fixture(scope='module')
def data_mentah(dataset_path):
    df = pd.read_csv(dataset_path)
    df['year_month'] = pd.to_datetime(df['tanggal'], dayfirst=True).dt.to_period('M')
    return df


# ---------------------------------------------------------------------------
# (a) Agregasi & fitur vs perhitungan pandas langsung
# ---------------------------------------------------------------------------
def test_fitur_agregasi_sama_dengan_perhitungan_langsung(df_agregat, data_mentah):
    g = data_mentah.groupby('sub_kategori')
    sistem = df_agregat.set_index('produk')
    assert (sistem['frekuensi_transaksi'] == g['no_invoice'].nunique()).all()
    assert (sistem['total_kuantitas'] == g['kuantitas_terjual'].sum()).all()
    assert np.allclose(sistem['total_pendapatan'], g['total_sales'].sum())
    assert np.allclose(sistem['avg_harga'], g['harga_per_unit'].mean())
    assert (sistem['n_bulan_aktif'] == g['year_month'].nunique()).all()


def test_jumlah_baris_invoice_produk(df_agregat, data_mentah):
    assert df_agregat.attrs['n_transaksi_setelah_cleaning'] == len(data_mentah)
    assert df_agregat.attrs['n_invoice'] == data_mentah['no_invoice'].nunique()
    assert len(df_agregat) == data_mentah['sub_kategori'].nunique()


# ---------------------------------------------------------------------------
# (b) Standarisasi & PCA vs scikit-learn
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def fitur_setelah_transformasi(df_agregat):
    """Ulangi winsorize + log1p secara independen untuk input pembanding."""
    X = df_agregat[preprocessing.CLUSTER_FEATURES].copy()
    for c in X.columns:
        lo, hi = X[c].quantile([preprocessing.WINSORIZE_LOWER, preprocessing.WINSORIZE_UPPER])
        X[c] = X[c].clip(lo, hi)
        X[c] = np.log1p(X[c] - X[c].min() + 1)
    return X.values.astype(float)


def test_standarisasi_sama_dengan_sklearn_standardscaler(df_agregat, fitur_setelah_transformasi):
    sc = StandardScaler().fit(fitur_setelah_transformasi)
    assert np.allclose(df_agregat.attrs['scale_mean'], sc.mean_)
    assert np.allclose(df_agregat.attrs['scale_std'], sc.scale_)


def test_pca_sama_dengan_sklearn(df_agregat, data_pca, fitur_setelah_transformasi):
    Z = StandardScaler().fit_transform(fitur_setelah_transformasi)
    pca = PCA(n_components=2).fit(Z)
    proyeksi = pca.transform(Z)
    assert np.allclose(df_agregat.attrs['pca_explained_variance'][:2], pca.explained_variance_ratio_)
    # Arah (tanda) eigenvector boleh terbalik -- secara matematis setara
    for i in range(2):
        assert np.allclose(data_pca[:, i], proyeksi[:, i]) or np.allclose(data_pca[:, i], -proyeksi[:, i])


# ---------------------------------------------------------------------------
# (b) Silhouette, WCSS, K-Means vs scikit-learn
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('k', [2, 3, 4, 5])
def test_silhouette_manual_sama_dengan_sklearn(data_pca, k):
    labels, _, _ = kmeans._run_single_kmeans(data_pca, k)
    assert kmeans.compute_silhouette(data_pca, labels) == pytest.approx(silhouette_score(data_pca, labels), abs=1e-9)


@pytest.mark.parametrize('k', [2, 3, 4, 5])
def test_wcss_manual_sama_dengan_inertia_sklearn(data_pca, k):
    # sklearn dijalankan dari centroid akhir sistem: label harus identik
    # dan WCSS manual harus sama dengan inertia_ hasil sklearn.
    labels, centroids, _ = kmeans._run_single_kmeans(data_pca, k)
    km = KMeans(n_clusters=k, init=centroids, n_init=1).fit(data_pca)
    assert adjusted_rand_score(labels, km.labels_) == pytest.approx(1.0)
    assert kmeans.compute_wcss(data_pca, labels, centroids) == pytest.approx(km.inertia_, rel=1e-9)


def test_kmeans_k4_identik_dengan_sklearn(data_pca, hasil_kmeans_k4):
    df_kmeans, _ = hasil_kmeans_k4
    km = KMeans(n_clusters=4, n_init=50, random_state=0).fit(data_pca)
    # ARI = 1.0 berarti pembagian klaster identik (nomor label boleh berbeda)
    assert adjusted_rand_score(df_kmeans['cluster'].values, km.labels_) == pytest.approx(1.0)


def test_k4_tetap_terbaik_dibanding_hasil_terbaik_sklearn(data_pca):
    skor = {k: silhouette_score(data_pca, KMeans(k, n_init=50, random_state=0).fit(data_pca).labels_) for k in range(2, 6)}
    assert max(skor, key=skor.get) == 4


# ---------------------------------------------------------------------------
# (a) ABC Analysis vs perhitungan independen
# ---------------------------------------------------------------------------
def test_abc_sama_dengan_perhitungan_independen(df_agregat, hasil_abc):
    t = df_agregat[['produk', 'total_pendapatan']].sort_values('total_pendapatan', ascending=False).reset_index(drop=True)
    share = t['total_pendapatan'] / t['total_pendapatan'].sum()
    sebelum = share.cumsum() - share
    t['kategori'] = np.where(sebelum < 0.80, 'A', np.where(sebelum < 0.95, 'B', 'C'))
    sistem = hasil_abc.set_index('produk')['prioritas_abc']
    assert (sistem == t.set_index('produk')['kategori']).all()


def test_abc_urutan_monoton(hasil_abc):
    # Produk kategori A selalu berpendapatan >= produk B, dan B >= C
    g = hasil_abc.groupby('prioritas_abc')['total_pendapatan']
    assert g.min()['A'] >= g.max()['B']
    assert g.min()['B'] >= g.max()['C']