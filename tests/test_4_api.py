"""
tests/test_4_api.py
PENGUJIAN TAHAP 4 -- Pengujian endpoint API (alur sistem dari sisi backend).

Menggunakan database MongoDB tiruan (mongomock) di memori, sehingga pengujian
TIDAK menyentuh database asli dan tidak perlu server MongoDB berjalan.
File log & upload selama pengujian disimpan di folder sementara.
"""

import datetime
import io
import logging
import os
from collections import Counter

import pytest

mongomock = pytest.importorskip('mongomock')


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    kerja = tmp_path_factory.mktemp('api')
    cwd_lama = os.getcwd()
    os.chdir(kerja)  # folder logs/ dibuat di sini, bukan di project
    logging.disable(logging.NOTSET)  # log analisis perlu ditulis untuk diuji
    os.environ.update(MONGO_URI='mongodb://localhost/uji', SECRET_KEY='uji',
                      UPLOAD_FOLDER=str(kerja / 'uploads'))

    import app as appmod
    import app.extensions as ext
    from app.config import Config
    Config.MONGO_URI, Config.SECRET_KEY, Config.UPLOAD_FOLDER = os.environ['MONGO_URI'], 'uji', os.environ['UPLOAD_FOLDER']

    mc = mongomock.MongoClient()

    def init_palsu(app):
        ext.mongo_client, ext._db = mc, mc['uji']
        return ext._db

    asli = appmod.init_mongo
    appmod.init_mongo = init_palsu
    try:
        app = appmod.create_app()
    finally:
        appmod.init_mongo = asli

    from werkzeug.security import generate_password_hash
    ext.get_db().users.insert_one({'username': 'admin', 'password': generate_password_hash('rahasia')})
    yield app.test_client()
    os.chdir(cwd_lama)
    logging.disable(logging.CRITICAL)


@pytest.fixture(scope='module')
def hasil_analisis(client, dataset_path):
    with open(dataset_path, 'rb') as f:
        up = client.post('/api/upload', data={'file': (f, 'dataPenjualanYudiMotor.csv')},
                         content_type='multipart/form-data').get_json()
    r = client.post('/api/analyze', json={'filepath': up['filepath'], 'nama_file_asli': up['nama_file_asli'],
                                          'tahun_awal': up['tahun_min'], 'tahun_akhir': up['tahun_max'], 'user_id': 'u1'})
    assert r.status_code == 200, r.get_json()
    return up, r.get_json()


# ---------------------------- Login ----------------------------------------
def test_login_valid(client):
    r = client.post('/api/auth/login', json={'username': 'admin', 'password': 'rahasia'})
    assert r.status_code == 200 and r.get_json()['user']['username'] == 'admin'


def test_login_password_salah(client):
    r = client.post('/api/auth/login', json={'username': 'admin', 'password': 'salah'})
    assert r.status_code == 401 and 'salah' in r.get_json()['error']


def test_login_kosong(client):
    r = client.post('/api/auth/login', json={'username': '', 'password': ''})
    assert r.status_code == 400 and 'wajib' in r.get_json()['error']


# ---------------------------- Upload ---------------------------------------
@pytest.mark.parametrize('nama', ['data.xlsx', 'data.pdf', 'data.txt'])
def test_upload_format_selain_csv_ditolak(client, nama):
    r = client.post('/api/upload', data={'file': (io.BytesIO(b'a,b\n1,2'), nama)}, content_type='multipart/form-data')
    assert r.status_code == 400 and 'Gunakan .csv' in r.get_json()['error']


def test_upload_tanpa_file(client):
    r = client.post('/api/upload', data={}, content_type='multipart/form-data')
    assert r.status_code == 400


def test_upload_lebih_dari_50mb_ditolak(client):
    besar = io.BytesIO(b'x' * (51 * 1024 * 1024))
    r = client.post('/api/upload', data={'file': (besar, 'besar.csv')}, content_type='multipart/form-data')
    assert r.status_code == 413 and '50MB' in r.get_json()['error']


def test_upload_csv_valid(hasil_analisis):
    up, _ = hasil_analisis
    assert up['total_rows'] == 66865
    assert up['total_invoice'] == 24540
    assert (up['tahun_min'], up['tahun_max']) == (2021, 2025)


def test_preview(client, hasil_analisis):
    up, _ = hasil_analisis
    r = client.post('/api/preview', json={'filepath': up['filepath'], 'tahun_awal': 2021, 'tahun_akhir': 2025})
    assert r.status_code == 200 and len(r.get_json()['sample_rows']) == 10


# ---------------------------- Analisis -------------------------------------
def test_analisis_berhasil_dan_sesuai_bab_iv(hasil_analisis):
    _, an = hasil_analisis
    assert an['status'] == 'Berhasil'
    assert an['jumlah_produk'] == 126
    assert an['jumlah_invoice'] == 24540
    assert an['optimal_k'] == 4
    assert round(an['silhouette_score'], 4) == 0.6608
    assert an['periode_data'] == '2021-2025'
    assert Counter(p['kondisi_penjualan'] for p in an['hasil_segmentasi']) == {
        'Produk Harian': 22, 'Produk Langka': 27, 'Produk Andalan': 56, 'Produk Premium': 21}


def test_tren_bulanan_konsisten_dengan_total(hasil_analisis):
    _, an = hasil_analisis
    assert len(an['tren_bulanan']) == 60
    total_tren = sum(t['total'] for t in an['tren_bulanan'])
    total_produk = sum(p['total_penjualan'] for p in an['hasil_segmentasi'])
    assert total_tren == pytest.approx(total_produk)


def test_log_analisis_mencatat_evaluasi(hasil_analisis):
    _, an = hasil_analisis
    isi = open(an['log_file'], encoding='utf-8').read()
    assert '[pca]' in isi and '84.14%' in isi
    assert 'Silhouette=0.6608' in isi


def test_analisis_file_tidak_ada(client):
    r = client.post('/api/analyze', json={'filepath': '/tidak/ada.csv'})
    assert r.status_code == 400


# ---------------------------- Dashboard & Produk ---------------------------
def test_dashboard_summary(client, hasil_analisis):
    r = client.get('/api/dashboard-summary').get_json()
    assert r['jumlah_produk'] == 126
    assert r['created_at'].endswith('Z')  # zona waktu UTC ditandai


@pytest.mark.parametrize('filter_, jumlah', [
    ({}, 126),
    ({'kondisi': 'Produk Harian', 'prioritas': 'C'}, 15),
    ({'kategori': 'Dinamo'}, 3),
    ({'prioritas': 'A'}, 77),
])
def test_filter_data_produk(client, hasil_analisis, filter_, jumlah):
    assert client.get('/api/products', query_string=filter_).get_json()['total'] == jumlah


def test_data_per_bulan(client, hasil_analisis):
    r = client.get('/api/products/monthly', query_string={'bulan_awal': '2025-01', 'bulan_akhir': '2025-12'})
    assert r.status_code == 200 and len(r.get_json()['months']) == 12


# ---------------------------- Riwayat --------------------------------------
def test_riwayat_dan_zona_waktu(client, hasil_analisis):
    h = client.get('/api/history').get_json()['history']
    assert len(h) >= 1 and h[0]['created_at'].endswith('Z')
    waktu = datetime.datetime.fromisoformat(h[0]['created_at'].rstrip('Z'))
    assert abs((datetime.datetime.utcnow() - waktu).total_seconds()) < 600


def test_riwayat_id_tidak_valid(client):
    assert client.get('/api/history/bukan-id').status_code == 400
    assert client.get('/api/history/0123456789abcdef01234567').status_code == 404


def test_hapus_riwayat(client, hasil_analisis):
    _, an = hasil_analisis
    assert client.delete(f"/api/history/{an['_id']}").status_code == 200
    ids = [d['_id'] for d in client.get('/api/history').get_json()['history']]
    assert an['_id'] not in ids