"""
blueprints/analysis.py
Endpoint utama sistem: upload dataset, preview, jalankan analisis,
dashboard summary, daftar produk, riwayat analisis.

Logic berat (preprocessing, K-Means, ABC, decision rules) TIDAK ditulis
di sini -- hanya dipanggil dari app/services/. Blueprint ini murni
orkestrasi + HTTP request/response handling.
"""

import os
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app

from app.extensions import get_db
from app.utils.logger import setup_analysis_logger
from app.services import preprocessing, kmeans, labeling, abc_analysis, decision_rules

analysis_bp = Blueprint('analysis', __name__)

ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}


def _allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@analysis_bp.route('/upload', methods=['POST'])
def upload_dataset():
    """
    Terima file csv/xlsx, simpan ke UPLOAD_FOLDER, return path + info dasar.
    Frontend memakai path yang dikembalikan untuk memanggil /preview.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nama file kosong."}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung. Gunakan .csv atau .xlsx."}), 400

    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(upload_folder, unique_name)
    file.save(filepath)

    try:
        info = preprocessing.preview_data(filepath, n=10)
    except Exception as e:
        os.remove(filepath)
        return jsonify({"error": f"Gagal membaca file: {str(e)}"}), 400

    return jsonify({
        "filepath": filepath,
        "nama_file_asli": file.filename,
        "columns": info["columns"],
        "total_rows": info["total_rows"],
        "tahun_min": info["tahun_min"],
        "tahun_max": info["tahun_max"],
    }), 200


@analysis_bp.route('/preview', methods=['POST'])
def preview_dataset():
    """
    Body JSON: { "filepath": "...", "tahun_awal": 2021, "tahun_akhir": 2025 }
    Return preview 10 baris + jumlah transaksi sesuai rentang tahun dipilih.
    """
    data = request.get_json()
    filepath = data.get('filepath')
    tahun_awal = data.get('tahun_awal')
    tahun_akhir = data.get('tahun_akhir')

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File tidak ditemukan. Silakan unggah ulang."}), 400

    try:
        info = preprocessing.preview_data(filepath, n=10)
        jumlah_transaksi = preprocessing.count_transaksi_in_range(filepath, tahun_awal, tahun_akhir)
    except Exception as e:
        return jsonify({"error": f"Gagal memproses preview: {str(e)}"}), 400

    return jsonify({
        "columns": info["columns"],
        "sample_rows": info["sample_rows"],
        "jumlah_transaksi_dipilih": jumlah_transaksi,
        "tahun_min": info["tahun_min"],
        "tahun_max": info["tahun_max"],
    }), 200


@analysis_bp.route('/analyze', methods=['POST'])
def run_analysis():
    """
    Body JSON: { "filepath": "...", "nama_file_asli": "...",
                 "tahun_awal": 2021, "tahun_akhir": 2025, "user_id": "..." }

    Menjalankan pipeline penuh: preprocessing -> K-Means -> labeling
    -> ABC Analysis -> decision rules -> simpan ke MongoDB.
    """
    data = request.get_json()
    filepath = data.get('filepath')
    nama_file_asli = data.get('nama_file_asli', os.path.basename(filepath) if filepath else '-')
    tahun_awal = data.get('tahun_awal')
    tahun_akhir = data.get('tahun_akhir')
    user_id = data.get('user_id')

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File tidak ditemukan. Silakan unggah ulang."}), 400

    logger, log_filepath = setup_analysis_logger()
    logger.info(f"=== Mulai analisis: {nama_file_asli} | periode {tahun_awal}-{tahun_akhir} ===")

    analysis_doc = {
        "nama_file": nama_file_asli,
        "filepath": filepath,
        "periode_data": f"{tahun_awal}-{tahun_akhir}",
        "created_at": datetime.utcnow(),
        "user_id": user_id,
        "status": "Gagal",
        "log_file": log_filepath,
    }

    try:
        df = preprocessing.aggregate_data(filepath, tahun_awal=tahun_awal, tahun_akhir=tahun_akhir)
        # feature_cols_raw: 5 fitur asli, dipakai labeling.py untuk bikin arketipe
        feature_cols_raw = df.attrs['cluster_features']
        # feature_cols_pca: 2 komponen utama hasil PCA, INI yang dipakai K-Means
        feature_cols_pca = df.attrs['pca_feature_cols']
        jumlah_transaksi = df.attrs['n_transaksi_setelah_cleaning']

        tren_bulanan = preprocessing.monthly_revenue(filepath, tahun_awal=tahun_awal, tahun_akhir=tahun_akhir)

        optimal_result = kmeans.find_optimal_k(df, feature_cols_pca)
        df_kmeans, centroids = kmeans.run_kmeans(df, feature_cols_pca, optimal_result['optimal_k'])

        df_labeled = labeling.label_clusters(df_kmeans, feature_cols_raw, centroids)

        df_abc = abc_analysis.classify_abc(df)

        df_gabungan = df_labeled[['produk', 'kategori', 'kondisi_penjualan']].merge(
            df_abc[['produk', 'total_pendapatan', 'prioritas_abc']], on='produk'
        )

        info_tambahan = df[['produk', 'frekuensi_transaksi']].copy()
        info_tambahan = info_tambahan.merge(
            df[['produk', 'avg_qty_per_transaksi']], on='produk'
        )
        info_tambahan['total_terjual'] = (
            info_tambahan['avg_qty_per_transaksi'] * info_tambahan['frekuensi_transaksi']
        ).round().astype(int)

        df_gabungan = df_gabungan.merge(
            info_tambahan[['produk', 'frekuensi_transaksi', 'total_terjual']], on='produk'
        )

        df_final = decision_rules.apply_decision_rules(df_gabungan)

        hasil_segmentasi = []
        for _, row in df_final.iterrows():
            hasil_segmentasi.append({
                "nama_produk": row['produk'],
                "kategori": row['kategori'],
                "kondisi_penjualan": row['kondisi_penjualan'],
                "prioritas_abc": row['prioritas_abc'],
                "total_terjual": int(row['total_terjual']),
                "frekuensi_transaksi": int(row['frekuensi_transaksi']),
                "total_penjualan": float(row['total_pendapatan']),
                "rekomendasi": row['rekomendasi'],
            })

        analysis_doc.update({
            "status": "Berhasil",
            "jumlah_produk": len(hasil_segmentasi),
            "jumlah_transaksi": int(jumlah_transaksi),
            "optimal_k": optimal_result['optimal_k'],
            "silhouette_score": max(optimal_result['silhouette_data']['scores']),
            "hasil_segmentasi": hasil_segmentasi,
            "tren_bulanan": tren_bulanan,
        })

        logger.info(f"=== Analisis selesai. {len(hasil_segmentasi)} produk berhasil dianalisis. ===")

    except Exception as e:
        logger.error(f"Analisis gagal: {str(e)}")
        analysis_doc["error_message"] = str(e)
        get_db().analyses.insert_one(analysis_doc)
        return jsonify({"error": f"Analisis gagal: {str(e)}"}), 500

    result = get_db().analyses.insert_one(analysis_doc)
    analysis_doc['_id'] = str(result.inserted_id)

    return jsonify(analysis_doc), 200


@analysis_bp.route('/dashboard-summary', methods=['GET'])
def dashboard_summary():
    """Ringkasan dari analisis TERAKHIR yang berhasil (dipakai halaman Dashboard)."""
    latest = get_db().analyses.find_one({"status": "Berhasil"}, sort=[("created_at", -1)])
    if latest is None:
        return jsonify({"error": "Belum ada analisis yang berhasil dijalankan."}), 404

    hasil = latest.get('hasil_segmentasi', [])

    summary = {
        "jumlah_produk": len(hasil),
        "total_penjualan": sum(p['total_penjualan'] for p in hasil),
        "komposisi_abc": {
            "A": sum(1 for p in hasil if p['prioritas_abc'] == 'A'),
            "B": sum(1 for p in hasil if p['prioritas_abc'] == 'B'),
            "C": sum(1 for p in hasil if p['prioritas_abc'] == 'C'),
        },
        "komposisi_kondisi": {},
        "periode_data": latest.get('periode_data'),
        "created_at": latest.get('created_at').isoformat() if latest.get('created_at') else None,
        "tren_bulanan": latest.get('tren_bulanan', []),
    }

    for p in hasil:
        kondisi = p['kondisi_penjualan']
        summary["komposisi_kondisi"][kondisi] = summary["komposisi_kondisi"].get(kondisi, 0) + 1

    return jsonify(summary), 200


@analysis_bp.route('/products', methods=['GET'])
def get_products():
    """
    Query params (opsional): kategori, kondisi, prioritas, search
    Selalu ambil dari analisis TERAKHIR yang berhasil.
    """
    latest = get_db().analyses.find_one({"status": "Berhasil"}, sort=[("created_at", -1)])
    if latest is None:
        return jsonify({"error": "Belum ada analisis yang berhasil dijalankan."}), 404

    produk_list = latest.get('hasil_segmentasi', [])

    kategori = request.args.get('kategori')
    kondisi = request.args.get('kondisi')
    prioritas = request.args.get('prioritas')
    search = request.args.get('search', '').lower()

    if kategori:
        produk_list = [p for p in produk_list if p['kategori'] == kategori]
    if kondisi:
        produk_list = [p for p in produk_list if p['kondisi_penjualan'] == kondisi]
    if prioritas:
        produk_list = [p for p in produk_list if p['prioritas_abc'] == prioritas]
    if search:
        produk_list = [p for p in produk_list if search in p['nama_produk'].lower()]

    return jsonify({"products": produk_list, "total": len(produk_list)}), 200


@analysis_bp.route('/history', methods=['GET'])
def get_history():
    """List semua analisis (terbaru dulu), tanpa detail hasil_segmentasi penuh."""
    docs = list(get_db().analyses.find({}, {"hasil_segmentasi": 0}).sort("created_at", -1))
    for doc in docs:
        doc['_id'] = str(doc['_id'])
        if doc.get('created_at'):
            doc['created_at'] = doc['created_at'].isoformat()
    return jsonify({"history": docs}), 200


@analysis_bp.route('/history/<analysis_id>', methods=['GET'])
def get_history_detail(analysis_id):
    """Buka kembali hasil analisis spesifik dari riwayat."""
    from bson.objectid import ObjectId
    try:
        doc = get_db().analyses.find_one({"_id": ObjectId(analysis_id)})
    except Exception:
        return jsonify({"error": "ID analisis tidak valid."}), 400

    if doc is None:
        return jsonify({"error": "Analisis tidak ditemukan."}), 404

    doc['_id'] = str(doc['_id'])
    if doc.get('created_at'):
        doc['created_at'] = doc['created_at'].isoformat()

    return jsonify(doc), 200
# Tambahkan di app/blueprints/analysis.py, setelah route
# @analysis_bp.route('/history/<analysis_id>', methods=['GET']) yang sudah ada:

@analysis_bp.route('/history/<analysis_id>', methods=['DELETE'])
def delete_history(analysis_id):
    """Hapus satu riwayat analisis (dan filenya kalau masih ada) dari database."""
    from bson.objectid import ObjectId
    try:
        oid = ObjectId(analysis_id)
    except Exception:
        return jsonify({"error": "ID analisis tidak valid."}), 400

    doc = get_db().analyses.find_one({"_id": oid})
    if doc is None:
        return jsonify({"error": "Analisis tidak ditemukan."}), 404

    result = get_db().analyses.delete_one({"_id": oid})
    if result.deleted_count == 0:
        return jsonify({"error": "Gagal menghapus analisis."}), 500

    return jsonify({"status": "ok", "deleted_id": analysis_id}), 200

@analysis_bp.route('/ping', methods=['GET'])
def ping():
    return {"status": "analysis blueprint aktif"}
@analysis_bp.route('/products/monthly', methods=['GET'])
def get_products_monthly():
    """
    Query params (wajib): bulan_awal, bulan_akhir (format 'YYYY-MM')
    Query params (opsional): kategori, kondisi, prioritas, search

    Membaca ULANG file dataset asli dari analisis terakhir yang berhasil,
    lalu menghitung total_sales per produk per bulan pada rentang yang
    diminta. Berbeda dari /products (yang membaca ringkasan tersimpan),
    endpoint ini butuh data mentah karena breakdown per bulan tidak
    disimpan di hasil_segmentasi.
    """
    bulan_awal = request.args.get('bulan_awal')
    bulan_akhir = request.args.get('bulan_akhir')
    kategori = request.args.get('kategori')
    kondisi = request.args.get('kondisi')
    prioritas = request.args.get('prioritas')
    search = request.args.get('search', '').lower()

    if not bulan_awal or not bulan_akhir:
        return jsonify({"error": "Parameter bulan_awal dan bulan_akhir wajib diisi."}), 400

    latest = get_db().analyses.find_one({"status": "Berhasil"}, sort=[("created_at", -1)])
    if latest is None:
        return jsonify({"error": "Belum ada analisis yang berhasil dijalankan."}), 404

    filepath = latest.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File dataset asli tidak ditemukan. Silakan jalankan analisis ulang."}), 404

    try:
        monthly_result = preprocessing.monthly_products(filepath, bulan_awal, bulan_akhir)
    except Exception as e:
        return jsonify({"error": f"Gagal memproses data bulanan: {str(e)}"}), 400

    produk_list = latest.get('hasil_segmentasi', [])
    produk_lookup = {p['nama_produk']: p for p in produk_list}

    result_products = []
    for nama_produk, monthly_data in monthly_result['data'].items():
        info = produk_lookup.get(nama_produk)
        if info is None:
            continue

        if kategori and info['kategori'] != kategori:
            continue
        if kondisi and info['kondisi_penjualan'] != kondisi:
            continue
        if prioritas and info['prioritas_abc'] != prioritas:
            continue
        if search and search not in nama_produk.lower():
            continue

        result_products.append({
            "nama_produk": nama_produk,
            "kategori": info['kategori'],
            "kondisi_penjualan": info['kondisi_penjualan'],
            "prioritas_abc": info['prioritas_abc'],
            "monthly": monthly_data,
            "total": sum(monthly_data.values()),
        })

    result_products.sort(key=lambda p: p['nama_produk'])

    return jsonify({"months": monthly_result['months'], "products": result_products}), 200