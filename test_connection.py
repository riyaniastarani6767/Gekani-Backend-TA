from pymongo import MongoClient
from dotenv import load_dotenv
import os
import certifi

load_dotenv()
uri = os.getenv('MONGO_URI')

print(f"Mencoba konek ke: {uri[:40]}...")

# Pakai TLS cuma kalau ini koneksi ke Atlas (mongodb+srv://)
# Localhost MongoDB jalan tanpa SSL, jadi tidak perlu TLS
if uri.startswith('mongodb+srv://'):
    client = MongoClient(uri, tlsCAFile=certifi.where())
else:
    client = MongoClient(uri)

try:
    client.admin.command('ping')
    print('Koneksi ke MongoDB berhasil!')
except Exception as e:
    print('Gagal konek:', e)