"""
create_admin.py
Script one-time untuk membuat akun admin pertama (dijalankan manual sekali
lewat terminal, bukan lewat endpoint API -- sistem ini tidak punya
registrasi publik, sesuai teks UI Login).

Cara pakai:
    python create_admin.py
"""

from werkzeug.security import generate_password_hash
from app import create_app
from app.extensions import get_db


def create_admin():
    app = create_app()

    with app.app_context():
        username = input("Username admin: ").strip()
        password = input("Password admin: ").strip()

        if get_db().users.find_one({"username": username}):
            print(f"User '{username}' sudah ada. Batal.")
            return

        hashed = generate_password_hash(password)
        get_db().users.insert_one({"username": username, "password": hashed})
        print(f"User '{username}' berhasil dibuat.")


if __name__ == '__main__':
    create_admin()
