import sqlite3

DB_PATH = "face_database.db"

def init_db():
    """Membuat ulang database dari awal"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS faces")  # Hapus database lama
    cursor.execute('''
        CREATE TABLE faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT,
            image BLOB,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()
    print("Database baru berhasil dibuat!")

init_db()
