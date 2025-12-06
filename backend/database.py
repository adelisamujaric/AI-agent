import sqlite3
from datetime import datetime

DB_PATH = "backend/parking.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1️⃣ Tabela VOZAC
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vozac (
        vozac_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ime TEXT,
        tablica TEXT UNIQUE,
        auto_tip TEXT,
        invalid INTEGER DEFAULT 0,
        rezervacija INTEGER DEFAULT 0
    )
    """)

    # 2️⃣ Tabela PREKRSAJI (definicije)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prekrsaji (
        prekrsaj_id INTEGER PRIMARY KEY AUTOINCREMENT,
        opis TEXT,
        kazna INTEGER
    )
    """)

    # 3️⃣ Tabela DETEKTOVANO (many-to-many)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detektovano (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vozac_id INTEGER,
        prekrsaj_id INTEGER,
        vrijeme TEXT,
        slika1 TEXT,
        slika2 TEXT,
        FOREIGN KEY(vozac_id) REFERENCES vozac(vozac_id),
        FOREIGN KEY(prekrsaj_id) REFERENCES prekrsaji(prekrsaj_id)
    )
    """)

    conn.commit()
    conn.close()
