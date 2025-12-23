"""
Infrastructure sloj - Database Context
Svi DB operacije za parking agent
"""
import sqlite3
from typing import Optional, List
from datetime import datetime
import sys

sys.path.append('..')
from parking_agent.domain.entities import Driver, Violation, ViolationRecord


class ParkingDbContext:
    """
    Database context - svi upiti prema SQLite bazi
    Izvučeno iz main_old_notInUse.py - sve što je bilo cursor.execute()
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_driver_by_plate(self, plate: str) -> Optional[Driver]:
        """Pronalazi vozača po tablici"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM vozac WHERE tablica = ?", (plate,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return Driver(
            vozac_id=row[0],
            ime=row[1],
            tablica=row[2],
            auto_tip=row[3],
            invalid=bool(row[4]),
            rezervacija=bool(row[5])
        )

    def get_violation_by_id(self, prekrsaj_id: int) -> Optional[Violation]:
        """Dohvata prekršaj po ID-u"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT prekrsaj_id, opis, kazna FROM prekrsaji WHERE prekrsaj_id = ?",
            (prekrsaj_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return Violation(
            prekrsaj_id=row[0],
            opis=row[1],
            kazna=row[2]
        )

    def get_violation_by_description(self, opis: str) -> Optional[Violation]:
        """Pronalazi prekršaj po opisu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT prekrsaj_id, opis, kazna FROM prekrsaji WHERE opis = ?",
            (opis,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return Violation(
            prekrsaj_id=row[0],
            opis=row[1],
            kazna=row[2]
        )

    def save_violation_record(self, record: ViolationRecord) -> None:
        """Evidentira prekršaj u bazu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = record.vrijeme.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO detektovano (vozac_id, prekrsaj_id, vrijeme, slika1, slika2)
            VALUES (?, ?, ?, ?, ?)
        """, (record.vozac_id, record.prekrsaj_id, timestamp, record.slika1, record.slika2))

        conn.commit()
        conn.close()

    def get_all_drivers(self) -> List[Driver]:
        """Vraća sve vozače"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vozac")
        rows = cursor.fetchall()
        conn.close()

        return [
            Driver(
                vozac_id=r[0],
                ime=r[1],
                tablica=r[2],
                auto_tip=r[3],
                invalid=bool(r[4]),
                rezervacija=bool(r[5])
            )
            for r in rows
        ]

    def get_all_violations(self) -> List[Violation]:
        """Vraća sve tipove prekršaja"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT prekrsaj_id, opis, kazna FROM prekrsaji")
        rows = cursor.fetchall()
        conn.close()

        return [
            Violation(prekrsaj_id=r[0], opis=r[1], kazna=r[2])
            for r in rows
        ]

    def add_driver(self, driver: Driver) -> None:
        """Dodaje novog vozača"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO vozac (ime, tablica, auto_tip, invalid, rezervacija)
            VALUES (?, ?, ?, ?, ?)
        """, (driver.ime, driver.tablica, driver.auto_tip,
              int(driver.invalid), int(driver.rezervacija)))
        conn.commit()
        conn.close()

    def add_violation_type(self, violation: Violation) -> None:
        """Dodaje novi tip prekršaja"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prekrsaji (opis, kazna)
            VALUES (?, ?)
        """, (violation.opis, violation.kazna))
        conn.commit()
        conn.close()