"""
Domain sloj - Entiteti
Ovi entiteti predstavljaju domenske koncepte parking enforcement-a
"""
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from .enums import ViolationType, DetectionStatus


@dataclass
class Detection:
    """
    Jedna detekcija sa slike (rezultat YOLO-a)
    """
    image_path: str
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


@dataclass
class ViolationAnalysis:
    """
    Rezultat analize prekršaja sa slike
    Ovo je "Think" output - šta agent misli o slici
    """
    status: DetectionStatus
    detected_violation: Optional[ViolationType] = None
    on_reservation: bool = False
    has_reservation_sign: bool = False
    has_auto: bool = False
    has_occupied_spot: bool = False
    violations: List[str] = None
    prekrsaj_id: Optional[int] = None
    message: str = ""

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


@dataclass
class PlateRecognition:
    """
    Rezultat OCR-a tablice
    """
    plate_text: str
    bbox: List[float]
    confidence: float = 1.0


@dataclass
class Driver:
    """
    Vozač iz baze
    """
    vozac_id: int
    ime: str
    tablica: str
    auto_tip: str
    invalid: bool
    rezervacija: bool


@dataclass
class Violation:
    """
    Prekršaj iz baze
    """
    prekrsaj_id: int
    opis: str
    kazna: int


@dataclass
class ViolationRecord:
    """
    Zapis o evidentiranom prekršaju
    """
    vozac_id: int
    prekrsaj_id: int
    vrijeme: datetime
    slika1: str
    slika2: Optional[str] = None


@dataclass
class ModelVersion:
    """
    Verzija ML modela
    """
    version_id: str
    timestamp: datetime
    map50: float
    backup_path: str
    is_active: bool = False


@dataclass
class SystemSettings:
    """
    Sistemske postavke za učenje
    """
    min_images_for_retrain: int = 20
    retrain_enabled: bool = True
    new_confirmed_since_last_train: int = 0