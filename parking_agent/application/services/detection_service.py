"""
Application Layer - Detection Service
Logika za analizu parking prekršaja (izvučeno iz main_old_notInUse.py)
"""
from typing import Optional, List
import sys

sys.path.append('..')
from parking_agent.domain.entities import ViolationAnalysis, Detection, Driver
from parking_agent.domain.enums import DetectionStatus, ViolationType
from parking_agent.ML.yolo_classifier import YoloClassifier
from parking_agent.infrastructure.database import ParkingDbContext
from backend.ocr import read_plate
from backend.utils import crop_plate


class DetectionService:
    """
    Servis za detekciju parking prekršaja

    IZVUČENO IZ main_old_notInUse.py:
    - @app.post("/analyze_first_image")
    - @app.post("/analyze_zoom_image")
    - Sva poslovna pravila za parking enforcement
    """

    def __init__(
            self,
            classifier: YoloClassifier,
            db_context: ParkingDbContext
    ):
        self.classifier = classifier
        self.db = db_context

    async def analyze_first_image(self, image_path: str) -> ViolationAnalysis:
        """
        Analizira prvu sliku (široki kadar) i detektuje prekršaje

        OVO JE BILO U main_old_notInUse.py @app.post("/analyze_first_image"):
            results = model(first_path)
            if car_on_reservation and violations:
                ...
        """
        # Dohvati detekcije sa slike
        detections = await self.classifier.predict(image_path)

        # Analiziraj šta je detektovano
        analysis = self._analyze_detections(detections)

        # Primjeni poslovna pravila
        return self._apply_violation_rules(analysis)

    def _analyze_detections(self, detections: List[Detection]) -> ViolationAnalysis:
        """
        Analizira sirove detekcije i izvlači relevantne informacije
        """
        has_reservation_sign = False
        has_auto = False
        has_occupied_spot = False
        violations = []

        for det in detections:
            cls_name = det.class_name

            if cls_name == "RezervacijaOznaka":
                has_reservation_sign = True
            elif cls_name == "Auto":
                has_auto = True
            elif cls_name == "ZauzetoMjesto":
                has_occupied_spot = True
            elif cls_name.startswith("NepropisnoParkirano"):
                violations.append(cls_name)

        return ViolationAnalysis(
            status=DetectionStatus.OK,
            has_reservation_sign=has_reservation_sign,
            has_auto=has_auto,
            has_occupied_spot=has_occupied_spot,
            violations=violations
        )

    def _apply_violation_rules(self, analysis: ViolationAnalysis) -> ViolationAnalysis:
        """
        Primjenjuje poslovna pravila za parking prekršaje
        KLJUČNA DOMENSKA LOGIKA - izvučena iz main_old_notInUse.py!
        """
        car_on_reservation = (
                analysis.has_reservation_sign and
                analysis.has_auto and
                analysis.has_occupied_spot
        )

        # PRAVILO 1: Auto na rezervaciji + ima standardni prekršaj
        if car_on_reservation and analysis.violations:
            main_violation = analysis.violations[0]
            violation = self.db.get_violation_by_description(main_violation)

            if violation:
                analysis.status = DetectionStatus.NEEDS_ZOOM
                analysis.detected_violation = ViolationType(main_violation)
                analysis.on_reservation = True
                analysis.prekrsaj_id = violation.prekrsaj_id
                analysis.message = f"⚠️ Prekršaj: {main_violation} + Auto na rezervaciji - provjeri tablicu"
                return analysis

        # PRAVILO 2: Auto na rezervaciji (pravilno parkiran, ali možda nema pravo)
        if car_on_reservation:
            violation = self.db.get_violation_by_description("Parkiranje_na_rezervisanom_mjestu")

            if violation:
                analysis.status = DetectionStatus.NEEDS_ZOOM
                analysis.detected_violation = ViolationType.PARKIRANJE_NA_REZERVACIJI
                analysis.on_reservation = True
                analysis.prekrsaj_id = violation.prekrsaj_id
                analysis.message = "🅿️ Auto na rezervacijskom mestu - provjeri tablicu"
                return analysis

        # PRAVILO 3: Standardni prekršaji (ne na rezervaciji)
        if analysis.violations:
            main_violation = analysis.violations[0]
            violation = self.db.get_violation_by_description(main_violation)

            if violation:
                analysis.status = DetectionStatus.NEEDS_ZOOM
                analysis.detected_violation = ViolationType(main_violation)
                analysis.on_reservation = False
                analysis.prekrsaj_id = violation.prekrsaj_id
                analysis.message = f"⚠️ Prekršaj: {main_violation} - približi se za tablicu"
                return analysis

        # PRAVILO 4: Sve je OK - nema prekršaja
        analysis.status = DetectionStatus.OK
        analysis.message = "✅ Nema prekršaja - pravilno parkirano"
        return analysis

    async def analyze_zoom_image(
            self,
            image_path: str,
            prekrsaj_id: int,
            on_reservation: bool
    ) -> dict:
        """
        Analizira zoom sliku (tablica) i vraća kompletne informacije za potvrdu

        OVO JE BILO U main_old_notInUse.py @app.post("/analyze_zoom_image")
        """
        # Detektuj tablicu
        detections = await self.classifier.predict(image_path)

        plate_box = None
        for det in detections:
            if det.class_name.lower() == "tablica":
                plate_box = det.bbox
                break

        if not plate_box:
            return {"status": "NO_PLATE"}

        # OCR - pročitaj tablicu
        crop_path = crop_plate(image_path, plate_box)
        plate_text = read_plate(crop_path) or "Unknown"

        # Pronađi vozača
        driver = self.db.get_driver_by_plate(plate_text)
        if not driver:
            return {"status": "NO_DRIVER", "plate": plate_text}

        # Dohvati prekršaj
        violation = self.db.get_violation_by_id(prekrsaj_id)
        if not violation:
            return {"status": "ERROR", "message": "Prekršaj nije pronađen"}

        # Primjeni logiku za rezervaciju
        if on_reservation:
            return self._handle_reservation_violation(driver, violation, plate_text, image_path)

        # Standardni prekršaj
        return {
            "status": "READY_TO_CONFIRM",
            "plate": plate_text,
            "vozac": self._driver_to_dict(driver),
            "prekrsaj_opis": violation.opis,
            "prekrsaj_kazna": violation.kazna,
            "prekrsaj_id": prekrsaj_id,
            "slika2": image_path
        }

    def _handle_reservation_violation(
            self,
            driver: Driver,
            violation,
            plate_text: str,
            image_path: str
    ) -> dict:
        """
        Logika za prekršaje na rezervacijskom mestu
        """
        # Vozač IMA rezervaciju - parkiranje OK
        if driver.rezervacija:
            return {
                "status": "OK_WITH_RESERVATION",
                "message": f"✅ Vozač {driver.ime} ima rezervaciju - parkiranje dozvoljeno",
                "plate": plate_text,
                "vozac": self._driver_to_dict(driver)
            }

        # Vozač NEMA rezervaciju - dodatna kazna!
        reservation_violation = self.db.get_violation_by_description(
            "Parkiranje_na_rezervisanom_mjestu"
        )

        total_fine = violation.kazna
        if reservation_violation:
            total_fine += reservation_violation.kazna

        return {
            "status": "READY_TO_CONFIRM",
            "plate": plate_text,
            "vozac": self._driver_to_dict(driver),
            "prekrsaj_opis": f"{violation.opis} + Parkiranje bez rezervacije",
            "prekrsaj_kazna": total_fine,
            "prekrsaj_id": violation.prekrsaj_id,
            "extra_violation_id": reservation_violation.prekrsaj_id if reservation_violation else None,
            "slika2": image_path
        }

    def _driver_to_dict(self, driver: Driver) -> dict:
        """Helper za konverziju Driver → dict"""
        return {
            "vozac_id": driver.vozac_id,
            "ime": driver.ime,
            "tablica": driver.tablica,
            "auto_tip": driver.auto_tip,
            "invalid": driver.invalid,
            "rezervacija": driver.rezervacija
        }