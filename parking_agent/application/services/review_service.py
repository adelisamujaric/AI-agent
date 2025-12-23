"""
Application Layer - Review Service
Logika za čuvanje potvrđenih/odbijenih detekcija (učenje)
"""
from typing import Optional
from datetime import datetime
import sys

sys.path.append('..')
from parking_agent.domain.entities import ViolationRecord
from parking_agent.infrastructure.database import ParkingDbContext
from parking_agent.infrastructure.file_storage import FileStorage
from parking_agent.ML.yolo_classifier import YoloClassifier


class ReviewService:
    """
    Servis za Review i Learning

    IZVUČENO IZ main_old_notInUse.py:
    - @app.post("/record_violation")
    - @app.post("/record_ok_detection")
    - @app.post("/reject_detection")
    - @app.get("/learning_stats")
    """

    def __init__(
            self,
            db_context: ParkingDbContext,
            file_storage: FileStorage,
            classifier: YoloClassifier
    ):
        self.db = db_context
        self.storage = file_storage
        self.classifier = classifier

    async def save_confirmed_violation(
            self,
            vozac_id: int,
            prekrsaj_id: int,
            slika1: str,
            slika2: Optional[str] = None
    ) -> dict:
        """
        Evidentira potvrđeni prekršaj u bazu + čuva za učenje

        OVO JE BILO U main_old_notInUse.py @app.post("/record_violation")
        """
        # 1. Sačuvaj u bazu
        record = ViolationRecord(
            vozac_id=vozac_id,
            prekrsaj_id=prekrsaj_id,
            vrijeme=datetime.now(),
            slika1=slika1,
            slika2=slika2
        )
        self.db.save_violation_record(record)

        # 2. Čuvanje za učenje - generisanje YOLO labela
        await self._save_for_learning(slika1, "first")

        if slika2:
            await self._save_for_learning(slika2, "zoom")

        return {
            "status": "success",
            "message": "Prekršaj evidentiran i dodan u trening dataset"
        }

    async def save_confirmed_ok_detection(self, image_path: str) -> dict:
        """
        Čuva potvrđenu OK detekciju (nema prekršaja)
        Ovo je također važno za učenje - negativni primjeri!

        OVO JE BILO U main_old_notInUse.py @app.post("/record_ok_detection")
        """
        await self._save_for_learning(image_path, "ok")

        return {
            "status": "success",
            "message": "OK detekcija sačuvana za učenje"
        }

    async def save_rejected_detection(
            self,
            image_path: str,
            second_image_path: Optional[str] = None
    ) -> dict:
        """
        Čuva odbijenu detekciju (false positive)

        OVO JE BILO U main_old_notInUse.py @app.post("/reject_detection")
        """
        saved_count = 0

        # Sačuvaj first image
        self.storage.save_rejected_image(image_path, "first")
        saved_count += 1

        # Sačuvaj zoom image (ako postoji)
        if second_image_path:
            self.storage.save_rejected_image(second_image_path, "zoom")
            saved_count += 1

        return {
            "status": "success",
            "message": f"Odbačeno {saved_count} slika",
            "count": saved_count
        }

    async def _save_for_learning(self, image_path: str, suffix: str):
        """
        Pomoćna metoda - čuva sliku + generiše YOLO labele
        """
        # Dohvati detekcije sa slike
        detections = await self.classifier.predict(image_path)

        # Sačuvaj sliku + labele kroz FileStorage
        self.storage.save_confirmed_image(image_path, detections, suffix)

    def get_learning_stats(self) -> dict:
        """
        Vraća statistiku spremnih podataka za učenje

        OVO JE BILO U main_old_notInUse.py @app.get("/learning_stats")
        """
        confirmed_count = self.storage.count_confirmed_images()
        rejected_first, rejected_zoom = self.storage.count_rejected_images()

        return {
            "confirmed_images": confirmed_count,
            "rejected_first": rejected_first,
            "rejected_zoom": rejected_zoom,
            "ready_for_retraining": confirmed_count >= 20
        }