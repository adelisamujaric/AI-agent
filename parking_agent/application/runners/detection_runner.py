"""
Application Layer - Detection Runner
Agent ciklus za detekciju parking prekršaja: Sense → Think → Act
"""
from typing import Optional
import sys

sys.path.append('../..')
from core.software_agent import SoftwareAgent
from parking_agent.application.services.detection_service import DetectionService
from parking_agent.application.services.review_service import ReviewService


class DetectionRunner(SoftwareAgent):
    """
    Runner za detekciju prekršaja

    Implementira agent ciklus:
    - SENSE: Prima sliku
    - THINK: Analizira kroz DetectionService
    - ACT: Vraća rezultat

    OVO JE KLJUČNA RAZLIKA OD STAROG main_old_notInUse.py!
    Umjesto da endpoint direktno poziva YOLO i DB,
    sada poziva Runner koji orkestrira servise.
    """

    def __init__(
            self,
            detection_service: DetectionService,
            review_service: ReviewService
    ):
        self.detection_service = detection_service
        self.review_service = review_service

    async def step_async(self, image_path: str, step_type: str = "first") -> dict:
        """
        Jedan korak detekcije

        PRIJE (u main_old_notInUse.py):
            @app.post("/analyze_first_image")
            async def analyze_first_image(file):
                # SVA logika ovdje - 50+ linija

        POSLIJE (refaktorisano):
            @app.post("/analyze_first_image")
            async def analyze_first_image(file):
                result = await detection_runner.step_async(file_path, "first")
                return result

        Args:
            image_path: Putanja do slike
            step_type: "first" ili "zoom"

        Returns:
            dict: Rezultat detekcije
        """

        if step_type == "first":
            return await self._analyze_first_step(image_path)
        elif step_type == "zoom":
            # Za zoom step treba dodatni parametri
            # Ovo će biti pozvano iz endpoint-a sa više parametara
            return {"status": "error", "message": "Use analyze_zoom_step for zoom images"}

        return {"status": "error", "message": "Invalid step type"}

    async def _analyze_first_step(self, image_path: str) -> dict:
        """
        SENSE → THINK → ACT za prvu sliku (široki kadar)
        """
        # SENSE: Slika je input (parametar)

        # THINK: Analiziraj kroz servis
        analysis = await self.detection_service.analyze_first_image(image_path)

        # ACT: Vrati rezultat
        return {
            "status": analysis.status.value,
            "message": analysis.message,
            "prekrsaj_id": analysis.prekrsaj_id,
            "detected_violation": analysis.detected_violation.value if analysis.detected_violation else None,
            "on_reservation": analysis.on_reservation
        }

    async def analyze_zoom_step(
            self,
            image_path: str,
            prekrsaj_id: int,
            on_reservation: bool,
            first_image_path: str
    ) -> dict:
        """
        SENSE → THINK → ACT za zoom sliku (tablica)

        Args:
            image_path: Zoom slika
            prekrsaj_id: ID prekršaja sa prve slike
            on_reservation: Da li je auto na rezervaciji
            first_image_path: Putanja do prve slike (za rezultat)

        Returns:
            dict: Kompletan rezultat spremni za potvrdu
        """
        # SENSE: Zoom slika + kontekst (prekrsaj_id, on_reservation)

        # THINK: Analiziraj zoom sliku
        result = await self.detection_service.analyze_zoom_image(
            image_path,
            prekrsaj_id,
            on_reservation
        )

        # ACT: Dodaj prvu sliku u rezultat
        if result.get("status") == "READY_TO_CONFIRM":
            result["slika1"] = first_image_path

        return result

    async def confirm_detection(
            self,
            vozac_id: int,
            prekrsaj_id: int,
            slika1: str,
            slika2: str = None
    ) -> dict:
        """
        Potvrđuje detekciju i čuva za učenje

        SENSE → THINK → ACT → LEARN
        """
        # SENSE: Potvrđeni podaci

        # THINK: Validacija (opciono)

        # ACT: Evidentira u bazu
        # LEARN: Čuva za retraining
        result = await self.review_service.save_confirmed_violation(
            vozac_id, prekrsaj_id, slika1, slika2
        )

        return result

    async def reject_detection(
            self,
            image_path: str,
            second_image_path: str = None
    ) -> dict:
        """
        Odbija detekciju (false positive)

        SENSE → THINK → ACT → LEARN
        """
        # SENSE: Odbijeni podaci

        # THINK: (nema - direktno čuvamo)

        # ACT & LEARN: Čuva kao odbijen (za buduću analizu)
        result = await self.review_service.save_rejected_detection(
            image_path, second_image_path
        )

        return result

    async def confirm_ok_detection(self, image_path: str) -> dict:
        """
        Potvrđuje OK detekciju (nema prekršaja)

        SENSE → THINK → ACT → LEARN
        """
        # SENSE: OK slika

        # THINK: (nema prekršaja)

        # ACT & LEARN: Čuva kao negativan primjer
        result = await self.review_service.save_confirmed_ok_detection(image_path)

        return result