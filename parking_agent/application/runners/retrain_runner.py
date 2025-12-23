"""
Application Layer - Retrain Runner
Agent ciklus za retraining modela: Sense → Think → Act → Learn
"""
from typing import Optional
import sys

sys.path.append('../..')
from core.software_agent import SoftwareAgent
from parking_agent.application.services.training_service import TrainingService
from parking_agent.application.services.review_service import ReviewService


class RetrainRunner(SoftwareAgent):
    """
    Runner za retraining modela

    Implementira agent ciklus:
    - SENSE: Provjeri broj confirmed slika
    - THINK: Da li ima dovoljno podataka za retraining? (>= 20)
    - ACT: Pokreni retraining (ako treba)
    - LEARN: Ažuriraj model, arhiviraj podatke

    OVO JE KLJUČ PROFESOROVOG ZAHTJEVA!
    Agent MORA imati jasno odvojen Sense→Think→Act→Learn ciklus.
    """

    def __init__(
            self,
            training_service: TrainingService,
            review_service: ReviewService,
            min_images: int = 20
    ):
        self.training_service = training_service
        self.review_service = review_service
        self.min_images = min_images

    async def step_async(self, cancellation_token=None) -> Optional[dict]:
        """
        Jedan korak retraining ciklusa

        PRIJE (u main_old_notInUse.py):
            @app.post("/retrain_model")
            async def retrain_model():
                # 150+ linija logike ovdje
                if len(images) < 20:
                    return ...
                model.train(...)
                if new_map50 > old_map50:
                    ...

        POSLIJE (refaktorisano):
            @app.post("/retrain_model")
            async def retrain_model():
                result = await retrain_runner.step_async()
                return result  # 2 linije!

        Returns:
            dict: Rezultat retraining-a (ili None ako nema posla)
        """

        # ============================================
        # SENSE: Očitaj stanje svijeta
        # ============================================
        stats = self.review_service.get_learning_stats()
        confirmed_count = stats["confirmed_images"]

        print(f"📊 SENSE: Detektovano {confirmed_count} confirmed slika")

        # ============================================
        # THINK: Da li treba retraining?
        # ============================================
        should_retrain = self._should_retrain(confirmed_count)

        if not should_retrain:
            print(f"⏸️ THINK: Ne treba retraining (ima samo {confirmed_count}/{self.min_images})")
            return {
                "status": "NOT_ENOUGH_DATA",
                "message": f"Potrebno je minimum {self.min_images} slika, trenutno ima {confirmed_count}",
                "confirmed_count": confirmed_count,
                "required_count": self.min_images
            }

        print(f"✅ THINK: Pokrećem retraining sa {confirmed_count} slika")

        # ============================================
        # ACT: Pokreni retraining
        # ============================================
        result = await self.training_service.retrain_model()

        # ============================================
        # LEARN: Ažuriraj znanje (desilo se u TrainingService)
        # ============================================
        # - Model je reload-ovan (ako je bolji)
        # - Confirmed slike su arhivirane
        # - Brojač je resetovan

        print(f"🎓 LEARN: Status = {result.get('status')}")

        return result

    def _should_retrain(self, confirmed_count: int) -> bool:
        """
        THINK logika: Odlučuje da li treba pokrenuti retraining

        Ovo je KLJUČNO poslovno pravilo koje MORA biti u Application sloju,
        NE u Web layer-u!

        Args:
            confirmed_count: Broj confirmed slika

        Returns:
            bool: True ako treba retraining
        """
        return confirmed_count >= self.min_images

    def has_work(self) -> bool:
        """
        Override bazne klase - brža provjera bez izvršavanja step-a

        Returns:
            bool: True ako ima dovoljno slika za retraining
        """
        stats = self.review_service.get_learning_stats()
        return stats["ready_for_retraining"]

    def get_learning_stats(self) -> dict:
        """
        Pomoćna metoda - vraća statistiku spremnosti za učenje

        Returns:
            dict: Broj confirmed/rejected slika, spremnost za retraining
        """
        return self.review_service.get_learning_stats()