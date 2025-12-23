"""
Application Layer - Training Service
Logika za retraining modela (izvučeno iz main_old_notInUse.py)
"""
import os
import yaml
import shutil
from datetime import datetime
import sys

sys.path.append('..')
from parking_agent.domain.enums import LearningStatus
from parking_agent.ML.yolo_classifier import YoloClassifier
from parking_agent.infrastructure.file_storage import FileStorage


class TrainingService:
    """
    Servis za treniranje i evaluaciju ML modela

    IZVUČENO IZ main_old_notInUse.py:
    - @app.post("/retrain_model")
    - Kompletan retraining ciklus sa evaluacijom
    """

    def __init__(
            self,
            classifier: YoloClassifier,
            file_storage: FileStorage,
            confirmed_dir: str = "backend/confirmed",
            weights_dir: str = "backend/weights"
    ):
        self.classifier = classifier
        self.storage = file_storage
        self.confirmed_dir = confirmed_dir
        self.weights_dir = weights_dir

    async def retrain_model(self) -> dict:
        """
        Glavni metod za retraining - kompletan ciklus:
        1. Provjera podataka
        2. Backup starog modela
        3. Fine-tuning
        4. Evaluacija
        5. Odluka: zamijeniti ili zadržati stari

        OVO JE BILO U main_old_notInUse.py @app.post("/retrain_model")
        """
        # SENSE: Provjeri broj slika
        num_images = self.storage.count_confirmed_images()

        if num_images < 20:
            return {
                "status": LearningStatus.NOT_ENOUGH_DATA.value,
                "message": f"Potrebno je minimum 20 slika, trenutno ima {num_images}"
            }

        print(f"🚀 Pokrećem retraining sa {num_images} novih slika...")

        try:
            # 1. Kreiraj config za YOLO
            config_path = self._create_training_config()

            # 2. Backup trenutnog modela
            current_model_path = os.path.join(self.weights_dir, "best.pt")
            backup_path = self.storage.backup_model(current_model_path)
            print(f"💾 Backup starog modela: {backup_path}")

            # 3. Fine-tuning
            print("🏋️ Pokrećem treniranje...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            await self.classifier.train(
                config_path=config_path,
                epochs=5,
                imgsz=640,
                batch=8,
                lr0=0.0001,
                freeze=10,
                project='backend/retraining_runs',
                name=f'retrain_{timestamp}'
            )

            # 4. Evaluacija NOVOG modela
            print("📊 Evaluiram novi model...")
            new_map50 = await self.classifier.evaluate(config_path)

            # 5. Evaluacija STAROG modela (na istim podacima!)
            print("📊 Evaluiram stari model...")
            old_classifier = YoloClassifier(backup_path)
            old_map50 = await old_classifier.evaluate(config_path)

            print(f"📈 Stari model mAP50: {old_map50:.3f}")
            print(f"📈 Novi model mAP50: {new_map50:.3f}")

            # 6. THINK: Da li je novi model bolji?
            if new_map50 > old_map50:
                return await self._activate_new_model(
                    timestamp, new_map50, old_map50, current_model_path
                )
            else:
                return await self._keep_old_model(
                    backup_path, current_model_path, new_map50, old_map50
                )

        except Exception as e:
            print(f"❌ Greška pri treniranju: {e}")
            return {
                "status": LearningStatus.ERROR.value,
                "message": f"Greška pri treniranju: {str(e)}"
            }

    def _create_training_config(self) -> str:
        """
        Generiše data.yaml config za YOLO trening
        """
        config = {
            'path': os.path.abspath(self.confirmed_dir),
            'train': 'images',
            'val': 'images',
            'nc': 10,
            'names': [
                'Auto',
                'RezervacijaOznaka',
                'ZauzetoMjesto',
                'NepropisnoParkirano_naInvalidskomMjesto',
                'NepropisnoParkirano_prekoLinije',
                'NepropisnoParkirano_vanOkviraParkinga',
                'InvalidskaOznaka',
                'InvalidskoMjesto',
                'Tablica',
                'PravilnoParkirano'
            ]
        }

        config_path = os.path.abspath("backend/data_retrain.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return config_path

    async def _activate_new_model(
            self,
            timestamp: str,
            new_map50: float,
            old_map50: float,
            target_model_path: str
    ) -> dict:
        """
        Aktivira novi model (zamjenjuje stari)
        """
        print(f"✅ Novi model je bolji! Ažuriram...")

        # Kopiraj novi model preko starog
        new_model_path = f"backend/retraining_runs/retrain_{timestamp}/weights/best.pt"
        self.storage.replace_model(new_model_path, target_model_path)

        # Reload model u memoriji
        self.classifier.reload_model(target_model_path)

        # LEARN: Arhiviraj korištene slike
        self.storage.archive_confirmed_data()

        return {
            "status": LearningStatus.SUCCESS.value,
            "message": f"Model uspješno ažuriran! mAP50: {old_map50:.3f} → {new_map50:.3f}",
            "old_map50": float(old_map50),
            "new_map50": float(new_map50),
            "improvement": float(new_map50 - old_map50)
        }

    async def _keep_old_model(
            self,
            backup_path: str,
            target_model_path: str,
            new_map50: float,
            old_map50: float
    ) -> dict:
        """
        Zadržava stari model (novi nije bolji)
        """
        print(f"⚠️ Novi model nije bolji. Zadržavam stari.")

        # Restore backup
        shutil.copy(backup_path, target_model_path)

        return {
            "status": LearningStatus.NO_IMPROVEMENT.value,
            "message": "Novi model nije pokazao poboljšanje. Zadržan stari model.",
            "old_map50": float(old_map50),
            "new_map50": float(new_map50)
        }