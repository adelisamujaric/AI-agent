"""
Infrastructure sloj - File Storage
Sve operacije sa fajlovima (čuvanje slika, labela, modela)
"""
import os
import shutil
from typing import List, Tuple
from datetime import datetime
import sys

sys.path.append('..')
from parking_agent.domain.entities import Detection


class FileStorage:
    """
    Helper za sve file operacije parking agenta
    Izvučeno iz main_old_notInUse.py - sve što je bilo shutil.copy, os.makedirs...
    """

    def __init__(
            self,
            confirmed_dir: str = "backend/confirmed",
            rejected_dir: str = "backend/rejected",
            uploads_dir: str = "backend/uploads",
            weights_dir: str = "backend/weights"
    ):
        self.confirmed_dir = confirmed_dir
        self.rejected_dir = rejected_dir
        self.uploads_dir = uploads_dir
        self.weights_dir = weights_dir

        # Kreiraj potrebne foldere
        self._ensure_directories()

    def _ensure_directories(self):
        """Kreira sve potrebne direktorije"""
        os.makedirs(os.path.join(self.confirmed_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.confirmed_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.rejected_dir, "first"), exist_ok=True)
        os.makedirs(os.path.join(self.rejected_dir, "zoom"), exist_ok=True)
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

    def save_confirmed_image(
            self,
            source_path: str,
            detections: List[Detection],
            suffix: str = "first"
    ) -> Tuple[str, str]:
        """
        Čuva potvrđenu sliku + generiše YOLO labele
        Returns: (image_path, label_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Kopiraj sliku
        new_img_name = f"confirmed_{timestamp}_{suffix}.jpg"
        new_img_path = os.path.join(self.confirmed_dir, "images", new_img_name)
        shutil.copy(source_path, new_img_path)

        # Generiši YOLO label
        label_name = new_img_name.replace('.jpg', '.txt')
        label_path = os.path.join(self.confirmed_dir, "labels", label_name)
        self._save_yolo_labels(detections, label_path, source_path)

        return new_img_path, label_path

    def save_rejected_image(self, source_path: str, image_type: str = "first") -> str:
        """
        Čuva odbijenu sliku
        image_type: 'first' ili 'zoom'
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"rejected_{image_type}_{timestamp}.jpg"
        dest_path = os.path.join(self.rejected_dir, image_type, new_name)
        shutil.copy(source_path, dest_path)
        return dest_path

    def count_confirmed_images(self) -> int:
        """Broji koliko ima confirmed slika"""
        images_dir = os.path.join(self.confirmed_dir, "images")
        return len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

    def count_rejected_images(self) -> Tuple[int, int]:
        """Broji odbijene slike (first, zoom)"""
        first_count = len([
            f for f in os.listdir(os.path.join(self.rejected_dir, "first"))
            if f.endswith('.jpg')
        ])
        zoom_count = len([
            f for f in os.listdir(os.path.join(self.rejected_dir, "zoom"))
            if f.endswith('.jpg')
        ])
        return first_count, zoom_count

    def backup_model(self, model_path: str) -> str:
        """Kreira backup trenutnog modela"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.weights_dir, f"backup_model_{timestamp}.pt")
        shutil.copy(model_path, backup_path)
        return backup_path

    def replace_model(self, new_model_path: str, target_path: str) -> None:
        """Zamjenjuje stari model sa novim"""
        shutil.copy(new_model_path, target_path)

    def archive_confirmed_data(self) -> str:
        """Arhivira confirmed podatke nakon uspješnog retraining-a"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = f"backend/confirmed_archive/{timestamp}"
        os.makedirs(archive_dir, exist_ok=True)

        # Premjesti images i labels
        shutil.move(
            os.path.join(self.confirmed_dir, "images"),
            os.path.join(archive_dir, "images")
        )
        shutil.move(
            os.path.join(self.confirmed_dir, "labels"),
            os.path.join(archive_dir, "labels")
        )

        # Kreiraj nove prazne foldere
        os.makedirs(os.path.join(self.confirmed_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.confirmed_dir, "labels"), exist_ok=True)

        return archive_dir

    def _save_yolo_labels(
            self,
            detections: List[Detection],
            label_path: str,
            original_image_path: str
    ):
        """
        Generiše YOLO format labele iz detekcija
        IZVUČENO IZ main_old_notInUse.py - funkcija save_yolo_labels()
        """
        # Mapiranje klasa - SAMO PARKING RELEVANTNE
        class_mapping = {
            'Auto': 0,
            'RezervacijaOznaka': 1,
            'ZauzetoMjesto': 2,
            'NepropisnoParkirano_naInvalidskomMjesto': 3,
            'NepropisnoParkirano_prekoLinije': 4,
            'NepropisnoParkirano_vanOkviraParkinga': 5,
            'InvalidskaOznaka': 6,
            'InvalidskoMjesto': 7,
            'Tablica': 8,
            'PravilnoParkirano': 9
        }

        # Filtriraj samo relevantne detekcije
        valid_detections = [
            d for d in detections
            if d.class_name in class_mapping
        ]

        if len(valid_detections) == 0:
            # Prazna label datoteka
            with open(label_path, 'w') as f:
                pass
            return

        # Učitaj dimenzije slike (potrebno za normalizaciju)
        import cv2
        img = cv2.imread(original_image_path)
        img_h, img_w = img.shape[:2]

        with open(label_path, 'w') as f:
            for det in valid_detections:
                new_cls_id = class_mapping[det.class_name]
                x1, y1, x2, y2 = det.bbox

                # YOLO format: center_x center_y width height (normalized)
                center_x = ((x1 + x2) / 2) / img_w
                center_y = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                f.write(f"{new_cls_id} {center_x} {center_y} {width} {height}\n")