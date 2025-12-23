"""
ML sloj - YOLO Classifier
Wrapper oko Ultralytics YOLOv8s modela
"""
from typing import List
from ultralytics import YOLO
import sys

sys.path.append('..')
from parking_agent.domain.entities import Detection


class YoloClassifier:
    """
    YOLO wrapper za parking detekciju
    Izvučeno iz main_old_notInUse.py - sva YOLO logika
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path)

    async def predict(self, image_path: str) -> List[Detection]:
        """
        Detektuje objekte na slici
        Vraća samo "sirove" detekcije - nema domenskih odluka!

        OVO JE BILO U main_old_notInUse.py:
            results = model(image_path)
            for box in results[0].boxes:
                ...
        """
        results = self.model(image_path)
        detections = []

        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            detections.append(Detection(
                image_path=image_path,
                class_name=cls_name,
                confidence=conf,
                bbox=xyxy
            ))

        return detections

    async def train(
            self,
            config_path: str,
            epochs: int = 5,
            imgsz: int = 640,
            batch: int = 8,
            lr0: float = 0.0001,
            freeze: int = 10,
            project: str = 'backend/retraining_runs',
            name: str = 'retrain',
            exist_ok: bool = True
    ):
        """
        Fine-tuning postojećeg modela

        OVO JE BILO U main_old_notInUse.py:
            current_model.train(data=config_path, epochs=5, ...)
        """
        results = self.model.train(
            data=config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            freeze=freeze,
            project=project,
            name=name,
            exist_ok=exist_ok
        )
        return results

    async def evaluate(self, config_path: str) -> float:
        """
        Evaluira model na datasetu i vraća mAP50

        OVO JE BILO U main_old_notInUse.py:
            metrics = model.val(data=config_path)
            map50 = metrics.box.map50
        """
        metrics = self.model.val(data=config_path)
        return float(metrics.box.map50)

    def get_class_names(self) -> dict:
        """Vraća dictionary class_id -> class_name"""
        return self.model.names

    def reload_model(self, new_model_path: str):
        """
        Učitava novi model (nakon retraining-a)

        OVO JE BILO U main_old_notInUse.py:
            global model
            model = YOLO("backend/weights/best.pt")
        """
        self.model_path = new_model_path
        self.model = YOLO(new_model_path)