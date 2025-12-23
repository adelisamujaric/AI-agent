"""
ParkingAgent Web Layer - main.py
TANKI HOST - samo API endpoints, poziva Runnere
"""
import os
import sys
import shutil
from datetime import datetime

# ===================================
# PATH FIX - KRITIČNO!
# ===================================
# Dodaj project root u Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # parking_agent_web/
project_root = os.path.dirname(current_dir)  # ai_agent/
sys.path.insert(0, project_root)

print(f"📁 Project root: {project_root}")

# Sada može da importuje backend i parking_agent
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# ===================================
# DEPENDENCY INJECTION - Inicijalizacija
# ===================================
from backend.database import init_db, DB_PATH
from parking_agent.ML.yolo_classifier import YoloClassifier
from parking_agent.infrastructure.database import ParkingDbContext
from parking_agent.infrastructure.file_storage import FileStorage
from parking_agent.application.services.detection_service import DetectionService
from parking_agent.application.services.review_service import ReviewService
from parking_agent.application.services.training_service import TrainingService
from parking_agent.application.runners.detection_runner import DetectionRunner
from parking_agent.application.runners.retrain_runner import RetrainRunner

# ===================================
# SETUP
# ===================================
app = FastAPI()
init_db()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload folder
UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===================================
# DEPENDENCY INJECTION - Kreiraj instancu jednom!
# ===================================
print("🚀 Inicijalizujem ParkSmart AI Agent...")

# Infrastructure
classifier = YoloClassifier("backend/weights/best.pt")
db_context = ParkingDbContext(DB_PATH)
file_storage = FileStorage()

# Services
detection_service = DetectionService(classifier, db_context)
review_service = ReviewService(db_context, file_storage, classifier)
training_service = TrainingService(classifier, file_storage)

# Runners (⭐ KLJUČNO!)
detection_runner = DetectionRunner(detection_service, review_service)
retrain_runner = RetrainRunner(training_service, review_service)

print("✅ ParkSmart AI Agent spreman!")


# ===================================
# DTOs (Pydantic modeli)
# ===================================
class Vozac(BaseModel):
    ime: str
    tablica: str
    auto_tip: str
    invalid: bool = False
    rezervacija: bool = False


class Prekrsaj(BaseModel):
    opis: str
    kazna: int


class Detektovano(BaseModel):
    vozac_id: int
    prekrsaj_id: int
    slika1: str
    slika2: str | None = None


# ===================================
# ENDPOINTS - TANKI! Samo pozivaju Runnere
# ===================================

@app.get("/")
def root():
    return {"message": "ParkSmart AI Agent backend is running!"}


# --------------------------------------------------------
# DETECTION ENDPOINTS - samo pozivaju DetectionRunner
# --------------------------------------------------------
@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    """
    Osnovni YOLO detect za frontend prikaz bbox-ova
    """
    temp_path = os.path.join(UPLOAD_DIR, "temp_image.jpg")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # TANKO - pozovi klasifikator direktno (ovo je OK, nema poslovne logike)
    detections = await classifier.predict(temp_path)

    return {
        "detections": [
            {
                "box": det.bbox,
                "class": det.class_name,
                "confidence": det.confidence
            }
            for det in detections
        ]
    }


@app.post("/analyze_first_image")
async def analyze_first_image(file: UploadFile = File(...)):
    """
    Analizira prvu sliku (široki kadar)

    PRIJE: 50+ linija logike ovdje
    POSLIJE: 3 linije - poziv Runner-a!
    """
    first_path = os.path.join(UPLOAD_DIR, "first_image.jpg")
    with open(first_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ Samo pozovi Runner!
    result = await detection_runner.step_async(first_path, "first")
    print("🔍 BACKEND VRAĆA:", result)
    return result


@app.post("/analyze_zoom_image")
async def analyze_zoom_image(
        file: UploadFile = File(...),
        prekrsaj_id: int = Form(...),
        on_reservation: bool = Form(False)
):
    """
    Analizira zoom sliku (tablica)

    PRIJE: 60+ linija logike
    POSLIJE: 4 linije - poziv Runner-a!
    """
    zoom_path = os.path.join(UPLOAD_DIR, "zoom_image.jpg")
    with open(zoom_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    first_path = os.path.join(UPLOAD_DIR, "first_image.jpg")

    # ✅ Samo pozovi Runner!
    result = await detection_runner.analyze_zoom_step(
        zoom_path,
        prekrsaj_id,
        on_reservation,
        first_path
    )
    return result


# --------------------------------------------------------
# REVIEW ENDPOINTS - samo pozivaju DetectionRunner
# --------------------------------------------------------
@app.post("/record_violation")
async def record_violation(d: Detektovano):
    """
    Potvrđuje prekršaj i čuva za učenje

    PRIJE: 40+ linija logike
    POSLIJE: 1 linija - poziv Runner-a!
    """
    result = await detection_runner.confirm_detection(
        d.vozac_id, d.prekrsaj_id, d.slika1, d.slika2
    )
    return result


@app.post("/record_ok_detection")
async def record_ok_detection(image_path: str = Form(...)):
    """
    Čuva OK detekciju (nema prekršaja)

    PRIJE: 20+ linija logike
    POSLIJE: 1 linija - poziv Runner-a!
    """
    result = await detection_runner.confirm_ok_detection(image_path)
    return result


@app.post("/reject_detection")
async def reject_detection(
        image_path: str = Form(...),
        second_image_path: str = Form(None)
):
    """
    Odbija detekciju (false positive)

    PRIJE: 20+ linija logike
    POSLIJE: 1 linija - poziv Runner-a!
    """
    result = await detection_runner.reject_detection(image_path, second_image_path)
    return result


@app.get("/learning_stats")
def get_learning_stats():
    """
    Statistika spremnosti za učenje

    PRIJE: direktan pristup file system-u
    POSLIJE: poziv Runner-a!
    """
    return retrain_runner.get_learning_stats()


# --------------------------------------------------------
# RETRAINING ENDPOINT - samo poziva RetrainRunner
# --------------------------------------------------------
@app.post("/retrain_model")
async def retrain_model():
    """
    Pokreće retraining modela

    PRIJE: 150+ linija logike ovdje ❌
    POSLIJE: 1 linija - poziv Runner-a! ✅

    OVO JE KLJUČNA RAZLIKA!
    Sva logika (threshold pravila, trening, evaluacija)
    je u RetrainRunner i TrainingService!
    """
    result = await retrain_runner.step_async()
    return result


# --------------------------------------------------------
# CRUD ENDPOINTS - direktan pristup DB (OK za CRUD)
# --------------------------------------------------------
@app.get("/driver/{plate}")
def get_driver(plate: str):
    """Traži vozača po tablici"""
    driver = db_context.get_driver_by_plate(plate)

    if driver:
        return {
            "vozac_id": driver.vozac_id,
            "ime": driver.ime,
            "tablica": driver.tablica,
            "auto_tip": driver.auto_tip,
            "invalid": driver.invalid,
            "rezervacija": driver.rezervacija
        }

    return {"error": "Driver not found"}


@app.post("/add_driver")
def add_driver(driver: Vozac):
    """Dodaje novog vozača"""
    from parking_agent.domain.entities import Driver

    new_driver = Driver(
        vozac_id=0,  # DB će generisati
        ime=driver.ime,
        tablica=driver.tablica,
        auto_tip=driver.auto_tip,
        invalid=driver.invalid,
        rezervacija=driver.rezervacija
    )
    db_context.add_driver(new_driver)
    return {"message": "Vozač uspješno dodan."}


@app.post("/add_violation_type")
def add_violation_type(v: Prekrsaj):
    """Dodaje novi tip prekršaja"""
    from parking_agent.domain.entities import Violation

    violation = Violation(
        prekrsaj_id=0,  # DB će generisati
        opis=v.opis,
        kazna=v.kazna
    )
    db_context.add_violation_type(violation)
    return {"message": "Prekršaj dodan."}


@app.get("/vozaci")
def list_vozaci():
    """Lista svih vozača"""
    drivers = db_context.get_all_drivers()

    return [
        {
            "vozac_id": d.vozac_id,
            "ime": d.ime,
            "tablica": d.tablica,
            "auto_tip": d.auto_tip,
            "invalid": d.invalid,
            "rezervacija": d.rezervacija
        }
        for d in drivers
    ]


@app.get("/prekrsaji")
def list_prekrsaji():
    """Lista svih prekršaja"""
    violations = db_context.get_all_violations()

    return [
        {"prekrsaj_id": v.prekrsaj_id, "opis": v.opis, "kazna": v.kazna}
        for v in violations
    ]


# --------------------------------------------------------
# RUN SERVER
# --------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)