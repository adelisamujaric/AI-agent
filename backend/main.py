import os
import sqlite3
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
from backend.database import init_db, DB_PATH
from pydantic import BaseModel
from backend.ocr import read_plate
from backend.utils import crop_plate
from ultralytics import YOLO
import yaml

# Folder za odbijene detekcije
REJECTED_DIR = "backend/rejected"
os.makedirs(os.path.join(REJECTED_DIR, "first"), exist_ok=True)
os.makedirs(os.path.join(REJECTED_DIR, "zoom"), exist_ok=True)

# 🆕 Folder za potvrđene detekcije (učenje)
CONFIRMED_DIR = "backend/confirmed"
os.makedirs(os.path.join(CONFIRMED_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(CONFIRMED_DIR, "labels"), exist_ok=True)

app = FastAPI()
init_db()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kreiraj uploads folder ako ne postoji
UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# YOLO model
model = YOLO("backend/weights/best.pt")


# --------------------------------------------------------
# BASIC TEST ROUTE
# --------------------------------------------------------
@app.get("/")
def root():
    return {"message": "AI Parking Agent backend is running!"}


# --------------------------------------------------------
# YOLO DETECT – for bounding boxes on frontend
# --------------------------------------------------------
@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, "temp_image.jpg")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_path)
    detections = []

    for box in results[0].boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        detections.append({
            "box": xyxy,
            "class": cls_name,
            "confidence": conf
        })

    return {"detections": detections}


# --------------------------------------------------------
# OCR DETECTION (optional)
# --------------------------------------------------------
@app.post("/detect_plate")
async def detect_plate(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, "temp_plate_source.jpg")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_path)
    boxes = results[0].boxes

    plate_box = None
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        if cls_name.lower() == "tablica":
            plate_box = box.xyxy[0].tolist()
            break

    if plate_box is None:
        return {"plate": None, "error": "Plate not detected"}

    crop_path = crop_plate(temp_path, plate_box)
    plate_text = read_plate(crop_path)

    return {
        "plate": plate_text if plate_text else "Not detected",
        "bbox": plate_box
    }


# --------------------------------------------------------
# DRIVER LOOKUP
# --------------------------------------------------------
@app.get("/driver/{plate}")
def get_driver(plate: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM vozac WHERE tablica = ?", (plate,))
    driver = cursor.fetchone()
    conn.close()

    if driver:
        return {
            "vozac_id": driver[0],
            "ime": driver[1],
            "tablica": driver[2],
            "auto_tip": driver[3],
            "invalid": bool(driver[4]),
            "rezervacija": bool(driver[5])
        }

    return {"error": "Driver not found"}


# --------------------------------------------------------
# ADD DRIVER
# --------------------------------------------------------
class Vozac(BaseModel):
    ime: str
    tablica: str
    auto_tip: str
    invalid: bool = False
    rezervacija: bool = False


@app.post("/add_driver")
def add_driver(driver: Vozac):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO vozac (ime, tablica, auto_tip, invalid, rezervacija)
        VALUES (?, ?, ?, ?, ?)
    """, (driver.ime, driver.tablica, driver.auto_tip, int(driver.invalid), int(driver.rezervacija)))
    conn.commit()
    conn.close()
    return {"message": "Vozac uspješno dodan."}


# --------------------------------------------------------
# ADD VIOLATION TYPE
# --------------------------------------------------------
class Prekrsaj(BaseModel):
    opis: str
    kazna: int


@app.post("/add_violation_type")
def add_violation_type(v: Prekrsaj):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prekrsaji (opis, kazna)
        VALUES (?, ?)
    """, (v.opis, v.kazna))
    conn.commit()
    conn.close()
    return {"message": "Prekrsaj dodan."}


# --------------------------------------------------------
# NEW: GET ALL DRIVERS
# --------------------------------------------------------
@app.get("/vozaci")
def list_vozaci():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vozac")
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "vozac_id": r[0],
            "ime": r[1],
            "tablica": r[2],
            "auto_tip": r[3],
            "invalid": bool(r[4]),
            "rezervacija": bool(r[5])
        }
        for r in rows
    ]


# --------------------------------------------------------
# NEW: GET ALL VIOLATIONS
# --------------------------------------------------------
@app.get("/prekrsaji")
def list_prekrsaji():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT prekrsaj_id, opis, kazna FROM prekrsaji")
    rows = cursor.fetchall()
    conn.close()

    return [
        {"prekrsaj_id": r[0], "opis": r[1], "kazna": r[2]}
        for r in rows
    ]


# --------------------------------------------------------
# 🆕 HELPER: Save YOLO labels from detections
# --------------------------------------------------------
def save_yolo_labels(image_path: str, label_path: str):
    """
    Generise YOLO anotacije za KEY PARKING KLASE
    """
    results = model(image_path)
    boxes = results[0].boxes

    # 🔴 PARKING ENFORCEMENT KLASE
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

    old_class_names = model.names

    valid_boxes = []
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = old_class_names[cls_id]

        if cls_name in class_mapping:
            new_cls_id = class_mapping[cls_name]
            valid_boxes.append((box, new_cls_id))

    if len(valid_boxes) == 0:
        print(f"⚠️ Nema relevantnih parking detekcija u {image_path}")
        with open(label_path, 'w') as f:
            pass
        return

    with open(label_path, 'w') as f:
        for box, new_cls_id in valid_boxes:
            xyxy = box.xyxy[0].tolist()
            img_h, img_w = results[0].orig_shape
            x1, y1, x2, y2 = xyxy

            center_x = ((x1 + x2) / 2) / img_w
            center_y = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            f.write(f"{new_cls_id} {center_x} {center_y} {width} {height}\n")

    print(f"✅ Sačuvano {len(valid_boxes)} parking-relevantnih detekcija")
# --------------------------------------------------------
# RECORD CONFIRMED VIOLATION (🆕 SA UČENJEM)
# --------------------------------------------------------
class Detektovano(BaseModel):
    vozac_id: int
    prekrsaj_id: int
    slika1: str
    slika2: str | None = None


@app.post("/record_violation")
def record_violation(d: Detektovano):
    """
    Evidentira prekršaj U BAZU + čuva slike za učenje
    """
    # 1. Sačuvaj u bazu
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO detektovano (vozac_id, prekrsaj_id, vrijeme, slika1, slika2)
        VALUES (?, ?, ?, ?, ?)
    """, (d.vozac_id, d.prekrsaj_id, timestamp, d.slika1, d.slika2))
    conn.commit()
    conn.close()

    # 2. 🆕 ČUVANJE ZA UČENJE - Kopiraj slike + generiši YOLO labele
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prva slika
    if d.slika1 and os.path.exists(d.slika1):
        new_img_name = f"confirmed_{timestamp_file}_first.jpg"
        new_img_path = os.path.join(CONFIRMED_DIR, "images", new_img_name)
        shutil.copy(d.slika1, new_img_path)

        # Generiši YOLO label
        label_name = new_img_name.replace('.jpg', '.txt')
        label_path = os.path.join(CONFIRMED_DIR, "labels", label_name)
        save_yolo_labels(d.slika1, label_path)

        print(f"✅ Sačuvana prva slika za učenje: {new_img_name}")

    # Druga slika (zoom)
    if d.slika2 and os.path.exists(d.slika2):
        new_img_name = f"confirmed_{timestamp_file}_zoom.jpg"
        new_img_path = os.path.join(CONFIRMED_DIR, "images", new_img_name)
        shutil.copy(d.slika2, new_img_path)

        label_name = new_img_name.replace('.jpg', '.txt')
        label_path = os.path.join(CONFIRMED_DIR, "labels", label_name)
        save_yolo_labels(d.slika2, label_path)

        print(f"✅ Sačuvana druga slika za učenje: {new_img_name}")

    return {"message": "Prekršaj evidentiran i dodan u trening dataset."}


# --------------------------------------------------------
# 🆕 RECORD OK DETECTION (nema prekršaja)
# --------------------------------------------------------
@app.post("/record_ok_detection")
async def record_ok_detection(image_path: str = Form(...)):
    """
    Čuva sliku kad nema prekršaja (potvrđena OK detekcija)
    """
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Slika ne postoji"}

    # Timestamp za ime fajla
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Kopiraj sliku
    new_img_name = f"confirmed_{timestamp_file}_ok.jpg"
    new_img_path = os.path.join(CONFIRMED_DIR, "images", new_img_name)
    shutil.copy(image_path, new_img_path)

    # Generiši YOLO label
    label_name = new_img_name.replace('.jpg', '.txt')
    label_path = os.path.join(CONFIRMED_DIR, "labels", label_name)
    save_yolo_labels(image_path, label_path)

    print(f"✅ Sačuvana OK detekcija za učenje: {new_img_name}")

    return {"status": "success", "message": "OK detekcija sačuvana za učenje"}
# --------------------------------------------------------
# REJECT VIOLATION (save for later analysis)
# --------------------------------------------------------
@app.post("/reject_violation")
def reject_violation(d: Detektovano):
    """
    Kopiraj slike u rejected folder za buduće treniranje modela
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Kopiraj prvu sliku
    if d.slika1 and os.path.exists(d.slika1):
        new_name = f"rejected_first_{timestamp}.jpg"
        shutil.copy(d.slika1, os.path.join(REJECTED_DIR, "first", new_name))

    # Kopiraj zoom sliku
    if d.slika2 and os.path.exists(d.slika2):
        new_name = f"rejected_zoom_{timestamp}.jpg"
        shutil.copy(d.slika2, os.path.join(REJECTED_DIR, "zoom", new_name))

    return {"message": "Odbijene slike sačuvane za treniranje."}


# --------------------------------------------------------
# REJECT VIOLATION (save for later analysis)
# --------------------------------------------------------
# --------------------------------------------------------
# 🆕 REJECT DETECTION - Pojednostavljen (UVIJEK RADI!)
# --------------------------------------------------------
@app.post("/reject_detection")
async def reject_detection(
        image_path: str = Form(...),
        second_image_path: str = Form(None)
):
    """
    Čuva odbijene slike - radi uvijek bez obzira na state!
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_count = 0

    # Kopiraj first image
    if os.path.exists(image_path):
        new_name = f"rejected_first_{timestamp}.jpg"
        dest_path = os.path.join(REJECTED_DIR, "first", new_name)
        shutil.copy(image_path, dest_path)
        print(f"✅ Odbačena first image: {new_name}")
        saved_count += 1
    else:
        print(f"⚠️ Slika ne postoji: {image_path}")

    # Kopiraj second image (ako postoji)
    if second_image_path and os.path.exists(second_image_path):
        new_name = f"rejected_zoom_{timestamp}.jpg"
        dest_path = os.path.join(REJECTED_DIR, "zoom", new_name)
        shutil.copy(second_image_path, dest_path)
        print(f"✅ Odbačena zoom image: {new_name}")
        saved_count += 1

    return {
        "status": "success",
        "message": f"Odbačeno {saved_count} slika",
        "count": saved_count
    }

# --------------------------------------------------------
# 🆕 LEARNING STATS - Check how many images are ready
# --------------------------------------------------------
@app.get("/learning_stats")
def get_learning_stats():
    """
    Provjerava koliko novih slika ima za učenje
    """
    confirmed_images = len([f for f in os.listdir(os.path.join(CONFIRMED_DIR, "images")) if f.endswith('.jpg')])
    rejected_first = len([f for f in os.listdir(os.path.join(REJECTED_DIR, "first")) if f.endswith('.jpg')])
    rejected_zoom = len([f for f in os.listdir(os.path.join(REJECTED_DIR, "zoom")) if f.endswith('.jpg')])

    return {
        "confirmed_images": confirmed_images,
        "rejected_first": rejected_first,
        "rejected_zoom": rejected_zoom,
        "ready_for_retraining": confirmed_images >= 10  # 🔴 PROMIJENJEN PRAG SA 50 NA 10
    }


# --------------------------------------------------------
# 🆕 RETRAIN MODEL - Main learning endpoint
# --------------------------------------------------------
# --------------------------------------------------------
# 🆕 RETRAIN MODEL - Main learning endpoint
# --------------------------------------------------------
@app.post("/retrain_model")
async def retrain_model():
    """
    Pokreće retraining modela sa novim podacima iz confirmed foldera
    """
    confirmed_images = os.listdir(os.path.join(CONFIRMED_DIR, "images"))

    if len(confirmed_images) < 10:
        return {
            "status": "NOT_ENOUGH_DATA",
            "message": f"Potrebno je minimum 10 slika, trenutno ima {len(confirmed_images)}"
        }

    print(f"🚀 Pokrećem retraining sa {len(confirmed_images)} novih slika...")

    # 1. Kreiraj data.yaml config sa APSOLUTNIM putanjama
    config = {
        'path': os.path.abspath(CONFIRMED_DIR),  # ← APSOLUTNA!
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

    print(f"📝 Config path: {config_path}")
    print(f"📁 Dataset path: {config['path']}")

    # 2. Učitaj trenutni model
    current_model = YOLO("backend/weights/best.pt")

    # 3. Sačuvaj backup starog modela
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"backend/weights/backup_model_{timestamp}.pt"
    shutil.copy("backend/weights/best.pt", backup_path)

    print(f"💾 Backup starog modela: {backup_path}")

    # 4. Fine-tuning sa novim podacima
    print("🏋️ Pokrećem treniranje...")

    try:
        results = current_model.train(
            data=config_path,
            epochs=20,
            imgsz=640,
            batch=8,
            project='backend/retraining_runs',
            name=f'retrain_{timestamp}',
            exist_ok=True
        )

        # 5. Evaluacija NOVOG modela na novim podacima
        print("📊 Evaluiram novi model...")
        new_metrics = current_model.val(data=config_path)
        new_map50 = new_metrics.box.map50

        # 6. Evaluacija STAROG modela TAKOĐE na novim podacima
        print("📊 Evaluiram stari model na istim podacima...")
        old_model = YOLO(backup_path)
        old_metrics = old_model.val(data=config_path)  # ← ISTO config!
        old_map50 = old_metrics.box.map50

        print(f"📈 Stari model mAP50: {old_map50:.3f}")
        print(f"📈 Novi model mAP50: {new_map50:.3f}")

        # 7. Zamijeni model ako je bolji
        if new_map50 > old_map50:
            print(f"✅ Novi model je bolji! Ažuriram...")

            # Kopiraj novi model preko starog
            new_model_path = f"backend/retraining_runs/retrain_{timestamp}/weights/best.pt"
            shutil.copy(new_model_path, "backend/weights/best.pt")

            # Reload model u memoriji
            global model
            model = YOLO("backend/weights/best.pt")

            # Arhiviraj korištene confirmed slike
            archive_dir = f"backend/confirmed_archive/{timestamp}"
            os.makedirs(archive_dir, exist_ok=True)

            shutil.move(os.path.join(CONFIRMED_DIR, "images"), os.path.join(archive_dir, "images"))
            shutil.move(os.path.join(CONFIRMED_DIR, "labels"), os.path.join(archive_dir, "labels"))

            # Kreiraj nove prazne foldere
            os.makedirs(os.path.join(CONFIRMED_DIR, "images"), exist_ok=True)
            os.makedirs(os.path.join(CONFIRMED_DIR, "labels"), exist_ok=True)

            return {
                "status": "SUCCESS",
                "message": f"Model uspješno ažuriran! mAP50: {old_map50:.3f} → {new_map50:.3f}",
                "old_map50": float(old_map50),
                "new_map50": float(new_map50),
                "improvement": float(new_map50 - old_map50)
            }
        else:
            print(f"⚠️ Novi model nije bolji. Zadržavam stari.")

            # Restore backup
            shutil.copy(backup_path, "backend/weights/best.pt")

            return {
                "status": "NO_IMPROVEMENT",
                "message": f"Novi model nije pokazao poboljšanje. Zadržan stari model.",
                "old_map50": float(old_map50),
                "new_map50": float(new_map50)
            }

    except Exception as e:
        print(f"❌ Greška pri treniranju: {e}")

        # Restore backup ako je nešto pošlo po zlu
        if os.path.exists(backup_path):
            shutil.copy(backup_path, "backend/weights/best.pt")

        return {
            "status": "ERROR",
            "message": f"Greška pri treniranju: {str(e)}"
        }
# --------------------------------------------------------
# FIRST IMAGE – CHECK VIOLATION
# --------------------------------------------------------
@app.post("/analyze_first_image")
async def analyze_first_image(file: UploadFile = File(...)):
    first_path = os.path.join(UPLOAD_DIR, "first_image.jpg")
    with open(first_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(first_path)
    boxes = results[0].boxes
    class_names = model.names

    # Skupljamo detekcije
    has_reservation_sign = False  # RezervacijaOznaka
    has_auto = False  # Auto
    has_occupied_spot = False  # ZauzetoMjesto
    violations = []

    for box in boxes:
        cls_name = class_names[int(box.cls[0])]

        if cls_name == "RezervacijaOznaka":
            has_reservation_sign = True
        elif cls_name == "Auto":
            has_auto = True
        elif cls_name == "ZauzetoMjesto":
            has_occupied_spot = True
        elif cls_name.startswith("NepropisnoParkirano"):
            violations.append(cls_name)

    # Proveri da li je auto NA REZERVACIJI
    car_on_reservation = (has_reservation_sign and has_auto and has_occupied_spot)

    # --- LOGIKA DETEKCIJE ---

    # 1. Auto na rezervaciji + IMA prekršaj (nepravilno parkiran)
    if car_on_reservation and violations:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        main_violation = violations[0]
        cursor.execute("SELECT prekrsaj_id FROM prekrsaji WHERE opis = ?", (main_violation,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "status": "NEEDS_ZOOM",
                "prekrsaj_id": row[0],
                "detected_violation": main_violation,
                "on_reservation": True,
                "message": f"⚠️ Prekršaj detektovan: {main_violation} + Auto je na rezervacijskom mestu - provjeri tablicu"
            }

    # 2. Auto na rezervaciji + BEZ prekršaja (pravilno parkiran)
    if car_on_reservation:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT prekrsaj_id FROM prekrsaji WHERE opis = ?", ("Parkiranje_na_rezervisanom_mjestu",))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "status": "NEEDS_ZOOM",
                "prekrsaj_id": row[0],
                "detected_violation": "Parkiranje_na_rezervisanom_mjestu",
                "on_reservation": True,
                "message": "🅿️ Auto na rezervacijskom mestu - provjeri tablicu"
            }

    # 3. Standardni prekršaji (ne na rezervaciji)
    if violations:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        main_violation = violations[0]
        cursor.execute("SELECT prekrsaj_id FROM prekrsaji WHERE opis = ?", (main_violation,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "status": "NEEDS_ZOOM",
                "prekrsaj_id": row[0],
                "detected_violation": main_violation,
                "on_reservation": False,
                "message": f"⚠️ Prekršaj detektovan: {main_violation} - približi se za tablicu"
            }

    # 4. Sve je OK
    return {"status": "OK", "message": "✅ Nema prekršaja - pravilno parkirano"}


# --------------------------------------------------------
# ZOOM IMAGE
# --------------------------------------------------------
@app.post("/analyze_zoom_image")
async def analyze_zoom_image(
        file: UploadFile = File(...),
        prekrsaj_id: int = Form(...),
        on_reservation: bool = Form(False)
):
    zoom_path = os.path.join(UPLOAD_DIR, "zoom_image.jpg")
    with open(zoom_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(zoom_path)
    boxes = results[0].boxes

    plate_box = None
    for box in boxes:
        cls_name = model.names[int(box.cls[0])].lower()
        if cls_name == "tablica":
            plate_box = box.xyxy[0].tolist()
            break

    if not plate_box:
        return {"status": "NO_PLATE"}

    crop_path = crop_plate(zoom_path, plate_box)
    plate_text = read_plate(crop_path) or "Unknown"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM vozac WHERE tablica = ?", (plate_text,))
    driver = cursor.fetchone()

    cursor.execute("SELECT opis, kazna FROM prekrsaji WHERE prekrsaj_id = ?", (prekrsaj_id,))
    prek = cursor.fetchone()

    if not driver:
        conn.close()
        return {"status": "NO_DRIVER", "plate": plate_text}

    # PROVERA ZA REZERVACIJU
    if on_reservation:
        # Vozač IMA rezervaciju - parkiranje OK
        if driver[5]:
            conn.close()
            return {
                "status": "OK_WITH_RESERVATION",
                "message": f"✅ Vozač {driver[1]} ima rezervaciju - parkiranje dozvoljeno",
                "plate": plate_text,
                "vozac": {
                    "vozac_id": driver[0],
                    "ime": driver[1],
                    "tablica": driver[2],
                    "auto_tip": driver[3],
                    "invalid": bool(driver[4]),
                    "rezervacija": bool(driver[5])
                }
            }

        # Vozač NEMA rezervaciju - DODATNA kazna!
        cursor.execute("SELECT prekrsaj_id, kazna FROM prekrsaji WHERE opis = ?",
                       ("Parkiranje_na_rezervisanom_mjestu",))
        reservation_violation = cursor.fetchone()

        conn.close()

        # Saberi kazne
        total_fine = prek[1] + (reservation_violation[1] if reservation_violation else 0)

        return {
            "status": "READY_TO_CONFIRM",
            "plate": plate_text,
            "vozac": {
                "vozac_id": driver[0],
                "ime": driver[1],
                "tablica": driver[2],
                "auto": driver[3],
                "invalid": bool(driver[4]),
                "rezervacija": bool(driver[5])
            },
            "prekrsaj_opis": f"{prek[0]} + Parkiranje bez rezervacije",
            "prekrsaj_kazna": total_fine,
            "prekrsaj_id": prekrsaj_id,
            "extra_violation_id": reservation_violation[0] if reservation_violation else None,
            "slika1": os.path.join(UPLOAD_DIR, "first_image.jpg"),
            "slika2": zoom_path,
        }

    # Standardni prekršaj (bez rezervacije)
    conn.close()
    return {
        "status": "READY_TO_CONFIRM",
        "plate": plate_text,
        "vozac": {
            "vozac_id": driver[0],
            "ime": driver[1],
            "tablica": driver[2],
            "auto_tip": driver[3],
            "invalid": bool(driver[4]),
            "rezervacija": bool(driver[5])
        },
        "prekrsaj_opis": prek[0],
        "prekrsaj_kazna": prek[1],
        "prekrsaj_id": prekrsaj_id,
        "slika1": os.path.join(UPLOAD_DIR, "first_image.jpg"),
        "slika2": zoom_path,
    }


# RUN SERVER
# --------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)