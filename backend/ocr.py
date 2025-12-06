import easyocr
import cv2
import numpy as np
import re


reader = easyocr.Reader(['en'], gpu=False)


def normalize_plate(text):
    """
    Normalizuje bosanske registarske tablice u format:
    XXX-X-XXX
    i ispravlja OCR greške.
    """

    if not text:
        return None

    # 1️⃣ ukloni razmake i posebne znakove
    cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()

    # 2️⃣ ispravi zamjene O ↔ 0 (česte OCR greške)
    cleaned = cleaned.replace("O0", "O")
    cleaned = cleaned.replace("0O", "0")

    # Ako OCR pročita O umjesto 0 u numeričkom dijelu
    if len(cleaned) >= 3 and cleaned[2] == 'O':
        cleaned = cleaned[:2] + '0' + cleaned[3:]

    # 3️⃣ Ako OCR pročita slovo umjesto broja na kraju
    if len(cleaned) >= 6 and cleaned[-3:].isalpha():
        cleaned = cleaned[:-3] + ''.join('0' if c.isalpha() else c for c in cleaned[-3:])

    # 4️⃣ Ako OCR fali znakova → popuniti nulama
    while len(cleaned) < 7:
        cleaned += "0"

    # 5️⃣ Format u oblik XXX-X-XXX
    plate = f"{cleaned[0:3]}-{cleaned[3]}-{cleaned[4:7]}"

    return plate


def read_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 1️⃣ Resize (OCR radi bolje na većoj slici)
    img = cv2.resize(img, None, fx=2.3, fy=2.3)

    # 2️⃣ Pretvori u grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3️⃣ Pojačaj kontrast
    enhanced = cv2.equalizeHist(gray)

    # 4️⃣ Pošalji OCR-u poboljšanu sliku
    results = reader.readtext(enhanced)

    if not results:
        return None

    # Izaberemo NAJDULJI string → obično je to prava tablica
    texts = [r[1] for r in results]
    texts_sorted = sorted(texts, key=len, reverse=True)

    normalized = normalize_plate(texts_sorted[0])
    return normalized
