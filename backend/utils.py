import os
import cv2

def crop_plate(image_path, bbox, margin=15):
    """
    Crops plate from image with extra margin to improve OCR.
    bbox = [x1, y1, x2, y2]
    """

    img = cv2.imread(image_path)
    if img is None:
        return None

    x1, y1, x2, y2 = map(int, bbox)

    # Add margin around the crop
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img.shape[1], x2 + margin)
    y2 = min(img.shape[0], y2 + margin)

    crop = img[y1:y2, x1:x2]

    # Save to uploads folder
    crop_path = os.path.join("backend", "uploads", "temp_plate_crop.jpg")
    cv2.imwrite(crop_path, crop)

    return crop_path