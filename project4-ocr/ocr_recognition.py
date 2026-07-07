"""
Project 4 (Optional): Image / Text Recognition (Basic)
DecodeLabs Industrial Training Kit

Path 1: OCR using pytesseract (Google's Tesseract engine).
Pipeline: Grayscale -> Blur -> Adaptive Threshold -> OCR -> 80% confidence gate.
"""

from pathlib import Path

import cv2
import pytesseract

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "sample_text.png"     # replace with any image containing text
OUTPUT_PATH = BASE_DIR / "processed_output.png"
CONFIDENCE_THRESHOLD = 80          # per spec: 80% is the minimum standard


def preprocess_image(path):
    """Grayscale -> Gaussian blur -> adaptive threshold (Otsu's method)."""
    image_path = Path(path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # Step 1: Grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)            # Step 2: Blur (denoise)
    _, thresh = cv2.threshold(                             # Step 3: Adaptive/Otsu threshold
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


def run_ocr(processed_img):
    """Run Tesseract and return recognized text + per-word confidence data."""
    data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
    return data


def main():
    processed = preprocess_image(IMAGE_PATH)
    cv2.imwrite(str(OUTPUT_PATH), processed)  # visual confirmation artifact

    data = run_ocr(processed)

    print("=== OCR Results (80% Confidence Gate) ===\n")
    accepted_words = []
    for i, word in enumerate(data["text"]):
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else -1
        if word.strip() == "" or conf < 0:
            continue

        if conf >= CONFIDENCE_THRESHOLD:
            print(f"ACCEPTED  '{word}'  (confidence: {conf}%)")
            accepted_words.append(word)
        else:
            print(f"DROPPED   '{word}'  (confidence: {conf}% < {CONFIDENCE_THRESHOLD}%)")

    print("\n--- Final Recognized Text (validated >= 80%) ---")
    print(" ".join(accepted_words) if accepted_words else "(no text passed the confidence gate)")


if __name__ == "__main__":
    main()
