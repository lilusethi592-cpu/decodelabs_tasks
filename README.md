# Artificial Intelligence — Industrial Training Kit (Batch 2026)
**DecodeLabs**

Four-project internship track: rule-based logic → supervised classification →
content-based recommendation → basic image/text recognition (optional).

## Structure
```
ai-internship/
├── project1-chatbot/          # Rule-Based AI Chatbot
│   └── chatbot.py
├── project2-classification/   # Data Classification Using AI (Iris + KNN)
│   └── classifier.py
├── project3-recommender/      # AI Recommendation Logic (TF-IDF + Cosine Similarity)
│   ├── recommender.py
│   └── raw_skills.csv
├── project4-ocr/              # Image/Text Recognition (Optional)
│   ├── ocr_recognition.py
│   └── sample_text.png
├── requirements.txt
└── .gitignore
```

## Setup
```bash
pip install -r requirements.txt
```
Project 4 also needs the Tesseract OCR system binary:
```bash
sudo apt install tesseract-ocr
```

## Run each project
```bash
python3 project1-chatbot/chatbot.py
python3 project2-classification/classifier.py
python3 project3-recommender/recommender.py
python3 project4-ocr/ocr_recognition.py
```

## Project Summaries

**Project 1 — Rule-Based AI Chatbot**
Dictionary-based intent matching (`responses.get()`), continuous input loop,
clean exit command. O(1) lookup instead of an if-elif ladder.

**Project 2 — Data Classification Using AI**
Iris dataset (150 samples, 3 classes) → StandardScaler → 80/20 train-test split
→ K-Nearest Neighbors (k=5) → confusion matrix + F1 score.

**Project 3 — AI Recommendation Logic**
Content-based filtering. User skills and job-role skills are vectorized into
a shared TF-IDF space; Cosine Similarity ranks the closest matches (angle, not
raw distance, so vector length doesn't skew results).

**Project 4 — Image/Text Recognition (Optional)**
OCR path: grayscale → Gaussian blur → Otsu adaptive threshold → Tesseract →
per-word confidence gate (drops anything under 80%).
