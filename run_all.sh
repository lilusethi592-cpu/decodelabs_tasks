#!/usr/bin/env bash
set -euo pipefail

# Run all projects using the repository .venv python. This script uses absolute
# path to the venv located at the repo root. It runs each project from the
# appropriate working directory when necessary.

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$ROOT/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "Virtual environment python not found at $VENV_PY"
  echo "Create the venv and install requirements:"
  echo "  python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

echo "--- Running Project 1: Chatbot ---"
printf "hello
bye
" | "$VENV_PY" "$ROOT/project1-chatbot/chatbot.py"

echo "--- Running Project 2: Classification ---"
"$VENV_PY" "$ROOT/project2-classification/classifier.py"

echo "--- Running Project 3: Recommender ---"
(
  cd "$ROOT/project3-recommender"
  "$VENV_PY" recommender.py
)

echo "--- Running Project 4: OCR ---"
"$VENV_PY" "$ROOT/project4-ocr/ocr_recognition.py"

echo "--- All projects ran successfully ---"
