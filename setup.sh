#!/bin/bash
# Quick setup script for manga text extraction

echo "=================================="
echo "Manga Text Extraction - Setup"
echo "=================================="

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q easyocr scikit-learn scipy tqdm opencv-python numpy pandas matplotlib seaborn

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import cv2, numpy, sklearn, easyocr, scipy, tqdm; print('✓ All dependencies installed successfully!')"

# Test on one sequence
echo ""
echo "Testing pipeline on one development sequence..."
python -c "
from src.pipeline import MangaPipeline
import json

pipeline = MangaPipeline({'ocr_confidence': 0.3})
print('✓ Pipeline initialized successfully!')
print('')
print('Note: First run will download EasyOCR models (~150MB)')
print('This is a one-time download and may take a few minutes.')
"

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Train and validate: python train.py"
echo "2. Generate test predictions: python inference.py"
echo "3. Evaluate: python Dataset/score.py --references Dataset/development/labels.jsonl --predictions dev_predictions.jsonl --output scores.json"
echo ""
