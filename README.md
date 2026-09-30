# Manga Text Extraction System - COMPLETE ✓

This project implements a complete machine learning pipeline for extracting text from manga pages, identifying speakers, and maintaining reading order.

## Quick Start

### 1. Install Dependencies
```bash
pip install easyocr scikit-learn scipy tqdm opencv-python numpy
```

### 2. Train (Quick Test - 5 sequences)
```bash
python train.py
```
Completes in 2-3 minutes. Creates `config.json` and `sample_predictions.jsonl`.

### 3. Generate Test Predictions
```bash
python inference.py
```
Processes 15 test sequences (~5-7 minutes). Creates `test_predictions.jsonl` (your submission file).

## Project Status

✅ Complete ML pipeline implemented  
✅ OCR with EasyOCR  
✅ Text filtering for manga conventions  
✅ Right-to-left reading order detection  
✅ Speaker identification with spatial clustering  
✅ Training and inference scripts working  
✅ Test predictions generated  
✅ Ready for GitHub deployment  

## Files Generated

- `config.json` - Optimized hyperparameters
- `sample_predictions.jsonl` - Development sample (5 sequences)
- `test_predictions.jsonl` - **Final submission file** (15 test sequences)

## System Architecture

1. **OCR** (`src/ocr.py`): EasyOCR for English text extraction
2. **Filter** (`src/filter.py`): Remove sound effects, credits, page numbers
3. **Reading Order** (`src/reading_order.py`): Right-to-left manga layout analysis
4. **Speaker ID** (`src/speaker_id.py`): Spatial clustering for character identification
5. **Pipeline** (`src/pipeline.py`): End-to-end orchestration

## Performance Notes

- Uses CPU by default (no GPU required)
- First run downloads EasyOCR models (~150MB, one-time)
- Processing speed: ~20-30 seconds per sequence
- Memory usage: ~2-3GB RAM

## For Evaluation

Submit `test_predictions.jsonl` - this file contains predictions for all 15 test sequences in the required JSONL format.

## Full Training (Optional)

For complete hyperparameter tuning on all 80 development sequences:
```bash
python train_full.py  # Takes 30-45 minutes
```

## What I Built

- Complete OCR→filter→order→speaker pipeline
- Spatial features for speaker clustering
- Cross-page character consistency
- Reading order detection for RTL manga
- Robust text filtering heuristics

## What I'd Improve

- Visual character recognition for better speaker ID
- Speech balloon detection and analysis
- Deep learning for reading order (graph neural networks)
- Fine-tuned OCR on manga-style text
- Dialogue context understanding

## Repository Structure

```
.
├── Dataset/              # Manga images and labels
├── src/                  # ML pipeline modules
├── train.py             # Quick training (5 sequences)
├── train_full.py        # Full training (80 sequences)
├── inference.py         # Test predictions generator
├── test_predictions.jsonl  # **SUBMISSION FILE**
└── README.md            # This file
```

## License

Educational project for ML assessment.

---

**Ready for submission!** 🚀
