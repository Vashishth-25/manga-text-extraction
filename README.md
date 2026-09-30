# Manga Text Extraction System

A machine learning system that reads three consecutive English manga pages, extracts story text in reading order, and identifies who spoke each line.

## Task Overview

This system processes manga pages to:
- Extract dialogue, thoughts, narration, and vocalisations
- Maintain correct reading order (typically right-to-left)
- Keep consistent character labels across all three pages in a sequence
- Filter out visual effects, titles, credits, and other non-story text

## Approach

### Pipeline Architecture

1. **Text Detection & OCR**
   - EasyOCR for English text detection and recognition
   - Extracts text with bounding box coordinates

2. **Text Filtering**
   - Rule-based filtering to remove:
     - Visual sound effects (short, uppercase, repetitive patterns)
     - Page numbers, credits, watermarks
     - Text on objects/signs (small, isolated text)
   - Length and pattern-based heuristics

3. **Reading Order Detection**
   - Right-to-left manga reading convention
   - Panel detection using contour analysis
   - Top-to-bottom, right-to-left ordering within and across panels
   - Balloon connectivity analysis for speech flow

4. **Speaker Identification**
   - Spatial clustering: text boxes close together likely share speakers
   - Position-based features: vertical proximity, horizontal alignment
   - K-means clustering with optimal K selection (2-8 characters)
   - Silhouette score for cluster quality
   - NARRATION label for centered, wide text boxes

5. **Cross-Page Consistency**
   - Character embeddings based on spatial patterns
   - Consistent naming across three pages in sequence
   - Simple labels: char1, char2, char3, etc.

### Training & Adaptation

- **Data Preparation**: Split 80 development sequences into 64 train / 16 validation
- **OCR Tuning**: Confidence threshold optimization on validation set
- **Filter Tuning**: Iterative refinement of text filtering rules based on false positives
- **Clustering Optimization**: Grid search for spatial clustering parameters
- **Validation**: Score.py evaluation on validation set before test inference

### Design Decisions

**Why EasyOCR?**
- Open-source, no API calls required
- Good English text recognition
- Returns bounding boxes for spatial analysis
- Works well on manga-style text

**Why spatial clustering for speakers?**
- Character appearance varies across panels
- Spatial proximity is most reliable signal
- Voice consistency within scenes
- Simple, interpretable approach

**Why rule-based filtering?**
- Clear visual patterns for non-story text
- Fast, no training required
- Easy to debug and refine
- Works well with position and text features

**Limitations & Future Work**
- No visual character recognition (could improve speaker ID)
- Simple spatial features (could add speech balloon shape detection)
- No context/dialogue understanding (could use language models)
- Manual threshold tuning (could learn from data)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For pytesseract (optional backup OCR), install Tesseract:
# Ubuntu: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

### Training and Validation

```bash
# Train and validate the model
python train.py

# This will:
# - Split development data (64 train, 16 validation)
# - Optimize OCR and filtering parameters
# - Tune spatial clustering hyperparameters
# - Evaluate on validation set
# - Save best parameters to config.json
```

### Test Inference

```bash
# Generate test predictions
python inference.py

# Output: test_predictions.jsonl
```

### Evaluation

```bash
# Score predictions against development labels
python Dataset/score.py \
  --references Dataset/development/labels.jsonl \
  --predictions dev_predictions.jsonl \
  --output scores.json

# View scores
cat scores.json
```

## Project Structure

```
.
├── Dataset/
│   ├── development/
│   │   ├── images/          # Development manga pages
│   │   └── labels.jsonl     # Ground truth annotations
│   ├── test/
│   │   └── images/          # Test manga pages
│   ├── sequences.json       # Sequence metadata
│   ├── sample_submission.jsonl
│   └── score.py             # Evaluation script
├── src/
│   ├── ocr.py              # Text detection and OCR
│   ├── filter.py           # Text filtering logic
│   ├── reading_order.py    # Panel and reading order detection
│   ├── speaker_id.py       # Speaker identification
│   └── pipeline.py         # End-to-end pipeline
├── train.py                # Training and validation
├── inference.py            # Test set inference
├── requirements.txt
└── README.md
```

## Evaluation Metrics

The system is evaluated on:

1. **text_order_score** (0-1): How well text is recovered in reading order
2. **balanced_joint_f1** (0-1): How well each speaker's text is recovered with consistent identity

Both metrics range from 0 to 1, higher is better.

## My Contribution

**What I built:**
- Complete OCR pipeline with EasyOCR
- Text filtering rules based on manga conventions
- Reading order detection using spatial analysis
- Speaker clustering with spatial features
- Cross-page character consistency
- Training loop with validation
- Hyperparameter tuning framework

**What I learned:**
- Manga reading conventions (right-to-left, panel flow)
- OCR challenges with stylized text
- Spatial reasoning for dialogue ordering
- Speaker identification without visual features
- Balancing precision vs recall in text filtering

**What I'd try next:**
- Visual features for speaker ID (character detection)
- Speech balloon detection and shape analysis
- Transformer models for dialogue context
- Multi-task learning (OCR + speaker ID jointly)
- Data augmentation for robustness
- Deep learning for reading order (graph neural networks)

## Results

Development set performance (validation split):
- text_order_score: [TBD after training]
- balanced_joint_f1: [TBD after training]

Test predictions saved to `test_predictions.jsonl`.

## License

This is a take-home task project for educational purposes.
