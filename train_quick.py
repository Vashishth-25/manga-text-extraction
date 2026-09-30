"""Simple training script - runs just a few sequences to validate the system."""
import sys
import json
from pathlib import Path

# Ensure utf-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.pipeline import MangaPipeline, save_config

print("=" * 60)
print("Manga Text Extraction - Quick Training")
print("=" * 60)

# Use a simple, tested configuration
config = {
    'ocr_confidence': 0.3,
    'rtl_threshold': 0.5,
    'min_speakers': 2,
    'max_speakers': 8
}

print("\nUsing configuration:")
print(json.dumps(config, indent=2))

# Save config
save_config(config, 'config.json')
print("\nSaved configuration to config.json")

# Create pipeline
print("\nInitializing pipeline...")
pipeline = MangaPipeline(config)
print("Pipeline initialized!")

# Process a few development sequences as validation
print("\nProcessing sample development sequences...")
with open('Dataset/sequences.json', 'r', encoding='utf-8') as f:
    sequences = json.load(f)

dev_sequences = [seq for seq in sequences if seq['split'] == 'development'][:5]

predictions = []
for i, seq_info in enumerate(dev_sequences, 1):
    print(f"  Processing {i}/{len(dev_sequences)}: {seq_info['sequence_id']}")
    try:
        result = pipeline.process_sequence(seq_info, 'Dataset')
        predictions.append(result)
    except Exception as e:
        print(f"    Error: {e}")
        predictions.append({
            'sequence_id': seq_info['sequence_id'],
            'pages': [[], [], []]
        })

# Save sample predictions
output_path = 'sample_predictions.jsonl'
with open(output_path, 'w', encoding='utf-8') as f:
    for pred in predictions:
        f.write(json.dumps(pred) + '\n')

print(f"\nSaved sample predictions to {output_path}")

print("\n" + "=" * 60)
print("Quick training complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Review sample_predictions.jsonl to see the output format")
print("2. Run full training: python train_full.py (processes all 80 sequences)")
print("3. Generate test predictions: python inference.py")
print("\nConfiguration saved to config.json and ready for inference!")
