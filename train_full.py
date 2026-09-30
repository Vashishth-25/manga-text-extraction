"""Training script with validation."""
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import subprocess

# Ensure utf-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.pipeline import MangaPipeline, load_config, save_config


def split_development_data(sequences_path: str, labels_path: str,
                          train_ratio: float = 0.8, random_state: int = 42):
    """Split development data into train and validation sets.

    Args:
        sequences_path: Path to sequences.json
        labels_path: Path to development labels.jsonl
        train_ratio: Ratio of data to use for training
        random_state: Random seed

    Returns:
        train_sequences, val_sequences, train_labels, val_labels
    """
    # Load sequences
    with open(sequences_path, 'r', encoding='utf-8') as f:
        sequences = json.load(f)

    # Filter development sequences
    dev_sequences = [seq for seq in sequences if seq['split'] == 'development']

    # Load labels
    labels = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                label = json.loads(line)
                labels[label['sequence_id']] = label

    # Split sequences
    sequence_ids = [seq['sequence_id'] for seq in dev_sequences]
    train_ids, val_ids = train_test_split(
        sequence_ids,
        train_size=train_ratio,
        random_state=random_state
    )

    # Split sequences and labels
    train_sequences = [seq for seq in dev_sequences if seq['sequence_id'] in train_ids]
    val_sequences = [seq for seq in dev_sequences if seq['sequence_id'] in val_ids]

    train_labels = {sid: labels[sid] for sid in train_ids}
    val_labels = {sid: labels[sid] for sid in val_ids}

    return train_sequences, val_sequences, train_labels, val_labels


def evaluate_predictions(predictions_path: str, references_path: str, output_path: str):
    """Run evaluation script and return scores.

    Args:
        predictions_path: Path to predictions JSONL
        references_path: Path to ground truth JSONL
        output_path: Path to save scores

    Returns:
        Dictionary with macro scores
    """
    # Run evaluation script
    cmd = [
        'python', 'Dataset/score.py',
        '--references', references_path,
        '--predictions', predictions_path,
        '--output', output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Evaluation failed: {result.stderr}")
        return None

    # Load scores
    with open(output_path, 'r', encoding='utf-8') as f:
        scores = json.load(f)

    return scores['macro']


def tune_hyperparameters(train_sequences, val_sequences, val_labels_path, base_path):
    """Simple hyperparameter tuning on validation set.

    Args:
        train_sequences: Training sequences (not used for now)
        val_sequences: Validation sequences
        val_labels_path: Path to validation labels
        base_path: Base path for dataset

    Returns:
        Best configuration
    """
    print("Tuning hyperparameters on validation set...")

    # Grid of hyperparameters to try
    param_grid = {
        'ocr_confidence': [0.2, 0.3, 0.4],
        'min_speakers': [2],
        'max_speakers': [6, 8, 10]
    }

    best_score = -1
    best_config = None

    # Try each configuration
    for ocr_conf in param_grid['ocr_confidence']:
        for max_spk in param_grid['max_speakers']:
            config = {
                'ocr_confidence': ocr_conf,
                'rtl_threshold': 0.5,
                'min_speakers': param_grid['min_speakers'][0],
                'max_speakers': max_spk
            }

            print(f"\nTrying config: {config}")

            # Create pipeline with this config
            pipeline = MangaPipeline(config)

            # Save validation sequences to temp file
            val_sequences_path = 'temp_val_sequences.json'
            with open(val_sequences_path, 'w', encoding='utf-8') as f:
                json.dump(val_sequences, f)

            # Process validation set
            val_predictions_path = 'temp_val_predictions.jsonl'
            predictions = []
            for seq_info in val_sequences:
                try:
                    result = pipeline.process_sequence(seq_info, base_path)
                    predictions.append(result)
                except Exception as e:
                    print(f"Error processing {seq_info['sequence_id']}: {e}")
                    predictions.append({
                        'sequence_id': seq_info['sequence_id'],
                        'pages': [[], [], []]
                    })

            # Save predictions
            with open(val_predictions_path, 'w', encoding='utf-8') as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + '\n')

            # Evaluate
            val_scores_path = 'temp_val_scores.json'
            scores = evaluate_predictions(val_predictions_path, val_labels_path, val_scores_path)

            if scores:
                # Use balanced_joint_f1 as primary metric
                score = scores.get('balanced_joint_f1', 0) or 0
                text_score = scores.get('text_order_score', 0) or 0

                print(f"balanced_joint_f1: {score:.4f}, text_order_score: {text_score:.4f}")

                # Combined score
                combined_score = 0.6 * score + 0.4 * text_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_config = config
                    print(f"New best config! Combined score: {combined_score:.4f}")

    print(f"\nBest configuration: {best_config}")
    print(f"Best combined score: {best_score:.4f}")

    return best_config


def main():
    """Main training script."""
    print("=" * 60)
    print("Manga Text Extraction - Training and Validation")
    print("=" * 60)

    # Paths
    sequences_path = 'Dataset/sequences.json'
    labels_path = 'Dataset/development/labels.jsonl'
    base_path = 'Dataset'

    # Split data
    print("\n1. Splitting development data (80% train, 20% validation)...")
    train_sequences, val_sequences, train_labels, val_labels = split_development_data(
        sequences_path, labels_path, train_ratio=0.8, random_state=42
    )

    print(f"   Train sequences: {len(train_sequences)}")
    print(f"   Validation sequences: {len(val_sequences)}")

    # Save validation labels
    val_labels_path = 'val_labels.jsonl'
    with open(val_labels_path, 'w', encoding='utf-8') as f:
        for label in val_labels.values():
            f.write(json.dumps(label) + '\n')

    print(f"   Saved validation labels to {val_labels_path}")

    # Tune hyperparameters (simple grid search on validation set)
    print("\n2. Tuning hyperparameters...")
    best_config = tune_hyperparameters(train_sequences, val_sequences, val_labels_path, base_path)

    # Save best configuration
    config_path = 'config.json'
    save_config(best_config, config_path)
    print(f"\n3. Saved best configuration to {config_path}")

    # Final evaluation on full development set
    print("\n4. Evaluating on full development set with best config...")
    pipeline = MangaPipeline(best_config)
    pipeline.process_dataset(
        sequences_path=sequences_path,
        base_path=base_path,
        output_path='dev_predictions.jsonl',
        split='development'
    )

    # Evaluate
    final_scores = evaluate_predictions(
        'dev_predictions.jsonl',
        labels_path,
        'dev_scores.json'
    )

    print("\n" + "=" * 60)
    print("FINAL DEVELOPMENT SET RESULTS")
    print("=" * 60)
    if final_scores:
        print(f"text_order_score: {final_scores.get('text_order_score', 0):.4f}")
        print(f"balanced_joint_f1: {final_scores.get('balanced_joint_f1', 0):.4f}")
        print(f"speaker_accuracy_on_matched: {final_scores.get('speaker_accuracy_on_matched', 0):.4f}")
        print(f"joint_f1: {final_scores.get('joint_f1', 0):.4f}")
    print("=" * 60)

    print("\nTraining complete! Use inference.py to generate test predictions.")


if __name__ == '__main__':
    main()
