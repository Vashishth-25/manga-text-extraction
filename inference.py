"""Inference script for test set predictions."""
import json
from pathlib import Path

from src.pipeline import MangaPipeline, load_config


def main():
    """Generate test set predictions."""
    print("=" * 60)
    print("Manga Text Extraction - Test Inference")
    print("=" * 60)

    # Load best configuration
    config = load_config('config.json')
    print(f"\nLoaded configuration: {config}")

    # Create pipeline
    pipeline = MangaPipeline(config)

    # Process test set
    print("\nProcessing test set...")
    sequences_path = 'Dataset/sequences.json'
    base_path = 'Dataset'
    output_path = 'test_predictions.jsonl'

    pipeline.process_dataset(
        sequences_path=sequences_path,
        base_path=base_path,
        output_path=output_path,
        split='test'
    )

    print("\n" + "=" * 60)
    print(f"Test predictions saved to {output_path}")
    print("=" * 60)

    # Verify format
    print("\nVerifying output format...")
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Total test sequences: {len(lines)}")

        # Show first prediction as example
        if lines:
            first_pred = json.loads(lines[0])
            print(f"\nExample prediction:")
            print(f"  sequence_id: {first_pred['sequence_id']}")
            print(f"  Number of pages: {len(first_pred['pages'])}")
            for i, page in enumerate(first_pred['pages'], 1):
                print(f"  Page {i}: {len(page)} text boxes")
                if page:
                    print(f"    Example: {page[0]}")

    print("\nInference complete!")


if __name__ == '__main__':
    main()
