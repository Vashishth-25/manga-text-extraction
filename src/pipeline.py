"""End-to-end pipeline for manga text extraction."""
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from src.ocr import MangaOCR, get_image_dimensions
from src.filter import TextFilter
from src.reading_order import ReadingOrderDetector
from src.speaker_id import SpeakerIdentifier


class MangaPipeline:
    """Complete pipeline for manga text extraction."""

    def __init__(self, config: Dict = None):
        """Initialize pipeline with components.

        Args:
            config: Configuration dictionary with hyperparameters
        """
        self.config = config or {}

        # Initialize components
        self.ocr = MangaOCR(
            confidence_threshold=self.config.get('ocr_confidence', 0.3)
        )
        self.filter = TextFilter()
        self.reading_order = ReadingOrderDetector(
            rtl_threshold=self.config.get('rtl_threshold', 0.5)
        )
        self.speaker_id = SpeakerIdentifier(
            min_clusters=self.config.get('min_speakers', 2),
            max_clusters=self.config.get('max_speakers', 8)
        )

    def process_sequence(self, sequence_info: Dict, base_path: str) -> Dict:
        """Process a three-page manga sequence.

        Args:
            sequence_info: Dict with 'sequence_id' and 'images' list
            base_path: Base path for dataset

        Returns:
            Dict with 'sequence_id' and 'pages' (list of 3 page results)
        """
        sequence_id = sequence_info['sequence_id']
        image_paths = [str(Path(base_path) / img) for img in sequence_info['images']]

        # Process each page
        pages_data = []
        for image_path in image_paths:
            page_data = self._process_page(image_path, sequence_id)
            pages_data.append(page_data)

        # Ensure speaker consistency across pages
        pages_data = self.speaker_id.ensure_consistency_across_pages(pages_data)

        # Format output
        result = {
            'sequence_id': sequence_id,
            'pages': []
        }

        for page_data in pages_data:
            page_result = []
            for box in page_data:
                page_result.append({
                    'speaker': box['speaker'],
                    'text': box['text']
                })
            result['pages'].append(page_result)

        return result

    def _process_page(self, image_path: str, sequence_id: str) -> List[Dict]:
        """Process a single manga page.

        Args:
            image_path: Path to manga page image
            sequence_id: Sequence ID for speaker identification

        Returns:
            List of text boxes with speaker labels
        """
        # Get image dimensions
        image_width, image_height = get_image_dimensions(image_path)

        # Extract text
        text_boxes = self.ocr.extract_text(image_path)

        # Filter non-story text
        text_boxes = self.filter.filter_text_boxes(text_boxes, image_width, image_height)

        # Order by reading order
        text_boxes = self.reading_order.order_text_boxes(text_boxes, image_width, image_height)

        # Identify speakers
        text_boxes = self.speaker_id.identify_speakers(text_boxes, sequence_id)

        return text_boxes

    def process_dataset(self, sequences_path: str, base_path: str, output_path: str, split: str = 'development'):
        """Process entire dataset and save predictions.

        Args:
            sequences_path: Path to sequences.json
            base_path: Base path for dataset
            output_path: Path to save predictions JSONL
            split: Which split to process ('development' or 'test')
        """
        # Load sequences
        with open(sequences_path, 'r', encoding='utf-8') as f:
            sequences = json.load(f)

        # Filter by split
        sequences = [seq for seq in sequences if seq['split'] == split]

        # Process each sequence
        predictions = []
        for sequence_info in tqdm(sequences, desc=f'Processing {split}'):
            try:
                result = self.process_sequence(sequence_info, base_path)
                predictions.append(result)
            except Exception as e:
                print(f"Error processing {sequence_info['sequence_id']}: {e}")
                # Add empty prediction
                predictions.append({
                    'sequence_id': sequence_info['sequence_id'],
                    'pages': [[], [], []]
                })

        # Save predictions
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')

        print(f"Saved predictions to {output_path}")


def load_config(config_path: str = 'config.json') -> Dict:
    """Load configuration from JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            'ocr_confidence': 0.3,
            'rtl_threshold': 0.5,
            'min_speakers': 2,
            'max_speakers': 8
        }


def save_config(config: Dict, config_path: str = 'config.json'):
    """Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
