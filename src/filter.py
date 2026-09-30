"""Text filtering to remove non-story elements."""
import re
from typing import List, Dict


class TextFilter:
    """Filter non-story text from manga pages."""

    def __init__(self):
        """Initialize text filter with rules."""
        # Sound effect patterns (uppercase, short, repetitive)
        self.sound_effect_patterns = [
            r'^[A-Z]{1,3}$',  # Single letters: "H", "AH"
            r'^[A-Z\s\-!?.]*$',  # Only uppercase and punctuation
            r'(.)\1{2,}',  # Repeated characters: "AAAA", "!!!!!"
        ]

        # Credit/metadata keywords
        self.metadata_keywords = [
            'page', 'chapter', 'volume', 'translation', 'translator',
            'editor', 'scanlation', 'scan', 'raw', 'source', 'credit',
            'typesetter', 'cleaner', 'redraw', 'quality', 'version',
            'www.', 'http', '.com', '.net', '.org', 'copyright', '©'
        ]

    def filter_text_boxes(self, text_boxes: List[Dict], image_width: int, image_height: int) -> List[Dict]:
        """Filter out non-story text.

        Args:
            text_boxes: List of text boxes from OCR
            image_width: Width of manga page
            image_height: Height of manga page

        Returns:
            Filtered list of text boxes
        """
        filtered = []

        for box in text_boxes:
            text = box['text']

            # Skip empty or very short text
            if len(text.strip()) == 0:
                continue

            # Check if it's a sound effect
            if self._is_sound_effect(text):
                continue

            # Check if it's metadata/credits
            if self._is_metadata(text):
                continue

            # Check if it's at page edges (likely page number or watermark)
            if self._is_at_page_edge(box, image_width, image_height):
                continue

            # Check if it's very small (likely background text)
            if self._is_too_small(box, image_width, image_height):
                continue

            filtered.append(box)

        return filtered

    def _is_sound_effect(self, text: str) -> bool:
        """Check if text is likely a sound effect."""
        # Very short text
        if len(text) <= 2:
            return True

        # All uppercase and short
        if text.isupper() and len(text) <= 6:
            # But allow common dialogue words
            dialogue_words = {'NO', 'YES', 'HUH', 'WHAT', 'WHY', 'WHO', 'HOW', 'WHEN', 'WHERE'}
            if text in dialogue_words:
                return False
            return True

        # Check against patterns
        for pattern in self.sound_effect_patterns:
            if re.search(pattern, text):
                # But keep if it has multiple words
                if ' ' in text and len(text.split()) >= 2:
                    return False
                return True

        return False

    def _is_metadata(self, text: str) -> bool:
        """Check if text is metadata/credits."""
        text_lower = text.lower()

        # Check for metadata keywords
        for keyword in self.metadata_keywords:
            if keyword in text_lower:
                return True

        return False

    def _is_at_page_edge(self, box: Dict, image_width: int, image_height: int, margin: float = 0.05) -> bool:
        """Check if text box is at page edge (likely page number)."""
        center_x, center_y = box['center']

        # Define edge margins (5% of image size)
        x_margin = image_width * margin
        y_margin = image_height * margin

        # Check if too close to edges
        if center_x < x_margin or center_x > image_width - x_margin:
            if center_y < y_margin or center_y > image_height - y_margin:
                return True

        return False

    def _is_too_small(self, box: Dict, image_width: int, image_height: int, min_ratio: float = 0.01) -> bool:
        """Check if text box is too small (likely background detail)."""
        width = box['width']
        height = box['height']

        # Too small relative to page size
        if width < image_width * min_ratio or height < image_height * min_ratio:
            return True

        return False
