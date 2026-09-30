"""OCR module for text detection and extraction from manga pages."""
import sys
import cv2
import numpy as np
from typing import List, Dict, Tuple
import easyocr

# Ensure utf-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Global cached reader so models are loaded only once into memory
_GLOBAL_READER = None


def get_easyocr_reader():
    global _GLOBAL_READER
    if _GLOBAL_READER is None:
        _GLOBAL_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _GLOBAL_READER


class MangaOCR:
    """OCR system for manga text extraction."""

    def __init__(self, confidence_threshold=0.3, reader=None):
        """Initialize OCR reader.

        Args:
            confidence_threshold: Minimum confidence score for text detection
            reader: Optional pre-instantiated EasyOCR Reader
        """
        self.confidence_threshold = confidence_threshold
        self.reader = reader or get_easyocr_reader()

    def extract_text(self, image_path: str) -> List[Dict]:
        """Extract text from manga page with bounding boxes.

        Args:
            image_path: Path to manga page image

        Returns:
            List of dicts with keys: text, bbox, confidence
            bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Run OCR
        results = self.reader.readtext(image)

        # Filter by confidence and format output
        extracted = []
        for bbox, text, confidence in results:
            if confidence >= self.confidence_threshold:
                extracted.append({
                    'text': text.strip(),
                    'bbox': bbox,
                    'confidence': confidence,
                    'center': self._get_bbox_center(bbox),
                    'width': self._get_bbox_width(bbox),
                    'height': self._get_bbox_height(bbox)
                })

        return extracted

    def _get_bbox_center(self, bbox) -> Tuple[float, float]:
        """Get center point of bounding box."""
        points = np.array(bbox)
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        return (center_x, center_y)

    def _get_bbox_width(self, bbox) -> float:
        """Get width of bounding box."""
        points = np.array(bbox)
        return np.max(points[:, 0]) - np.min(points[:, 0])

    def _get_bbox_height(self, bbox) -> float:
        """Get height of bounding box."""
        points = np.array(bbox)
        return np.max(points[:, 1]) - np.min(points[:, 1])


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image width and height.

    Args:
        image_path: Path to image

    Returns:
        (width, height) tuple
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    height, width = image.shape[:2]
    return width, height
