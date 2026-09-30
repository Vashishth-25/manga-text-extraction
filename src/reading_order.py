"""Reading order detection for manga pages."""
import numpy as np
from typing import List, Dict, Tuple
import cv2


class ReadingOrderDetector:
    """Detect reading order of text boxes in manga."""

    def __init__(self, rtl_threshold=0.5):
        """Initialize reading order detector.

        Args:
            rtl_threshold: Threshold for determining if page is right-to-left
        """
        self.rtl_threshold = rtl_threshold

    def order_text_boxes(self, text_boxes: List[Dict], image_width: int, image_height: int) -> List[Dict]:
        """Order text boxes by reading order.

        Args:
            text_boxes: List of text boxes from OCR
            image_width: Width of manga page
            image_height: Height of manga page

        Returns:
            Text boxes sorted by reading order
        """
        if not text_boxes:
            return []

        # Detect if page is RTL or LTR
        is_rtl = self._detect_rtl(text_boxes, image_width)

        # Group text boxes into rows (horizontal bands)
        rows = self._group_into_rows(text_boxes, image_height)

        # Sort rows top to bottom
        rows.sort(key=lambda row: np.mean([box['center'][1] for box in row]))

        # Within each row, sort based on RTL/LTR
        ordered = []
        for row in rows:
            if is_rtl:
                # Right to left: sort by x descending
                row.sort(key=lambda box: box['center'][0], reverse=True)
            else:
                # Left to right: sort by x ascending
                row.sort(key=lambda box: box['center'][0])
            ordered.extend(row)

        return ordered

    def _detect_rtl(self, text_boxes: List[Dict], image_width: int) -> bool:
        """Detect if page is right-to-left (typical manga) or left-to-right.

        Args:
            text_boxes: List of text boxes
            image_width: Width of image

        Returns:
            True if right-to-left, False if left-to-right
        """
        if not text_boxes:
            return True  # Default to RTL for manga

        # Count text boxes on left vs right half of page
        left_half = sum(1 for box in text_boxes if box['center'][0] < image_width / 2)
        right_half = len(text_boxes) - left_half

        # If more text on right side, likely RTL
        if right_half > left_half * 1.2:
            return True

        # Default to RTL for manga
        return True

    def _group_into_rows(self, text_boxes: List[Dict], image_height: int, row_threshold: float = 0.03) -> List[List[Dict]]:
        """Group text boxes into horizontal rows.

        Args:
            text_boxes: List of text boxes
            image_height: Height of image
            row_threshold: Threshold for grouping into same row (as fraction of image height)

        Returns:
            List of rows, where each row is a list of text boxes
        """
        if not text_boxes:
            return []

        # Sort by y coordinate
        sorted_boxes = sorted(text_boxes, key=lambda box: box['center'][1])

        # Group into rows
        rows = []
        current_row = [sorted_boxes[0]]
        threshold_px = image_height * row_threshold

        for box in sorted_boxes[1:]:
            # If y is close to current row's average y, add to row
            avg_y = np.mean([b['center'][1] for b in current_row])
            if abs(box['center'][1] - avg_y) < threshold_px:
                current_row.append(box)
            else:
                # Start new row
                rows.append(current_row)
                current_row = [box]

        # Add last row
        if current_row:
            rows.append(current_row)

        return rows


def detect_panels(image_path: str) -> List[Tuple[int, int, int, int]]:
    """Detect manga panels in image (optional, for advanced ordering).

    Args:
        image_path: Path to manga page

    Returns:
        List of panel bounding boxes as (x, y, w, h)
    """
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []

    # Apply threshold to get black borders
    _, binary = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)

    # Invert (panels are white, borders are black)
    binary = cv2.bitwise_not(binary)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    panels = []
    min_area = image.shape[0] * image.shape[1] * 0.01  # At least 1% of image

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            panels.append((x, y, w, h))

    return panels
