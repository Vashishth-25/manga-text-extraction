"""Speaker identification for manga dialogue."""
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter


class SpeakerIdentifier:
    """Identify speakers in manga pages."""

    def __init__(self, min_clusters=2, max_clusters=8):
        """Initialize speaker identifier.

        Args:
            min_clusters: Minimum number of speakers to consider
            max_clusters: Maximum number of speakers to consider
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

    def identify_speakers(self, text_boxes: List[Dict], sequence_id: str) -> List[Dict]:
        """Identify speaker for each text box.

        Args:
            text_boxes: List of text boxes in reading order
            sequence_id: ID of the sequence for consistent naming

        Returns:
            Text boxes with added 'speaker' field
        """
        if not text_boxes:
            return []

        # Extract features for clustering
        features = self._extract_features(text_boxes)

        # Handle special case: narration
        narration_mask = self._detect_narration(text_boxes)

        # Filter out narration for clustering
        dialogue_indices = [i for i, is_narr in enumerate(narration_mask) if not is_narr]

        if not dialogue_indices:
            # All narration
            for box in text_boxes:
                box['speaker'] = 'NARRATION'
            return text_boxes

        dialogue_features = features[dialogue_indices]

        # Determine optimal number of clusters
        n_clusters = self._find_optimal_clusters(dialogue_features)

        # Cluster dialogue
        if n_clusters == 1:
            dialogue_labels = [0] * len(dialogue_indices)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            dialogue_labels = kmeans.fit_predict(dialogue_features)

        # Assign speaker labels
        all_labels = []
        dialogue_idx = 0

        for i, is_narr in enumerate(narration_mask):
            if is_narr:
                all_labels.append(-1)  # -1 for narration
            else:
                all_labels.append(dialogue_labels[dialogue_idx])
                dialogue_idx += 1

        # Convert cluster IDs to speaker names
        for box, label in zip(text_boxes, all_labels):
            if label == -1:
                box['speaker'] = 'NARRATION'
            else:
                box['speaker'] = f'char{label + 1}'

        return text_boxes

    def _extract_features(self, text_boxes: List[Dict]) -> np.ndarray:
        """Extract spatial features for clustering.

        Args:
            text_boxes: List of text boxes

        Returns:
            Feature matrix (n_boxes, n_features)
        """
        features = []

        for box in text_boxes:
            center_x, center_y = box['center']
            width = box['width']
            height = box['height']

            # Spatial features
            feat = [
                center_x,
                center_y,
                width,
                height,
                width / height,  # Aspect ratio
            ]

            features.append(feat)

        features = np.array(features)

        # Normalize features
        if len(features) > 0:
            mean = features.mean(axis=0)
            std = features.std(axis=0) + 1e-8
            features = (features - mean) / std

        return features

    def _detect_narration(self, text_boxes: List[Dict]) -> List[bool]:
        """Detect which text boxes are narration.

        Narration is typically:
        - Wider boxes (spanning panel)
        - Not in speech balloons
        - Often at top or bottom of panel

        Args:
            text_boxes: List of text boxes

        Returns:
            Boolean mask indicating narration boxes
        """
        if not text_boxes:
            return []

        narration_mask = []

        # Calculate statistics
        widths = [box['width'] for box in text_boxes]
        mean_width = np.mean(widths)

        for box in text_boxes:
            # Wide boxes are likely narration
            is_wide = box['width'] > mean_width * 1.5

            # Long text is more likely dialogue
            is_long = len(box['text']) > 50

            # Simple heuristic: wide and not too long
            is_narration = is_wide and not is_long

            narration_mask.append(is_narration)

        return narration_mask

    def _find_optimal_clusters(self, features: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette score.

        Args:
            features: Feature matrix

        Returns:
            Optimal number of clusters
        """
        n_samples = len(features)

        # Need at least 2 samples to cluster
        if n_samples < 2:
            return 1

        # Try different numbers of clusters
        max_k = min(self.max_clusters, n_samples)
        min_k = min(self.min_clusters, max_k)

        if max_k < 2:
            return 1

        best_score = -1
        best_k = min_k

        for k in range(min_k, max_k + 1):
            if k >= n_samples:
                break

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)

            # Calculate silhouette score
            try:
                score = silhouette_score(features, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                pass

        return best_k

    def ensure_consistency_across_pages(self, pages_data: List[List[Dict]]) -> List[List[Dict]]:
        """Ensure speaker labels are consistent across three pages.

        Args:
            pages_data: List of 3 lists, each containing text boxes for one page

        Returns:
            Updated pages_data with consistent speaker labels
        """
        # Collect all speaker labels across pages
        all_speakers = set()
        for page in pages_data:
            for box in page:
                speaker = box.get('speaker', 'char1')
                if speaker != 'NARRATION':
                    all_speakers.add(speaker)

        # Create mapping to consistent labels
        sorted_speakers = sorted(all_speakers)
        speaker_mapping = {old: f'char{i+1}' for i, old in enumerate(sorted_speakers)}
        speaker_mapping['NARRATION'] = 'NARRATION'

        # Apply mapping
        for page in pages_data:
            for box in page:
                old_speaker = box.get('speaker', 'char1')
                box['speaker'] = speaker_mapping.get(old_speaker, old_speaker)

        return pages_data
