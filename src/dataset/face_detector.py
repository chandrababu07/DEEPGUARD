"""Face detector interfaces and implementations for DeepGuard baseline preprocessing."""

import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import cv2
import numpy as np


class BaseFaceDetector(ABC):
    """Abstract base class for face detectors."""

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the primary face in a BGR frame.

        Args:
            image_bgr: Input image in BGR format (height, width, 3).

        Returns:
            Tuple of (x, y, w, h) bounding box, or None if no face is detected.
        """
        pass


class HaarCascadeFaceDetector(BaseFaceDetector):
    """OpenCV Haar Cascade frontal face detector."""

    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize the Haar Cascade detector.

        Args:
            cascade_path: Optional path to haarcascade_frontalface_default.xml.
                          If None, checks OpenCV default data path.
        """
        self.cascade_path = cascade_path
        if self.cascade_path is None and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            candidate = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(candidate):
                self.cascade_path = candidate

        if not self.cascade_path or not os.path.exists(self.cascade_path):
            self.classifier = None
        else:
            self.classifier = cv2.CascadeClassifier(self.cascade_path)

    @property
    def is_available(self) -> bool:
        """Check whether the cascade classifier is loaded."""
        return self.classifier is not None and not self.classifier.empty()

    def detect(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using Haar Cascade."""
        if not self.is_available:
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if len(faces) == 0:
            return None

        # Return the largest face box by area (w * h)
        faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = faces_sorted[0]
        return int(x), int(y), int(w), int(h)


class SyntheticFaceDetector(BaseFaceDetector):
    """
    Deterministic detector for synthetic test fixtures and unit tests.
    Detects non-background patterns (e.g., drawn circles, squares, or synthetic face markers).
    """

    def __init__(self, min_area: int = 100):
        self.min_area = min_area

    def detect(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect foreground synthetic face marker by intensity contrast/contours."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Check if the frame is completely uniform (e.g. all zeros or single color)
        if np.std(gray) < 1.0:
            return None

        # Find non-zero / foreground region
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Select largest contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        return int(x), int(y), int(w), int(h)


class CenterCropFaceDetector(BaseFaceDetector):
    """
    Fixed center-crop detector used strictly when face detection is bypassed or
    as an explicit fallback for benchmark comparisons.
    """

    def __init__(self, crop_ratio: float = 0.5):
        self.crop_ratio = crop_ratio

    def detect(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = image_bgr.shape[:2]
        crop_w = int(w * self.crop_ratio)
        crop_h = int(h * self.crop_ratio)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2
        return int(x), int(y), int(crop_w), int(crop_h)
