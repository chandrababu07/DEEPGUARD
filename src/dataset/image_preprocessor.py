"""
Image preprocessing and validation pipeline for DeepGuard.

Implements:
1. Decoding validation and RGB color conversion.
2. Aspect-preserving / centered resize to model target dimensions.
3. Unit interval normalization [0.0, 1.0] matching DeepGuard conventions.
4. Input tensor shape and finite value validation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class ImagePreprocessorConfig:
    """Configuration contract for image deepfake preprocessing."""

    target_height: int = 128
    target_width: int = 128
    channels: int = 3
    normalization: str = "unit_interval"  # [0.0, 1.0] float32

    @property
    def target_shape(self) -> Tuple[int, int, int]:
        return (self.target_height, self.target_width, self.channels)


@dataclass
class ImagePreprocessingResult:
    """Result container for processed image frames."""

    status: str  # "SUCCESS", "DECODE_FAILED", "INVALID_DIMENSIONS", "NON_FINITE"
    image_tensor: Optional[np.ndarray] = None  # (H, W, C) float32 in [0.0, 1.0]
    original_dimensions: Optional[Tuple[int, int, int]] = None  # (H, W, C)
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status == "SUCCESS" and self.image_tensor is not None


class ImagePreprocessor:
    """
    Standard preprocessor for single-image deepfake analysis.
    """

    def __init__(self, config: Optional[ImagePreprocessorConfig] = None):
        self.config = config or ImagePreprocessorConfig()

    def process_rgb_array(self, rgb_array: np.ndarray) -> ImagePreprocessingResult:
        """
        Preprocess an in-memory RGB uint8 numpy array.

        Args:
            rgb_array: uint8 image array with shape (H, W, 3) in RGB order.

        Returns:
            ImagePreprocessingResult containing (target_H, target_W, 3) float32 tensor.
        """
        if not isinstance(rgb_array, np.ndarray):
            return ImagePreprocessingResult(
                status="INVALID_INPUT",
                error_message=f"Expected numpy.ndarray, got {type(rgb_array).__name__}",
            )

        if rgb_array.ndim != 3 or rgb_array.shape[2] != self.config.channels:
            return ImagePreprocessingResult(
                status="INVALID_DIMENSIONS",
                error_message=(
                    f"Expected 3-channel image with shape (H, W, {self.config.channels}), "
                    f"got shape {rgb_array.shape}"
                ),
            )

        orig_h, orig_w, orig_c = rgb_array.shape
        if orig_h <= 0 or orig_w <= 0:
            return ImagePreprocessingResult(
                status="INVALID_DIMENSIONS",
                error_message=f"Invalid image dimensions: {orig_h}x{orig_w}",
            )

        try:
            # Resize to target resolution with bilinear interpolation
            resized = cv2.resize(
                rgb_array,
                (self.config.target_width, self.config.target_height),
                interpolation=cv2.INTER_LINEAR,
            )

            # Convert to float32 and scale to [0.0, 1.0]
            float_img = resized.astype(np.float32) / 255.0

            # Guard against NaN / Inf values
            if not np.all(np.isfinite(float_img)):
                return ImagePreprocessingResult(
                    status="NON_FINITE",
                    error_message="Preprocessed image contains non-finite values (NaN or Inf).",
                )

            return ImagePreprocessingResult(
                status="SUCCESS",
                image_tensor=float_img,
                original_dimensions=(orig_h, orig_w, orig_c),
            )

        except Exception as e:
            return ImagePreprocessingResult(
                status="ERROR",
                error_message=f"Failed during image preprocessing: {str(e)}",
            )

    def process_image_file(self, image_path: str) -> ImagePreprocessingResult:
        """
        Load an image file from disk, convert to RGB, and preprocess.
        """
        # Read with OpenCV (respects EXIF orientation by default in OpenCV 4.x)
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return ImagePreprocessingResult(
                status="DECODE_FAILED",
                error_message=f"OpenCV failed to decode image at '{image_path}'. File may be corrupted or unsupported.",
            )

        # Convert BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.process_rgb_array(rgb)
