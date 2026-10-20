"""
Image model serving and lifecycle management service for DeepGuard.

Implements:
1. Thread-safe dedicated image model lifecycle (loaded once per process).
2. Explicit detection of unconfigured or missing image model checkpoints.
3. Clean HTTP 503 response when image inference is requested without a trained checkpoint.
4. Prevention of fabricated or hallucinated predictions.
5. Integration with ImagePredictor and ImagePreprocessor.
"""

import os
import threading
import time
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from src.dataset.image_preprocessor import ImagePreprocessor
from src.model.image_inference import ImagePredictor, ImageInferenceResult
from src.model.loader import compute_file_sha256, HAS_TF
from .config import settings
from .schemas import ImagePredictionResponse


class ImageModelService:
    """
    Thread-safe serving manager for dedicated single-image deepfake detection.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        decision_threshold: Optional[float] = None,
        predictor: Optional[ImagePredictor] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
    ):
        self.checkpoint_path = (
            checkpoint_path
            if checkpoint_path is not None
            else settings.image_checkpoint_path
        )
        self.decision_threshold = (
            decision_threshold
            if decision_threshold is not None
            else settings.image_decision_threshold
        )
        self.preprocessor = preprocessor or ImagePreprocessor()
        self._predictor = predictor
        self._lock = threading.Lock()
        self._initialized = predictor is not None
        self._readiness_error: Optional[str] = None
        self._checkpoint_sha256: str = "unknown"

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                self._checkpoint_sha256 = compute_file_sha256(self.checkpoint_path)
            except Exception:
                self._checkpoint_sha256 = "error"

    def initialize(self) -> bool:
        """
        Thread-safe lazy initialization of the image model and predictor.
        """
        if self._initialized:
            return self._predictor is not None

        with self._lock:
            if self._initialized:
                return self._predictor is not None

            # 1. Check if an image checkpoint is configured
            if not self.checkpoint_path or not self.checkpoint_path.strip():
                self._readiness_error = (
                    "Image model checkpoint is not configured. Dedicated image analysis "
                    "requires a separately trained image checkpoint via DEEPGUARD_IMAGE_CHECKPOINT_PATH."
                )
                self._initialized = True
                return False

            # 2. Check TensorFlow runtime availability
            if not HAS_TF:
                self._readiness_error = (
                    "TensorFlow is not installed in the current Python environment. "
                    "Live image inference requires TensorFlow/Keras."
                )
                self._initialized = True
                return False

            # 3. Check checkpoint file existence on disk
            if not os.path.exists(self.checkpoint_path):
                self._readiness_error = (
                    f"Image model checkpoint file not found at: '{self.checkpoint_path}'. "
                    "Please verify DEEPGUARD_IMAGE_CHECKPOINT_PATH configuration."
                )
                self._initialized = True
                return False

            # 4. Attempt model loading
            try:
                self._predictor = ImagePredictor(
                    checkpoint_path=self.checkpoint_path,
                    preprocessor=self.preprocessor,
                    decision_threshold=self.decision_threshold,
                )
                self._checkpoint_sha256 = compute_file_sha256(self.checkpoint_path)
                self._readiness_error = None
                self._initialized = True
                return True

            except Exception as e:
                self._readiness_error = f"Failed to initialize image model checkpoint: {str(e)}"
                self._predictor = None
                self._initialized = True
                return False

    def is_ready(self) -> bool:
        """Returns True if the image model is loaded and ready for live predictions."""
        self.initialize()
        return self._predictor is not None and self._readiness_error is None

    def get_status_info(self) -> Dict[str, Any]:
        """Provides image model status dictionary for API probes."""
        self.initialize()
        ready = self.is_ready()
        if ready:
            status_str = "ready"
        elif not self.checkpoint_path or not self.checkpoint_path.strip():
            status_str = "unconfigured"
        else:
            status_str = "unavailable"

        return {
            "ready": ready,
            "status": status_str,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self._checkpoint_sha256,
            "error": self._readiness_error,
        }

    def predict(
        self,
        image_path: str,
        request_id: str,
        image_filename: str,
    ) -> ImagePredictionResponse:
        """
        Run deepfake analysis on an uploaded image file.
        """
        self.initialize()
        if self._predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Image inference service is unavailable: "
                    f"{self._readiness_error or 'Image deepfake model checkpoint is not configured.'}"
                ),
            )

        start_time = time.perf_counter()
        inf_res: ImageInferenceResult = self._predictor.predict_image(
            image_path=image_path,
            sample_id=request_id,
        )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        if inf_res.predicted_label == "ERROR" or inf_res.fake_score is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Image prediction failed: {inf_res.error or 'Failed to analyze image content.'}",
            )

        fake_score = float(inf_res.fake_score)
        real_score = float(1.0 - fake_score)
        is_manipulated = fake_score >= self.decision_threshold

        predicted_label = "MANIPULATED" if is_manipulated else "AUTHENTIC"
        label_code = 1 if is_manipulated else 0

        # Preserve original image dimensions if returned
        dims = inf_res.original_dimensions or [128, 128, 3]

        return ImagePredictionResponse(
            request_id=request_id,
            image_filename=image_filename,
            predicted_label=predicted_label,
            label_code=label_code,
            fake_score=round(fake_score, 4),
            real_score=round(real_score, 4),
            decision_threshold=round(self.decision_threshold, 4),
            score_calibration=inf_res.score_calibration,
            model_version=inf_res.model_version,
            checkpoint_sha256=self._checkpoint_sha256,
            processing_time_ms=elapsed_ms,
            image_dimensions=dims,
            warnings=inf_res.warnings,
        )


# Global default image service instance
default_image_service = ImageModelService()
