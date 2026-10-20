"""
DeepGuard Dedicated Image Deepfake Inference Engine and Contracts.

Provides:
1. Versioned ImageInferenceResult contract.
2. Production-ready ImagePredictor interface.
3. Strict validation of output scores (finite, bounded in [0.0, 1.0]).
4. Prevention of fabricated predictions when checkpoints are unavailable.
"""

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Dict, List, Optional
import numpy as np

from src.dataset.image_preprocessor import ImagePreprocessor, ImagePreprocessingResult
from src.model.loader import compute_file_sha256, HAS_TF

try:
    import tensorflow as tf
except ImportError:
    tf = None


@dataclass
class ImageInferenceResult:
    """
    Stable inference output contract for single-image deepfake analysis.
    """

    sample_id: str
    image_path: str
    predicted_label: str  # "REAL", "FAKE", or "ERROR"
    fake_score: Optional[float] = None  # Score in [0.0, 1.0]
    real_score: Optional[float] = None  # 1.0 - fake_score
    predicted_class_score: Optional[float] = None
    decision_threshold: float = 0.5
    model_version: str = "deepguard-image-classifier-v1"
    checkpoint_hash: str = "unknown"
    preprocessing_status: str = "PENDING"
    image_tensor_shape: Optional[List[int]] = None
    original_dimensions: Optional[List[int]] = None
    score_calibration: str = "Uncalibrated sigmoid output. Reflects raw model confidence, not certified certainty."
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ImagePredictor:
    """
    Dedicated single-image deepfake predictor.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[object] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        decision_threshold: float = 0.5,
        model_version: str = "deepguard-image-classifier-v1",
    ):
        self.checkpoint_path = checkpoint_path
        self.decision_threshold = decision_threshold
        self.model_version = model_version
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.checkpoint_hash = "unloaded"
        self.model = model

        if self.model is None and self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self.model, self.checkpoint_hash = self._load_checkpoint(self.checkpoint_path)

    def _load_checkpoint(self, path: str):
        """Loads and validates a dedicated image classifier checkpoint."""
        if not HAS_TF or tf is None:
            raise ImportError("TensorFlow is required to load the image model checkpoint.")
        file_hash = compute_file_sha256(path)
        loaded = tf.keras.models.load_model(path, compile=False)
        return loaded, file_hash

    def predict_tensor(
        self,
        image_tensor: np.ndarray,
        sample_id: str = "image_tensor",
        image_path: str = "in_memory",
        original_dimensions: Optional[List[int]] = None,
    ) -> ImageInferenceResult:
        """
        Run inference directly on a preprocessed (H, W, 3) or (1, H, W, 3) float32 tensor.
        """
        standard_warnings = [
            "Dedicated image deepfake analysis requires a specifically trained image classification checkpoint.",
            "Prediction score is an uncalibrated sigmoid output.",
        ]

        if not isinstance(image_tensor, np.ndarray):
            return ImageInferenceResult(
                sample_id=sample_id,
                image_path=image_path,
                predicted_label="ERROR",
                decision_threshold=self.decision_threshold,
                model_version=self.model_version,
                checkpoint_hash=self.checkpoint_hash,
                preprocessing_status="INVALID_TENSOR",
                warnings=standard_warnings,
                error=f"Expected numpy ndarray, got {type(image_tensor).__name__}",
            )

        if image_tensor.ndim == 3:
            batch_tensor = np.expand_dims(image_tensor, axis=0)
        elif image_tensor.ndim == 4:
            batch_tensor = image_tensor
        else:
            return ImageInferenceResult(
                sample_id=sample_id,
                image_path=image_path,
                predicted_label="ERROR",
                decision_threshold=self.decision_threshold,
                model_version=self.model_version,
                checkpoint_hash=self.checkpoint_hash,
                preprocessing_status="INVALID_TENSOR",
                warnings=standard_warnings,
                error=f"Invalid tensor rank {image_tensor.ndim}. Expected 3D (H, W, 3) or 4D (1, H, W, 3).",
            )

        if self.model is None:
            return ImageInferenceResult(
                sample_id=sample_id,
                image_path=image_path,
                predicted_label="ERROR",
                decision_threshold=self.decision_threshold,
                model_version=self.model_version,
                checkpoint_hash=self.checkpoint_hash,
                preprocessing_status="MODEL_UNAVAILABLE",
                warnings=standard_warnings,
                error="Image model is not loaded or checkpoint is unavailable.",
            )

        try:
            # Model forward pass
            raw_output = self.model(batch_tensor, training=False)
            if hasattr(raw_output, "numpy"):
                raw_score = float(raw_output.numpy()[0][0])
            else:
                raw_score = float(raw_output[0][0])

            # Validate finite value and bounds
            if not np.isfinite(raw_score):
                return ImageInferenceResult(
                    sample_id=sample_id,
                    image_path=image_path,
                    predicted_label="ERROR",
                    decision_threshold=self.decision_threshold,
                    model_version=self.model_version,
                    checkpoint_hash=self.checkpoint_hash,
                    preprocessing_status="INVALID_SCORE",
                    warnings=standard_warnings,
                    error=f"Model output score is non-finite: {raw_score}",
                )

            if raw_score < 0.0 or raw_score > 1.0:
                return ImageInferenceResult(
                    sample_id=sample_id,
                    image_path=image_path,
                    predicted_label="ERROR",
                    decision_threshold=self.decision_threshold,
                    model_version=self.model_version,
                    checkpoint_hash=self.checkpoint_hash,
                    preprocessing_status="INVALID_SCORE",
                    warnings=standard_warnings,
                    error=f"Model output score out of range [0.0, 1.0]: {raw_score}",
                )

            fake_score = float(raw_score)
            real_score = float(1.0 - fake_score)
            predicted_label = "FAKE" if fake_score >= self.decision_threshold else "REAL"
            predicted_class_score = fake_score if predicted_label == "FAKE" else real_score

            return ImageInferenceResult(
                sample_id=sample_id,
                image_path=image_path,
                predicted_label=predicted_label,
                fake_score=fake_score,
                real_score=real_score,
                predicted_class_score=predicted_class_score,
                decision_threshold=self.decision_threshold,
                model_version=self.model_version,
                checkpoint_hash=self.checkpoint_hash,
                preprocessing_status="SUCCESS",
                image_tensor_shape=list(batch_tensor.shape[1:]),
                original_dimensions=original_dimensions,
                warnings=standard_warnings,
            )

        except Exception as e:
            return ImageInferenceResult(
                sample_id=sample_id,
                image_path=image_path,
                predicted_label="ERROR",
                decision_threshold=self.decision_threshold,
                model_version=self.model_version,
                checkpoint_hash=self.checkpoint_hash,
                preprocessing_status="INFERENCE_FAILED",
                warnings=standard_warnings,
                error=f"Image model forward pass failed: {str(e)}",
            )

    def predict_image(self, image_path: str, sample_id: str = "") -> ImageInferenceResult:
        """
        Preprocess image file from disk and execute inference.
        """
        if not sample_id:
            sample_id = os.path.basename(image_path)

        prep_result: ImagePreprocessingResult = self.preprocessor.process_image_file(image_path)
        if not prep_result.is_success or prep_result.image_tensor is None:
            return ImageInferenceResult(
                sample_id=sample_id,
                image_path=image_path,
                predicted_label="ERROR",
                decision_threshold=self.decision_threshold,
                model_version=self.model_version,
                checkpoint_hash=self.checkpoint_hash,
                preprocessing_status=prep_result.status,
                warnings=[
                    "Image preprocessing failed before model inference.",
                ],
                error=prep_result.error_message,
            )

        orig_dims = list(prep_result.original_dimensions) if prep_result.original_dimensions else None
        res = self.predict_tensor(
            image_tensor=prep_result.image_tensor,
            sample_id=sample_id,
            image_path=image_path,
            original_dimensions=orig_dims,
        )
        res.preprocessing_status = prep_result.status
        return res
