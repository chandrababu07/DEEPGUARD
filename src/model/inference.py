"""
DeepGuard Inference Pipeline and Stable API Contract.

Prepares the model pipeline for future Phase 6 frontend and REST API integration.
Provides a reusable Python inference entry point decoupled from notebook-specific code.
"""

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Dict, List, Optional
import numpy as np

from src.dataset.preprocessor import VideoPreprocessor, PreprocessingResult
from src.model.loader import load_deepguard_checkpoint, HAS_TF


@dataclass
class InferenceResult:
    """
    Stable, versioned inference output contract for DeepGuard.
    """

    sample_id: str
    video_path: str
    predicted_label: str  # "REAL" or "FAKE" or "ERROR"
    fake_score: Optional[float] = None  # Raw sigmoid output in [0.0, 1.0]
    real_score: Optional[float] = None  # Complementary score (1.0 - fake_score)
    predicted_class_score: Optional[float] = None  # Confidence of winning class
    decision_threshold: float = 0.5
    model_version: str = "deepguard-mobilenetv2-lstm-v1"
    checkpoint_hash: str = "unknown"
    preprocessing_status: str = "PENDING"
    frame_tensor_shape: Optional[List[int]] = None
    score_calibration: str = "Not calibrated"
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary matching JSON API expectations."""
        return asdict(self)


class DeepGuardPredictor:
    """
    Production-ready inference engine interface for DeepGuard.
    Accepts video files or raw 5D frame tensors, executes preprocessing,
    and returns an InferenceResult.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[object] = None,
        preprocessor: Optional[VideoPreprocessor] = None,
        decision_threshold: float = 0.5,
        model_version: str = "deepguard-mobilenetv2-lstm-v1",
    ):
        """
        Initialize the predictor.

        Args:
            checkpoint_path: Path to trained .keras checkpoint.
            model: Optional pre-loaded tf.keras.Model.
            preprocessor: VideoPreprocessor instance. If None, default preprocessor is used.
            decision_threshold: Binary decision threshold for fake class (default 0.5).
            model_version: String tag describing model version.
        """
        self.checkpoint_path = checkpoint_path
        self.decision_threshold = decision_threshold
        self.model_version = model_version
        self.preprocessor = preprocessor or VideoPreprocessor()
        self.checkpoint_hash = "unloaded"

        self.model = model
        if self.model is None and self.checkpoint_path:
            if os.path.exists(self.checkpoint_path):
                self.model, self.checkpoint_hash = load_deepguard_checkpoint(self.checkpoint_path)

    def predict_frames(
        self,
        frames: np.ndarray,
        sample_id: str = "frame_tensor",
        video_path: str = "in_memory",
    ) -> InferenceResult:
        """
        Run inference directly on a pre-processed (8, 128, 128, 3) or (1, 8, 128, 128, 3) tensor.
        """
        standard_warnings = [
            "Experimental prediction only. Small development dataset does not establish reliable generalization.",
            "Prediction score is an uncalibrated sigmoid output.",
        ]

        # Verify tensor shape
        if frames.ndim == 4:
            # Add batch dimension: (8, 128, 128, 3) -> (1, 8, 128, 128, 3)
            tensor_batch = np.expand_dims(frames, axis=0)
        elif frames.ndim == 5:
            tensor_batch = frames
        else:
            return InferenceResult(
                sample_id=sample_id,
                video_path=video_path,
                predicted_label="ERROR",
                decision_threshold=self.decision_threshold,
                model_version=self.model_version,
                checkpoint_hash=self.checkpoint_hash,
                preprocessing_status="INVALID_TENSOR",
                warnings=standard_warnings,
                error=f"Invalid tensor shape {frames.shape}. Expected (8, 128, 128, 3) or (1, 8, 128, 128, 3).",
            )

        if tensor_batch.shape[1:] != (8, 128, 128, 3):
            return InferenceResult(
                sample_id=sample_id,
                video_path=video_path,
                predicted_label="ERROR",
                decision_threshold=self.decision_threshold,
                model_version=self.model_version,
                checkpoint_hash=self.checkpoint_hash,
                preprocessing_status="INVALID_TENSOR",
                warnings=standard_warnings,
                error=f"Invalid tensor dimensions {tensor_batch.shape}. Expected (B, 8, 128, 128, 3).",
            )

        if self.model is None:
            # Check if model can be loaded
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                self.model, self.checkpoint_hash = load_deepguard_checkpoint(self.checkpoint_path)
            else:
                return InferenceResult(
                    sample_id=sample_id,
                    video_path=video_path,
                    predicted_label="ERROR",
                    decision_threshold=self.decision_threshold,
                    model_version=self.model_version,
                    checkpoint_hash=self.checkpoint_hash,
                    preprocessing_status="SUCCESS",
                    frame_tensor_shape=list(tensor_batch.shape[1:]),
                    warnings=standard_warnings,
                    error=f"Model not loaded. Checkpoint path '{self.checkpoint_path}' does not exist or TensorFlow is unavailable.",
                )

        # Execute model forward pass
        raw_pred = self.model(tensor_batch, training=False)
        if hasattr(raw_pred, "numpy"):
            raw_score = float(raw_pred.numpy()[0][0])
        else:
            raw_score = float(raw_pred[0][0])

        fake_score = float(np.clip(raw_score, 0.0, 1.0))
        real_score = float(1.0 - fake_score)

        predicted_label = "FAKE" if fake_score >= self.decision_threshold else "REAL"
        predicted_class_score = fake_score if predicted_label == "FAKE" else real_score

        return InferenceResult(
            sample_id=sample_id,
            video_path=video_path,
            predicted_label=predicted_label,
            fake_score=fake_score,
            real_score=real_score,
            predicted_class_score=predicted_class_score,
            decision_threshold=self.decision_threshold,
            model_version=self.model_version,
            checkpoint_hash=self.checkpoint_hash,
            preprocessing_status="SUCCESS",
            frame_tensor_shape=list(tensor_batch.shape[1:]),
            score_calibration="Not calibrated",
            warnings=standard_warnings,
        )

    def predict_video(self, video_path: str, sample_id: str = "") -> InferenceResult:
        """
        Preprocess video and run inference.
        """
        if not sample_id:
            sample_id = os.path.basename(video_path)

        standard_warnings = [
            "Experimental prediction only. Small development dataset does not establish reliable generalization.",
            "Prediction score is an uncalibrated sigmoid output.",
        ]

        # Step 1: Preprocess video
        prep_result: PreprocessingResult = self.preprocessor.process_video(
            video_path, sample_id=sample_id
        )

        if not prep_result.is_success or prep_result.frames is None:
            return InferenceResult(
                sample_id=sample_id,
                video_path=video_path,
                predicted_label="ERROR",
                decision_threshold=self.decision_threshold,
                model_version=self.model_version,
                checkpoint_hash=self.checkpoint_hash,
                preprocessing_status=prep_result.status,
                warnings=standard_warnings,
                error=prep_result.error_message,
            )

        # Step 2: Predict on extracted frames
        res = self.predict_frames(
            frames=prep_result.frames,
            sample_id=sample_id,
            video_path=video_path,
        )
        res.preprocessing_status = prep_result.status
        return res
