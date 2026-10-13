"""
Model serving and lifecycle management service for DeepGuard.

Implements:
1. Singleton thread-safe model lifecycle (loaded once per process).
2. Distinction between application health and model inference readiness.
3. Decoupled dependency injection for deterministic unit testing.
4. Prevention of fake or fabricated predictions when dependencies are missing.
5. Integration with DeepGuardPredictor and DeepGuardGradCAM.
"""

import os
import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import HTTPException, status

from src.dataset.preprocessor import VideoPreprocessor
from src.explainability.gradcam import DeepGuardGradCAM, EXPLANATION_DISCLAIMER, GradCAMExplanation
from src.model.baseline import HAS_TF
from src.model.inference import DeepGuardPredictor, InferenceResult
from src.model.loader import compute_file_sha256
from .config import settings
from .schemas import (
    ExplanationResponse,
    FrameExplanationItem,
    PredictionResponse,
    ReadinessResponse,
)


class ModelService:
    """
    Thread-safe model serving and explanation manager.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        decision_threshold: Optional[float] = None,
        predictor: Optional[DeepGuardPredictor] = None,
        explainer: Optional[DeepGuardGradCAM] = None,
        preprocessor: Optional[VideoPreprocessor] = None,
    ):
        self.checkpoint_path = checkpoint_path or settings.checkpoint_path
        self.decision_threshold = (
            decision_threshold if decision_threshold is not None else settings.decision_threshold
        )
        self.preprocessor = preprocessor or VideoPreprocessor()
        self._predictor = predictor
        self._explainer = explainer
        self._lock = threading.Lock()
        self._initialized = predictor is not None
        self._readiness_error: Optional[str] = None
        self._checkpoint_sha256: str = "unknown"

        # Checkpoint hash
        if os.path.exists(self.checkpoint_path):
            self._checkpoint_sha256 = compute_file_sha256(self.checkpoint_path)

    def initialize(self) -> bool:
        """
        Thread-safe lazy initialization of model and predictor.
        """
        if self._initialized:
            return self._predictor is not None

        with self._lock:
            if self._initialized:
                return self._predictor is not None

            if not HAS_TF:
                self._readiness_error = (
                    "TensorFlow is not installed in the current Python environment. "
                    "Live model inference requires TensorFlow/Keras (run in Google Colab GPU environment)."
                )
                self._initialized = True
                return False

            if not os.path.exists(self.checkpoint_path):
                self._readiness_error = (
                    f"Model checkpoint file not found at: '{self.checkpoint_path}'. "
                    "Please verify DEEPGUARD_CHECKPOINT_PATH configuration."
                )
                self._initialized = True
                return False

            try:
                self._predictor = DeepGuardPredictor(
                    checkpoint_path=self.checkpoint_path,
                    preprocessor=self.preprocessor,
                    decision_threshold=self.decision_threshold,
                )
                self._checkpoint_sha256 = compute_file_sha256(self.checkpoint_path)

                # Initialize explainer with loaded model
                if hasattr(self._predictor, "model") and self._predictor.model is not None:
                    self._explainer = DeepGuardGradCAM(
                        model=self._predictor.model,
                        target_layer_name="Conv_1",
                    )

                self._readiness_error = None
                self._initialized = True
                return True

            except Exception as e:
                self._readiness_error = f"Failed to initialize model checkpoint: {str(e)}"
                self._predictor = None
                self._explainer = None
                self._initialized = True
                return False

    def get_readiness(self) -> ReadinessResponse:
        """
        Check whether dependencies and model checkpoint are ready for inference.
        """
        self.initialize()
        is_ready = self._predictor is not None and self._readiness_error is None

        warnings = []
        if not HAS_TF:
            warnings.append(
                "TensorFlow is unavailable locally. Server running in API/UI preview mode."
            )
        if not os.path.exists(self.checkpoint_path):
            warnings.append(f"Checkpoint not found at '{self.checkpoint_path}'.")

        return ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            model_loaded=is_ready,
            tensorflow_available=HAS_TF,
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=self._checkpoint_sha256,
            decision_threshold=self.decision_threshold,
            expected_input_shape=[8, 128, 128, 3],
            device="GPU" if (HAS_TF and hasattr(self, "_is_gpu") and self._is_gpu) else "CPU/Unloaded",
            warnings=warnings,
            error=self._readiness_error,
        )

    def predict(
        self,
        video_path: str,
        request_id: str,
        video_filename: str,
    ) -> PredictionResponse:
        """
        Run inference over an uploaded video.
        """
        self.initialize()
        if self._predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model inference service is not ready: {self._readiness_error or 'Model not loaded.'}",
            )

        start_time = time.perf_counter()
        inf_res: InferenceResult = self._predictor.predict_video(
            video_path=video_path,
            sample_id=request_id,
        )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        if inf_res.predicted_label == "ERROR" or inf_res.fake_score is None:
            raise HTTPException(
                status_code=422,
                detail=f"Prediction preprocessing error: {inf_res.error or 'Failed to process video faces.'}",
            )

        # Standardize forensic label semantics
        # 0 = AUTHENTIC (Real), 1 = MANIPULATED (Fake)
        fake_score = float(inf_res.fake_score)
        real_score = float(1.0 - fake_score)
        is_manipulated = fake_score >= self.decision_threshold

        predicted_label = "MANIPULATED" if is_manipulated else "AUTHENTIC"
        label_code = 1 if is_manipulated else 0

        return PredictionResponse(
            request_id=request_id,
            video_filename=video_filename,
            predicted_label=predicted_label,
            label_code=label_code,
            fake_score=round(fake_score, 4),
            real_score=round(real_score, 4),
            decision_threshold=round(self.decision_threshold, 4),
            score_calibration=(
                "Uncalibrated sigmoid output. Reflects raw model output, not certified statistical confidence."
            ),
            model_version=inf_res.model_version,
            checkpoint_sha256=self._checkpoint_sha256,
            processing_time_ms=elapsed_ms,
            frame_tensor_shape=inf_res.frame_tensor_shape or [8, 128, 128, 3],
            warnings=inf_res.warnings,
        )

    def explain(
        self,
        video_path: str,
        request_id: str,
        video_filename: str,
        artifact_dir: str,
        base_artifact_url: str = "/api/v1/artifacts",
    ) -> ExplanationResponse:
        """
        Generate spatial attribution Grad-CAM overlays for key frames of the video.
        """
        self.initialize()
        if self._predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model is not ready: {self._readiness_error or 'Model not loaded.'}",
            )

        if self._explainer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Grad-CAM explainability module is unavailable for the current model runtime.",
            )

        # 1. Preprocess video frames
        prep_res = self.preprocessor.process_video(video_path)
        if prep_res.status != "SUCCESS" or prep_res.frames is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Video preprocessing failed for explainability: {prep_res.error_message}",
            )

        os.makedirs(artifact_dir, exist_ok=True)

        # 2. Compute inference prediction
        pred_res = self.predict(video_path, request_id, video_filename)

        # 3. Generate Grad-CAM overlays for early, mid, late frames (indices 0, 3, 7)
        frame_indices = [0, 3, 7]
        items: List[FrameExplanationItem] = []

        try:
            explanations: List[GradCAMExplanation] = self._explainer.explain_video(
                frames_tensor=prep_res.frames,
                output_dir=artifact_dir,
                sample_id=request_id,
                frame_indices=frame_indices,
            )

            for exp in explanations:
                fname = os.path.basename(exp.output_image_path)
                url = f"{base_artifact_url}/{fname}"
                items.append(
                    FrameExplanationItem(
                        frame_index=exp.frame_index,
                        overlay_url=url,
                        peak_activation_coords=list(exp.peak_activation_coords),
                        mean_activation_intensity=round(exp.mean_activation_intensity, 4),
                        predicted_score=round(exp.predicted_score, 4),
                    )
                )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate Grad-CAM heatmaps: {str(e)}",
            )

        return ExplanationResponse(
            request_id=request_id,
            video_filename=video_filename,
            predicted_label=pred_res.predicted_label,
            fake_score=pred_res.fake_score,
            target_layer=self._explainer.target_layer_name,
            frames_explained=len(items),
            explanations=items,
            disclaimer=EXPLANATION_DISCLAIMER,
            warnings=pred_res.warnings,
        )


# Global default service instance
default_model_service = ModelService()
