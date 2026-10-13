"""
Pydantic schemas for DeepGuard API request and response contracts.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness check response."""

    status: str = Field(default="healthy", description="Application service status")
    service: str = Field(default="DeepGuard Deepfake Detection API")
    version: str = Field(default="1.0.0")
    timestamp: str


class ReadinessResponse(BaseModel):
    """Model dependency and checkpoint readiness check response."""

    status: str = Field(description="'ready' if model is loaded and inference-capable; else 'not_ready'")
    model_loaded: bool
    tensorflow_available: bool
    checkpoint_path: str
    checkpoint_sha256: str
    decision_threshold: float
    expected_input_shape: List[int]
    device: str
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class PredictionResponse(BaseModel):
    """Structured video prediction response."""

    request_id: str = Field(description="Unique UUID tracking the request")
    video_filename: str = Field(description="Original uploaded video filename")
    predicted_label: str = Field(description="'AUTHENTIC' or 'MANIPULATED'")
    label_code: int = Field(description="0 = Authentic / Real, 1 = Manipulated / Deepfake")
    fake_score: float = Field(description="Continuous probability score for manipulation in [0.0, 1.0]")
    real_score: float = Field(description="Complement probability score (1.0 - fake_score)")
    decision_threshold: float = Field(description="Decision threshold used to separate classes")
    score_calibration: str = Field(
        default="Uncalibrated sigmoid output. Do not treat as certified posterior probability."
    )
    model_version: str
    checkpoint_sha256: str
    processing_time_ms: float
    frame_tensor_shape: List[int]
    warnings: List[str] = Field(default_factory=list)


class FrameExplanationItem(BaseModel):
    """Per-frame spatial attribution explanation artifact."""

    frame_index: int
    overlay_url: str
    peak_activation_coords: List[int]  # [x, y]
    mean_activation_intensity: float
    predicted_score: float


class ExplanationResponse(BaseModel):
    """Grad-CAM spatial attribution response."""

    request_id: str
    video_filename: str
    predicted_label: str
    fake_score: float
    target_layer: str
    frames_explained: int
    explanations: List[FrameExplanationItem]
    disclaimer: str
    warnings: List[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Sanitized API error response schema."""

    error: str
    detail: str
    request_id: str
    status_code: int
    timestamp: str
