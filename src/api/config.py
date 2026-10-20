"""
Configuration settings for DeepGuard API and Model Serving.

Loads configuration from environment variables with safe defaults.
"""

from dataclasses import dataclass, field
import os
from typing import List, Optional, Set


@dataclass
class APISettings:
    """DeepGuard API configuration settings."""

    # Server settings
    host: str = os.getenv("DEEPGUARD_HOST", "127.0.0.1")
    port: int = int(os.getenv("DEEPGUARD_PORT", "8000"))
    debug: bool = os.getenv("DEEPGUARD_DEBUG", "false").lower() in ("true", "1", "yes")

    # Model & Inference settings
    checkpoint_path: str = os.getenv(
        "DEEPGUARD_CHECKPOINT_PATH",
        "experiments/mini_baseline/checkpoints/deepguard_mobilenetv2_lstm_baseline.keras",
    )
    decision_threshold: float = float(os.getenv("DEEPGUARD_DECISION_THRESHOLD", "0.5"))

    # Upload & Security constraints
    max_upload_size_bytes: int = int(
        os.getenv("DEEPGUARD_MAX_UPLOAD_SIZE", str(100 * 1024 * 1024))
    )  # 100 MB default
    max_video_duration_seconds: float = float(
        os.getenv("DEEPGUARD_MAX_VIDEO_DURATION", "180.0")
    )  # 3 minutes default
    min_required_frames: int = 8

    # Allowed extensions
    allowed_extensions: Set[str] = field(
        default_factory=lambda: {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    )

    # Dedicated Image Analysis settings
    image_checkpoint_path: Optional[str] = os.getenv("DEEPGUARD_IMAGE_CHECKPOINT_PATH", "")
    image_decision_threshold: float = float(
        os.getenv("DEEPGUARD_IMAGE_DECISION_THRESHOLD", "0.5")
    )
    max_image_upload_size_bytes: int = int(
        os.getenv("DEEPGUARD_MAX_IMAGE_UPLOAD_SIZE", str(20 * 1024 * 1024))
    )  # 20 MB default
    max_image_dimension: int = int(
        os.getenv("DEEPGUARD_MAX_IMAGE_DIMENSION", "8192")
    )  # 8192 px maximum dimension
    allowed_image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".webp"}
    )

    # Storage paths
    upload_tmp_dir: str = os.getenv(
        "DEEPGUARD_UPLOAD_TMP_DIR", "experiments/temp_uploads"
    )
    artifact_storage_dir: str = os.getenv(
        "DEEPGUARD_ARTIFACT_DIR", "experiments/api_artifacts"
    )

    # CORS origins
    cors_allowed_origins: List[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "DEEPGUARD_CORS_ORIGINS",
                "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000,http://127.0.0.1:3000",
            ).split(",")
            if origin.strip()
        ]
    )


# Global settings instance
settings = APISettings()
