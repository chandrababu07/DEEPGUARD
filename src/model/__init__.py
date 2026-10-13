"""Model architecture components for DeepGuard deepfake video detection."""

from .baseline import (
    DeepGuardModelConfig,
    build_deepguard_model,
    build_frame_feature_extractor,
    HAS_TF,
)

__all__ = [
    "DeepGuardModelConfig",
    "build_deepguard_model",
    "build_frame_feature_extractor",
    "HAS_TF",
]
