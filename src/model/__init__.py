"""Model architecture components, loader, and inference interfaces for DeepGuard."""

from .baseline import (
    DeepGuardModelConfig,
    build_deepguard_model,
    build_frame_feature_extractor,
    HAS_TF,
)
from .loader import (
    load_deepguard_checkpoint,
    compute_file_sha256,
)
from .inference import (
    DeepGuardPredictor,
    InferenceResult,
)

__all__ = [
    "DeepGuardModelConfig",
    "build_deepguard_model",
    "build_frame_feature_extractor",
    "HAS_TF",
    "load_deepguard_checkpoint",
    "compute_file_sha256",
    "DeepGuardPredictor",
    "InferenceResult",
]
