"""
DeepGuard Robustness and Generalization Module (Phase 5).
"""

from .evaluator import (
    ConditionResult,
    DeepGuardRobustnessEvaluator,
    RobustnessReport,
)
from .generalization import (
    CrossDatasetEvaluator,
    CrossDatasetManifestSummary,
    GeneralizationComparisonReport,
)
from .perturbations import (
    apply_downsampling,
    apply_gaussian_blur,
    apply_jpeg_compression,
    apply_perturbation,
    get_standard_phase5_perturbations,
    validate_frame_tensor,
    PerturbationConfig,
)

__all__ = [
    "apply_jpeg_compression",
    "apply_gaussian_blur",
    "apply_downsampling",
    "apply_perturbation",
    "get_standard_phase5_perturbations",
    "validate_frame_tensor",
    "PerturbationConfig",
    "DeepGuardRobustnessEvaluator",
    "ConditionResult",
    "RobustnessReport",
    "CrossDatasetEvaluator",
    "CrossDatasetManifestSummary",
    "GeneralizationComparisonReport",
]
