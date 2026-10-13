"""Dataset preparation, manifest management, and baseline preprocessing for DeepGuard."""

from .manifest import ManifestRecord, ValidationReport, ManifestManager
from .preprocessor import VideoPreprocessor, PreprocessingResult
from .face_detector import BaseFaceDetector, HaarCascadeFaceDetector, SyntheticFaceDetector

__all__ = [
    "ManifestRecord",
    "ValidationReport",
    "ManifestManager",
    "VideoPreprocessor",
    "PreprocessingResult",
    "BaseFaceDetector",
    "HaarCascadeFaceDetector",
    "SyntheticFaceDetector",
]
