"""DeepGuard evaluation and calibration package."""

from .metrics import (
    ConfusionMatrix,
    EvaluationMetrics,
    compute_confusion_matrix,
    compute_roc_auc,
    compute_pr_auc,
    evaluate_predictions,
)
from .calibration import (
    CalibrationBin,
    CalibrationReport,
    brier_score,
    compute_reliability_bins,
    evaluate_calibration,
    optimize_decision_threshold,
    PlattCalibrator,
)
from .evaluator import (
    DeepGuardComprehensiveEvaluator,
    ComprehensiveEvaluationReport,
)

__all__ = [
    "ConfusionMatrix",
    "EvaluationMetrics",
    "compute_confusion_matrix",
    "compute_roc_auc",
    "compute_pr_auc",
    "evaluate_predictions",
    "CalibrationBin",
    "CalibrationReport",
    "brier_score",
    "compute_reliability_bins",
    "evaluate_calibration",
    "optimize_decision_threshold",
    "PlattCalibrator",
    "DeepGuardComprehensiveEvaluator",
    "ComprehensiveEvaluationReport",
]
