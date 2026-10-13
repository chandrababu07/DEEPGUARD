"""
Comprehensive evaluation metrics engine for DeepGuard deepfake video detection.

Forensic Semantics:
- Positive Class (1): Manipulated / Deepfake Video
- Negative Class (0): Authentic / Real Video
- Unit of Analysis: Video level (one prediction per video sequence).
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ConfusionMatrix:
    """Standard 2x2 confusion matrix container."""

    tp: int  # True Positive: Fake correctly flagged as Fake
    tn: int  # True Negative: Real correctly verified as Real
    fp: int  # False Positive: Real incorrectly flagged as Fake (False Alarm)
    fn: int  # False Negative: Fake incorrectly missed as Real (Missed Detection)

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def total_samples(self) -> int:
        return self.total

    def to_matrix(self) -> List[List[int]]:
        """Returns [[TN, FP], [FN, TP]] standard format."""
        return [[self.tn, self.fp], [self.fn, self.tp]]

    @property
    def matrix_list(self) -> List[List[int]]:
        return self.to_matrix()

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics collection."""

    total_samples: int
    accuracy: float
    balanced_accuracy: float
    fake_precision: float
    fake_recall: float
    fake_f1: float
    real_precision: float
    real_recall: float  # Specificity
    real_f1: float
    macro_f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion_matrix: ConfusionMatrix
    class_distribution: Dict[str, int]
    warnings: List[str]

    @property
    def num_real(self) -> int:
        return self.class_distribution.get("real", 0)

    @property
    def num_fake(self) -> int:
        return self.class_distribution.get("fake", 0)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confusion_matrix"] = self.confusion_matrix.to_matrix()
        d["confusion_matrix_details"] = self.confusion_matrix.to_dict()
        d["labeled_confusion_matrix"] = {
            "TN": self.confusion_matrix.tn,
            "FP": self.confusion_matrix.fp,
            "FN": self.confusion_matrix.fn,
            "TP": self.confusion_matrix.tp,
        }
        return d


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrix:
    """
    Compute TP, TN, FP, FN counts from 1D binary numpy arrays.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """
    Compute exact ROC-AUC using Mann-Whitney U rank formulation.

    Returns:
        float AUC score in [0.0, 1.0], or None if only a single class is present.
    """
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return None  # ROC-AUC is mathematically undefined with single class

    pos_scores = y_prob[pos_mask]
    neg_scores = y_prob[neg_mask]

    # Rank-sum calculation
    u_sum = 0.0
    for p in pos_scores:
        u_sum += np.sum(p > neg_scores) + 0.5 * np.sum(p == neg_scores)

    return float(u_sum / (n_pos * n_neg))


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """
    Compute Precision-Recall Area Under Curve (PR-AUC) using trapezoidal integration.

    Returns:
        float PR-AUC score in [0.0, 1.0], or None if no positive examples exist.
    """
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return None

    # Sort by descending probability
    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]

    # True positives and false positives cumulative sums
    tp_cumsum = np.cumsum(y_true_sorted == 1)
    fp_cumsum = np.cumsum(y_true_sorted == 0)

    recalls = tp_cumsum / n_pos
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Prepend recall=0, precision=first precision
    recalls = np.concatenate(([0.0], recalls))
    precisions = np.concatenate(([precisions[0] if len(precisions) > 0 else 1.0], precisions))

    # Numerical integration (trapezoidal rule)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(precisions, recalls))
    return float(np.trapz(precisions, recalls))


def compute_precision_recall_f1(cm: ConfusionMatrix) -> Dict[str, float]:
    """Compute per-class and macro precision, recall, and F1 from ConfusionMatrix."""
    # Fake (Class 1)
    fake_precision = float(cm.tp / (cm.tp + cm.fp)) if (cm.tp + cm.fp) > 0 else 0.0
    fake_recall = float(cm.tp / (cm.tp + cm.fn)) if (cm.tp + cm.fn) > 0 else 0.0
    fake_f1 = (
        float(2 * fake_precision * fake_recall / (fake_precision + fake_recall))
        if (fake_precision + fake_recall) > 0
        else 0.0
    )

    # Real (Class 0)
    real_precision = float(cm.tn / (cm.tn + cm.fn)) if (cm.tn + cm.fn) > 0 else 0.0
    real_recall = float(cm.tn / (cm.tn + cm.fp)) if (cm.tn + cm.fp) > 0 else 0.0
    real_f1 = (
        float(2 * real_precision * real_recall / (real_precision + real_recall))
        if (real_precision + real_recall) > 0
        else 0.0
    )

    macro_f1 = float(0.5 * (fake_f1 + real_f1))

    return {
        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,
        "real_precision": real_precision,
        "real_recall": real_recall,
        "real_f1": real_f1,
        "macro_f1": macro_f1,
    }


def compute_balanced_accuracy(cm: ConfusionMatrix) -> float:
    """Compute balanced accuracy from ConfusionMatrix as average of per-class recalls."""
    n_real = cm.tn + cm.fp
    n_fake = cm.tp + cm.fn

    fake_recall = float(cm.tp / n_fake) if n_fake > 0 else 0.0
    real_recall = float(cm.tn / n_real) if n_real > 0 else 0.0

    if n_real > 0 and n_fake > 0:
        return float(0.5 * (fake_recall + real_recall))
    elif n_fake > 0:
        return float(fake_recall)
    else:
        return float(real_recall)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """
    Compute comprehensive evaluation metrics for binary deepfake detection.

    Args:
        y_true: Ground truth binary labels (0 = Real, 1 = Fake).
        y_prob: Continuous predicted probabilities in [0.0, 1.0] for Class 1 (Fake).
        threshold: Decision threshold for positive classification (default 0.5).

    Returns:
        EvaluationMetrics instance.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty predictions array.")

    if len(y_true) != len(y_prob):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_prob)={len(y_prob)}")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError(f"Ground truth y_true must contain only binary labels (0 or 1), got {np.unique(y_true)}")

    warnings: List[str] = []
    y_pred = (y_prob >= threshold).astype(int)

    cm = compute_confusion_matrix(y_true, y_pred)
    total = cm.total

    # Class distribution
    n_real = int(np.sum(y_true == 0))
    n_fake = int(np.sum(y_true == 1))
    class_dist = {"real": n_real, "fake": n_fake}

    if n_real == 0 or n_fake == 0:
        warnings.append(
            f"Only a single class present in evaluation set ({'Fake only' if n_real == 0 else 'Real only'}). "
            "ROC-AUC and balanced accuracy cannot be fully defined."
        )

    # Accuracy
    accuracy = float((cm.tp + cm.tn) / total) if total > 0 else 0.0

    # Per-class metrics and macro averages
    prf = compute_precision_recall_f1(cm)
    fake_precision = prf["fake_precision"]
    fake_recall = prf["fake_recall"]
    fake_f1 = prf["fake_f1"]
    real_precision = prf["real_precision"]
    real_recall = prf["real_recall"]
    real_f1 = prf["real_f1"]
    macro_f1 = prf["macro_f1"]

    # Balanced Accuracy
    balanced_accuracy = compute_balanced_accuracy(cm)

    # ROC-AUC & PR-AUC
    roc_auc = compute_roc_auc(y_true, y_prob)
    pr_auc = compute_pr_auc(y_true, y_prob)

    if roc_auc is not None:
        roc_auc = round(roc_auc, 4)
    if pr_auc is not None:
        pr_auc = round(pr_auc, 4)

    if total < 30:
        warnings.append(
            f"Evaluation set has very few samples (N={total}). Metrics exhibit high variance."
        )

    return EvaluationMetrics(
        total_samples=total,
        accuracy=round(accuracy, 4),
        balanced_accuracy=round(balanced_accuracy, 4),
        fake_precision=round(fake_precision, 4),
        fake_recall=round(fake_recall, 4),
        fake_f1=round(fake_f1, 4),
        real_precision=round(real_precision, 4),
        real_recall=round(real_recall, 4),
        real_f1=round(real_f1, 4),
        macro_f1=round(macro_f1, 4),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm,
        class_distribution=class_dist,
        warnings=warnings,
    )
