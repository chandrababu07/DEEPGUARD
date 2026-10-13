"""
Confidence calibration, reliability analysis, and threshold optimization for DeepGuard.

Distinguishes:
1. Model Score: Raw sigmoid activation output from the classification head.
2. Probability Estimate: Uncalibrated score treated heuristically as a likelihood.
3. Calibrated Probability: Post-hoc mapped score aligned empirically with true accuracy.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CalibrationBin:
    """Bin summary for reliability diagrams."""

    bin_index: int
    bin_lower: float
    bin_upper: float
    sample_count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_gap: float  # |mean_confidence - empirical_accuracy|

    @property
    def bin_center(self) -> float:
        return (self.bin_lower + self.bin_upper) / 2.0

    @property
    def observed_frequency(self) -> float:
        return self.empirical_accuracy

    @property
    def mean_predicted_confidence(self) -> float:
        return self.mean_confidence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationReport:
    """Confidence calibration metrics container."""

    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    num_bins: int
    bins: List[CalibrationBin]
    sample_size: int
    is_calibrated: bool
    calibration_method: str
    warnings: List[str]

    @property
    def reliability_bins(self) -> List[CalibrationBin]:
        return self.bins

    @property
    def total_samples(self) -> int:
        return self.sample_size

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bins"] = [b.to_dict() for b in self.bins]
        d["reliability_bins"] = d["bins"]
        return d


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes mean squared error of predicted probabilities:
    Brier Score = (1 / N) * sum((p_i - y_i)^2).
    Lower is better (0.0 is perfect calibration).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


compute_brier_score = brier_score


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10
) -> float:
    """Computes Expected Calibration Error across equal-width bins."""
    _, ece, _ = compute_reliability_bins(y_true, y_prob, num_bins=num_bins)
    return ece


def compute_reliability_diagram_data(
    y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10
) -> List[CalibrationBin]:
    """Returns reliability diagram bin summaries."""
    bins_data, _, _ = compute_reliability_bins(y_true, y_prob, num_bins=num_bins)
    return bins_data


def compute_reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10,
) -> Tuple[List[CalibrationBin], float, float]:
    """
    Partition predicted probabilities into bins and compute ECE and MCE.

    Returns:
        Tuple of (bins_list, expected_calibration_error, maximum_calibration_error)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n_samples = len(y_true)

    if n_samples == 0:
        return [], 0.0, 0.0

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins_data: List[CalibrationBin] = []
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):
        lower = float(bin_edges[i])
        upper = float(bin_edges[i + 1])

        if i == num_bins - 1:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        else:
            in_bin = (y_prob >= lower) & (y_prob < upper)

        count = int(np.sum(in_bin))
        if count > 0:
            mean_conf = float(np.mean(y_prob[in_bin]))
            emp_acc = float(np.mean(y_true[in_bin]))
            gap = abs(mean_conf - emp_acc)
            ece += (count / n_samples) * gap
            mce = max(mce, gap)
        else:
            mean_conf = (lower + upper) / 2.0
            emp_acc = 0.0
            gap = 0.0

        bins_data.append(
            CalibrationBin(
                bin_index=i,
                bin_lower=round(lower, 2),
                bin_upper=round(upper, 2),
                sample_count=count,
                mean_confidence=round(mean_conf, 4),
                empirical_accuracy=round(emp_acc, 4),
                calibration_gap=round(gap, 4),
            )
        )

    return bins_data, float(ece), float(mce)


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10,
    is_calibrated: bool = False,
    calibration_method: str = "Uncalibrated Sigmoid",
) -> CalibrationReport:
    """
    Compute full calibration report including Brier Score, ECE, and MCE.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_samples = len(y_true)

    warnings: List[str] = []
    if n_samples < 50:
        warnings.append(
            f"Sample size (N={n_samples}) is small for calibration analysis. "
            f"Bins contain few samples, leading to noisy ECE estimates."
        )

    if not is_calibrated:
        warnings.append(
            "Prediction scores are raw uncalibrated sigmoid outputs. "
            "Do not treat scores as verified posterior probabilities."
        )

    bs = brier_score(y_true, y_prob)
    bins_data, ece, mce = compute_reliability_bins(y_true, y_prob, num_bins=num_bins)

    return CalibrationReport(
        brier_score=round(bs, 4),
        expected_calibration_error=round(ece, 4),
        maximum_calibration_error=round(mce, 4),
        num_bins=num_bins,
        bins=bins_data,
        sample_size=n_samples,
        is_calibrated=is_calibrated,
        calibration_method=calibration_method,
        warnings=warnings,
    )


def optimize_decision_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    metric: str = "f1",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    steps: int = 91,
    search_steps: Optional[int] = None,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Select optimal decision threshold using strictly VALIDATION data (zero test leakage).

    Args:
        y_val: Validation ground truth binary labels.
        p_val: Validation predicted probabilities for Class 1 (Fake).
        metric: Optimization target ('f1' or 'balanced_acc').
        min_threshold: Minimum threshold to evaluate.
        max_threshold: Maximum threshold to evaluate.
        steps: Number of threshold evaluation steps.
        search_steps: Alias for steps.

    Returns:
        Tuple of (best_threshold, best_metric_value, search_history_list)
    """
    if search_steps is not None:
        steps = search_steps

    y_val = np.asarray(y_val, dtype=int)
    p_val = np.asarray(p_val, dtype=float)

    thresholds = np.linspace(min_threshold, max_threshold, steps)
    best_thresh = 0.5
    best_score = -1.0
    history: List[Dict[str, float]] = []

    for t in thresholds:
        t = round(float(t), 4)
        pred = (p_val >= t).astype(int)

        tp = np.sum((y_val == 1) & (pred == 1))
        tn = np.sum((y_val == 0) & (pred == 0))
        fp = np.sum((y_val == 0) & (pred == 1))
        fn = np.sum((y_val == 1) & (pred == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = 0.5 * (rec + spec)

        score = f1 if metric == "f1" else bal_acc
        history.append({
            "threshold": t,
            "f1": round(f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        })

        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, round(best_score, 4), history


class PlattCalibrator:
    """
    Logistic calibration (Platt scaling) fitted on validation predictions.
    Applies sigmoid transformation over logit scores:
    P_calibrated = 1 / (1 + exp(-(a * z + b)))

    Sample Size Guard:
    Requires at least 20 independent validation samples with both classes represented.
    If sample size is insufficient, raises an error or falls back with clear warning.
    """

    def __init__(self, min_samples: int = 20):
        self.min_samples = min_samples
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        self.is_fitted: bool = False
        self.fit_warning: Optional[str] = None

    def fit(
        self,
        y_val: np.ndarray,
        p_val: np.ndarray,
        strict: bool = False,
    ) -> bool:
        """
        Fit Platt scaling on validation data.

        Returns True if successfully fitted; False if sample size is insufficient.
        If strict=True, raises ValueError on insufficient sample size or missing class.
        """
        y_val = np.asarray(y_val, dtype=float)
        p_val = np.asarray(p_val, dtype=float)

        if len(y_val) < self.min_samples:
            self.fit_warning = (
                f"Insufficient validation samples for reliable calibration fitting "
                f"(N={len(y_val)}, required minimum is {self.min_samples}). "
                f"Falling back to uncalibrated probabilities."
            )
            self.is_fitted = False
            if strict:
                raise ValueError(self.fit_warning)
            return False

        n_pos = np.sum(y_val == 1)
        n_neg = np.sum(y_val == 0)
        if n_pos < 5 or n_neg < 5:
            self.fit_warning = (
                f"Imbalanced or sparse validation classes (pos={n_pos}, neg={n_neg}). "
                f"Platt scaling requires at least 5 examples of both classes."
            )
            self.is_fitted = False
            if strict:
                raise ValueError(self.fit_warning)
            return False

        # Transform probabilities to logits, avoiding log(0)
        eps = 1e-7
        p_clipped = np.clip(p_val, eps, 1.0 - eps)
        logits = np.log(p_clipped / (1.0 - p_clipped))

        # Fit simple 1D logistic regression via gradient descent
        a, b = 1.0, 0.0
        lr = 0.05
        for _ in range(200):
            z = a * logits + b
            pred = 1.0 / (1.0 + np.exp(-z))
            grad_a = np.mean((pred - y_val) * logits)
            grad_b = np.mean(pred - y_val)
            a -= lr * grad_a
            b -= lr * grad_b

        self.a = float(a)
        self.b = float(b)
        self.is_fitted = True
        self.fit_warning = None
        return True

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        """Apply calibration mapping to input probabilities."""
        p = np.asarray(p, dtype=float)
        if not self.is_fitted or self.a is None or self.b is None:
            return p  # Return raw uncalibrated probabilities as safe fallback

        eps = 1e-7
        p_clipped = np.clip(p, eps, 1.0 - eps)
        logits = np.log(p_clipped / (1.0 - p_clipped))
        z = self.a * logits + self.b
        return 1.0 / (1.0 + np.exp(-z))
