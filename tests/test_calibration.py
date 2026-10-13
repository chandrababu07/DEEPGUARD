"""
Unit tests for DeepGuard confidence calibration and decision threshold module.

Tests cover:
- Brier score: perfect, worst, and uniform predictions
- Expected Calibration Error (ECE) and binning logic
- Reliability diagram bins schema and edge cases
- Leakage-safe threshold optimization on validation partitions
- PlattCalibrator fitting, validation guards (sample size < 20, single class), and inference
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.calibration import (
    compute_brier_score,
    compute_expected_calibration_error,
    compute_reliability_diagram_data,
    evaluate_calibration,
    optimize_decision_threshold,
    CalibrationReport,
    PlattCalibrator,
)


class TestCalibration(unittest.TestCase):
    """Test suite for calibration and threshold optimization."""

    def test_brier_score_perfect(self):
        """Verify Brier score is 0.0 for perfectly calibrated predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        score = compute_brier_score(y_true, y_prob)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_brier_score_worst(self):
        """Verify Brier score is 1.0 for completely inverted predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        score = compute_brier_score(y_true, y_prob)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_brier_score_uniform_guesses(self):
        """Verify Brier score is 0.25 when model outputs 0.5 for all samples."""
        y_true = np.array([0, 1])
        y_prob = np.array([0.5, 0.5])
        score = compute_brier_score(y_true, y_prob)
        # (0-0.5)^2 + (1-0.5)^2 = 0.25 + 0.25 = 0.5 / 2 = 0.25
        self.assertAlmostEqual(score, 0.25, places=5)

    def test_ece_perfect_calibration(self):
        """Verify ECE is approximately 0.0 when probabilities match empirical frequency."""
        # 10 samples with 0.8 prob where 8 are positive
        # 10 samples with 0.2 prob where 2 are positive
        y_true = np.array([1] * 8 + [0] * 2 + [1] * 2 + [0] * 8)
        y_prob = np.array([0.8] * 10 + [0.2] * 10)

        ece = compute_expected_calibration_error(y_true, y_prob, num_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=4)

    def test_reliability_diagram_bins_structure(self):
        """Verify reliability bins output schema and bin count."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        bins = compute_reliability_diagram_data(y_true, y_prob, num_bins=5)
        self.assertEqual(len(bins), 5)
        total_samples_counted = sum(b.sample_count for b in bins)
        self.assertEqual(total_samples_counted, 6)

        for b in bins:
            self.assertGreaterEqual(b.bin_upper, b.bin_lower)
            self.assertTrue(0.0 <= b.bin_center <= 1.0)
            self.assertTrue(0.0 <= b.observed_frequency <= 1.0)
            self.assertTrue(0.0 <= b.mean_predicted_confidence <= 1.0)

    def test_evaluate_calibration_full_report(self):
        """Verify evaluate_calibration generates a complete CalibrationReport dataclass."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.15, 0.25, 0.85, 0.95])

        report = evaluate_calibration(y_true, y_prob, num_bins=10)
        self.assertIsInstance(report, CalibrationReport)
        self.assertEqual(report.total_samples, 4)
        self.assertIsNotNone(report.brier_score)
        self.assertIsNotNone(report.expected_calibration_error)
        self.assertEqual(len(report.reliability_bins), 10)

        d = report.to_dict()
        self.assertIn("brier_score", d)
        self.assertIn("expected_calibration_error", d)
        self.assertIn("reliability_bins", d)

    def test_threshold_optimization_finds_best_threshold(self):
        """Verify threshold optimization finds the threshold maximizing F1 on validation data."""
        # Ground truth: 0, 0, 1, 1
        # Model predictions: 0.2, 0.35, 0.4, 0.45
        # Standard threshold 0.5 gives 0 F1 (no fakes predicted).
        # Optimal threshold around 0.38 gives F1 = 1.0.
        y_val = np.array([0, 0, 1, 1])
        y_prob_val = np.array([0.2, 0.35, 0.4, 0.45])

        best_t, best_score, history = optimize_decision_threshold(
            y_val, y_prob_val, metric="f1", search_steps=50
        )
        self.assertGreaterEqual(best_score, 0.99)
        self.assertTrue(0.35 < best_t <= 0.4)
        self.assertGreater(len(history), 0)

    def test_platt_calibrator_sample_size_guard(self):
        """Verify PlattCalibrator rejects fitting on insufficient samples (N < 20)."""
        calibrator = PlattCalibrator(min_samples=20)
        y_small = np.array([0, 0, 1, 1])  # Only 4 samples
        raw_scores_small = np.array([0.2, 0.3, 0.7, 0.8])

        # Strict mode raises ValueError
        with self.assertRaises(ValueError) as ctx:
            calibrator.fit(y_small, raw_scores_small, strict=True)
        self.assertIn("Insufficient validation samples", str(ctx.exception))

        # Non-strict mode returns False with warning
        res = calibrator.fit(y_small, raw_scores_small, strict=False)
        self.assertFalse(res)
        self.assertFalse(calibrator.is_fitted)
        self.assertIsNotNone(calibrator.fit_warning)

    def test_platt_calibrator_single_class_guard(self):
        """Verify PlattCalibrator rejects fitting when validation data has only one class."""
        calibrator = PlattCalibrator(min_samples=10)
        y_single = np.array([0] * 20)  # 20 samples, but all real (no fake)
        raw_scores = np.linspace(0.1, 0.9, 20)

        # Strict mode raises ValueError
        with self.assertRaises(ValueError) as ctx:
            calibrator.fit(y_single, raw_scores, strict=True)
        self.assertIn("both classes", str(ctx.exception))

        # Non-strict mode returns False with warning
        res = calibrator.fit(y_single, raw_scores, strict=False)
        self.assertFalse(res)
        self.assertFalse(calibrator.is_fitted)
        self.assertIsNotNone(calibrator.fit_warning)

    def test_platt_calibrator_successful_fit_and_predict(self):
        """Verify PlattCalibrator fits successfully when sufficient samples from both classes exist."""
        calibrator = PlattCalibrator(min_samples=20)

        # 30 samples: 15 real (low scores), 15 fake (high scores)
        np.random.seed(42)
        real_scores = np.random.uniform(0.1, 0.4, size=15)
        fake_scores = np.random.uniform(0.6, 0.9, size=15)
        y_val = np.array([0] * 15 + [1] * 15)
        raw_scores = np.concatenate([real_scores, fake_scores])

        calibrator.fit(y_val, raw_scores)
        self.assertTrue(calibrator.is_fitted)

        # Predictions should be valid probabilities in [0, 1]
        test_inputs = np.array([0.2, 0.5, 0.8])
        calibrated_probs = calibrator.predict_proba(test_inputs)

        self.assertEqual(len(calibrated_probs), 3)
        for p in calibrated_probs:
            self.assertTrue(0.0 <= p <= 1.0)
        # Monotonicity: higher raw score should map to higher calibrated probability
        self.assertLess(calibrated_probs[0], calibrated_probs[1])
        self.assertLess(calibrated_probs[1], calibrated_probs[2])


if __name__ == "__main__":
    unittest.main()
