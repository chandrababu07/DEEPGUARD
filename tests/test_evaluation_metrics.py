"""
Unit tests for DeepGuard evaluation metrics module.

Tests cover:
- Binary classification metrics: Accuracy, Balanced Accuracy, Precision, Recall, F1
- Positive-class convention: Fake = 1, Real = 0
- Per-class metrics and Macro averages
- Exact ROC-AUC (Mann-Whitney U) and PR-AUC
- Edge cases: Single class (all 0s or all 1s), empty input, zero divisions, mismatched shapes
- Input validation and exception handling
"""

import math
import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.metrics import (
    compute_balanced_accuracy,
    compute_confusion_matrix,
    compute_pr_auc,
    compute_precision_recall_f1,
    compute_roc_auc,
    evaluate_predictions,
    EvaluationMetrics,
)


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for DeepGuard evaluation metrics."""

    def test_confusion_matrix_perfect_prediction(self):
        """Verify confusion matrix for perfectly separated predictions."""
        # 3 Real (0), 3 Fake (1)
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])

        cm = compute_confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.tn, 3)
        self.assertEqual(cm.fp, 0)
        self.assertEqual(cm.fn, 0)
        self.assertEqual(cm.tp, 3)
        self.assertEqual(cm.total_samples, 6)
        self.assertEqual(cm.matrix_list, [[3, 0], [0, 3]])

    def test_confusion_matrix_mixed_errors(self):
        """Verify confusion matrix correctly counts FP and FN."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        # Real (0): one predicted 1 -> 1 FP, 2 TN
        # Fake (1): one predicted 0 -> 1 FN, 2 TP

        cm = compute_confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.tn, 2)
        self.assertEqual(cm.fp, 1)
        self.assertEqual(cm.fn, 1)
        self.assertEqual(cm.tp, 2)

    def test_precision_recall_f1_calculation(self):
        """Verify per-class and macro precision, recall, and F1 values."""
        cm = compute_confusion_matrix(
            y_true=np.array([0, 0, 0, 1, 1, 1]),
            y_pred=np.array([0, 1, 0, 1, 0, 1]),
        )
        prf = compute_precision_recall_f1(cm)

        # Fake class (positive = 1): TP=2, FP=1, FN=1
        # Precision = 2/3, Recall = 2/3, F1 = 2/3
        self.assertAlmostEqual(prf["fake_precision"], 2 / 3, places=5)
        self.assertAlmostEqual(prf["fake_recall"], 2 / 3, places=5)
        self.assertAlmostEqual(prf["fake_f1"], 2 / 3, places=5)

        # Real class (negative = 0): TN=2, FN=1 (predicted real when fake), FP=1
        # Real Precision = TN / (TN + FN) = 2/3
        # Real Recall = TN / (TN + FP) = 2/3
        self.assertAlmostEqual(prf["real_precision"], 2 / 3, places=5)
        self.assertAlmostEqual(prf["real_recall"], 2 / 3, places=5)
        self.assertAlmostEqual(prf["real_f1"], 2 / 3, places=5)
        self.assertAlmostEqual(prf["macro_f1"], 2 / 3, places=5)

    def test_balanced_accuracy(self):
        """Verify balanced accuracy on an imbalanced dataset."""
        # 10 Real, 2 Fake
        # Model predicts all Real (0)
        y_true = np.array([0] * 10 + [1] * 2)
        y_pred = np.array([0] * 12)

        cm = compute_confusion_matrix(y_true, y_pred)
        # Real recall = 10/10 = 1.0, Fake recall = 0/2 = 0.0
        # Balanced accuracy = (1.0 + 0.0) / 2 = 0.5
        bal_acc = compute_balanced_accuracy(cm)
        self.assertAlmostEqual(bal_acc, 0.5, places=5)

    def test_roc_auc_perfect(self):
        """Verify ROC-AUC returns 1.0 for perfectly ranked probabilities."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        auc = compute_roc_auc(y_true, y_score)
        self.assertIsNotNone(auc)
        self.assertAlmostEqual(auc, 1.0, places=5)

    def test_roc_auc_inverted(self):
        """Verify ROC-AUC returns 0.0 for completely inverted probabilities."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])

        auc = compute_roc_auc(y_true, y_score)
        self.assertIsNotNone(auc)
        self.assertAlmostEqual(auc, 0.0, places=5)

    def test_roc_auc_single_class_edge_case(self):
        """Verify ROC-AUC returns None when ground truth has only one class."""
        y_true_all_zeros = np.array([0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4])

        auc = compute_roc_auc(y_true_all_zeros, y_score)
        self.assertIsNone(auc)

        y_true_all_ones = np.array([1, 1, 1, 1])
        auc_ones = compute_roc_auc(y_true_all_ones, y_score)
        self.assertIsNone(auc_ones)

    def test_pr_auc_perfect(self):
        """Verify PR-AUC returns 1.0 for perfect ranking."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])

        pr_auc = compute_pr_auc(y_true, y_score)
        self.assertIsNotNone(pr_auc)
        self.assertAlmostEqual(pr_auc, 1.0, places=5)

    def test_pr_auc_single_class_edge_case(self):
        """Verify PR-AUC returns None when only one class is present."""
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.2, 0.3, 0.4])
        self.assertIsNone(compute_pr_auc(y_true, y_score))

    def test_evaluate_predictions_full_pipeline(self):
        """Verify evaluate_predictions generates complete valid EvaluationMetrics dataclass."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.35, 0.65, 0.8, 0.9])

        metrics = evaluate_predictions(y_true, y_score, threshold=0.5)

        self.assertIsInstance(metrics, EvaluationMetrics)
        self.assertEqual(metrics.total_samples, 6)
        self.assertEqual(metrics.num_real, 3)
        self.assertEqual(metrics.num_fake, 3)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.balanced_accuracy, 1.0)
        self.assertEqual(metrics.fake_precision, 1.0)
        self.assertEqual(metrics.fake_recall, 1.0)
        self.assertEqual(metrics.fake_f1, 1.0)
        self.assertEqual(metrics.roc_auc, 1.0)
        self.assertEqual(metrics.pr_auc, 1.0)

        # Dictionary serialization
        d = metrics.to_dict()
        self.assertIn("accuracy", d)
        self.assertIn("confusion_matrix", d)
        self.assertIn("labeled_confusion_matrix", d)

    def test_evaluate_predictions_custom_threshold(self):
        """Verify threshold shifting changes predictions correctly."""
        y_true = np.array([0, 1])
        y_score = np.array([0.3, 0.4])

        # At threshold 0.5: both predicted 0 (Accuracy = 0.5, TP = 0)
        m_default = evaluate_predictions(y_true, y_score, threshold=0.5)
        self.assertEqual(m_default.confusion_matrix.tp, 0)
        self.assertEqual(m_default.accuracy, 0.5)

        # At threshold 0.35: sample 2 predicted 1 (Accuracy = 1.0, TP = 1)
        m_lower = evaluate_predictions(y_true, y_score, threshold=0.35)
        self.assertEqual(m_lower.confusion_matrix.tp, 1)
        self.assertEqual(m_lower.accuracy, 1.0)

    def test_input_validation_empty_arrays(self):
        """Verify ValueError is raised on empty input arrays."""
        with self.assertRaises(ValueError):
            evaluate_predictions(np.array([]), np.array([]))

    def test_input_validation_mismatched_lengths(self):
        """Verify ValueError is raised when y_true and y_score have different lengths."""
        with self.assertRaises(ValueError):
            evaluate_predictions(np.array([0, 1]), np.array([0.5]))

    def test_input_validation_invalid_labels(self):
        """Verify ValueError is raised when y_true contains values other than 0 and 1."""
        with self.assertRaises(ValueError):
            evaluate_predictions(np.array([0, 2]), np.array([0.5, 0.8]))

    def test_zero_division_safety(self):
        """Verify all zero predictions (no positive predictions) does not cause ZeroDivisionError."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 0])
        cm = compute_confusion_matrix(y_true, y_pred)
        prf = compute_precision_recall_f1(cm)

        self.assertEqual(prf["fake_precision"], 0.0)
        self.assertEqual(prf["fake_recall"], 0.0)
        self.assertEqual(prf["fake_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
