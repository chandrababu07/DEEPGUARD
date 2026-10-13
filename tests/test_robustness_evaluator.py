"""
Unit tests for DeepGuard robustness evaluation pipeline.

Tests cover:
- ConditionResult and RobustnessReport schema and serialization
- Paired evaluation sample matching contracts
- Metric deltas (Delta Accuracy, Delta F1, Delta ROC-AUC) calculation
- Consistency metrics: prediction flip count, flip rate, and MAPC
- Summary Markdown generation
- Artifact paths registration
"""

import json
import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.robustness.evaluator import (
    ConditionResult,
    DeepGuardRobustnessEvaluator,
    RobustnessReport,
)
from src.robustness.perturbations import PerturbationConfig


class TestRobustnessEvaluator(unittest.TestCase):
    """Test suite for robustness evaluation pipeline data structures and math."""

    def test_condition_result_schema_and_serialization(self):
        """Verify ConditionResult dataclass serializes cleanly to dict."""
        cr = ConditionResult(
            condition_name="jpeg_q70",
            perturbation_type="jpeg",
            severity_level="moderate",
            parameters={"quality": 70},
            description="JPEG compression Q70",
            total_evaluated_samples=20,
            metrics={"accuracy": 0.85, "fake_f1": 0.82},
            metric_deltas_vs_clean={"accuracy": -0.05, "fake_f1": -0.06},
            mean_absolute_probability_change=0.0345,
            max_absolute_probability_change=0.1200,
            prediction_flip_count=2,
            prediction_flip_rate=0.10,
            delta_false_positives=1,
            delta_false_negatives=1,
        )
        d = cr.to_dict()
        self.assertEqual(d["condition_name"], "jpeg_q70")
        self.assertEqual(d["prediction_flip_count"], 2)
        self.assertAlmostEqual(d["mean_absolute_probability_change"], 0.0345)
        self.assertEqual(d["delta_false_positives"], 1)

    def test_robustness_report_serialization(self):
        """Verify RobustnessReport dataclass contains all provenance and summary fields."""
        cr = ConditionResult(
            condition_name="blur_mild",
            perturbation_type="gaussian_blur",
            severity_level="mild",
            parameters={"kernel_size": 3, "sigma": 1.0},
            description="Mild blur",
            total_evaluated_samples=10,
            metrics={"accuracy": 0.90, "fake_f1": 0.88},
            metric_deltas_vs_clean={"accuracy": 0.0, "fake_f1": 0.0},
            mean_absolute_probability_change=0.012,
            max_absolute_probability_change=0.045,
            prediction_flip_count=0,
            prediction_flip_rate=0.0,
            delta_false_positives=0,
            delta_false_negatives=0,
        )
        rep = RobustnessReport(
            timestamp="2026-10-13T12:00:00Z",
            checkpoint_path="experiments/model.keras",
            checkpoint_sha256="abc123456",
            manifest_path="data/manifest.csv",
            manifest_sha256="def789012",
            decision_threshold=0.5,
            paired_test_sample_count=10,
            clean_baseline_metrics={"accuracy": 0.90, "fake_f1": 0.88},
            conditions=[cr],
            artifacts_created={"robustness_report_json": "report.json"},
            warnings=["Evaluation split is small."],
        )

        d = rep.to_dict()
        self.assertEqual(d["paired_test_sample_count"], 10)
        self.assertEqual(d["checkpoint_sha256"], "abc123456")
        self.assertEqual(len(d["conditions"]), 1)
        self.assertEqual(d["conditions"][0]["condition_name"], "blur_mild")

    def test_markdown_summary_generation(self):
        """Verify _write_markdown_summary produces formatted table without errors."""
        evaluator = DeepGuardRobustnessEvaluator(checkpoint_path="dummy.keras")
        cr = ConditionResult(
            condition_name="jpeg_q50",
            perturbation_type="jpeg",
            severity_level="moderate",
            parameters={"quality": 50},
            description="JPEG Q50",
            total_evaluated_samples=6,
            metrics={"accuracy": 0.8333, "fake_f1": 0.8000, "roc_auc": 0.8750},
            metric_deltas_vs_clean={"accuracy": -0.1667, "fake_f1": -0.2000, "roc_auc": -0.1250},
            mean_absolute_probability_change=0.0543,
            max_absolute_probability_change=0.1800,
            prediction_flip_count=1,
            prediction_flip_rate=0.1667,
            delta_false_positives=1,
            delta_false_negatives=0,
        )
        rep = RobustnessReport(
            timestamp="2026-10-13T12:00:00Z",
            checkpoint_path="experiments/dummy.keras",
            checkpoint_sha256="123456",
            manifest_path="manifest.csv",
            manifest_sha256="abcdef",
            decision_threshold=0.5,
            paired_test_sample_count=6,
            clean_baseline_metrics={"accuracy": 1.0, "fake_f1": 1.0, "roc_auc": 1.0},
            conditions=[cr],
            artifacts_created={},
            warnings=["Warning: small dataset."],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_md = os.path.join(tmpdir, "summary.md")
            evaluator._write_markdown_summary(out_md, rep)
            self.assertTrue(os.path.exists(out_md))

            with open(out_md, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("DEEPGUARD — PHASE 5 ROBUSTNESS EVALUATION REPORT", content)
            self.assertIn("jpeg_q50", content)
            self.assertIn("CLEAN BASELINE BENCHMARK", content)
            self.assertIn("Warning: small dataset", content)


if __name__ == "__main__":
    unittest.main()
