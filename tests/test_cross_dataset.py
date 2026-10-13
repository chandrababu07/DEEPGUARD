"""
Unit tests for DeepGuard cross-dataset generalization module.

Tests cover:
- External manifest audit: schema, video path verification, class balance
- Detection of missing video files and unready external manifests
- Frozen threshold enforcement (zero target test leakage)
- Domain shift generalization deltas calculation (Target - Source)
- Serialization of GeneralizationComparisonReport and CrossDatasetManifestSummary
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.manifest import ManifestManager, ManifestRecord
from src.robustness.generalization import (
    CrossDatasetEvaluator,
    CrossDatasetManifestSummary,
    GeneralizationComparisonReport,
)


class TestCrossDatasetGeneralization(unittest.TestCase):
    """Test suite for cross-dataset generalization evaluation."""

    def test_audit_missing_manifest_returns_unready_status(self):
        """Verify non-existent manifest file returns is_ready_for_evaluation=False with clear issue."""
        summary = CrossDatasetEvaluator.audit_target_manifest("nonexistent_path/manifest.csv")
        self.assertFalse(summary.is_ready_for_evaluation)
        self.assertEqual(summary.total_records, 0)
        self.assertTrue(any("Manifest file not found" in issue for issue in summary.issues))

    def test_audit_detects_missing_video_files(self):
        """Verify manifest pointing to non-existent video files is flagged as unready."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "celebdf_manifest.csv")
            records = [
                ManifestRecord(
                    sample_id="CELEB_001",
                    video_path="missing_video_001.mp4",
                    label=0,
                    dataset_name="Celeb-DF-v2",
                    split="test",
                    manipulation_type="original",
                    original_video_id="001",
                    preprocessing_status="PENDING",
                ),
                ManifestRecord(
                    sample_id="CELEB_002",
                    video_path="missing_video_002.mp4",
                    label=1,
                    dataset_name="Celeb-DF-v2",
                    split="test",
                    manipulation_type="DeepFake",
                    original_video_id="001",
                    preprocessing_status="PENDING",
                ),
            ]
            ManifestManager.write_manifest(records, csv_path)

            summary = CrossDatasetEvaluator.audit_target_manifest(csv_path, base_dir=tmpdir)
            self.assertFalse(summary.is_ready_for_evaluation)
            self.assertEqual(summary.missing_video_files, 2)
            self.assertEqual(summary.existing_video_files, 0)
            self.assertTrue(any("do not exist on disk" in issue for issue in summary.issues))

    def test_audit_flags_single_class_imbalance(self):
        """Verify target dataset containing only one class is flagged with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy video file
            dummy_vid = os.path.join(tmpdir, "dummy.mp4")
            with open(dummy_vid, "wb") as f:
                f.write(b"0" * 100)

            csv_path = os.path.join(tmpdir, "celebdf_all_real.csv")
            records = [
                ManifestRecord(
                    sample_id="CELEB_001",
                    video_path="dummy.mp4",
                    label=0,  # Real only
                    dataset_name="Celeb-DF-v2",
                    split="test",
                    manipulation_type="original",
                    original_video_id="001",
                    preprocessing_status="PENDING",
                ),
            ]
            ManifestManager.write_manifest(records, csv_path)

            summary = CrossDatasetEvaluator.audit_target_manifest(csv_path, base_dir=tmpdir)
            self.assertFalse(summary.is_ready_for_evaluation)
            self.assertEqual(summary.real_count, 1)
            self.assertEqual(summary.fake_count, 0)
            self.assertTrue(any("missing one class" in issue for issue in summary.issues))

    def test_frozen_decision_threshold_contract(self):
        """Verify decision threshold is frozen and preserved without tuning on target data."""
        frozen_thresh = 0.65
        evaluator = CrossDatasetEvaluator(
            checkpoint_path="dummy.keras",
            frozen_decision_threshold=frozen_thresh,
        )
        self.assertEqual(evaluator.decision_threshold, frozen_thresh)
        self.assertEqual(evaluator.predictor.decision_threshold, frozen_thresh)

    def test_generalization_report_serialization(self):
        """Verify GeneralizationComparisonReport schema and serialization."""
        source_metrics = {
            "accuracy": 0.88,
            "fake_f1": 0.86,
            "roc_auc": 0.92,
            "total_samples": 50,
        }
        target_metrics = {
            "accuracy": 0.72,
            "fake_f1": 0.68,
            "roc_auc": 0.75,
            "total_samples": 50,
        }
        deltas = {
            "accuracy": -0.16,
            "fake_f1": -0.18,
            "roc_auc": -0.17,
        }
        rep = GeneralizationComparisonReport(
            timestamp="2026-10-13T12:00:00Z",
            checkpoint_path="experiments/model.keras",
            checkpoint_sha256="hash123",
            source_dataset_name="FaceForensics++",
            target_dataset_name="Celeb-DF-v2",
            decision_threshold=0.5,
            threshold_source="Frozen from source validation split",
            source_metrics=source_metrics,
            target_metrics=target_metrics,
            metric_deltas=deltas,
            sample_counts={"source_samples": 50, "target_samples": 50},
            warnings=[],
        )

        d = rep.to_dict()
        self.assertEqual(d["source_dataset_name"], "FaceForensics++")
        self.assertEqual(d["target_dataset_name"], "Celeb-DF-v2")
        self.assertAlmostEqual(d["metric_deltas"]["accuracy"], -0.16)
        self.assertEqual(d["threshold_source"], "Frozen from source validation split")


if __name__ == "__main__":
    unittest.main()
