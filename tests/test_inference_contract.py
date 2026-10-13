"""
Unit tests for DeepGuard inference engine, checkpoint loader, and stable API contract.
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

from src.model.inference import DeepGuardPredictor, InferenceResult
from src.model.loader import compute_file_sha256, load_deepguard_checkpoint


class TestInferenceContract(unittest.TestCase):
    """Test suite for InferenceResult schema and DeepGuardPredictor contract."""

    def test_inference_result_schema_and_serialization(self):
        """Verify InferenceResult contains all required contract fields and serializes to dict."""
        res = InferenceResult(
            sample_id="SAMPLE_TEST_001",
            video_path="path/to/test.mp4",
            predicted_label="REAL",
            fake_score=0.15,
            real_score=0.85,
            predicted_class_score=0.85,
            decision_threshold=0.5,
            model_version="deepguard-mobilenetv2-lstm-v1",
            checkpoint_hash="abcdef123456",
            preprocessing_status="SUCCESS",
            frame_tensor_shape=[8, 128, 128, 3],
            score_calibration="Not calibrated",
            warnings=["Experimental prediction only."],
        )
        data = res.to_dict()

        expected_fields = [
            "sample_id",
            "video_path",
            "predicted_label",
            "fake_score",
            "real_score",
            "predicted_class_score",
            "decision_threshold",
            "model_version",
            "checkpoint_hash",
            "preprocessing_status",
            "frame_tensor_shape",
            "score_calibration",
            "warnings",
            "error",
        ]
        for field_name in expected_fields:
            self.assertIn(field_name, data)

        self.assertEqual(data["predicted_label"], "REAL")
        self.assertEqual(data["score_calibration"], "Not calibrated")
        self.assertEqual(data["fake_score"], 0.15)
        self.assertEqual(data["real_score"], 0.85)

    def test_predictor_rejects_invalid_tensor_rank(self):
        """Verify predictor returns clean ERROR result when input tensor rank is invalid."""
        predictor = DeepGuardPredictor(checkpoint_path="dummy_path.keras")
        # 3D tensor instead of 4D or 5D
        invalid_3d = np.zeros((128, 128, 3), dtype=np.float32)
        res = predictor.predict_frames(invalid_3d)

        self.assertEqual(res.predicted_label, "ERROR")
        self.assertEqual(res.preprocessing_status, "INVALID_TENSOR")
        self.assertIn("Invalid tensor shape", res.error)

    def test_predictor_rejects_invalid_tensor_dimensions(self):
        """Verify predictor returns clean ERROR result when frame count or resolution is wrong."""
        predictor = DeepGuardPredictor(checkpoint_path="dummy_path.keras")
        # 16 frames instead of 8
        wrong_dims = np.zeros((1, 16, 128, 128, 3), dtype=np.float32)
        res = predictor.predict_frames(wrong_dims)

        self.assertEqual(res.predicted_label, "ERROR")
        self.assertEqual(res.preprocessing_status, "INVALID_TENSOR")
        self.assertIn("Invalid tensor dimensions", res.error)

    def test_predictor_reports_missing_checkpoint_error(self):
        """Verify predictor reports explicit error when checkpoint file does not exist on disk."""
        predictor = DeepGuardPredictor(checkpoint_path="non_existent_model_checkpoint.keras")
        valid_frames = np.zeros((8, 128, 128, 3), dtype=np.float32)
        res = predictor.predict_frames(valid_frames)

        self.assertEqual(res.predicted_label, "ERROR")
        self.assertIn("Model not loaded", res.error)

    def test_compute_file_sha256(self):
        """Verify compute_file_sha256 correctly hashes a file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"DeepGuard Phase 3 Checkpoint Hash Verification")
            tmp_path = tmp.name

        try:
            sha = compute_file_sha256(tmp_path)
            self.assertIsInstance(sha, str)
            self.assertEqual(len(sha), 64)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_checkpoint_missing_file_raises_filenotfound(self):
        """Verify load_deepguard_checkpoint raises FileNotFoundError for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_deepguard_checkpoint("non_existent_file.keras")


if __name__ == "__main__":
    unittest.main()
