"""
Unit tests for DeepGuard video prediction API (/api/v1/predict).

Tests cover:
- Empty upload rejection (400)
- Unsupported container / magic byte rejection (415)
- File size boundary enforcement (413)
- Valid prediction with injected deterministic predictor (200)
- Correct label mapping (AUTHENTIC = 0, MANIPULATED = 1)
- Handling of predictor errors and unready services (422, 503)
"""

import io
import os
import sys
import tempfile
import unittest
import cv2
import numpy as np
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.app import create_app
from src.api.config import APISettings
from src.api.service import ModelService
from src.model.inference import InferenceResult


def create_synthetic_mp4_bytes(num_frames: int = 8, width: int = 128, height: int = 128) -> bytes:
    """Helper to generate a small valid in-memory MP4 video file using OpenCV."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
        for _ in range(num_frames):
            writer.write(np.zeros((height, width, 3), dtype=np.uint8))
        writer.release()

        with open(tmp_path, "rb") as f:
            data = f.read()
        return data
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class TestAPIPredict(unittest.TestCase):
    """Test suite for /api/v1/predict endpoint."""

    def setUp(self):
        self.synth_video_bytes = create_synthetic_mp4_bytes(num_frames=8)

    def test_empty_file_upload_returns_400(self):
        """Verify empty file upload is rejected with 400 Bad Request."""
        app = create_app()
        client = TestClient(app)

        files = {"file": ("empty.mp4", b"", "video/mp4")}
        resp = client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertIn("empty", data["detail"].lower())

    def test_invalid_magic_bytes_rejected_with_415(self):
        """Verify file with non-video magic bytes is rejected with 415 Unsupported Media Type."""
        app = create_app()
        client = TestClient(app)

        # Text file masked as .mp4
        files = {"file": ("fake_video.mp4", b"This is plain text content, not an MP4 video.", "video/mp4")}
        resp = client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 415)
        data = resp.json()
        self.assertIn("not a recognized or supported video container", data["detail"])

    def test_oversized_file_rejected_with_413(self):
        """Verify file exceeding max_upload_size_bytes is rejected with 413."""
        # Custom setting with tiny limit of 500 bytes
        custom_settings = APISettings(max_upload_size_bytes=500)
        app = create_app(custom_settings=custom_settings)
        client = TestClient(app)

        # Video bytes > 500 bytes
        files = {"file": ("oversized.mp4", self.synth_video_bytes, "video/mp4")}
        resp = client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 413)
        data = resp.json()
        self.assertIn("exceeds maximum allowed limit", data["detail"])

    def test_prediction_success_manipulated(self):
        """Verify prediction returning fake_score >= threshold returns MANIPULATED (code 1)."""
        class MockManipulatedPredictor:
            def predict_video(self, video_path: str, sample_id: str):
                return InferenceResult(
                    sample_id=sample_id,
                    video_path=video_path,
                    predicted_label="FAKE",
                    fake_score=0.875,
                    real_score=0.125,
                    decision_threshold=0.5,
                    model_version="deepguard-v1-test",
                    checkpoint_hash="hash_mock_fake",
                    frame_tensor_shape=[8, 128, 128, 3],
                )

        mock_service = ModelService(predictor=MockManipulatedPredictor(), decision_threshold=0.5)
        app = create_app(model_service=mock_service)
        client = TestClient(app)

        files = {"file": ("test_vid.mp4", self.synth_video_bytes, "video/mp4")}
        resp = client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["predicted_label"], "MANIPULATED")
        self.assertEqual(data["label_code"], 1)
        self.assertAlmostEqual(data["fake_score"], 0.875, places=3)
        self.assertAlmostEqual(data["real_score"], 0.125, places=3)
        self.assertEqual(data["decision_threshold"], 0.5)
        self.assertIn("request_id", data)
        self.assertIn("processing_time_ms", data)
        self.assertEqual(data["frame_tensor_shape"], [8, 128, 128, 3])

    def test_prediction_success_authentic(self):
        """Verify prediction returning fake_score < threshold returns AUTHENTIC (code 0)."""
        class MockAuthenticPredictor:
            def predict_video(self, video_path: str, sample_id: str):
                return InferenceResult(
                    sample_id=sample_id,
                    video_path=video_path,
                    predicted_label="REAL",
                    fake_score=0.120,
                    real_score=0.880,
                    decision_threshold=0.5,
                    model_version="deepguard-v1-test",
                    checkpoint_hash="hash_mock_real",
                    frame_tensor_shape=[8, 128, 128, 3],
                )

        mock_service = ModelService(predictor=MockAuthenticPredictor(), decision_threshold=0.5)
        app = create_app(model_service=mock_service)
        client = TestClient(app)

        files = {"file": ("test_real.mp4", self.synth_video_bytes, "video/mp4")}
        resp = client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["predicted_label"], "AUTHENTIC")
        self.assertEqual(data["label_code"], 0)
        self.assertAlmostEqual(data["fake_score"], 0.120, places=3)
        self.assertAlmostEqual(data["real_score"], 0.880, places=3)

    def test_predictor_error_returns_422(self):
        """Verify predictor error (e.g. face detection failure) returns 422 Unprocessable Entity."""
        class MockErrorPredictor:
            def predict_video(self, video_path: str, sample_id: str):
                return InferenceResult(
                    sample_id=sample_id,
                    video_path=video_path,
                    predicted_label="ERROR",
                    error="No detectable face in video frames.",
                )

        mock_service = ModelService(predictor=MockErrorPredictor())
        app = create_app(model_service=mock_service)
        client = TestClient(app)

        files = {"file": ("test_noface.mp4", self.synth_video_bytes, "video/mp4")}
        resp = client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertIn("No detectable face", data["detail"])

    def test_unready_service_returns_503(self):
        """Verify that when no predictor is loaded, service returns 503 instead of fabricating scores."""
        # Uninitialized service with non-existent checkpoint and no mock
        unready_service = ModelService(checkpoint_path="nonexistent_checkpoint.keras")
        app = create_app(model_service=unready_service)
        client = TestClient(app)

        files = {"file": ("test.mp4", self.synth_video_bytes, "video/mp4")}
        resp = client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 503)
        data = resp.json()
        self.assertIn("service is not ready", data["detail"])


if __name__ == "__main__":
    unittest.main()
