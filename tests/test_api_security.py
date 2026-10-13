"""
Unit tests for DeepGuard API security, temporary file lifecycle, and error sanitization.
"""

import os
import sys
import tempfile
import unittest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.app import create_app
from src.api.config import APISettings
from src.api.service import ModelService
from src.model.inference import InferenceResult
from tests.test_api_predict import create_synthetic_mp4_bytes


class TestAPISecurityAndCleanup(unittest.TestCase):
    """Test suite for safe file handling, cleanup on success and error, and sanitized errors."""

    def setUp(self):
        self.synth_video_bytes = create_synthetic_mp4_bytes(num_frames=8)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.upload_dir = os.path.join(self.tmp_dir.name, "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        self.custom_settings = APISettings(upload_tmp_dir=self.upload_dir)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_temp_file_removed_after_successful_prediction(self):
        """Verify uploaded temporary video file is removed from disk after successful prediction."""
        class MockPredictor:
            def predict_video(self, video_path: str, sample_id: str):
                # Verify that while prediction runs, the temp file actually exists
                if not os.path.exists(video_path):
                    raise RuntimeError("Temp file does not exist during prediction!")
                return InferenceResult(
                    sample_id=sample_id,
                    video_path=video_path,
                    predicted_label="REAL",
                    fake_score=0.10,
                )

        mock_service = ModelService(predictor=MockPredictor())
        app = create_app(model_service=mock_service, custom_settings=self.custom_settings)
        client = TestClient(app)

        files = {"file": ("test_clean.mp4", self.synth_video_bytes, "video/mp4")}
        resp = client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 200)

        # After request completes, the upload directory must be completely empty
        remaining_files = os.listdir(self.upload_dir)
        self.assertEqual(len(remaining_files), 0, f"Leaked temporary files found: {remaining_files}")

    def test_temp_file_removed_after_prediction_exception(self):
        """Verify uploaded temporary video file is cleaned up even if predictor raises an exception."""
        class MockFailingPredictor:
            def predict_video(self, video_path: str, sample_id: str):
                raise RuntimeError("Simulated internal exception during inference!")

        mock_service = ModelService(predictor=MockFailingPredictor())
        app = create_app(model_service=mock_service, custom_settings=self.custom_settings)
        client = TestClient(app)

        files = {"file": ("test_fail.mp4", self.synth_video_bytes, "video/mp4")}
        resp = client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 500)

        # Temp file must still be safely removed in finally block
        remaining_files = os.listdir(self.upload_dir)
        self.assertEqual(len(remaining_files), 0, f"Leaked temporary files on failure: {remaining_files}")

    def test_client_filename_path_traversal_sanitized(self):
        """Verify malicious client filename with directory traversal patterns cannot escape upload dir."""
        class MockPredictor:
            def predict_video(self, video_path: str, sample_id: str):
                # File must be inside upload_dir
                assert os.path.dirname(video_path) == os.path.abspath(self_upload_dir)
                return InferenceResult(sample_id=sample_id, video_path=video_path, predicted_label="REAL", fake_score=0.2)

        self_upload_dir = self.upload_dir
        mock_service = ModelService(predictor=MockPredictor())
        app = create_app(model_service=mock_service, custom_settings=self.custom_settings)
        client = TestClient(app)

        malicious_filename = "../../../../../etc/passwd.mp4"
        files = {"file": (malicious_filename, self.synth_video_bytes, "video/mp4")}
        resp = client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 200)

    def test_error_responses_do_not_leak_internal_paths_or_tracebacks(self):
        """Verify 500 error responses return sanitized ErrorResponse without Python tracebacks."""
        class MockCrashPredictor:
            def predict_video(self, video_path: str, sample_id: str):
                raise ValueError("Secret database password or path D:\\secrets\\key.pem exposed!")

        mock_service = ModelService(predictor=MockCrashPredictor())
        app = create_app(model_service=mock_service, custom_settings=self.custom_settings)
        client = TestClient(app)

        files = {"file": ("test.mp4", self.synth_video_bytes, "video/mp4")}
        resp = client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 500)

        data = resp.json()
        self.assertIn("error", data)
        self.assertIn("detail", data)
        self.assertIn("request_id", data)
        # Verify internal secret/path is NOT leaked to client
        self.assertNotIn("D:\\secrets", data["detail"])
        self.assertNotIn("Traceback", resp.text)


if __name__ == "__main__":
    unittest.main()
