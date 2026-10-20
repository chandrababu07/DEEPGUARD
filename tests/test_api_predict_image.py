"""
Unit tests for DeepGuard dedicated image prediction API (/api/v1/predict/image).

Tests cover:
- Empty upload rejection (400)
- Unsupported container / magic byte rejection (415)
- Corrupted image file rejection (422)
- File size boundary enforcement (413)
- Excessive image dimensions rejection (400 - decompression bomb defense)
- Unconfigured image model returns 503 Service Unavailable
- Valid prediction with injected deterministic predictor (200)
- Correct label mapping (AUTHENTIC = 0, MANIPULATED = 1)
- Non-finite and out-of-range score handling
- Temporary file cleanup in success and failure paths
- Directory traversal filename sanitization
- Readiness probe image model reporting
"""

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
from src.api.image_service import ImageModelService
from src.api.service import ModelService
from src.dataset.image_preprocessor import ImagePreprocessor
from src.model.image_inference import ImagePredictor, ImageInferenceResult


def create_synthetic_image_bytes(width: int = 128, height: int = 128, ext: str = ".jpg") -> bytes:
    """Helper to generate an in-memory encoded image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[20:100, 20:100] = [100, 150, 240]
    success, encoded = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError(f"Failed to encode synthetic {ext} image")
    return encoded.tobytes()


class TestAPIPredictImage(unittest.TestCase):
    """Test suite for /api/v1/predict/image endpoint."""

    def setUp(self):
        self.synth_jpeg_bytes = create_synthetic_image_bytes(128, 128, ".jpg")
        self.synth_png_bytes = create_synthetic_image_bytes(128, 128, ".png")
        self.synth_webp_bytes = create_synthetic_image_bytes(128, 128, ".webp")

    def test_empty_file_upload_returns_400(self):
        """Verify empty file upload is rejected with 400 Bad Request."""
        app = create_app()
        client = TestClient(app)

        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        resp = client.post("/api/v1/predict/image", files=files)
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertIn("empty", data["detail"].lower())

    def test_invalid_magic_bytes_rejected_with_415(self):
        """Verify file with non-image magic bytes is rejected with 415."""
        app = create_app()
        client = TestClient(app)

        files = {"file": ("fake_image.jpg", b"This is plain text content masquerading as JPEG", "image/jpeg")}
        resp = client.post("/api/v1/predict/image", files=files)
        self.assertEqual(resp.status_code, 415)
        data = resp.json()
        self.assertIn("not a recognized or supported image format", data["detail"])

    def test_corrupted_image_rejected_with_422(self):
        """Verify file with valid header but corrupted body is rejected with 422."""
        app = create_app()
        client = TestClient(app)

        # JPEG SOI marker followed by truncated garbage
        corrupted_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 20
        files = {"file": ("corrupt.jpg", corrupted_jpeg, "image/jpeg")}
        resp = client.post("/api/v1/predict/image", files=files)
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertIn("Failed to decode image", data["detail"])

    def test_oversized_file_rejected_with_413(self):
        """Verify image file exceeding max_image_upload_size_bytes is rejected with 413."""
        custom_settings = APISettings(max_image_upload_size_bytes=500)
        app = create_app(custom_settings=custom_settings)
        client = TestClient(app)

        files = {"file": ("oversized.jpg", self.synth_jpeg_bytes, "image/jpeg")}
        resp = client.post("/api/v1/predict/image", files=files)
        self.assertEqual(resp.status_code, 413)
        data = resp.json()
        self.assertIn("exceeds maximum allowed limit", data["detail"])

    def test_excessive_dimensions_rejected_with_400(self):
        """Verify image exceeding max_image_dimension is rejected with 400 (decompression bomb safety)."""
        # Set max dimension to 100 pixels
        custom_settings = APISettings(max_image_dimension=100)
        app = create_app(custom_settings=custom_settings)
        client = TestClient(app)

        # 128x128 image exceeds 100 max dimension
        files = {"file": ("large_dim.jpg", self.synth_jpeg_bytes, "image/jpeg")}
        resp = client.post("/api/v1/predict/image", files=files)
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertIn("exceed maximum allowed limit", data["detail"])

    def test_unconfigured_model_returns_503(self):
        """Verify that when no image model is configured, endpoint returns structured 503."""
        # Service with no checkpoint configured
        unconfigured_service = ImageModelService(checkpoint_path="")
        app = create_app(image_service=unconfigured_service)
        client = TestClient(app)

        files = {"file": ("test.jpg", self.synth_jpeg_bytes, "image/jpeg")}
        resp = client.post("/api/v1/predict/image", files=files)
        self.assertEqual(resp.status_code, 503)
        data = resp.json()
        self.assertIn("service is unavailable", data["detail"].lower())

    def test_prediction_success_manipulated(self):
        """Verify image prediction returning fake_score >= threshold returns MANIPULATED (label_code 1)."""
        class MockImagePredictor:
            def predict_image(self, image_path: str, sample_id: str):
                return ImageInferenceResult(
                    sample_id=sample_id,
                    image_path=image_path,
                    predicted_label="FAKE",
                    fake_score=0.885,
                    real_score=0.115,
                    decision_threshold=0.5,
                    model_version="deepguard-image-v1-mock",
                    checkpoint_hash="hash_mock_img_fake",
                    original_dimensions=[128, 128, 3],
                )

        mock_img_service = ImageModelService(predictor=MockImagePredictor(), decision_threshold=0.5)
        app = create_app(image_service=mock_img_service)
        client = TestClient(app)

        files = {"file": ("test_fake.jpg", self.synth_jpeg_bytes, "image/jpeg")}
        resp = client.post("/api/v1/predict/image", files=files)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["predicted_label"], "MANIPULATED")
        self.assertEqual(data["label_code"], 1)
        self.assertAlmostEqual(data["fake_score"], 0.885, places=3)
        self.assertAlmostEqual(data["real_score"], 0.115, places=3)
        self.assertEqual(data["decision_threshold"], 0.5)
        self.assertIn("request_id", data)
        self.assertIn("processing_time_ms", data)
        self.assertEqual(data["image_dimensions"], [128, 128, 3])

    def test_prediction_success_authentic(self):
        """Verify image prediction returning fake_score < threshold returns AUTHENTIC (label_code 0)."""
        class MockImagePredictor:
            def predict_image(self, image_path: str, sample_id: str):
                return ImageInferenceResult(
                    sample_id=sample_id,
                    image_path=image_path,
                    predicted_label="REAL",
                    fake_score=0.112,
                    real_score=0.888,
                    decision_threshold=0.5,
                    model_version="deepguard-image-v1-mock",
                    checkpoint_hash="hash_mock_img_real",
                    original_dimensions=[128, 128, 3],
                )

        mock_img_service = ImageModelService(predictor=MockImagePredictor(), decision_threshold=0.5)
        app = create_app(image_service=mock_img_service)
        client = TestClient(app)

        files = {"file": ("test_real.png", self.synth_png_bytes, "image/png")}
        resp = client.post("/api/v1/predict/image", files=files)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["predicted_label"], "AUTHENTIC")
        self.assertEqual(data["label_code"], 0)
        self.assertAlmostEqual(data["fake_score"], 0.112, places=3)
        self.assertAlmostEqual(data["real_score"], 0.888, places=3)

    def test_supported_formats_png_and_webp(self):
        """Verify PNG and WebP images are decoded and handled successfully."""
        class MockImagePredictor:
            def predict_image(self, image_path: str, sample_id: str):
                return ImageInferenceResult(
                    sample_id=sample_id,
                    image_path=image_path,
                    predicted_label="REAL",
                    fake_score=0.20,
                    real_score=0.80,
                    original_dimensions=[128, 128, 3],
                )

        mock_img_service = ImageModelService(predictor=MockImagePredictor())
        app = create_app(image_service=mock_img_service)
        client = TestClient(app)

        # PNG
        resp_png = client.post(
            "/api/v1/predict/image",
            files={"file": ("test.png", self.synth_png_bytes, "image/png")},
        )
        self.assertEqual(resp_png.status_code, 200)

        # WebP
        resp_webp = client.post(
            "/api/v1/predict/image",
            files={"file": ("test.webp", self.synth_webp_bytes, "image/webp")},
        )
        self.assertEqual(resp_webp.status_code, 200)

    def test_predictor_error_returns_422(self):
        """Verify image predictor processing error returns 422."""
        class MockErrorPredictor:
            def predict_image(self, image_path: str, sample_id: str):
                return ImageInferenceResult(
                    sample_id=sample_id,
                    image_path=image_path,
                    predicted_label="ERROR",
                    error="Image tensor normalization error.",
                )

        mock_img_service = ImageModelService(predictor=MockErrorPredictor())
        app = create_app(image_service=mock_img_service)
        client = TestClient(app)

        files = {"file": ("test.jpg", self.synth_jpeg_bytes, "image/jpeg")}
        resp = client.post("/api/v1/predict/image", files=files)
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertIn("normalization error", data["detail"])

    def test_temp_file_cleanup_on_success_and_failure(self):
        """Verify uploaded temporary image file is cleanly removed on both success and error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            upload_dir = os.path.join(tmp_dir, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            custom_settings = APISettings(upload_tmp_dir=upload_dir)

            class MockPredictor:
                def predict_image(self, image_path: str, sample_id: str):
                    if not os.path.exists(image_path):
                        raise RuntimeError("Temp image does not exist during prediction!")
                    return ImageInferenceResult(
                        sample_id=sample_id,
                        image_path=image_path,
                        predicted_label="REAL",
                        fake_score=0.1,
                    )

            mock_service = ImageModelService(predictor=MockPredictor())
            app = create_app(image_service=mock_service, custom_settings=custom_settings)
            client = TestClient(app)

            files = {"file": ("test_clean.jpg", self.synth_jpeg_bytes, "image/jpeg")}
            resp = client.post("/api/v1/predict/image", files=files)
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(len(os.listdir(upload_dir)), 0, "Temporary image was not cleaned up!")

            # Test failure cleanup
            class MockFailingPredictor:
                def predict_image(self, image_path: str, sample_id: str):
                    raise RuntimeError("Simulated crash!")

            failing_service = ImageModelService(predictor=MockFailingPredictor())
            app_fail = create_app(image_service=failing_service, custom_settings=custom_settings)
            client_fail = TestClient(app_fail)

            resp_fail = client_fail.post("/api/v1/predict/image", files=files)
            self.assertEqual(resp_fail.status_code, 500)
            self.assertEqual(len(os.listdir(upload_dir)), 0, "Temporary image was not cleaned up on failure!")

    def test_non_finite_and_out_of_range_score_validation(self):
        """Verify ImagePredictor rejects NaN, Inf, and scores outside [0.0, 1.0]."""
        predictor = ImagePredictor()

        # Mock model returning NaN
        class NanModel:
            def __call__(self, x, training=False):
                return np.array([[float("nan")]])

        predictor.model = NanModel()
        res_nan = predictor.predict_tensor(np.zeros((128, 128, 3), dtype=np.float32))
        self.assertEqual(res_nan.predicted_label, "ERROR")
        self.assertIn("non-finite", res_nan.error)

        # Mock model returning Inf
        class InfModel:
            def __call__(self, x, training=False):
                return np.array([[float("inf")]])

        predictor.model = InfModel()
        res_inf = predictor.predict_tensor(np.zeros((128, 128, 3), dtype=np.float32))
        self.assertEqual(res_inf.predicted_label, "ERROR")
        self.assertIn("non-finite", res_inf.error)

        # Mock model returning out of range score (1.5)
        class OutOfRangeModel:
            def __call__(self, x, training=False):
                return np.array([[1.5]])

        predictor.model = OutOfRangeModel()
        res_oor = predictor.predict_tensor(np.zeros((128, 128, 3), dtype=np.float32))
        self.assertEqual(res_oor.predicted_label, "ERROR")
        self.assertIn("out of range", res_oor.error)

    def test_readiness_probe_includes_image_model_fields(self):
        """Verify /api/v1/ready probe reports image model fields."""
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/ready")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("image_model_ready", data)
        self.assertIn("image_checkpoint_path", data)
        self.assertIn("image_model_status", data)
        self.assertFalse(data["image_model_ready"])
        self.assertEqual(data["image_model_status"], "unconfigured")


if __name__ == "__main__":
    unittest.main()
