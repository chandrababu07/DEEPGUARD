"""
Unit tests for DeepGuard explainability API (/api/v1/explain) and artifact serving.

Tests cover:
- Grad-CAM explanation generation with mock explainer
- Unavailability handling when explainer is not loaded (503)
- Artifact image serving (/api/v1/artifacts/{filename})
- Path traversal protection on artifact resolution (400 / 403 / 404)
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
from src.api.service import ModelService
from src.dataset.preprocessor import PreprocessingResult
from src.explainability.gradcam import GradCAMExplanation
from src.model.inference import InferenceResult
from tests.test_api_predict import create_synthetic_mp4_bytes


class TestAPIExplain(unittest.TestCase):
    """Test suite for /api/v1/explain and /api/v1/artifacts/{filename}."""

    def setUp(self):
        self.synth_video_bytes = create_synthetic_mp4_bytes(num_frames=8)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.artifact_dir = os.path.join(self.tmp_dir.name, "artifacts")
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.custom_settings = APISettings(artifact_storage_dir=self.artifact_dir)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_explain_unready_returns_503(self):
        """Verify explain returns 503 when explainer is not available."""
        unready_service = ModelService(checkpoint_path="nonexistent.keras")
        app = create_app(model_service=unready_service, custom_settings=self.custom_settings)
        client = TestClient(app)

        files = {"file": ("test.mp4", self.synth_video_bytes, "video/mp4")}
        resp = client.post("/api/v1/explain", files=files)
        self.assertEqual(resp.status_code, 503)

    def test_explain_success_with_injected_explainer(self):
        """Verify explain returns valid ExplanationResponse and overlays when explainer is injected."""
        class MockPredictor:
            def predict_video(self, video_path: str, sample_id: str):
                return InferenceResult(
                    sample_id=sample_id,
                    video_path=video_path,
                    predicted_label="FAKE",
                    fake_score=0.91,
                    real_score=0.09,
                    decision_threshold=0.5,
                )

        class MockExplainer:
            target_layer_name = "Conv_1"

            def explain_video(self, frames_tensor, output_dir, sample_id, frame_indices):
                exps = []
                for idx in frame_indices:
                    img_path = os.path.join(output_dir, f"{sample_id}_frame_{idx}_gradcam.png")
                    # Write small dummy image file
                    cv2.imwrite(img_path, np.zeros((128, 128, 3), dtype=np.uint8))
                    exps.append(
                        GradCAMExplanation(
                            sample_id=sample_id,
                            frame_index=idx,
                            target_layer=self.target_layer_name,
                            output_image_path=img_path,
                            peak_activation_coords=(64, 50),
                            mean_activation_intensity=0.45,
                            predicted_score=0.91,
                        )
                    )
                return exps

        class MockPreprocessor:
            def process_video(self, video_path: str):
                return PreprocessingResult(
                    sample_id="sample_01",
                    video_path=video_path,
                    status="SUCCESS",
                    frames=np.zeros((8, 128, 128, 3), dtype=np.float32),
                )

        mock_service = ModelService(
            predictor=MockPredictor(),
            explainer=MockExplainer(),
            preprocessor=MockPreprocessor(),
        )
        app = create_app(model_service=mock_service, custom_settings=self.custom_settings)
        client = TestClient(app)

        files = {"file": ("test.mp4", self.synth_video_bytes, "video/mp4")}
        resp = client.post("/api/v1/explain", files=files)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["predicted_label"], "MANIPULATED")
        self.assertEqual(data["target_layer"], "Conv_1")
        self.assertEqual(data["frames_explained"], 3)
        self.assertIn("Preliminary development model", data["disclaimer"])
        self.assertEqual(len(data["explanations"]), 3)

        # Check first explanation item
        first_exp = data["explanations"][0]
        self.assertEqual(first_exp["frame_index"], 0)
        self.assertEqual(first_exp["peak_activation_coords"], [64, 50])
        self.assertIn("/api/v1/artifacts/", first_exp["overlay_url"])

    def test_artifact_retrieval_and_path_traversal_protection(self):
        """Verify artifact route safely serves images and blocks path traversal attempts."""
        app = create_app(custom_settings=self.custom_settings)
        client = TestClient(app)

        # 1. Create legitimate test image in artifact storage dir
        valid_img_path = os.path.join(self.artifact_dir, "test_artifact_01.png")
        cv2.imwrite(valid_img_path, np.zeros((10, 10, 3), dtype=np.uint8))

        # 2. Legitimate retrieval returns 200
        resp = client.get("/api/v1/artifacts/test_artifact_01.png")
        self.assertEqual(resp.status_code, 200)

        # 3. Path traversal returns 400 (bad request) or 404 (normalized by routing)
        resp_traversal = client.get("/api/v1/artifacts/..%2F..%2Fsecret.png")
        self.assertIn(resp_traversal.status_code, (400, 404))

        # 4. Non-existent artifact returns 404
        resp_404 = client.get("/api/v1/artifacts/nonexistent_image.png")
        self.assertEqual(resp_404.status_code, 404)


if __name__ == "__main__":
    unittest.main()
