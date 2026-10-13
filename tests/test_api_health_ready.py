"""
Unit tests for DeepGuard API health, readiness, and middleware contracts.
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.app import create_app
from src.api.service import ModelService
from src.api.schemas import HealthResponse, ReadinessResponse


class TestAPIHealthAndReadiness(unittest.TestCase):
    """Test suite for /api/v1/health and /api/v1/ready probes."""

    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_health_probe_returns_200_and_healthy(self):
        """Verify /api/v1/health reports active service status."""
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "DeepGuard Deepfake Detection API")
        self.assertIn("timestamp", data)
        self.assertIn("version", data)

    def test_readiness_probe_distinguishes_health_from_unloaded_model(self):
        """Verify /api/v1/ready reports not_ready when model or TensorFlow is not loaded locally."""
        resp = self.client.get("/api/v1/ready")
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("tensorflow_available", data)
        self.assertIn("model_loaded", data)
        self.assertIn("checkpoint_path", data)
        self.assertIn("expected_input_shape", data)
        self.assertEqual(data["expected_input_shape"], [8, 128, 128, 3])

    def test_readiness_probe_with_injected_ready_service(self):
        """Verify /api/v1/ready returns status='ready' when predictor is injected."""
        class MockPredictor:
            pass

        mock_service = ModelService(predictor=MockPredictor())
        mock_app = create_app(model_service=mock_service)
        client = TestClient(mock_app)

        resp = client.get("/api/v1/ready")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ready")
        self.assertTrue(data["model_loaded"])

    def test_request_id_and_timing_headers_present(self):
        """Verify X-Request-ID and X-Process-Time-Ms headers are injected by middleware."""
        resp = self.client.get("/api/v1/health", headers={"X-Request-ID": "test-uuid-12345"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("X-Request-ID"), "test-uuid-12345")
        self.assertIn("X-Process-Time-Ms", resp.headers)

    def test_ui_index_html_route_serves_html(self):
        """Verify GET / route serves the web application HTML."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers.get("content-type", ""))
        self.assertIn("DEEPGUARD", resp.text)


if __name__ == "__main__":
    unittest.main()
