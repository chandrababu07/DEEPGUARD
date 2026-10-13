"""
Unit tests for DeepGuard Grad-CAM explainability module.

Tests cover:
- Frame sequence tensor validation (8, 128, 128, 3)
- Handling of 5D single-video batch tensor (1, 8, 128, 128, 3)
- Rejection of invalid shapes (wrong frame count, resolution, channels, dimensions)
- GradCAMExplanation dataclass schema, disclaimer text, and GradCAMIndex catalog
- Graceful handling of missing TensorFlow on local environment
- Output contract verification
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explainability.gradcam import (
    DeepGuardGradCAM,
    GradCAMExplanation,
    GradCAMIndex,
    EXPLANATION_DISCLAIMER,
)
from src.model.baseline import HAS_TF


class TestExplainability(unittest.TestCase):
    """Test suite for Grad-CAM explainability module."""

    def setUp(self):
        self.explainer = DeepGuardGradCAM(model=None, target_layer_name="Conv_1")

    def test_tensor_validation_valid_4d_shape(self):
        """Verify (8, 128, 128, 3) array passes validation."""
        valid_tensor = np.zeros((8, 128, 128, 3), dtype=np.float32)
        validated = self.explainer.validate_tensor(valid_tensor)
        self.assertEqual(validated.shape, (8, 128, 128, 3))
        self.assertEqual(validated.dtype, np.float32)

    def test_tensor_validation_valid_5d_shape(self):
        """Verify (1, 8, 128, 128, 3) batch array is squeezed to (8, 128, 128, 3)."""
        valid_5d = np.zeros((1, 8, 128, 128, 3), dtype=np.float32)
        validated = self.explainer.validate_tensor(valid_5d)
        self.assertEqual(validated.shape, (8, 128, 128, 3))

    def test_tensor_validation_invalid_shapes(self):
        """Verify ValueError is raised for non-compliant tensor shapes."""
        invalid_shapes = [
            (128, 128, 3),          # 3D single frame
            (16, 128, 128, 3),         # 16 frames instead of 8
            (8, 224, 224, 3),          # 224x224 instead of 128x128
            (8, 128, 128, 1),          # Grayscale instead of 3 channels
            (2, 8, 128, 128, 3),       # Multiple videos in batch
        ]
        for shape in invalid_shapes:
            dummy = np.zeros(shape, dtype=np.float32)
            with self.assertRaises(ValueError, msg=f"Failed to reject shape {shape}"):
                self.explainer.validate_tensor(dummy)

    def test_explanation_dataclass_and_disclaimer(self):
        """Verify GradCAMExplanation dataclass schema, serialization, and mandatory disclaimer."""
        exp = GradCAMExplanation(
            sample_id="SAMPLE_001",
            frame_index=3,
            target_layer="Conv_1",
            output_image_path="experiments/gradcam/sample_001_frame_3.png",
            peak_activation_coords=(64, 45),
            mean_activation_intensity=0.321,
            predicted_score=0.875,
        )

        d = exp.to_dict()
        self.assertEqual(d["sample_id"], "SAMPLE_001")
        self.assertEqual(d["frame_index"], 3)
        self.assertEqual(d["target_layer"], "Conv_1")
        self.assertEqual(d["peak_activation_coords"], (64, 45))
        self.assertAlmostEqual(d["mean_activation_intensity"], 0.321)
        self.assertAlmostEqual(d["predicted_score"], 0.875)
        self.assertIn("Preliminary development model", d["disclaimer"])
        self.assertIn("not legal or forensic proof", d["disclaimer"])

    def test_explanation_index_serialization(self):
        """Verify GradCAMIndex catalog aggregates explanations into structured dictionary."""
        exp1 = GradCAMExplanation(
            sample_id="SAMPLE_001",
            frame_index=0,
            target_layer="Conv_1",
            output_image_path="path/to/img0.png",
            peak_activation_coords=(10, 10),
            mean_activation_intensity=0.2,
            predicted_score=0.6,
        )
        exp2 = GradCAMExplanation(
            sample_id="SAMPLE_001",
            frame_index=7,
            target_layer="Conv_1",
            output_image_path="path/to/img7.png",
            peak_activation_coords=(20, 20),
            mean_activation_intensity=0.4,
            predicted_score=0.6,
        )
        index = GradCAMIndex(
            timestamp="2026-10-13T12:00:00",
            target_layer="Conv_1",
            explanations=[exp1, exp2],
        )

        d = index.to_dict()
        self.assertEqual(d["timestamp"], "2026-10-13T12:00:00")
        self.assertEqual(d["target_layer"], "Conv_1")
        self.assertEqual(d["total_explanations"], 2)
        self.assertEqual(len(d["explanations"]), 2)

    def test_frame_index_bounds_checking(self):
        """Verify frame_idx outside [0, 7] is rejected."""
        valid_tensor = np.zeros((8, 128, 128, 3), dtype=np.float32)
        if not HAS_TF:
            # When TF is absent, compute_heatmap raises ImportError
            with self.assertRaises(ImportError):
                self.explainer.compute_heatmap(valid_tensor, frame_idx=0)
        else:
            with self.assertRaises(IndexError):
                self.explainer.compute_heatmap(valid_tensor, frame_idx=-1)
            with self.assertRaises(IndexError):
                self.explainer.compute_heatmap(valid_tensor, frame_idx=8)

    def test_missing_tensorflow_raises_clear_error(self):
        """Verify when TensorFlow is absent, explainer reports clear ImportError."""
        if not HAS_TF:
            valid_tensor = np.zeros((8, 128, 128, 3), dtype=np.float32)
            with self.assertRaises(ImportError) as ctx:
                self.explainer.compute_heatmap(valid_tensor, frame_idx=0)
            self.assertIn("TensorFlow is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
