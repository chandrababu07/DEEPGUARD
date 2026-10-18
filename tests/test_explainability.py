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
        """Verify frame_idx outside [0, 7] and non-integer indices are rejected with IndexError."""
        valid_tensor = np.zeros((8, 128, 128, 3), dtype=np.float32)

        # Boundary and out-of-bounds indices must raise IndexError before TF/model checks
        for invalid_idx in [-1, -5, 8, 9, 100]:
            with self.assertRaises(IndexError, msg=f"Expected IndexError for frame_idx={invalid_idx}"):
                self.explainer.compute_heatmap(valid_tensor, frame_idx=invalid_idx)

        # Non-integer types must raise IndexError
        for non_int_idx in [1.5, "0", None, True, False, [0]]:
            with self.assertRaises(IndexError, msg=f"Expected IndexError for non-integer frame_idx={non_int_idx}"):
                self.explainer.compute_heatmap(valid_tensor, frame_idx=non_int_idx)

    def test_frame_index_valid_indices_continue_pipeline(self):
        """Verify valid Python and NumPy integer indices pass frame bounds check to TF/model checks."""
        valid_tensor = np.zeros((8, 128, 128, 3), dtype=np.float32)
        valid_indices = [0, 7, np.int32(0), np.int64(7), np.int16(3)]
        for idx in valid_indices:
            if not HAS_TF:
                with self.assertRaises(ImportError):
                    self.explainer.compute_heatmap(valid_tensor, frame_idx=idx)
            else:
                with self.assertRaises(ValueError) as ctx:
                    self.explainer.compute_heatmap(valid_tensor, frame_idx=idx)
                self.assertIn("No model provided", str(ctx.exception))

    def test_missing_tensorflow_raises_clear_error(self):
        """Verify when TensorFlow is absent, explainer reports clear ImportError."""
        if not HAS_TF:
            valid_tensor = np.zeros((8, 128, 128, 3), dtype=np.float32)
            with self.assertRaises(ImportError) as ctx:
                self.explainer.compute_heatmap(valid_tensor, frame_idx=0)
            self.assertIn("TensorFlow is required", str(ctx.exception))

    def test_target_layer_selection_exact_and_nested_protection(self):
        """Verify _find_target_layer selects exact layer and never selects nested model containers."""
        class MockLayer:
            def __init__(self, name: str, layers: list = None):
                self.name = name
                if layers is not None:
                    self.layers = layers

        # Construct nested extractor hierarchy matching DeepGuard MobileNetV2 + GAP
        conv1 = MockLayer("Conv1")
        conv_1 = MockLayer("Conv_1")
        conv_1_bn = MockLayer("Conv_1_bn")
        out_relu = MockLayer("out_relu")
        backbone = MockLayer("mobilenetv2_1.00_128", layers=[conv1, conv_1, conv_1_bn, out_relu])
        gap = MockLayer("spatial_global_avg_pooling")
        extractor = MockLayer(
            "mobilenetv2_feature_extractor",
            layers=[MockLayer("single_frame_input"), backbone, gap],
        )

        # Exact match for Conv_1
        target, bb, post = self.explainer._find_target_layer(extractor, "Conv_1")
        self.assertEqual(target.name, "Conv_1")
        self.assertIs(bb, backbone)
        self.assertEqual(
            [l.name for l in post],
            ["Conv_1_bn", "out_relu", "spatial_global_avg_pooling"],
        )

        # Never select nested model container even if its name is targeted
        target_bb, _, _ = self.explainer._find_target_layer(extractor, "mobilenetv2_1.00_128")
        self.assertNotEqual(target_bb.name, "mobilenetv2_1.00_128")
        # Should fallback to last conv/relu layer (out_relu)
        self.assertEqual(target_bb.name, "out_relu")

        # Fallback when unknown layer name is requested
        target_fb, _, post_fb = self.explainer._find_target_layer(extractor, "nonexistent_layer")
        self.assertEqual(target_fb.name, "out_relu")
        self.assertEqual([l.name for l in post_fb], ["spatial_global_avg_pooling"])

        # ValueError when no conv/relu layer exists
        flat_linear_extractor = MockLayer(
            "flat_extractor", layers=[MockLayer("dense_1"), MockLayer("dense_2")]
        )
        with self.assertRaises(ValueError) as ctx:
            self.explainer._find_target_layer(flat_linear_extractor, "Conv_1")
        self.assertIn("Could not locate target layer", str(ctx.exception))

    def test_disconnected_gradients_raises_clear_error(self):
        """Verify that disconnected or None gradients raise clear error instead of fake uniform heatmap."""
        class DummyModel:
            pass

        explainer = DeepGuardGradCAM(model=DummyModel(), target_layer_name="Conv_1")
        with self.assertRaises((RuntimeError, ValueError, ImportError)):
            valid_tensor = np.zeros((8, 128, 128, 3), dtype=np.float32)
            explainer.compute_heatmap(valid_tensor, frame_idx=0)


    def test_target_class_selection_logic(self):
        """Verify _determine_target correctly targets fake vs authentic class based on prediction & threshold."""
        explainer = DeepGuardGradCAM(model=None, target_layer_name="Conv_1", decision_threshold=0.5)

        # 1. Prediction >= threshold targets FAKE (p)
        _, label, score = explainer._determine_target("p_tensor", 0.85, target_class=None)
        self.assertEqual(label, "FAKE")
        self.assertAlmostEqual(score, 0.85)

        # 2. Prediction < threshold targets REAL (1 - p)
        _, label, score = explainer._determine_target("p_tensor", 0.15, target_class=None)
        self.assertEqual(label, "REAL")
        self.assertAlmostEqual(score, 0.85)

        # 3. Custom decision threshold
        custom_explainer = DeepGuardGradCAM(model=None, target_layer_name="Conv_1", decision_threshold=0.2)
        _, label, score = custom_explainer._determine_target("p_tensor", 0.25, target_class=None)
        self.assertEqual(label, "FAKE")
        self.assertAlmostEqual(score, 0.25)

        _, label, score = custom_explainer._determine_target("p_tensor", 0.15, target_class=None)
        self.assertEqual(label, "REAL")
        self.assertAlmostEqual(score, 0.85)

        # 4. Explicit target_class overrides prediction
        _, label_fake, score_fake = explainer._determine_target("p_tensor", 0.15, target_class="FAKE")
        self.assertEqual(label_fake, "FAKE")
        self.assertAlmostEqual(score_fake, 0.15)

        _, label_real, score_real = explainer._determine_target("p_tensor", 0.85, target_class="REAL")
        self.assertEqual(label_real, "REAL")
        self.assertAlmostEqual(score_real, 0.15)

        # 5. Invalid target_class raises ValueError
        with self.assertRaises(ValueError):
            explainer._determine_target("p_tensor", 0.5, target_class="UNKNOWN_CLASS")


@unittest.skipIf(not HAS_TF, "TensorFlow not installed in current environment. Run in Google Colab to execute.")
class TestExplainabilityTensorFlow(unittest.TestCase):
    """
    TensorFlow live gradient backpropagation and contract verification for Grad-CAM.
    Runs when TensorFlow is available (such as in Google Colab).
    """

    @classmethod
    def setUpClass(cls):
        from src.model.baseline import DeepGuardModelConfig, build_deepguard_model
        cls.config = DeepGuardModelConfig(weights=None, trainable_backbone=False)
        cls.model = build_deepguard_model(cls.config)

    def test_gradcam_with_mobilenetv2_lstm_architecture(self):
        """Verify end-to-end Grad-CAM with full MobileNetV2 + LSTM architecture and numerical diagnostics."""
        explainer = DeepGuardGradCAM(model=self.model, target_layer_name="Conv_1", decision_threshold=0.5)
        dummy_tensor = np.random.uniform(0.1, 0.9, size=(8, 128, 128, 3)).astype(np.float32)

        # Test both target classes
        for target_cls in ["REAL", "FAKE", None]:
            heatmap, pred_score = explainer.compute_heatmap(dummy_tensor, frame_idx=0, target_class=target_cls)

            # Output shape contract
            self.assertEqual(heatmap.shape, (128, 128))
            self.assertEqual(heatmap.dtype, np.float32)

            # Finite, nonnegative, normalized range
            self.assertTrue(np.all(np.isfinite(heatmap)))
            self.assertTrue(np.all(heatmap >= 0.0))
            self.assertTrue(np.all(heatmap <= 1.0))
            self.assertTrue(float(np.max(heatmap)) > float(np.min(heatmap)), "Heatmap must have nonzero variation")

            # Prediction score contract
            self.assertIsInstance(pred_score, float)
            self.assertTrue(np.isfinite(pred_score))
            self.assertTrue(0.0 <= pred_score <= 1.0)

            # Numerical diagnostics verification
            diag = explainer.last_diagnostics
            self.assertFalse(diag["grads_none"])
            self.assertTrue(diag["grads_finite"])
            self.assertTrue(np.isfinite(diag["grads_min"]))
            self.assertTrue(np.isfinite(diag["grads_max"]))
            self.assertTrue(np.isfinite(diag["grads_mean"]))
            self.assertTrue(np.isfinite(diag["activations_min"]))
            self.assertTrue(np.isfinite(diag["activations_max"]))
            self.assertTrue(np.isfinite(diag["pooled_grads_min"]))
            self.assertTrue(np.isfinite(diag["raw_cam_min"]))
            self.assertTrue(np.isfinite(diag["raw_cam_max"]))
            if target_cls is not None:
                self.assertEqual(diag["target_label"], target_cls)

        # Overlay generation
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_img = os.path.join(tmp_dir, "test_overlay.png")
            exp = explainer.generate_and_save_overlay(
                frames_tensor=dummy_tensor,
                frame_idx=3,
                output_image_path=out_img,
                sample_id="test_sample",
            )
            self.assertTrue(os.path.exists(out_img))
            self.assertEqual(exp.frame_index, 3)
            self.assertEqual(exp.target_layer, "Conv_1")
            self.assertEqual(len(exp.peak_activation_coords), 2)
            self.assertTrue(0 <= exp.peak_activation_coords[0] <= 128)
            self.assertTrue(0 <= exp.peak_activation_coords[1] <= 128)
            self.assertTrue(0.0 <= exp.mean_activation_intensity <= 1.0)
            self.assertTrue(0.0 <= exp.predicted_score <= 1.0)

    def test_gradcam_with_loaded_checkpoint_if_present(self):
        """Verify Grad-CAM against trained checkpoint if available on disk for both authentic and manipulated targets."""
        checkpoint_path = "experiments/mini_baseline/best_model.keras"
        if not os.path.exists(checkpoint_path):
            self.skipTest(f"Checkpoint not found at {checkpoint_path}")

        import tempfile
        from src.model.loader import load_deepguard_checkpoint
        loaded_model, chk_hash = load_deepguard_checkpoint(checkpoint_path)
        explainer = DeepGuardGradCAM(model=loaded_model, target_layer_name="Conv_1", decision_threshold=0.5)

        dummy_tensor = np.random.uniform(0.1, 0.9, size=(8, 128, 128, 3)).astype(np.float32)

        # Verify frames 0, 3, 7 have nonconstant maps and finite values
        exps = explainer.explain_video(
            frames_tensor=dummy_tensor,
            output_dir=tempfile.mkdtemp(),
            sample_id="checkpoint_test",
            frame_indices=[0, 3, 7],
        )

        self.assertEqual(len(exps), 3)
        for exp in exps:
            self.assertEqual(exp.target_layer, "Conv_1")
            self.assertTrue(np.isfinite(exp.predicted_score))
            self.assertTrue(0.0 <= exp.predicted_score <= 1.0)
            self.assertTrue(np.isfinite(exp.mean_activation_intensity))
            self.assertTrue(0.0 <= exp.mean_activation_intensity <= 1.0)

        # Verify explicit authentic target and manipulated target produce valid diagnostics
        for target_cls in ["REAL", "FAKE"]:
            heatmap, _ = explainer.compute_heatmap(dummy_tensor, frame_idx=0, target_class=target_cls)
            self.assertEqual(heatmap.shape, (128, 128))
            self.assertTrue(np.all(np.isfinite(heatmap)))
            self.assertTrue(explainer.last_diagnostics["grads_finite"])


if __name__ == "__main__":
    unittest.main()
