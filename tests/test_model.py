"""
Unit tests and architecture verification for DeepGuard MobileNetV2 + LSTM model.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.baseline import (
    DeepGuardModelConfig,
    build_deepguard_model,
    build_frame_feature_extractor,
    HAS_TF,
)


class TestModelConfigurationAndContracts(unittest.TestCase):
    """
    Contract and configuration tests. These run in all environments (including local Python
    without TensorFlow) to ensure architecture invariants and input validation rules are strictly enforced.
    """

    def setUp(self):
        self.config = DeepGuardModelConfig()

    def test_default_config_contract(self):
        """Verify baseline architecture configuration parameters against reference paper & spec."""
        self.assertEqual(self.config.sequence_length, 8)
        self.assertEqual(self.config.frame_shape, (128, 128, 3))
        self.assertEqual(self.config.video_input_shape, (8, 128, 128, 3))
        self.assertEqual(self.config.feature_dim, 1280)
        self.assertEqual(self.config.lstm_units, 64)
        self.assertEqual(self.config.dropout_rate, 0.3)
        self.assertEqual(self.config.learning_rate, 0.0001)
        self.assertFalse(self.config.trainable_backbone)
        self.assertEqual(self.config.input_normalization, "unit_interval")

    def test_shape_validation_accepts_valid_5d(self):
        """Verify that valid 5D video input shape is accepted without error."""
        valid_shape = (2, 8, 128, 128, 3)
        # Should not raise
        self.config.validate_tensor_shape(valid_shape)

    def test_shape_validation_rejects_non_5d(self):
        """Verify that invalid tensor rank (e.g. 4D image tensor) is rejected with clear error."""
        invalid_4d = (2, 128, 128, 3)
        with self.assertRaises(ValueError) as ctx:
            self.config.validate_tensor_shape(invalid_4d)
        self.assertIn("Expected 5D tensor", str(ctx.exception))

    def test_shape_validation_rejects_wrong_sequence_length(self):
        """Verify that inputs with mismatched sequence lengths (e.g. 16 frames instead of 8) are rejected."""
        wrong_frames = (2, 16, 128, 128, 3)
        with self.assertRaises(ValueError) as ctx:
            self.config.validate_tensor_shape(wrong_frames)
        self.assertIn("Invalid sequence length", str(ctx.exception))

    def test_shape_validation_rejects_wrong_resolution(self):
        """Verify that inputs with mismatched spatial resolutions are rejected."""
        wrong_res = (2, 8, 224, 224, 3)
        with self.assertRaises(ValueError) as ctx:
            self.config.validate_tensor_shape(wrong_res)
        self.assertIn("Invalid frame dimensions", str(ctx.exception))

    def test_informative_error_when_tf_missing(self):
        """Verify that if TensorFlow is absent, an informative error is raised rather than an unhandled crash."""
        if not HAS_TF:
            with self.assertRaises(ImportError) as ctx:
                build_deepguard_model(self.config)
            self.assertIn("TensorFlow is required", str(ctx.exception))
            self.assertIn("Google Colab", str(ctx.exception))


@unittest.skipIf(not HAS_TF, "TensorFlow not installed in current environment. Run in Google Colab to execute.")
class TestModelArchitectureTensorFlow(unittest.TestCase):
    """
    TensorFlow forward-pass smoke tests and shape assertions.
    Runs when TensorFlow is available (such as in Google Colab).
    """

    @classmethod
    def setUpClass(cls):
        # Configure lightweight weights for fast unit testing in CI / local test
        cls.config = DeepGuardModelConfig(weights=None, trainable_backbone=False)

    def test_feature_extractor_output_dimension(self):
        """Verify MobileNetV2 + GAP produces exactly 1,280 features per frame."""
        extractor = build_frame_feature_extractor(self.config)
        self.assertEqual(extractor.output_shape, (None, 1280))

        # Forward pass single dummy frame
        dummy_frame = np.zeros((1, 128, 128, 3), dtype=np.float32)
        out = extractor(dummy_frame, training=False).numpy()
        self.assertEqual(out.shape, (1, 1280))

    def test_model_instantiation_and_tensor_shapes(self):
        """Verify full model input (None, 8, 128, 128, 3) and output (None, 1)."""
        model = build_deepguard_model(self.config)
        self.assertEqual(model.input_shape, (None, 8, 128, 128, 3))
        self.assertEqual(model.output_shape, (None, 1))

    def test_forward_pass_smoke_test(self):
        """
        Verify forward-pass smoke test on a small dummy batch:
        - 5D shape accepted
        - One sigmoid probability returned per video
        - Values are finite and strictly within [0.0, 1.0]
        """
        model = build_deepguard_model(self.config)
        batch_size = 2
        # Inputs normalized in [0.0, 1.0] matching preprocessor
        dummy_batch = np.random.uniform(0.0, 1.0, size=(batch_size, 8, 128, 128, 3)).astype(np.float32)

        predictions = model(dummy_batch, training=False).numpy()

        # Shape verification: (B, 1)
        self.assertEqual(predictions.shape, (batch_size, 1))

        # Finite value check (no NaN, no Inf)
        self.assertTrue(np.all(np.isfinite(predictions)))

        # Sigmoid range check: [0.0, 1.0]
        self.assertTrue(np.all(predictions >= 0.0))
        self.assertTrue(np.all(predictions <= 1.0))

    def test_backbone_frozen_by_default(self):
        """Verify that the spatial backbone is non-trainable by default."""
        model = build_deepguard_model(self.config)
        time_dist_layer = model.get_layer("time_distributed_spatial_extractor")
        self.assertFalse(time_dist_layer.trainable)


if __name__ == "__main__":
    unittest.main()
