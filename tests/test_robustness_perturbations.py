"""
Unit tests for DeepGuard controlled perturbations module.

Tests cover:
- JPEG compression: Q90, Q70, Q50, Q30 and invalid quality parameter guards
- Gaussian blur: kernel sizes, sigmas, and invalid parameter guards
- Downsampling: scale factors 0.75, 0.50, 0.25 and invalid scale guards
- Deterministic behavior: identical inputs produce bitwise-consistent outputs
- Clean-baseline preservation: input arrays are never mutated in-place
- Input shape and normalization validation: (8, 128, 128, 3) in [0.0, 1.0]
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.robustness.perturbations import (
    apply_downsampling,
    apply_gaussian_blur,
    apply_jpeg_compression,
    apply_perturbation,
    get_standard_phase5_perturbations,
    validate_frame_tensor,
    PerturbationConfig,
)


class TestRobustnessPerturbations(unittest.TestCase):
    """Test suite for image and frame perturbation functions."""

    def setUp(self):
        # Deterministic synthetic test sequence: 8 frames of 128x128 RGB in [0.0, 1.0]
        np.random.seed(42)
        self.sample_frames = np.random.uniform(0.0, 1.0, size=(8, 128, 128, 3)).astype(np.float32)

    def test_tensor_validation_accepts_valid_4d(self):
        """Verify (8, 128, 128, 3) tensor passes validation."""
        validated = validate_frame_tensor(self.sample_frames)
        self.assertEqual(validated.shape, (8, 128, 128, 3))
        self.assertEqual(validated.dtype, np.float32)

    def test_tensor_validation_accepts_valid_5d(self):
        """Verify (1, 8, 128, 128, 3) batch tensor is squeezed to 4D."""
        batch_5d = np.expand_dims(self.sample_frames, axis=0)
        validated = validate_frame_tensor(batch_5d)
        self.assertEqual(validated.shape, (8, 128, 128, 3))

    def test_tensor_validation_rejects_invalid_shapes(self):
        """Verify non-conforming shapes raise ValueError."""
        invalid_shapes = [
            (128, 128, 3),        # 3D
            (10, 128, 128, 3),       # 10 frames instead of 8
            (8, 64, 64, 3),          # 64x64 instead of 128x128
            (8, 128, 128, 1),        # Grayscale
        ]
        for shape in invalid_shapes:
            dummy = np.zeros(shape, dtype=np.float32)
            with self.assertRaises(ValueError):
                validate_frame_tensor(dummy)

    def test_tensor_validation_rejects_nan_and_inf(self):
        """Verify NaN and Inf values are rejected."""
        corrupted = self.sample_frames.copy()
        corrupted[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_frame_tensor(corrupted)

        corrupted[0, 0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            validate_frame_tensor(corrupted)

    def test_clean_baseline_preservation_no_inplace_mutation(self):
        """Verify input arrays are not mutated in-place by perturbations."""
        original_copy = self.sample_frames.copy()

        _ = apply_jpeg_compression(self.sample_frames, quality=50)
        np.testing.assert_array_equal(self.sample_frames, original_copy)

        _ = apply_gaussian_blur(self.sample_frames, kernel_size=5, sigma=2.0)
        np.testing.assert_array_equal(self.sample_frames, original_copy)

        _ = apply_downsampling(self.sample_frames, scale_factor=0.5)
        np.testing.assert_array_equal(self.sample_frames, original_copy)

    def test_jpeg_compression_valid_qualities(self):
        """Verify JPEG compression across Q90, Q70, Q50, and Q30."""
        for q in [90, 70, 50, 30]:
            perturbed = apply_jpeg_compression(self.sample_frames, quality=q)
            self.assertEqual(perturbed.shape, (8, 128, 128, 3))
            self.assertEqual(perturbed.dtype, np.float32)
            self.assertTrue(np.all(perturbed >= 0.0) and np.all(perturbed <= 1.0))
            # Verify distortion is actually introduced
            self.assertFalse(np.array_equal(perturbed, self.sample_frames))

    def test_jpeg_compression_invalid_qualities_raise_value_error(self):
        """Verify out-of-range quality values raise ValueError."""
        for invalid_q in [0, -10, 101, 200]:
            with self.assertRaises(ValueError):
                apply_jpeg_compression(self.sample_frames, quality=invalid_q)

    def test_gaussian_blur_valid_parameters(self):
        """Verify Gaussian blur with mild, moderate, and severe parameters."""
        test_params = [(3, 1.0), (5, 2.0), (7, 3.0)]
        for k, s in test_params:
            perturbed = apply_gaussian_blur(self.sample_frames, kernel_size=k, sigma=s)
            self.assertEqual(perturbed.shape, (8, 128, 128, 3))
            self.assertTrue(np.all(perturbed >= 0.0) and np.all(perturbed <= 1.0))
            self.assertFalse(np.array_equal(perturbed, self.sample_frames))

    def test_gaussian_blur_invalid_parameters_raise_value_error(self):
        """Verify even kernel sizes or non-positive sigmas raise ValueError."""
        # Even kernel sizes
        for k in [2, 4, 6]:
            with self.assertRaises(ValueError):
                apply_gaussian_blur(self.sample_frames, kernel_size=k, sigma=1.0)

        # Invalid sigmas
        for s in [0.0, -1.5]:
            with self.assertRaises(ValueError):
                apply_gaussian_blur(self.sample_frames, kernel_size=3, sigma=s)

    def test_downsampling_valid_scales(self):
        """Verify downsampling at scale 0.75, 0.50, and 0.25 restores shape (8, 128, 128, 3)."""
        for scale in [0.75, 0.50, 0.25]:
            perturbed = apply_downsampling(self.sample_frames, scale_factor=scale)
            self.assertEqual(perturbed.shape, (8, 128, 128, 3))
            self.assertTrue(np.all(perturbed >= 0.0) and np.all(perturbed <= 1.0))
            self.assertFalse(np.array_equal(perturbed, self.sample_frames))

    def test_downsampling_invalid_scales_raise_value_error(self):
        """Verify scale factors <= 0 or >= 1 raise ValueError."""
        for invalid_scale in [0.0, -0.2, 1.0, 1.5]:
            with self.assertRaises(ValueError):
                apply_downsampling(self.sample_frames, scale_factor=invalid_scale)

    def test_deterministic_reproducibility(self):
        """Verify identical inputs produce identical outputs across calls."""
        res1 = apply_jpeg_compression(self.sample_frames, quality=70)
        res2 = apply_jpeg_compression(self.sample_frames, quality=70)
        np.testing.assert_array_equal(res1, res2)

        blur1 = apply_gaussian_blur(self.sample_frames, kernel_size=5, sigma=2.0)
        blur2 = apply_gaussian_blur(self.sample_frames, kernel_size=5, sigma=2.0)
        np.testing.assert_array_equal(blur1, blur2)

    def test_standard_perturbation_registry(self):
        """Verify standard suite contains all Phase 5 required conditions."""
        suite = get_standard_phase5_perturbations()
        self.assertEqual(len(suite), 10)  # 4 JPEG + 3 Blur + 3 Downsample
        names = [c.name for c in suite]

        expected_names = [
            "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
            "blur_mild", "blur_moderate", "blur_severe",
            "downsample_s75", "downsample_s50", "downsample_s25",
        ]
        for exp in expected_names:
            self.assertIn(exp, names)

        # Test apply_perturbation dispatcher for each config
        for cfg in suite:
            out = apply_perturbation(self.sample_frames, cfg)
            self.assertEqual(out.shape, (8, 128, 128, 3))


if __name__ == "__main__":
    unittest.main()
