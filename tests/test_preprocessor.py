"""
Unit tests for DeepGuard video preprocessor, face cropping, resizing, and normalization.
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

from fixtures.generate_synthetic_fixture import (
    create_synthetic_face_video,
    generate_synthetic_fixture,
)
from src.dataset.face_detector import SyntheticFaceDetector
from src.dataset.preprocessor import (
    TARGET_CHANNELS,
    TARGET_RESOLUTION,
    TARGET_SEQUENCE_LENGTH,
    VideoPreprocessor,
)


class TestVideoPreprocessor(unittest.TestCase):
    """Test suite for the VideoPreprocessor baseline implementation."""

    @classmethod
    def setUpClass(cls):
        # Generate fixtures in a dedicated test fixture directory
        cls.fixture_dir = os.path.join(PROJECT_ROOT, "fixtures", "synthetic")
        cls.manifest_path = generate_synthetic_fixture(cls.fixture_dir)
        cls.preprocessor = VideoPreprocessor(face_detector=SyntheticFaceDetector())

    def test_uniform_indices_calculation(self):
        """Test equidistant frame index sampling across different video lengths."""
        # 80 frames -> 8 frames at [0, 10, 20, 30, 40, 50, 60, 70]
        indices = VideoPreprocessor.calculate_uniform_indices(total_frames=80, num_samples=8)
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices, [0, 10, 20, 30, 40, 50, 60, 70])

        # Exactly 8 frames -> [0, 1, 2, 3, 4, 5, 6, 7]
        indices_exact = VideoPreprocessor.calculate_uniform_indices(total_frames=8, num_samples=8)
        self.assertEqual(indices_exact, [0, 1, 2, 3, 4, 5, 6, 7])

        # Fewer than 8 frames returns empty list
        indices_too_few = VideoPreprocessor.calculate_uniform_indices(total_frames=6, num_samples=8)
        self.assertEqual(indices_too_few, [])

    def test_bbox_margin_expansion(self):
        """Test +20% margin expansion around face bounding box."""
        frame_shape = (200, 200)
        # Bbox: x=50, y=50, w=100, h=100. 20% margin is 20px.
        # Expected: x1=30, y1=30, x2=170, y2=170
        x1, y1, x2, y2 = self.preprocessor._apply_margin((50, 50, 100, 100), frame_shape)
        self.assertEqual((x1, y1, x2, y2), (30, 30, 170, 170))

        # Test boundary clamping at border
        x1_edge, y1_edge, x2_edge, y2_edge = self.preprocessor._apply_margin((5, 5, 50, 50), frame_shape)
        self.assertEqual(x1_edge, 0)  # max(0, 5 - 10) = 0
        self.assertEqual(y1_edge, 0)

    def test_process_valid_synthetic_video(self):
        """Test processing of a valid synthetic video matching paper specifications."""
        video_path = os.path.join(self.fixture_dir, "synth_real_01.mp4")
        result = self.preprocessor.process_video(video_path, sample_id="TEST_001")

        self.assertTrue(result.is_success)
        self.assertEqual(result.status, "SUCCESS")
        self.assertIsNotNone(result.frames)

        frames = result.frames
        # Verify tensor shape: (T=8, H=128, W=128, C=3)
        self.assertEqual(
            frames.shape,
            (TARGET_SEQUENCE_LENGTH, TARGET_RESOLUTION[1], TARGET_RESOLUTION[0], TARGET_CHANNELS),
        )
        # Verify tensor dtype is float32
        self.assertEqual(frames.dtype, np.float32)

        # Verify pixel intensity normalization to [0.0, 1.0] (Equation 1: x / 255.0)
        self.assertGreaterEqual(float(np.min(frames)), 0.0)
        self.assertLessEqual(float(np.max(frames)), 1.0)

    def test_process_corrupted_video_fails_explicitly(self):
        """Test that corrupted/unreadable video fails explicitly and is not marked success."""
        video_path = os.path.join(self.fixture_dir, "synth_corrupt.mp4")
        result = self.preprocessor.process_video(video_path, sample_id="CORRUPT_001")

        self.assertFalse(result.is_success)
        self.assertEqual(result.status, "FAILED_UNREADABLE")
        self.assertIsNone(result.frames)
        self.assertIsNotNone(result.error_message)

    def test_process_nonexistent_video_fails_explicitly(self):
        """Test that non-existent video path is flagged explicitly."""
        result = self.preprocessor.process_video("does_not_exist.mp4", sample_id="MISSING_001")
        self.assertFalse(result.is_success)
        self.assertEqual(result.status, "FAILED_UNREADABLE")
        self.assertIn("does not exist", result.error_message)

    def test_process_insufficient_frames_video_fails_explicitly(self):
        """Test that video with fewer than 8 frames fails explicitly."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create a 4-frame video (< 8 required frames)
            create_synthetic_face_video(tmp_path, num_frames=4, draw_face=True)
            result = self.preprocessor.process_video(tmp_path, sample_id="SHORT_001")

            self.assertFalse(result.is_success)
            self.assertEqual(result.status, "FAILED_INSUFFICIENT_FRAMES")
            self.assertIsNone(result.frames)
            self.assertIn("minimum required is 8", result.error_message)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_process_no_face_video_fails_explicitly(self):
        """Test that video without detectable face fails explicitly instead of silent pass."""
        video_path = os.path.join(self.fixture_dir, "synth_noface.mp4")
        result = self.preprocessor.process_video(video_path, sample_id="NOFACE_001")

        self.assertFalse(result.is_success)
        self.assertEqual(result.status, "FAILED_NO_FACE_DETECTED")
        self.assertIsNone(result.frames)
        self.assertIsNotNone(result.error_message)


if __name__ == "__main__":
    unittest.main()
