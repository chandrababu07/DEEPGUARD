"""
Unit tests for DeepGuard reproducible training configuration and metadata capture.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.manifest import ManifestRecord
from src.dataset.preprocessor import VideoPreprocessor
from src.training.trainer import (
    TrainingConfig,
    VideoDataGenerator,
    collect_environment_metadata,
    set_reproducible_seeds,
)


class TestTrainingPipeline(unittest.TestCase):
    """Test suite for TrainingConfig, seed setting, and data generation."""

    def test_training_config_defaults_and_dict(self):
        """Verify default training hyperparameter contracts and dictionary serialization."""
        config = TrainingConfig(
            manifest_path="path/to/manifest.csv",
            output_dir="experiments/run_test",
        )
        self.assertEqual(config.epochs, 15)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.learning_rate, 0.0001)
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(config.early_stopping_patience, 4)

        d = config.to_dict()
        self.assertEqual(d["manifest_path"], "path/to/manifest.csv")
        self.assertIn("model_config", d)
        self.assertEqual(d["model_config"]["feature_dim"], 1280)

    def test_set_reproducible_seeds(self):
        """Verify reproducible seed setting guarantees deterministic random numbers."""
        set_reproducible_seeds(1234)
        val1 = np.random.rand()
        set_reproducible_seeds(1234)
        val2 = np.random.rand()
        self.assertEqual(val1, val2)

    def test_collect_environment_metadata(self):
        """Verify environment metadata dictionary structure."""
        meta = collect_environment_metadata()
        self.assertIn("python_version", meta)
        self.assertIn("platform", meta)
        self.assertIn("numpy_version", meta)
        self.assertIn("opencv_version", meta)
        self.assertIn("gpu_available", meta)
        self.assertIn("gpu_devices", meta)


if __name__ == "__main__":
    unittest.main()
