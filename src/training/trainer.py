"""
Reproducible Training Pipeline for DeepGuard.

Specifications & Phase 3 Requirements:
1. Deterministic seed management (Python, NumPy, TensorFlow).
2. Manifest hash verification and provenance tracking.
3. Strict split isolation (train only for fitting, val only for tuning, test untouched).
4. GPU hardware logging and environment metadata capture.
5. Dual checkpoint persistence: best validation checkpoint and final epoch checkpoint.
6. Machine-readable experiment report (JSON) and training history (CSV/JSON).
"""

import csv
from dataclasses import asdict, dataclass, field
import json
import os
import platform
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.dataset.manifest import ManifestManager, ManifestRecord
from src.dataset.preprocessor import VideoPreprocessor
from src.model.baseline import DeepGuardModelConfig, build_deepguard_model, HAS_TF
from src.model.loader import compute_file_sha256

if HAS_TF:
    import tensorflow as tf
else:
    tf = None


def set_reproducible_seeds(seed: int = 42) -> None:
    """Set global deterministic random seeds across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_TF:
        tf.random.set_seed(seed)


def collect_environment_metadata() -> Dict[str, Any]:
    """Capture software library versions and compute hardware specs."""
    gpu_devices = []
    if HAS_TF:
        try:
            physical_gpus = tf.config.list_physical_devices("GPU")
            gpu_devices = [d.name for d in physical_gpus]
        except Exception:
            gpu_devices = ["Error querying physical devices"]

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__ if HAS_TF else "Not installed",
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "gpu_available": len(gpu_devices) > 0,
        "gpu_devices": gpu_devices,
    }


@dataclass
class TrainingConfig:
    """Configuration for reproducible DeepGuard model training."""

    manifest_path: str
    output_dir: str
    experiment_name: str = "deepguard_mobilenetv2_lstm"
    epochs: int = 15
    batch_size: int = 2
    learning_rate: float = 0.0001
    random_seed: int = 42
    early_stopping_patience: int = 4
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    model_config: DeepGuardModelConfig = field(default_factory=DeepGuardModelConfig)
    data_base_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        d = asdict(self)
        d["model_config"] = asdict(self.model_config)
        return d


class VideoDataGenerator:
    """
    Stream-safe video batch generator for training without loading entire datasets into RAM.
    """

    def __init__(
        self,
        records: List[ManifestRecord],
        preprocessor: VideoPreprocessor,
        batch_size: int = 2,
        base_dir: Optional[str] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.records = list(records)
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.base_dir = base_dir
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return max(1, len(self.records) // self.batch_size)

    def generate(self):
        """Infinite generator suitable for tf.data or model.fit."""
        indices = list(range(len(self.records)))
        while True:
            if self.shuffle:
                self.rng.shuffle(indices)

            batch_x: List[np.ndarray] = []
            batch_y: List[int] = []

            for idx in indices:
                rec = self.records[idx]
                video_full_path = rec.video_path
                if self.base_dir and not os.path.isabs(video_full_path):
                    video_full_path = os.path.join(self.base_dir, video_full_path)

                res = self.preprocessor.process_video(video_full_path, sample_id=rec.sample_id)
                if res.is_success and res.frames is not None:
                    batch_x.append(res.frames)
                    batch_y.append(rec.label)

                if len(batch_x) == self.batch_size:
                    yield np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)
                    batch_x = []
                    batch_y = []

            # Handle remaining incomplete batch if any
            if batch_x:
                yield np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)


class DeepGuardTrainer:
    """
    End-to-end reproducible training orchestrator.
    """

    def __init__(self, config: TrainingConfig, preprocessor: Optional[VideoPreprocessor] = None):
        self.config = config
        self.preprocessor = preprocessor or VideoPreprocessor()

    def train(self) -> Dict[str, Any]:
        """
        Execute training according to the configuration.
        """
        if not HAS_TF:
            raise ImportError(
                "TensorFlow is required to train the DeepGuard model. "
                "Execute in Google Colab or an environment with TensorFlow installed."
            )

        os.makedirs(self.config.output_dir, exist_ok=True)
        set_reproducible_seeds(self.config.random_seed)

        manifest_hash = compute_file_sha256(self.config.manifest_path)
        records = ManifestManager.read_manifest(self.config.manifest_path)

        # Validate manifest and check leakage safety
        report = ManifestManager.validate_manifest(records)
        if not report.is_valid:
            raise ValueError(
                f"Cannot train on invalid manifest:\n{report.summary_text()}"
            )
        if report.leakage_violations:
            raise ValueError(
                f"Leakage violations detected in manifest before training:\n"
                + "\n".join(report.leakage_violations)
            )

        # Split records
        train_records = [r for r in records if r.split == "train"]
        val_records = [r for r in records if r.split == "validation"]
        test_records = [r for r in records if r.split == "test"]

        if not train_records:
            raise ValueError("Training split contains 0 records!")
        if not val_records:
            raise ValueError("Validation split contains 0 records!")

        # Initialize Model
        model = build_deepguard_model(self.config.model_config)

        # Checkpoint destinations
        best_model_path = os.path.join(self.config.output_dir, "best_model.keras")
        final_model_path = os.path.join(self.config.output_dir, "final_model.keras")
        history_csv_path = os.path.join(self.config.output_dir, "training_history.csv")
        report_json_path = os.path.join(self.config.output_dir, "experiment_report.json")

        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_model_path,
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(history_csv_path, separator=",", append=False),
        ]

        # Prepare generators
        train_gen = VideoDataGenerator(
            records=train_records,
            preprocessor=self.preprocessor,
            batch_size=self.config.batch_size,
            base_dir=self.config.data_base_dir,
            shuffle=True,
            seed=self.config.random_seed,
        )
        val_gen = VideoDataGenerator(
            records=val_records,
            preprocessor=self.preprocessor,
            batch_size=self.config.batch_size,
            base_dir=self.config.data_base_dir,
            shuffle=False,
            seed=self.config.random_seed,
        )

        steps_per_epoch = max(1, len(train_records) // self.config.batch_size)
        val_steps = max(1, len(val_records) // self.config.batch_size)

        start_time = time.time()
        history = model.fit(
            train_gen.generate(),
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen.generate(),
            validation_steps=val_steps,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1,
        )
        training_time = time.time() - start_time

        # Save final model separately
        model.save(final_model_path)

        # Best model hash and final model hash
        best_model_hash = compute_file_sha256(best_model_path) if os.path.exists(best_model_path) else "not_saved"
        final_model_hash = compute_file_sha256(final_model_path) if os.path.exists(final_model_path) else "not_saved"

        # Assemble comprehensive experiment report
        hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        final_val_loss = hist_dict.get("val_loss", [None])[-1]
        final_val_acc = hist_dict.get("val_accuracy", [None])[-1]
        best_val_loss = min(hist_dict.get("val_loss", [float("inf")])) if "val_loss" in hist_dict else None

        experiment_summary = {
            "status": "COMPLETED",
            "experiment_name": self.config.experiment_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "training_duration_seconds": round(training_time, 2),
            "manifest_path": self.config.manifest_path,
            "manifest_sha256": manifest_hash,
            "random_seed": self.config.random_seed,
            "split_counts": {
                "train": len(train_records),
                "validation": len(val_records),
                "test": len(test_records),
            },
            "environment": collect_environment_metadata(),
            "model_architecture": {
                "backbone": "MobileNetV2 (pretrained ImageNet, GAP=1280)",
                "backbone_trainable": self.config.model_config.trainable_backbone,
                "recurrent": f"LSTM({self.config.model_config.lstm_units} units)",
                "dropout": self.config.model_config.dropout_rate,
                "total_parameters": model.count_params(),
            },
            "training_hyperparameters": {
                "epochs_configured": self.config.epochs,
                "epochs_completed": len(hist_dict.get("loss", [])),
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "optimizer": "Adam",
                "loss": "binary_crossentropy",
            },
            "metrics": {
                "final_train_loss": hist_dict.get("loss", [None])[-1],
                "final_train_accuracy": hist_dict.get("accuracy", [None])[-1],
                "final_val_loss": final_val_loss,
                "final_val_accuracy": final_val_acc,
                "best_val_loss": best_val_loss,
            },
            "artifacts": {
                "best_checkpoint": best_model_path,
                "best_checkpoint_sha256": best_model_hash,
                "final_checkpoint": final_model_path,
                "final_checkpoint_sha256": final_model_hash,
                "history_csv": history_csv_path,
                "report_json": report_json_path,
            },
            "notes": [
                "Test split remained completely isolated throughout training.",
                "Model score represents an uncalibrated sigmoid output.",
            ],
        }

        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(experiment_summary, f, indent=2)

        return experiment_summary
