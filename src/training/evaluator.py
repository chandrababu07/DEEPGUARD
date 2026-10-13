"""
Evaluation pipeline and report generation for DeepGuard test partitions.

Provides reproducible post-training evaluation without test-set contamination:
- Ground truth vs predicted probabilities.
- Classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
- Machine-readable JSON summary and CSV predictions table.
- Clear warnings regarding calibration and sample size limitations.
"""

import csv
from dataclasses import asdict, dataclass, field
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.dataset.manifest import ManifestManager, ManifestRecord
from src.dataset.preprocessor import VideoPreprocessor
from src.model.inference import DeepGuardPredictor, InferenceResult
from src.model.loader import compute_file_sha256


@dataclass
class EvaluationReport:
    """Detailed post-training evaluation report."""

    timestamp: str
    checkpoint_path: str
    checkpoint_sha256: str
    test_manifest_path: str
    total_test_samples: int
    evaluated_samples: int
    failed_preprocessing_samples: int
    decision_threshold: float
    accuracy: float
    fake_precision: float
    fake_recall: float
    fake_f1: float
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    test_class_distribution: Dict[str, int]
    predictions_csv_path: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeepGuardEvaluator:
    """
    Evaluates a trained model checkpoint on an isolated test partition.
    """

    def __init__(
        self,
        checkpoint_path: str,
        preprocessor: Optional[VideoPreprocessor] = None,
        decision_threshold: float = 0.5,
    ):
        self.checkpoint_path = checkpoint_path
        self.decision_threshold = decision_threshold
        self.preprocessor = preprocessor or VideoPreprocessor()
        self.predictor = DeepGuardPredictor(
            checkpoint_path=checkpoint_path,
            preprocessor=self.preprocessor,
            decision_threshold=decision_threshold,
        )

    def evaluate(
        self,
        manifest_path: str,
        output_dir: str,
        base_dir: Optional[str] = None,
    ) -> EvaluationReport:
        """
        Evaluate the test split from a manifest file.
        """
        os.makedirs(output_dir, exist_ok=True)
        records = ManifestManager.read_manifest(manifest_path)
        test_records = [r for r in records if r.split == "test"]

        if not test_records:
            raise ValueError(f"No records with split='test' found in {manifest_path}")

        checkpoint_hash = (
            compute_file_sha256(self.checkpoint_path)
            if os.path.exists(self.checkpoint_path)
            else "unknown"
        )

        predictions_csv_path = os.path.join(output_dir, "test_predictions.csv")
        report_json_path = os.path.join(output_dir, "evaluation_report.json")

        actual_labels: List[int] = []
        pred_labels: List[int] = []
        fake_probabilities: List[float] = []
        prediction_rows: List[Dict[str, Any]] = []
        failed_count = 0

        for rec in test_records:
            full_path = rec.video_path
            if base_dir and not os.path.isabs(full_path):
                full_path = os.path.join(base_dir, full_path)

            inf_res: InferenceResult = self.predictor.predict_video(
                video_path=full_path, sample_id=rec.sample_id
            )

            if inf_res.fake_score is None:
                failed_count += 1
                row = {
                    "sample_id": rec.sample_id,
                    "video_path": rec.video_path,
                    "actual_label": rec.label,
                    "predicted_label": -1,
                    "fake_probability": -1.0,
                    "status": inf_res.preprocessing_status,
                    "error": inf_res.error or "Prediction failed",
                }
            else:
                pred_binary = 1 if inf_res.fake_score >= self.decision_threshold else 0
                actual_labels.append(rec.label)
                pred_labels.append(pred_binary)
                fake_probabilities.append(inf_res.fake_score)

                row = {
                    "sample_id": rec.sample_id,
                    "video_path": rec.video_path,
                    "actual_label": rec.label,
                    "predicted_label": pred_binary,
                    "fake_probability": round(inf_res.fake_score, 6),
                    "status": inf_res.preprocessing_status,
                    "error": "",
                }

            prediction_rows.append(row)

        # Write predictions CSV
        with open(predictions_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "video_path",
                    "actual_label",
                    "predicted_label",
                    "fake_probability",
                    "status",
                    "error",
                ],
            )
            writer.writeheader()
            for r in prediction_rows:
                writer.writerow(r)

        # Compute confusion matrix and metrics
        y_true = np.array(actual_labels)
        y_pred = np.array(pred_labels)

        if len(y_true) > 0:
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))

            total_valid = len(y_true)
            accuracy = float((tp + tn) / total_valid)
            fake_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            fake_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            fake_f1 = (
                float(2 * fake_precision * fake_recall / (fake_precision + fake_recall))
                if (fake_precision + fake_recall) > 0
                else 0.0
            )
            confusion_mat = [[tn, fp], [fn, tp]]
        else:
            tp = tn = fp = fn = 0
            accuracy = fake_precision = fake_recall = fake_f1 = 0.0
            confusion_mat = [[0, 0], [0, 0]]

        warnings_list = [
            "Model score represents an uncalibrated sigmoid output.",
        ]
        if len(y_true) < 30:
            warnings_list.append(
                f"Small test set size (N={len(y_true)}). Metrics exhibit high statistical variance."
            )
        if failed_count > 0:
            warnings_list.append(
                f"{failed_count} test samples failed preprocessing and were excluded from metric calculations."
            )

        report = EvaluationReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=checkpoint_hash,
            test_manifest_path=manifest_path,
            total_test_samples=len(test_records),
            evaluated_samples=len(y_true),
            failed_preprocessing_samples=failed_count,
            decision_threshold=self.decision_threshold,
            accuracy=round(accuracy, 4),
            fake_precision=round(fake_precision, 4),
            fake_recall=round(fake_recall, 4),
            fake_f1=round(fake_f1, 4),
            confusion_matrix=confusion_mat,
            test_class_distribution={
                "real": int(np.sum(y_true == 0)),
                "fake": int(np.sum(y_true == 1)),
            },
            predictions_csv_path=predictions_csv_path,
            warnings=warnings_list,
        )

        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report
