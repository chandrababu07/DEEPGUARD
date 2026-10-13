"""
Comprehensive evaluation orchestrator for DeepGuard.

Generates all Phase 4 artifacts:
- Metrics report (JSON)
- Labeled Confusion Matrix (JSON)
- Video-level test predictions table (CSV)
- Misclassification and failure analysis report (JSON and CSV)
- Calibration reliability bins table (CSV)
- Human-readable evaluation summary (Markdown)
- Full cryptographic provenance tracking (Checkpoint SHA-256 and Manifest SHA-256)
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
from .calibration import evaluate_calibration, optimize_decision_threshold, PlattCalibrator
from .metrics import evaluate_predictions, EvaluationMetrics


@dataclass
class MisclassificationRecord:
    """Record of an incorrect video-level prediction."""

    sample_id: str
    video_path: str
    actual_label: int  # 0 = Real, 1 = Fake
    predicted_label: int
    fake_probability: float
    error_type: str  # "False Positive" or "False Negative"
    severity_category: str  # "High Confidence Error" or "Marginal Error"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComprehensiveEvaluationReport:
    """Complete evaluation results and artifact catalog."""

    timestamp: str
    checkpoint_path: str
    checkpoint_sha256: str
    manifest_path: str
    manifest_sha256: str
    decision_threshold: float
    threshold_optimized_on_validation: bool
    total_test_samples: int
    evaluated_samples: int
    failed_preprocessing_samples: int
    metrics: Dict[str, Any]
    calibration: Dict[str, Any]
    misclassification_counts: Dict[str, int]
    artifacts_created: Dict[str, str]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeepGuardComprehensiveEvaluator:
    """
    Executes full Phase 4 evaluation pipeline over an isolated test partition.
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

    def evaluate_manifest(
        self,
        manifest_path: str,
        output_dir: str,
        base_dir: Optional[str] = None,
        optimize_threshold_on_val: bool = False,
    ) -> ComprehensiveEvaluationReport:
        """
        Run complete evaluation pipeline.

        Args:
            manifest_path: Path to dataset manifest CSV.
            output_dir: Output folder for generated evaluation artifacts.
            base_dir: Optional base path to prepend to relative video paths.
            optimize_threshold_on_val: If True, uses the validation split to select the optimal threshold.
        """
        os.makedirs(output_dir, exist_ok=True)
        records = ManifestManager.read_manifest(manifest_path)

        manifest_hash = (
            compute_file_sha256(manifest_path) if os.path.exists(manifest_path) else "unknown"
        )
        checkpoint_hash = (
            compute_file_sha256(self.checkpoint_path)
            if os.path.exists(self.checkpoint_path)
            else "unknown"
        )

        test_records = [r for r in records if r.split == "test"]
        val_records = [r for r in records if r.split == "validation"]

        if not test_records:
            raise ValueError(f"No records with split='test' found in manifest {manifest_path}")

        # Optional threshold optimization strictly on validation partition
        active_threshold = self.decision_threshold
        threshold_optimized = False
        val_opt_history: List[Dict[str, float]] = []

        if optimize_threshold_on_val and len(val_records) >= 10:
            val_true: List[int] = []
            val_prob: List[float] = []
            for vr in val_records:
                full_v_path = vr.video_path
                if base_dir and not os.path.isabs(full_v_path):
                    full_v_path = os.path.join(base_dir, full_v_path)
                res = self.predictor.predict_video(full_v_path, sample_id=vr.sample_id)
                if res.fake_score is not None:
                    val_true.append(vr.label)
                    val_prob.append(res.fake_score)

            if len(val_true) >= 10:
                best_t, _, val_opt_history = optimize_decision_threshold(
                    np.array(val_true), np.array(val_prob), metric="f1"
                )
                active_threshold = best_t
                threshold_optimized = True

        # Process test partition
        test_predictions_list: List[Dict[str, Any]] = []
        actual_labels: List[int] = []
        fake_probabilities: List[float] = []
        failed_count = 0

        for tr in test_records:
            full_t_path = tr.video_path
            if base_dir and not os.path.isabs(full_t_path):
                full_t_path = os.path.join(base_dir, full_t_path)

            inf_res: InferenceResult = self.predictor.predict_video(
                video_path=full_t_path, sample_id=tr.sample_id
            )

            if inf_res.fake_score is None:
                failed_count += 1
                row = {
                    "sample_id": tr.sample_id,
                    "video_path": tr.video_path,
                    "actual_label": tr.label,
                    "predicted_label": -1,
                    "fake_probability": -1.0,
                    "is_correct": False,
                    "status": inf_res.preprocessing_status,
                    "error": inf_res.error or "Failed preprocessing",
                }
            else:
                pred_binary = 1 if inf_res.fake_score >= active_threshold else 0
                is_correct = bool(pred_binary == tr.label)

                actual_labels.append(tr.label)
                fake_probabilities.append(inf_res.fake_score)

                row = {
                    "sample_id": tr.sample_id,
                    "video_path": tr.video_path,
                    "actual_label": tr.label,
                    "predicted_label": pred_binary,
                    "fake_probability": round(inf_res.fake_score, 6),
                    "is_correct": is_correct,
                    "status": inf_res.preprocessing_status,
                    "error": "",
                }

            test_predictions_list.append(row)

        y_true = np.array(actual_labels, dtype=int)
        y_prob = np.array(fake_probabilities, dtype=float)

        # Artifact file paths
        predictions_csv = os.path.join(output_dir, "test_predictions.csv")
        metrics_json = os.path.join(output_dir, "metrics_report.json")
        confusion_json = os.path.join(output_dir, "confusion_matrix.json")
        misclass_csv = os.path.join(output_dir, "misclassifications.csv")
        misclass_json = os.path.join(output_dir, "misclassifications_report.json")
        calibration_csv = os.path.join(output_dir, "calibration_bins.csv")
        summary_md = os.path.join(output_dir, "evaluation_summary.md")

        # 1. Save test_predictions.csv
        with open(predictions_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "sample_id",
                "video_path",
                "actual_label",
                "predicted_label",
                "fake_probability",
                "is_correct",
                "status",
                "error",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in test_predictions_list:
                writer.writerow(r)

        # 2. Evaluate metrics
        if len(y_true) > 0:
            eval_metrics: EvaluationMetrics = evaluate_predictions(
                y_true=y_true, y_prob=y_prob, threshold=active_threshold
            )
            cal_report = evaluate_calibration(
                y_true=y_true, y_prob=y_prob, is_calibrated=False
            )
        else:
            raise ValueError("All test video samples failed preprocessing; 0 predictions available.")

        # 3. Misclassification and Failure Analysis
        misclassifications: List[MisclassificationRecord] = []
        for r in test_predictions_list:
            if r["predicted_label"] != -1 and not r["is_correct"]:
                prob = r["fake_probability"]
                if r["actual_label"] == 0 and r["predicted_label"] == 1:
                    err_type = "False Positive"
                    severity = "High Confidence Error" if prob >= 0.8 else "Marginal Error"
                else:
                    err_type = "False Negative"
                    severity = "High Confidence Error" if prob <= 0.2 else "Marginal Error"

                misclassifications.append(
                    MisclassificationRecord(
                        sample_id=r["sample_id"],
                        video_path=r["video_path"],
                        actual_label=r["actual_label"],
                        predicted_label=r["predicted_label"],
                        fake_probability=prob,
                        error_type=err_type,
                        severity_category=severity,
                    )
                )

        # Save misclassifications CSV & JSON
        with open(misclass_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "video_path",
                    "actual_label",
                    "predicted_label",
                    "fake_probability",
                    "error_type",
                    "severity_category",
                ],
            )
            writer.writeheader()
            for m in misclassifications:
                writer.writerow(m.to_dict())

        with open(misclass_json, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in misclassifications], f, indent=2)

        # 4. Save calibration_bins.csv
        with open(calibration_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "bin_index",
                    "bin_lower",
                    "bin_upper",
                    "sample_count",
                    "mean_confidence",
                    "empirical_accuracy",
                    "calibration_gap",
                ],
            )
            writer.writeheader()
            for b in cal_report.bins:
                writer.writerow(b.to_dict())

        # 5. Save metrics_report.json and confusion_matrix.json
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(eval_metrics.to_dict(), f, indent=2)

        with open(confusion_json, "w", encoding="utf-8") as f:
            json.dump(eval_metrics.confusion_matrix.to_dict(), f, indent=2)

        # 6. Save human-readable evaluation summary Markdown
        summary_content = [
            "# DEEPGUARD EVALUATION SUMMARY",
            f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
            f"**Checkpoint:** `{self.checkpoint_path}` (SHA256: `{checkpoint_hash[:12]}...`)  ",
            f"**Manifest:** `{manifest_path}` (SHA256: `{manifest_hash[:12]}...`)  ",
            f"**Decision Threshold:** `{active_threshold}` {'(Optimized on Validation)' if threshold_optimized else '(Default)'}  ",
            "",
            "## 1. Test Split Performance",
            "| Metric | Score | Note |",
            "| :--- | :--- | :--- |",
            f"| **Accuracy** | {eval_metrics.accuracy * 100:.2f}% | Overall correct video predictions |",
            f"| **Balanced Accuracy** | {eval_metrics.balanced_accuracy * 100:.2f}% | Mean of per-class sensitivities |",
            f"| **Fake Precision** | {eval_metrics.fake_precision * 100:.2f}% | True fakes out of flagged fakes |",
            f"| **Fake Recall (Sensitivity)** | {eval_metrics.fake_recall * 100:.2f}% | Detected fakes out of actual fakes |",
            f"| **Fake F1-Score** | {eval_metrics.fake_f1:.4f} | Harmonic mean |",
            f"| **Real Specificity** | {eval_metrics.real_recall * 100:.2f}% | Correctly identified authentic videos |",
            f"| **ROC-AUC** | {eval_metrics.roc_auc if eval_metrics.roc_auc is not None else 'N/A'} | Area under ROC curve |",
            f"| **PR-AUC** | {eval_metrics.pr_auc if eval_metrics.pr_auc is not None else 'N/A'} | Area under PR curve |",
            f"| **Brier Score** | {cal_report.brier_score:.4f} | Lower is better (0.0 = perfect) |",
            f"| **Expected Calibration Error (ECE)** | {cal_report.expected_calibration_error:.4f} | Mean calibration divergence |",
            "",
            "## 2. Confusion Matrix",
            f"* True Positives (Fake correctly identified) : {eval_metrics.confusion_matrix.tp}",
            f"* True Negatives (Real correctly verified)   : {eval_metrics.confusion_matrix.tn}",
            f"* False Positives (Real falsely flagged)     : {eval_metrics.confusion_matrix.fp}",
            f"* False Negatives (Fake missed as Real)      : {eval_metrics.confusion_matrix.fn}",
            "",
            "## 3. Failure & Misclassification Analysis",
            f"* Total Misclassifications: {len(misclassifications)} / {len(y_true)} videos",
            f"* False Positives (False Alarms): {eval_metrics.confusion_matrix.fp}",
            f"* False Negatives (Missed Manipulations): {eval_metrics.confusion_matrix.fn}",
            "",
            "## 4. Warnings & Methodological Limitations",
        ]
        for w in eval_metrics.warnings + cal_report.warnings:
            summary_content.append(f"* [!] {w}")

        with open(summary_md, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_content) + "\n")

        all_warnings = list(set(eval_metrics.warnings + cal_report.warnings))

        report_obj = ComprehensiveEvaluationReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=checkpoint_hash,
            manifest_path=manifest_path,
            manifest_sha256=manifest_hash,
            decision_threshold=active_threshold,
            threshold_optimized_on_validation=threshold_optimized,
            total_test_samples=len(test_records),
            evaluated_samples=len(y_true),
            failed_preprocessing_samples=failed_count,
            metrics=eval_metrics.to_dict(),
            calibration=cal_report.to_dict(),
            misclassification_counts={
                "total_errors": len(misclassifications),
                "false_positives": eval_metrics.confusion_matrix.fp,
                "false_negatives": eval_metrics.confusion_matrix.fn,
            },
            artifacts_created={
                "predictions_csv": predictions_csv,
                "metrics_json": metrics_json,
                "confusion_json": confusion_json,
                "misclassifications_csv": misclass_csv,
                "misclassifications_json": misclass_json,
                "calibration_csv": calibration_csv,
                "summary_md": summary_md,
            },
            warnings=all_warnings,
        )

        return report_obj
