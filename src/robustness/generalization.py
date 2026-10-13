"""
Cross-dataset generalization evaluation and domain-shift analysis for DeepGuard.

Evaluates an in-dataset trained model (e.g., FaceForensics++) against an external,
unseen dataset (e.g., Celeb-DF v2, DFDC, or WildDeepfake) under strict leakage safeguards:
1. Zero threshold tuning on external target data (threshold is strictly frozen from source validation).
2. Label normalization and validation: Real = 0, Fake = 1.
3. Quantifies domain gap via generalization metric drops (Delta Accuracy, Delta F1, Delta AUC).
4. Explicit reporting when target dataset access is pending or files are unavailable.
"""

from dataclasses import asdict, dataclass, field
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.dataset.manifest import ManifestManager, ManifestRecord
from src.evaluation.metrics import evaluate_predictions, EvaluationMetrics
from src.model.inference import DeepGuardPredictor, InferenceResult
from src.model.loader import compute_file_sha256


@dataclass
class CrossDatasetManifestSummary:
    """Summary of external target dataset manifest validation."""

    dataset_name: str
    manifest_path: str
    manifest_sha256: str
    total_records: int
    existing_video_files: int
    missing_video_files: int
    real_count: int
    fake_count: int
    is_ready_for_evaluation: bool
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneralizationComparisonReport:
    """Quantitative comparison between source in-dataset and target cross-dataset performance."""

    timestamp: str
    checkpoint_path: str
    checkpoint_sha256: str
    source_dataset_name: str
    target_dataset_name: str
    decision_threshold: float
    threshold_source: str  # e.g., "Frozen from source validation split"
    source_metrics: Dict[str, Any]
    target_metrics: Dict[str, Any]
    metric_deltas: Dict[str, float]  # target - source
    sample_counts: Dict[str, int]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrossDatasetEvaluator:
    """
    Evaluator for measuring generalization across domain shifts without target test leakage.
    """

    def __init__(
        self,
        checkpoint_path: str,
        predictor: Optional[DeepGuardPredictor] = None,
        frozen_decision_threshold: float = 0.5,
    ):
        self.checkpoint_path = checkpoint_path
        self.decision_threshold = frozen_decision_threshold
        self.predictor = predictor or DeepGuardPredictor(
            checkpoint_path=checkpoint_path,
            decision_threshold=frozen_decision_threshold,
        )

    @staticmethod
    def audit_target_manifest(
        manifest_path: str,
        base_dir: Optional[str] = None,
    ) -> CrossDatasetManifestSummary:
        """
        Inspect and validate an external dataset manifest before attempting evaluation.
        """
        if not os.path.exists(manifest_path):
            return CrossDatasetManifestSummary(
                dataset_name="Unknown",
                manifest_path=manifest_path,
                manifest_sha256="not_found",
                total_records=0,
                existing_video_files=0,
                missing_video_files=0,
                real_count=0,
                fake_count=0,
                is_ready_for_evaluation=False,
                issues=[f"Manifest file not found: {manifest_path}"],
            )

        manifest_sha256 = compute_file_sha256(manifest_path)
        records = ManifestManager.read_manifest(manifest_path)
        dataset_names = set(r.dataset_name for r in records)
        dataset_name = list(dataset_names)[0] if dataset_names else "Unknown"

        existing = 0
        missing = 0
        real_c = 0
        fake_c = 0
        issues = []

        for r in records:
            if r.label == 0:
                real_c += 1
            elif r.label == 1:
                fake_c += 1
            else:
                issues.append(f"Invalid label {r.label} for sample {r.sample_id}")

            vpath = r.video_path
            if base_dir and not os.path.isabs(vpath):
                vpath = os.path.join(base_dir, vpath)

            if os.path.exists(vpath):
                existing += 1
            else:
                missing += 1

        if missing > 0:
            issues.append(
                f"{missing} of {len(records)} video files referenced in manifest do not exist on disk."
            )

        if real_c == 0 or fake_c == 0:
            issues.append(
                f"Target dataset is missing one class (real={real_c}, fake={fake_c}). Both classes required for full evaluation."
            )

        is_ready = len(issues) == 0 and existing > 0

        return CrossDatasetManifestSummary(
            dataset_name=dataset_name,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            total_records=len(records),
            existing_video_files=existing,
            missing_video_files=missing,
            real_count=real_c,
            fake_count=fake_c,
            is_ready_for_evaluation=is_ready,
            issues=issues,
        )

    def evaluate_generalization_against_source(
        self,
        source_metrics: Dict[str, Any],
        target_manifest_path: str,
        output_dir: str,
        source_dataset_name: str = "FaceForensics++",
        base_dir: Optional[str] = None,
    ) -> GeneralizationComparisonReport:
        """
        Evaluate model on target dataset and compare with source benchmark metrics.

        Raises:
            FileNotFoundError if target manifest or required video files are missing.
        """
        audit = self.audit_target_manifest(target_manifest_path, base_dir=base_dir)
        if not audit.is_ready_for_evaluation:
            raise FileNotFoundError(
                f"Target dataset manifest {target_manifest_path} is not ready for evaluation: "
                + "; ".join(audit.issues)
            )

        os.makedirs(output_dir, exist_ok=True)
        records = ManifestManager.read_manifest(target_manifest_path)

        target_true: List[int] = []
        target_probs: List[float] = []
        target_predictions_records: List[Dict[str, Any]] = []

        for r in records:
            vpath = r.video_path
            if base_dir and not os.path.isabs(vpath):
                vpath = os.path.join(base_dir, vpath)

            res: InferenceResult = self.predictor.predict_video(vpath, sample_id=r.sample_id)
            if res.fake_score is not None:
                target_true.append(r.label)
                target_probs.append(res.fake_score)
                target_predictions_records.append({
                    "sample_id": r.sample_id,
                    "video_path": r.video_path,
                    "actual_label": r.label,
                    "predicted_label": 1 if res.fake_score >= self.decision_threshold else 0,
                    "fake_score": res.fake_score,
                })

        target_metrics_obj: EvaluationMetrics = evaluate_predictions(
            y_true=np.array(target_true),
            y_prob=np.array(target_probs),
            threshold=self.decision_threshold,
        )
        target_metrics_dict = target_metrics_obj.to_dict()

        # Compute metric deltas (target - source)
        deltas = {}
        for m_key in ["accuracy", "balanced_accuracy", "fake_precision", "fake_recall", "fake_f1", "macro_f1"]:
            if m_key in source_metrics and m_key in target_metrics_dict:
                deltas[m_key] = round(float(target_metrics_dict[m_key] - source_metrics[m_key]), 4)

        if source_metrics.get("roc_auc") is not None and target_metrics_dict.get("roc_auc") is not None:
            deltas["roc_auc"] = round(float(target_metrics_dict["roc_auc"] - source_metrics["roc_auc"]), 4)

        report = GeneralizationComparisonReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=compute_file_sha256(self.checkpoint_path) if os.path.exists(self.checkpoint_path) else "unknown",
            source_dataset_name=source_dataset_name,
            target_dataset_name=audit.dataset_name,
            decision_threshold=self.decision_threshold,
            threshold_source="Strictly frozen from source validation split (zero target leakage)",
            source_metrics=source_metrics,
            target_metrics=target_metrics_dict,
            metric_deltas=deltas,
            sample_counts={
                "source_samples": source_metrics.get("total_samples", 0),
                "target_samples": len(target_true),
            },
            warnings=audit.issues,
        )

        # Save report JSON
        out_json = os.path.join(output_dir, "generalization_comparison_report.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report
