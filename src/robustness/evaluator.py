"""
Comprehensive robustness evaluation engine for DeepGuard.

Orchestrates paired perturbation evaluations across:
- Clean baseline
- JPEG compression (Q90, Q70, Q50, Q30)
- Gaussian blur (mild, moderate, severe)
- Spatial downsampling (0.75, 0.50, 0.25)

Produces:
1. robustness_report.json
2. robustness_summary.csv
3. per_video_predictions.csv
4. probability_deltas.csv
5. robustness_summary.md

Guarantees:
- Paired comparison: Compares identical test samples under identical ground-truth labels.
- Delta metrics calculation: Delta Acc, Delta F1, Delta ROC-AUC relative to clean baseline.
- Prediction consistency tracking: Flipping rates and Mean Absolute Probability Change (MAPC).
- Full cryptographic provenance: Checkpoint SHA-256 and Manifest SHA-256.
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
from src.evaluation.metrics import evaluate_predictions, EvaluationMetrics
from src.model.inference import DeepGuardPredictor, InferenceResult
from src.model.loader import compute_file_sha256
from .perturbations import (
    apply_perturbation,
    get_standard_phase5_perturbations,
    PerturbationConfig,
)


@dataclass
class ConditionResult:
    """Evaluation metrics and perturbation metadata for a single test condition."""

    condition_name: str
    perturbation_type: str
    severity_level: str
    parameters: Dict[str, Any]
    description: str
    total_evaluated_samples: int
    metrics: Dict[str, Any]
    metric_deltas_vs_clean: Dict[str, float]
    mean_absolute_probability_change: float
    max_absolute_probability_change: float
    prediction_flip_count: int
    prediction_flip_rate: float
    delta_false_positives: int
    delta_false_negatives: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobustnessReport:
    """Comprehensive multi-perturbation robustness experiment report."""

    timestamp: str
    checkpoint_path: str
    checkpoint_sha256: str
    manifest_path: str
    manifest_sha256: str
    decision_threshold: float
    paired_test_sample_count: int
    clean_baseline_metrics: Dict[str, Any]
    conditions: List[ConditionResult]
    artifacts_created: Dict[str, str]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["conditions"] = [c.to_dict() for c in self.conditions]
        return d


class DeepGuardRobustnessEvaluator:
    """
    Executes controlled perturbation evaluations over isolated test split.
    """

    def __init__(
        self,
        checkpoint_path: str,
        decision_threshold: float = 0.5,
        preprocessor: Optional[VideoPreprocessor] = None,
        predictor: Optional[DeepGuardPredictor] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.decision_threshold = decision_threshold
        self.preprocessor = preprocessor or VideoPreprocessor()
        self.predictor = predictor or DeepGuardPredictor(
            checkpoint_path=checkpoint_path,
            preprocessor=self.preprocessor,
            decision_threshold=decision_threshold,
        )

    def evaluate_robustness(
        self,
        manifest_path: str,
        output_dir: str,
        perturbations: Optional[List[PerturbationConfig]] = None,
        base_dir: Optional[str] = None,
    ) -> RobustnessReport:
        """
        Run complete paired robustness evaluation pipeline.
        """
        os.makedirs(output_dir, exist_ok=True)
        if perturbations is None:
            perturbations = get_standard_phase5_perturbations()

        records = ManifestManager.read_manifest(manifest_path)
        test_records = [r for r in records if r.split == "test"]

        if not test_records:
            raise ValueError(f"No records with split='test' found in manifest {manifest_path}")

        manifest_hash = (
            compute_file_sha256(manifest_path) if os.path.exists(manifest_path) else "unknown"
        )
        checkpoint_hash = (
            compute_file_sha256(self.checkpoint_path)
            if os.path.exists(self.checkpoint_path)
            else "unknown"
        )

        warnings: List[str] = []
        if len(test_records) < 30:
            warnings.append(
                f"Evaluation test split has only {len(test_records)} videos. "
                "Statistical confidence intervals on robustness deltas are wide."
            )

        # 1. Preprocess all test videos into memory to guarantee strict paired comparisons
        # Dict mapping sample_id -> (clean_frames, label, video_path)
        clean_cache: Dict[str, Tuple[np.ndarray, int, str]] = {}
        for tr in test_records:
            full_path = tr.video_path
            if base_dir and not os.path.isabs(full_path):
                full_path = os.path.join(base_dir, full_path)

            prep_res = self.preprocessor.process_video(full_path)
            if prep_res.status == "SUCCESS" and prep_res.frames is not None:
                clean_cache[tr.sample_id] = (prep_res.frames, tr.label, tr.video_path)
            else:
                warnings.append(
                    f"Sample {tr.sample_id} failed preprocessing ({prep_res.status}: {prep_res.error_message}). Excluded from paired analysis."
                )

        if not clean_cache:
            raise RuntimeError("Zero test samples successfully preprocessed. Cannot run robustness evaluation.")

        paired_sample_ids = sorted(clean_cache.keys())
        n_paired = len(paired_sample_ids)
        y_true = np.array([clean_cache[sid][1] for sid in paired_sample_ids])

        # 2. Evaluate Clean Baseline
        clean_scores: List[float] = []
        for sid in paired_sample_ids:
            frames, _, _ = clean_cache[sid]
            # Predict from preprocessed frames
            inf_res = self.predictor.predict_frames(frames, sample_id=sid)
            clean_scores.append(float(inf_res.fake_score) if inf_res.fake_score is not None else 0.5)

        clean_scores_arr = np.array(clean_scores)
        clean_metrics_obj = evaluate_predictions(
            y_true=y_true,
            y_prob=clean_scores_arr,
            threshold=self.decision_threshold,
        )
        clean_metrics = clean_metrics_obj.to_dict()
        clean_preds = (clean_scores_arr >= self.decision_threshold).astype(int)

        # Store predictions per video for table exports
        # Dict: sample_id -> { "clean_score": float, "condition_scores": {cond_name: float} }
        per_video_data: Dict[str, Dict[str, Any]] = {}
        for i, sid in enumerate(paired_sample_ids):
            _, label, vpath = clean_cache[sid]
            per_video_data[sid] = {
                "sample_id": sid,
                "video_path": vpath,
                "label": label,
                "clean_score": round(float(clean_scores_arr[i]), 4),
                "clean_pred": int(clean_preds[i]),
                "condition_scores": {},
                "condition_preds": {},
            }

        # 3. Evaluate each perturbation condition
        condition_results: List[ConditionResult] = []

        for p_cfg in perturbations:
            cond_scores: List[float] = []
            for sid in paired_sample_ids:
                frames, _, _ = clean_cache[sid]
                # Apply deterministic perturbation
                perturbed_frames = apply_perturbation(frames, p_cfg)
                # Predict
                inf_res = self.predictor.predict_frames(perturbed_frames, sample_id=sid)
                s = float(inf_res.fake_score) if inf_res.fake_score is not None else 0.5
                cond_scores.append(s)

                per_video_data[sid]["condition_scores"][p_cfg.name] = round(s, 4)
                per_video_data[sid]["condition_preds"][p_cfg.name] = (
                    1 if s >= self.decision_threshold else 0
                )

            cond_scores_arr = np.array(cond_scores)
            cond_preds = (cond_scores_arr >= self.decision_threshold).astype(int)

            cond_metrics_obj = evaluate_predictions(
                y_true=y_true,
                y_prob=cond_scores_arr,
                threshold=self.decision_threshold,
            )
            cond_metrics = cond_metrics_obj.to_dict()

            # Metric deltas relative to clean baseline (cond - clean)
            deltas: Dict[str, float] = {}
            for k in [
                "accuracy",
                "balanced_accuracy",
                "fake_precision",
                "fake_recall",
                "fake_f1",
                "real_precision",
                "real_recall",
                "real_f1",
                "macro_f1",
            ]:
                if k in clean_metrics and k in cond_metrics:
                    deltas[k] = round(float(cond_metrics[k] - clean_metrics[k]), 4)

            # ROC-AUC and PR-AUC deltas (if both non-None)
            if clean_metrics.get("roc_auc") is not None and cond_metrics.get("roc_auc") is not None:
                deltas["roc_auc"] = round(float(cond_metrics["roc_auc"] - clean_metrics["roc_auc"]), 4)
            if clean_metrics.get("pr_auc") is not None and cond_metrics.get("pr_auc") is not None:
                deltas["pr_auc"] = round(float(cond_metrics["pr_auc"] - clean_metrics["pr_auc"]), 4)

            # Consistency metrics
            prob_diffs = np.abs(cond_scores_arr - clean_scores_arr)
            mapc = float(np.mean(prob_diffs))
            max_apc = float(np.max(prob_diffs)) if len(prob_diffs) > 0 else 0.0

            flips = int(np.sum(cond_preds != clean_preds))
            flip_rate = float(flips / n_paired) if n_paired > 0 else 0.0

            # Delta False Positives and False Negatives
            delta_fp = int(cond_metrics_obj.confusion_matrix.fp - clean_metrics_obj.confusion_matrix.fp)
            delta_fn = int(cond_metrics_obj.confusion_matrix.fn - clean_metrics_obj.confusion_matrix.fn)

            cond_res = ConditionResult(
                condition_name=p_cfg.name,
                perturbation_type=p_cfg.perturbation_type,
                severity_level=p_cfg.severity_level,
                parameters=p_cfg.parameters,
                description=p_cfg.description,
                total_evaluated_samples=n_paired,
                metrics=cond_metrics,
                metric_deltas_vs_clean=deltas,
                mean_absolute_probability_change=round(mapc, 4),
                max_absolute_probability_change=round(max_apc, 4),
                prediction_flip_count=flips,
                prediction_flip_rate=round(flip_rate, 4),
                delta_false_positives=delta_fp,
                delta_false_negatives=delta_fn,
            )
            condition_results.append(cond_res)

        # 4. Export Artifacts
        artifacts: Dict[str, str] = {}

        # 4.1 robustness_report.json
        report_json_path = os.path.join(output_dir, "robustness_report.json")
        report_obj = RobustnessReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=checkpoint_hash,
            manifest_path=manifest_path,
            manifest_sha256=manifest_hash,
            decision_threshold=self.decision_threshold,
            paired_test_sample_count=n_paired,
            clean_baseline_metrics=clean_metrics,
            conditions=condition_results,
            artifacts_created={},
            warnings=warnings,
        )
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_obj.to_dict(), f, indent=2)
        artifacts["robustness_report_json"] = report_json_path

        # 4.2 robustness_summary.csv
        summary_csv_path = os.path.join(output_dir, "robustness_summary.csv")
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "condition",
                "type",
                "severity",
                "accuracy",
                "delta_accuracy",
                "fake_f1",
                "delta_fake_f1",
                "roc_auc",
                "delta_roc_auc",
                "mapc",
                "prediction_flips",
                "flip_rate",
                "delta_fp",
                "delta_fn",
            ])
            # Write clean row first
            writer.writerow([
                "clean_baseline",
                "none",
                "baseline",
                clean_metrics.get("accuracy", ""),
                0.0,
                clean_metrics.get("fake_f1", ""),
                0.0,
                clean_metrics.get("roc_auc", ""),
                0.0,
                0.0,
                0,
                0.0,
                0,
                0,
            ])
            for cr in condition_results:
                writer.writerow([
                    cr.condition_name,
                    cr.perturbation_type,
                    cr.severity_level,
                    cr.metrics.get("accuracy", ""),
                    cr.metric_deltas_vs_clean.get("accuracy", 0.0),
                    cr.metrics.get("fake_f1", ""),
                    cr.metric_deltas_vs_clean.get("fake_f1", 0.0),
                    cr.metrics.get("roc_auc", ""),
                    cr.metric_deltas_vs_clean.get("roc_auc", ""),
                    cr.mean_absolute_probability_change,
                    cr.prediction_flip_count,
                    cr.prediction_flip_rate,
                    cr.delta_false_positives,
                    cr.delta_false_negatives,
                ])
        artifacts["robustness_summary_csv"] = summary_csv_path

        # 4.3 per_video_predictions.csv
        per_video_csv_path = os.path.join(output_dir, "per_video_predictions.csv")
        with open(per_video_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["sample_id", "video_path", "actual_label", "clean_score", "clean_pred"]
            for p_cfg in perturbations:
                header.extend([f"{p_cfg.name}_score", f"{p_cfg.name}_pred"])
            writer.writerow(header)

            for sid in paired_sample_ids:
                row_data = per_video_data[sid]
                row = [
                    row_data["sample_id"],
                    row_data["video_path"],
                    row_data["label"],
                    row_data["clean_score"],
                    row_data["clean_pred"],
                ]
                for p_cfg in perturbations:
                    row.extend([
                        row_data["condition_scores"].get(p_cfg.name, ""),
                        row_data["condition_preds"].get(p_cfg.name, ""),
                    ])
                writer.writerow(row)
        artifacts["per_video_predictions_csv"] = per_video_csv_path

        # 4.4 probability_deltas.csv
        prob_deltas_csv_path = os.path.join(output_dir, "probability_deltas.csv")
        with open(prob_deltas_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["sample_id", "actual_label", "clean_score"]
            for p_cfg in perturbations:
                header.append(f"{p_cfg.name}_delta")
            writer.writerow(header)

            for sid in paired_sample_ids:
                row_data = per_video_data[sid]
                clean_s = row_data["clean_score"]
                row = [sid, row_data["label"], clean_s]
                for p_cfg in perturbations:
                    cond_s = row_data["condition_scores"].get(p_cfg.name, clean_s)
                    diff = round(cond_s - clean_s, 4)
                    row.append(diff)
                writer.writerow(row)
        artifacts["probability_deltas_csv"] = prob_deltas_csv_path

        # 4.5 robustness_summary.md
        md_path = os.path.join(output_dir, "robustness_summary.md")
        self._write_markdown_summary(md_path, report_obj)
        artifacts["robustness_summary_md"] = md_path

        report_obj.artifacts_created = artifacts

        # Update JSON with populated artifact paths
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_obj.to_dict(), f, indent=2)

        return report_obj

    def _write_markdown_summary(self, filepath: str, report: RobustnessReport) -> None:
        """Writes human-readable Markdown summary of robustness analysis."""
        lines = [
            "# DEEPGUARD — PHASE 5 ROBUSTNESS EVALUATION REPORT",
            f"**Timestamp:** {report.timestamp}  ",
            f"**Model Checkpoint:** `{report.checkpoint_path}`  ",
            f"**Checkpoint SHA-256:** `{report.checkpoint_sha256}`  ",
            f"**Manifest:** `{report.manifest_path}`  ",
            f"**Manifest SHA-256:** `{report.manifest_sha256}`  ",
            f"**Paired Sample Count:** {report.paired_test_sample_count} videos  ",
            f"**Decision Threshold:** {report.decision_threshold}  ",
            "",
            "---",
            "",
            "## 1. CLEAN BASELINE BENCHMARK",
            f"- **Accuracy:** {report.clean_baseline_metrics.get('accuracy', 0.0):.4f}",
            f"- **Balanced Accuracy:** {report.clean_baseline_metrics.get('balanced_accuracy', 0.0):.4f}",
            f"- **Fake Precision:** {report.clean_baseline_metrics.get('fake_precision', 0.0):.4f}",
            f"- **Fake Recall:** {report.clean_baseline_metrics.get('fake_recall', 0.0):.4f}",
            f"- **Fake F1-Score:** {report.clean_baseline_metrics.get('fake_f1', 0.0):.4f}",
            f"- **ROC-AUC:** {report.clean_baseline_metrics.get('roc_auc', 'N/A')}",
            f"- **PR-AUC:** {report.clean_baseline_metrics.get('pr_auc', 'N/A')}",
            "",
            "---",
            "",
            "## 2. PERTURBATION SENSITIVITY SUMMARY",
            "",
            "| Condition | Type | Severity | Accuracy | Δ Acc | Fake F1 | Δ F1 | ROC-AUC | MAPC | Flips (Rate) | Δ FP | Δ FN |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        for cr in report.conditions:
            acc = cr.metrics.get("accuracy", 0.0)
            d_acc = cr.metric_deltas_vs_clean.get("accuracy", 0.0)
            f1 = cr.metrics.get("fake_f1", 0.0)
            d_f1 = cr.metric_deltas_vs_clean.get("fake_f1", 0.0)
            auc = cr.metrics.get("roc_auc", "N/A")
            mapc = cr.mean_absolute_probability_change
            flips = f"{cr.prediction_flip_count} ({cr.prediction_flip_rate * 100:.1f}%)"
            dfp = f"{cr.delta_false_positives:+d}"
            dfn = f"{cr.delta_false_negatives:+d}"

            d_acc_str = f"{d_acc:+.4f}" if d_acc != 0 else "0.0000"
            d_f1_str = f"{d_f1:+.4f}" if d_f1 != 0 else "0.0000"

            lines.append(
                f"| `{cr.condition_name}` | {cr.perturbation_type} | {cr.severity_level} | "
                f"{acc:.4f} | {d_acc_str} | {f1:.4f} | {d_f1_str} | {auc} | {mapc:.4f} | {flips} | {dfp} | {dfn} |"
            )

        if report.warnings:
            lines.extend([
                "",
                "---",
                "",
                "## 3. EXPERIMENTAL LIMITATIONS & DIAGNOSTIC WARNINGS",
            ])
            for w in report.warnings:
                lines.append(f"- [!] {w}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
