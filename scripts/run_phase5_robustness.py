"""
CLI script to run comprehensive Phase 5 robustness and cross-dataset generalization evaluation.

Evaluates:
1. Clean baseline benchmark.
2. Controlled JPEG compression (Q90, Q70, Q50, Q30).
3. Optical Gaussian blur (mild, moderate, severe).
4. Spatial downsampling (0.75, 0.50, 0.25).
5. Optional cross-dataset domain shift evaluation (e.g., Celeb-DF v2).

Outputs generated:
- robustness_report.json
- robustness_summary.csv
- per_video_predictions.csv
- probability_deltas.csv
- robustness_summary.md

Usage:
    python scripts/run_phase5_robustness.py \
        --checkpoint experiments/mini_baseline/checkpoints/deepguard_mobilenetv2_lstm_baseline.keras \
        --manifest experiments/mini_baseline/mini_manifest.csv \
        --output-dir experiments/phase5_robustness \
        --threshold 0.5
"""

import argparse
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.baseline import HAS_TF
from src.robustness.evaluator import DeepGuardRobustnessEvaluator
from src.robustness.generalization import CrossDatasetEvaluator
from src.robustness.perturbations import get_standard_phase5_perturbations


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepGuard Phase 5 Robustness & Cross-Dataset Generalization Engine"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained .keras model checkpoint",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to in-dataset test manifest CSV (must contain split='test')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/phase5_robustness",
        help="Output directory for robustness evaluation artifacts",
    )
    parser.add_argument(
        "--data-base-dir",
        type=str,
        default=None,
        help="Optional base directory to prepend to relative video paths",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary classification (default 0.5)",
    )
    parser.add_argument(
        "--cross-manifest",
        type=str,
        default=None,
        help="Optional path to external cross-dataset manifest CSV (e.g. Celeb-DF)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 76)
    print(" DEEPGUARD - PHASE 5 ROBUSTNESS & CROSS-DATASET GENERALIZATION")
    print("=" * 76)

    if not HAS_TF:
        print("[!] ERROR: TensorFlow is not installed in the current environment.")
        print("    Full model inference and robustness testing requires TensorFlow/Keras.")
        print("    Please run this script in an environment with TensorFlow (e.g., Google Colab GPU).")
        print("    Command:")
        print(f"      python scripts/run_phase5_robustness.py --checkpoint {args.checkpoint} --manifest {args.manifest} --output-dir {args.output_dir}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"[!] ERROR: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    if not os.path.exists(args.manifest):
        print(f"[!] ERROR: Manifest file not found: {args.manifest}")
        sys.exit(1)

    print(f"[*] Checkpoint : {args.checkpoint}")
    print(f"[*] Manifest   : {args.manifest}")
    print(f"[*] Output Dir : {args.output_dir}")
    print(f"[*] Threshold  : {args.threshold}")
    print("-" * 76)

    evaluator = DeepGuardRobustnessEvaluator(
        checkpoint_path=args.checkpoint,
        decision_threshold=args.threshold,
    )

    standard_perturbations = get_standard_phase5_perturbations()
    print(f"[*] Executing {len(standard_perturbations)} perturbation conditions on paired test videos...")

    report = evaluator.evaluate_robustness(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        perturbations=standard_perturbations,
        base_dir=args.data_base_dir,
    )

    print("\n[SUCCESS] Robustness evaluation completed successfully!")
    print("-" * 76)
    print(f"Paired Test Videos Evaluated : {report.paired_test_sample_count}")
    m = report.clean_baseline_metrics
    print(f"Clean Baseline Accuracy     : {m.get('accuracy', 0.0):.4f}")
    print(f"Clean Baseline Fake F1      : {m.get('fake_f1', 0.0):.4f}")
    print(f"Clean Baseline ROC-AUC      : {m.get('roc_auc', 'N/A')}")
    print("-" * 76)
    print(f"{'Condition':<18} | {'Type':<12} | {'Acc':<6} | {'Δ Acc':<7} | {'Fake F1':<7} | {'Δ F1':<7} | {'MAPC':<6} | {'Flips':<6}")
    print("-" * 76)
    for cr in report.conditions:
        acc = cr.metrics.get('accuracy', 0.0)
        d_acc = cr.metric_deltas_vs_clean.get('accuracy', 0.0)
        f1 = cr.metrics.get('fake_f1', 0.0)
        d_f1 = cr.metric_deltas_vs_clean.get('fake_f1', 0.0)
        mapc = cr.mean_absolute_probability_change
        flips = cr.prediction_flip_count
        print(f"{cr.condition_name:<18} | {cr.perturbation_type:<12} | {acc:.4f} | {d_acc:+.4f} | {f1:.4f} | {d_f1:+.4f} | {mapc:.4f} | {flips:<6}")

    print("-" * 76)
    print("Generated Artifacts:")
    for key, path in report.artifacts_created.items():
        print(f"  - {key:<25}: {path}")

    # Optional Cross-Dataset Evaluation
    if args.cross_manifest:
        print("\n" + "=" * 76)
        print(" CROSS-DATASET GENERALIZATION EVALUATION")
        print("=" * 76)
        print(f"[*] Target Manifest: {args.cross_manifest}")
        audit = CrossDatasetEvaluator.audit_target_manifest(args.cross_manifest, base_dir=args.data_base_dir)
        print(f"[*] Target Records : {audit.total_records} (Existing videos: {audit.existing_video_files}, Missing: {audit.missing_video_files})")
        if audit.is_ready_for_evaluation:
            cross_eval = CrossDatasetEvaluator(
                checkpoint_path=args.checkpoint,
                frozen_decision_threshold=args.threshold,
            )
            cross_rep = cross_eval.evaluate_generalization_against_source(
                source_metrics=report.clean_baseline_metrics,
                target_manifest_path=args.cross_manifest,
                output_dir=os.path.join(args.output_dir, "cross_dataset"),
                base_dir=args.data_base_dir,
            )
            print("[SUCCESS] Cross-dataset evaluation completed!")
            print(f"Target Accuracy : {cross_rep.target_metrics.get('accuracy', 0.0):.4f} (Δ vs Source: {cross_rep.metric_deltas.get('accuracy', 0.0):+.4f})")
            print(f"Target Fake F1  : {cross_rep.target_metrics.get('fake_f1', 0.0):.4f} (Δ vs Source: {cross_rep.metric_deltas.get('fake_f1', 0.0):+.4f})")
        else:
            print(f"[!] Cross-dataset evaluation skipped: {'; '.join(audit.issues)}")

    print("=" * 76)


if __name__ == "__main__":
    main()
