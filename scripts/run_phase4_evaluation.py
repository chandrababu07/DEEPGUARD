"""
CLI script to run full Phase 4 comprehensive evaluation on an isolated test split.

Outputs generated:
- Metrics report (JSON)
- Labeled Confusion Matrix (JSON)
- Video-level test predictions table (CSV)
- Misclassification and failure analysis (JSON and CSV)
- Calibration reliability bins table (CSV)
- Evaluation summary (Markdown)
- Full cryptographic provenance (Checkpoint SHA-256 and Manifest SHA-256)

Usage:
    python scripts/run_phase4_evaluation.py \
        --checkpoint experiments/mini_baseline/checkpoints/deepguard_mobilenetv2_lstm_baseline.keras \
        --manifest path/to/dataset_manifest.csv \
        --output-dir experiments/phase4_evaluation \
        --threshold 0.5 \
        --optimize-threshold
"""

import argparse
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.baseline import HAS_TF
from src.evaluation.evaluator import DeepGuardComprehensiveEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepGuard Phase 4 Comprehensive Test Evaluator & Calibration Engine"
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
        help="Path to dataset manifest CSV (must contain split='test')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/phase4_evaluation",
        help="Output directory for predictions, metrics, calibration, and reports",
    )
    parser.add_argument(
        "--data-base-dir",
        type=str,
        default=None,
        help="Optional base directory to prepend to relative video paths in manifest",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Initial binary decision threshold (default 0.5)",
    )
    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="Optimize decision threshold using validation split (never test split)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 72)
    print(" DEEPGUARD - PHASE 4 COMPREHENSIVE EVALUATION & EXPLAINABILITY")
    print("=" * 72)

    if not HAS_TF:
        print("[!] ERROR: TensorFlow is not installed in the current environment.")
        print("    Full model inference requires TensorFlow/Keras.")
        print("    Please run this script in an environment with TensorFlow (e.g., Google Colab GPU).")
        print("    Command:")
        print(f"      python scripts/run_phase4_evaluation.py --checkpoint {args.checkpoint} --manifest {args.manifest} --output-dir {args.output_dir}")
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
    print(f"[*] Base Thresh: {args.threshold}")
    print(f"[*] Opt Thresh : {'Validation split (leakage-safe)' if args.optimize_threshold else 'Disabled (fixed threshold)'}")
    print("-" * 72)

    evaluator = DeepGuardComprehensiveEvaluator(
        checkpoint_path=args.checkpoint,
        decision_threshold=args.threshold,
    )

    report = evaluator.evaluate_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        base_dir=args.data_base_dir,
        optimize_threshold_on_val=args.optimize_threshold,
    )

    print("\n[SUCCESS] Evaluation pipeline completed successfully!")
    print("-" * 72)
    print(f"Evaluated Test Samples : {report.evaluated_samples} / {report.total_test_samples}")
    print(f"Failed Preprocessing   : {report.failed_preprocessing_samples}")
    print(f"Active Threshold       : {report.decision_threshold:.4f} (Optimized: {report.threshold_optimized_on_validation})")

    m = report.metrics
    print(f"Accuracy               : {m.get('accuracy', 0.0):.4f}")
    print(f"Balanced Accuracy      : {m.get('balanced_accuracy', 0.0):.4f}")
    print(f"Fake Precision         : {m.get('fake_precision', 0.0):.4f}")
    print(f"Fake Recall            : {m.get('fake_recall', 0.0):.4f}")
    print(f"Fake F1-Score          : {m.get('fake_f1', 0.0):.4f}")

    roc = m.get('roc_auc')
    pr = m.get('pr_auc')
    roc_str = f"{roc:.4f}" if roc is not None else "N/A (single class)"
    pr_str = f"{pr:.4f}" if pr is not None else "N/A (single class)"
    print(f"ROC-AUC                : {roc_str}")
    print(f"PR-AUC                 : {pr_str}")

    cal = report.calibration
    brier = cal.get('brier_score')
    ece = cal.get('expected_calibration_error')
    brier_str = f"{brier:.4f}" if brier is not None else "N/A"
    ece_str = f"{ece:.4f}" if ece is not None else "N/A"
    print(f"Brier Score            : {brier_str}")
    print(f"ECE (10 bins)          : {ece_str}")
    print(f"Misclassifications     : {report.misclassification_counts.get('total_errors', 0)}")
    print("-" * 72)
    print("Generated Artifacts:")
    for key, path in report.artifacts_created.items():
        print(f"  - {key:<25}: {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
