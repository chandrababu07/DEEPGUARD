"""
CLI script to evaluate a DeepGuard model checkpoint on an isolated test split.

Usage in Google Colab / CLI:
    python scripts/evaluate_baseline.py --checkpoint experiments/run1/best_model.keras --manifest path/to/manifest.csv --output-dir experiments/run1/evaluation
"""

import argparse
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.baseline import HAS_TF
from src.training.evaluator import DeepGuardEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Test Split Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .keras model checkpoint")
    parser.add_argument("--manifest", type=str, required=True, help="Path to dataset manifest CSV (must contain split='test')")
    parser.add_argument("--output-dir", type=str, default="experiments/evaluation", help="Output directory for predictions & evaluation report")
    parser.add_argument("--data-base-dir", type=str, default=None, help="Optional base directory to prepend to relative video paths")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary decision threshold (default 0.5)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(" DEEPGUARD - PHASE 3 TEST EVALUATION RUNNER")
    print("=" * 70)

    if not HAS_TF:
        print("[!] ERROR: TensorFlow is not installed in the current Python environment.")
        print("    Please run this script in an environment with TensorFlow (e.g., Google Colab GPU).")
        sys.exit(1)

    evaluator = DeepGuardEvaluator(
        checkpoint_path=args.checkpoint,
        decision_threshold=args.threshold,
    )

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Manifest   : {args.manifest}")
    print(f"Output dir : {args.output_dir}")
    print(f"Threshold  : {args.threshold}")
    print("-" * 70)

    report = evaluator.evaluate(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        base_dir=args.data_base_dir,
    )

    print("\n[SUCCESS] Test evaluation completed!")
    print(f"Test Samples Evaluated: {report.evaluated_samples} / {report.total_test_samples}")
    print(f"Accuracy              : {report.accuracy:.4f}")
    print(f"Fake Precision        : {report.fake_precision:.4f}")
    print(f"Fake Recall           : {report.fake_recall:.4f}")
    print(f"Fake F1-Score         : {report.fake_f1:.4f}")
    print(f"Confusion Matrix      : {report.confusion_matrix}  [[TN, FP], [FN, TP]]")
    print(f"Predictions CSV       : {report.predictions_csv_path}")


if __name__ == "__main__":
    main()
