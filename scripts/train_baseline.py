"""
CLI script to run DeepGuard reproducible training.

Usage in Google Colab / CLI:
    python scripts/train_baseline.py --manifest path/to/manifest.csv --output-dir experiments/run1 --epochs 15
"""

import argparse
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.baseline import DeepGuardModelConfig, HAS_TF
from src.training.trainer import DeepGuardTrainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Reproducible Training Runner")
    parser.add_argument("--manifest", type=str, required=True, help="Path to validated dataset manifest CSV")
    parser.add_argument("--output-dir", type=str, default="experiments/baseline_run", help="Output directory for checkpoints & reports")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (2 for end-to-end memory safety)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--data-base-dir", type=str, default=None, help="Optional base directory to prepend to relative video paths")
    parser.add_argument("--trainable-backbone", action="store_true", help="Set backbone to trainable (default False: frozen)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(" DEEPGUARD - PHASE 3 REPRODUCIBLE TRAINING RUNNER")
    print("=" * 70)

    if not HAS_TF:
        print("[!] ERROR: TensorFlow is not installed in the current Python environment.")
        print("    Please run this script in an environment with TensorFlow (e.g., Google Colab GPU).")
        sys.exit(1)

    model_config = DeepGuardModelConfig(
        trainable_backbone=args.trainable_backbone,
        learning_rate=args.lr,
    )

    train_config = TrainingConfig(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        random_seed=args.seed,
        model_config=model_config,
        data_base_dir=args.data_base_dir,
    )

    trainer = DeepGuardTrainer(train_config)
    print(f"Manifest path   : {args.manifest}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Random seed     : {args.seed}")
    print(f"Backbone frozen : {not args.trainable_backbone}")
    print("-" * 70)

    summary = trainer.train()
    print("\n[SUCCESS] Training completed successfully!")
    print(f"Best checkpoint saved to: {summary['artifacts']['best_checkpoint']}")
    print(f"Final checkpoint saved to: {summary['artifacts']['final_checkpoint']}")
    print(f"Report saved to: {summary['artifacts']['report_json']}")


if __name__ == "__main__":
    main()
