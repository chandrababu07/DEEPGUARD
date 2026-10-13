# DEEPGUARD — PHASE 3 REPRODUCIBILITY GUIDE
**Project:** DeepGuard — Spatial-Temporal Deepfake Video Detection
**Environment:** Google Colab (Tesla T4 GPU) or Local Python Environment
**Version:** Phase 3 Reproducible Pipeline

---

## 1. OVERVIEW

This guide explains how to reproduce the video preprocessing, leakage-safe dataset splitting, baseline model training, and post-training evaluation of the **DeepGuard MobileNetV2 + LSTM** framework.

The pipeline is designed to run seamlessly from a fresh Google Colab GPU runtime while maintaining strict isolation between training, validation, and test splits.

---

## 2. GOOGLE COLAB SETUP

### 2.1 Enable GPU Acceleration
1. In Google Colab, navigate to **Runtime > Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Click **Save**.

### 2.2 Clone the Repository
In a notebook cell:
```bash
!git clone https://github.com/chandrababu07/DEEPGUARD.git
%cd DEEPGUARD
```

### 2.3 Verify Dependencies
Google Colab comes with TensorFlow and OpenCV pre-installed. Verify the environment:
```bash
!python -c "import tensorflow as tf; import cv2; print('TensorFlow:', tf.__version__, '| OpenCV:', cv2.__version__, '| GPUs:', tf.config.list_physical_devices('GPU'))"
```

---

## 3. DATASET MANIFEST PREPARATION & LEAKAGE-SAFE SPLITTING

### 3.1 Dataset Manifest Structure
DeepGuard manifests are standard CSV files with the following schema:
* `sample_id`: Unique video identifier (e.g. `FF_REAL_001`).
* `video_path`: Relative or absolute path to the `.mp4` video file.
* `label`: `0` for Authentic/Real, `1` for Manipulated/Deepfake.
* `dataset_name`: Name of dataset (e.g. `FaceForensics++`, `Celeb-DF-v2`).
* `split`: Partition assignment (`train`, `validation`, `test`).
* `manipulation_type`: E.g. `original`, `Deepfakes`, `Face2Face`, `FaceSwap`.
* `original_video_id`: Source video identifier (e.g. `001`).
* `preprocessing_status`: Processing status (`PENDING`, `SUCCESS`, etc.).
* `subject_id` *(optional)*: Actor/subject identity for identity-disjoint splitting.
* `provenance` *(optional)*: Source citation or URL.
* `error_detail` *(optional)*: Detailed error if preprocessing failed.

### 3.2 Generating Leakage-Safe Splits
To ensure that authentic videos, their manipulated counterparts, and identical actors never cross between train, validation, and test sets:

```python
from src.dataset.manifest import ManifestManager

# 1. Load raw manifest
records = ManifestManager.read_manifest("path/to/raw_manifest.csv")

# 2. Partition using Disjoint-Set connected components
split_records = ManifestManager.create_leakage_safe_splits(
    records,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    prioritize_identity=True
)

# 3. Verify that zero leakage exists
violations = ManifestManager.verify_leakage_safety(split_records)
if violations:
    print("[!] Leakage detected:", violations)
else:
    print("[PASS] Splits are completely disjoint and leakage-safe!")

# 4. Save partitioned manifest
ManifestManager.write_manifest(split_records, "experiments/dataset_manifest_split.csv")
```

---

## 4. REPRODUCIBLE MODEL TRAINING

DeepGuard provides a standalone training runner that sets deterministic seeds, records environment and manifest checksums, trains strictly on the `train` split, tunes on `validation`, and keeps `test` completely untouched.

### 4.1 Running Training
```bash
!python scripts/train_baseline.py \
    --manifest experiments/dataset_manifest_split.csv \
    --output-dir experiments/phase3_training_run \
    --epochs 15 \
    --batch-size 2 \
    --lr 0.0001 \
    --seed 42
```

### 4.2 Training Output Artifacts
The training script generates the following artifacts in `--output-dir`:
* `best_model.keras`: Checkpoint with lowest validation loss.
* `final_model.keras`: Checkpoint saved at the final training epoch.
* `training_history.csv`: Per-epoch training and validation loss/accuracy.
* `experiment_report.json`: Machine-readable metadata capturing:
  * Deterministic random seed
  * Manifest SHA256 checksum
  * Train/Val/Test sample counts
  * GPU hardware specifications
  * Software versions (Python, TensorFlow, NumPy, OpenCV)
  * Final loss and accuracy metrics

---

## 5. TEST SPLIT EVALUATION

After training completes, evaluate the saved `best_model.keras` on the untouched `test` split:

```bash
!python scripts/evaluate_baseline.py \
    --checkpoint experiments/phase3_training_run/best_model.keras \
    --manifest experiments/dataset_manifest_split.csv \
    --output-dir experiments/phase3_training_run/evaluation \
    --threshold 0.5
```

### 5.1 Evaluation Output Artifacts
* `test_predictions.csv`: Tabular predictions tracking:
  * `sample_id`
  * `actual_label`
  * `predicted_label`
  * `fake_probability`
* `evaluation_report.json`: Summary containing:
  * Evaluated sample counts
  * Accuracy, Fake Precision, Fake Recall, Fake F1-Score
  * 2x2 Confusion Matrix (`[[TN, FP], [FN, TP]]`)
  * Calibration warnings

---

## 6. STANDALONE PYTHON INFERENCE (FUTURE API / FASTAPI READY)

To run inference outside of notebooks, use the reusable `DeepGuardPredictor` interface:

```python
from src.model.inference import DeepGuardPredictor

# Initialize predictor with trained checkpoint
predictor = DeepGuardPredictor(
    checkpoint_path="experiments/phase3_training_run/best_model.keras",
    decision_threshold=0.5
)

# Predict single video
result = predictor.predict_video("path/to/query_video.mp4")

print(f"Predicted Class : {result.predicted_label}")
print(f"Fake Probability: {result.fake_score:.4f}")
print(f"Real Probability: {result.real_score:.4f}")
print(f"Warnings        : {result.warnings}")
```

---

## 7. RUNNING LOCAL UNIT TESTS

To verify all components locally without TensorFlow:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
All 35 local contract, splitting, preprocessing, and inference tests will pass, with 4 TensorFlow forward-pass tests skipped until run in Google Colab.
