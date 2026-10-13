# DEEPGUARD — PHASE 4: EVALUATION, CONFIDENCE CALIBRATION, AND EXPLAINABILITY
**Project:** DeepGuard — Spatial-Temporal Deepfake Video Detection
**Module:** Phase 4 Evaluation and Explainability Framework
**Status:** Completed & Verified

---

## 1. EXECUTIVE OVERVIEW

Phase 4 establishes the formal scientific evaluation, reliability assessment, and spatial attribution infrastructure for the DeepGuard deepfake video detection system.

The implementation enforces rigorous methodological principles:
1. **Video-Level Unit of Analysis:** Predictions and metrics are evaluated strictly per video sequence (8 temporal frames per video), preventing frame correlation from inflating statistical significance.
2. **Standard Forensic Labeling:** Positive class ($1$) denotes Manipulated / Deepfake video; Negative class ($0$) denotes Authentic / Real video.
3. **Leakage-Safe Decision Thresholds:** Optimal decision threshold tuning is performed exclusively on validation data and held fixed for the test split.
4. **Epistemic Humility in Calibration:** Raw sigmoid outputs are explicitly distinguished from calibrated posterior probabilities. Calibration models (Platt scaling) enforce strict minimum sample-size guards ($N \ge 20$).
5. **Non-Causal Spatial Attribution:** Grad-CAM saliency heatmaps visualize model feature correlations without claiming causal proof or forensic certainty.

---

## 2. EVALUATION METRICS ENGINE

Located at `src/evaluation/metrics.py`.

### 2.1 Supported Metrics
* **Confusion Matrix:** Standard 2×2 table tracking True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP). Format: `[[TN, FP], [FN, TP]]`.
* **Accuracy:** Overall proportion of correct video-level classifications.
* **Balanced Accuracy:** Unweighted average of Class 0 recall (specificity) and Class 1 recall (sensitivity):
  $$\text{Balanced Accuracy} = \frac{1}{2} \left( \frac{\text{TN}}{\text{TN} + \text{FP}} + \frac{\text{TP}}{\text{TP} + \text{FN}} \right)$$
* **Per-Class Metrics:**
  * **Fake (Class 1):** Precision, Recall, and $F_1$-score.
  * **Real (Class 0):** Precision, Recall (Specificity), and $F_1$-score.
* **Macro $F_1$:** Arithmetic mean of Real $F_1$ and Fake $F_1$.
* **ROC-AUC:** Area under Receiver Operating Characteristic curve computed via exact Mann-Whitney $U$ rank-sum formulation.
* **PR-AUC:** Area under Precision-Recall curve computed via numerical trapezoidal integration.

### 2.2 Edge Cases and Safeguards
* **Single-Class Datasets:** When the test partition contains only one class (e.g. only Real videos), ROC-AUC and PR-AUC are mathematically undefined. The evaluator safely outputs `None` with an explicit warning rather than raising unhandled exceptions or returning misleading values.
* **Zero-Division Handling:** When no positive predictions occur, precision and recall safely default to `0.0`.
* **Empty Input Rejection:** Evaluation calls on empty arrays immediately raise `ValueError`.

---

## 3. CONFIDENCE CALIBRATION & THRESHOLD OPTIMIZATION

Located at `src/evaluation/calibration.py`.

### 3.1 Score Taxonomy
To prevent overconfidence and misleading legal claims, DeepGuard distinguishes three levels of scores:
1. **Model Score:** Raw sigmoid activation $z \in [0.0, 1.0]$ output by the dense head.
2. **Probability Estimate:** Uncalibrated model score used as a heuristic likelihood.
3. **Calibrated Probability:** Monotonically transformed score empirically aligned with true error rates.

### 3.2 Calibration Metrics
* **Brier Score:** Mean squared error between predicted probabilities and binary ground truth:
  $$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
  Range: $[0.0, 1.0]$, where $0.0$ represents perfect calibration.
* **Expected Calibration Error (ECE):** Weighted average of absolute difference between empirical accuracy and mean confidence across $M$ equal-width bins (default $M=10$):
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
* **Reliability Diagram Data:** Exports sample counts, bin boundaries, empirical accuracies, and mean confidences for plotting calibration curves.

### 3.3 Leakage-Safe Threshold Optimization
* The function `optimize_decision_threshold(y_val, p_val, metric="f1")` searches the threshold space $t \in [0.05, 0.95]$ strictly over **validation split** predictions.
* The test split is **never** used for threshold selection, preventing optimistic performance bias.

### 3.4 Platt Scaling & Sample-Size Guard
* `PlattCalibrator` fits a univariate logistic regression $P_{\text{cal}} = \sigma(a \cdot \text{logit}(p) + b)$.
* **Safeguard:** Fitting requires at least 20 independent validation samples with at least 5 examples of both classes. If sample size is insufficient, the fitter refuses to overfit, falls back to uncalibrated scores, and logs a clear explanatory warning.

---

## 4. EXPLAINABILITY VIA GRAD-CAM

Located at `src/explainability/gradcam.py`.

### 4.1 Target Layer Selection
* **Architecture:** MobileNetV2 + LSTM.
* **Target Convolutional Layer:** `Conv_1` (or `out_relu`), the final $1{,}280$-filter pointwise convolution of the MobileNetV2 spatial backbone before Global Average Pooling.
* Spatial resolution at this layer is $4 \times 4$ for $128 \times 128$ input frames, preserving rich high-level spatial semantic features.

### 4.2 Pipeline Contract
1. **Input Tensor Validation:** Accepts video frame tensors of shape `(8, 128, 128, 3)` with pixel values normalized to $[0.0, 1.0]$.
2. **Gradient Extraction:** Utilizes `tf.GradientTape()` to calculate gradients of the video prediction score with respect to feature maps of the target layer for selected frames.
3. **Heatmap Generation:** Computes channel-wise pooled gradients, performs weighted linear combination, applies ReLU to discard negative contributions, and normalizes to $[0.0, 1.0]$.
4. **Overlay Generation:** Resizes heatmap to $128 \times 128$, applies OpenCV JET colormap, and blends with the original RGB video frame ($\alpha = 0.5$).
5. **Catalog Index:** Records peak activation $(x, y)$ coordinates, mean activation intensity, predicted score, output file path, and explanation disclaimer in `GradCAMIndex`.

### 4.3 Mandatory Epistemic Disclaimer
All generated explanation artifacts carry the mandatory disclaimer:
> *"Preliminary development model. Grad-CAM saliency highlights spatial feature activation correlations, not legal or forensic proof of video manipulation."*

---

## 5. REPRODUCIBLE ARTIFACT SPECIFICATION

The evaluation runner `scripts/run_phase4_evaluation.py` generates standard reproducible artifacts:

| Artifact Name | Format | Content Description |
|---|---|---|
| `metrics_report.json` | JSON | Accuracy, Balanced Accuracy, Precision, Recall, $F_1$, ROC-AUC, PR-AUC, Confusion Matrix, provenance hashes |
| `confusion_matrix.json` | JSON | Labeled TN, FP, FN, TP counts and 2×2 matrix list |
| `test_predictions.csv` | CSV | Video-level table: sample ID, video path, true label, predicted label, fake score, status |
| `misclassifications.csv` | CSV | Table of incorrect test predictions, error category, and severity |
| `misclassifications_report.json` | JSON | Structured failure analysis report with False Positive and False Negative summaries |
| `calibration_bins.csv` | CSV | Reliability diagram table with bin edges, counts, empirical accuracy, and mean confidence |
| `evaluation_summary.md` | Markdown | Executive summary suitable for academic project reports |

### Provenance Tracking
Every report embeds:
* `checkpoint_sha256`: Cryptographic hash of the evaluated `.keras` model weights.
* `manifest_sha256`: Cryptographic hash of the dataset manifest CSV.
* `timestamp`: ISO-8601 evaluation timestamp.

---

## 6. CLI AND COLAB EXECUTION GUIDE

### 6.1 Running Comprehensive Evaluation in Google Colab
```bash
# 1. Ensure GPU environment with TensorFlow
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__, '| GPU:', tf.config.list_physical_devices('GPU'))"

# 2. Run Comprehensive Phase 4 Evaluation
python scripts/run_phase4_evaluation.py \
    --checkpoint experiments/mini_baseline/checkpoints/deepguard_mobilenetv2_lstm_baseline.keras \
    --manifest experiments/mini_baseline/mini_manifest.csv \
    --output-dir experiments/phase4_evaluation \
    --threshold 0.5 \
    --optimize-threshold
```

### 6.2 Running Local Unit Tests (Pure Python / NumPy / OpenCV)
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
# or
pytest tests/ -v
```
*(TensorFlow-dependent model training/gradient tests are automatically skipped locally when TensorFlow is not installed.)*
