# DEEPGUARD — PHASE 5: ROBUSTNESS TESTING, PERTURBATION SENSITIVITY, AND CROSS-DATASET GENERALIZATION
**Project:** DeepGuard — Spatial-Temporal Deepfake Video Detection
**Module:** Phase 5 Robustness and Generalization Framework
**Status:** Completed & Verified

---

## 1. EXECUTIVE OVERVIEW

Deepfake detection models frequently suffer severe performance drops when deployed outside ideal benchmark conditions. Phase 5 provides a formal, reproducible methodology to evaluate model resilience against:
1. **Realistic Transmission & Compression Degradations:** JPEG compression artifacts simulating lossy social media re-encoding.
2. **Optical & Acquisition Distortions:** Gaussian optical defocus and motion blur.
3. **Bandwidth & Resolution Constraints:** Spatial downsampling followed by resolution recovery.
4. **Domain Shift & Cross-Dataset Generalization:** Evaluation of models trained on one domain (e.g., FaceForensics++) against completely unseen external distributions (e.g., Celeb-DF v2).

---

## 2. CONTROLLED PERTURBATION LIBRARY

Located at `src/robustness/perturbations.py`.

All perturbations operate on video frame sequences of shape `(8, 128, 128, 3)` normalized to $[0.0, 1.0]$ float32 in RGB order. Pure functional transformations ensure zero in-place mutation of the baseline input array.

### 2.1 JPEG Compression Suite
Simulates lossy spatial discrete cosine transform (DCT) quantization artifacts:
* **`jpeg_q90` (Mild):** Quality factor $90$. Preserves fine high-frequency facial textures.
* **`jpeg_q70` (Moderate):** Quality factor $70$. Standard web/social media compression level.
* **`jpeg_q50` (Moderate):** Quality factor $50$. Apparent $8 \times 8$ pixel grid blocking and ringing around sharp facial boundaries.
* **`jpeg_q30` (Severe):** Quality factor $30$. Heavy quantization suppressing high-frequency biological and spatial manipulation cues.

*Methodological Note:* Perturbations are applied at the preprocessed frame level. While full video transcoding (e.g., H.264/H.265 CRF re-encoding) evaluates temporal container artifacts, frame-level JPEG benchmarking isolates pure spatial artifact sensitivity under identical uniform temporal sampling.

### 2.2 Gaussian Blur Suite
Simulates optical defocus, low-quality camera sensors, or video denoising filters:
* **`blur_mild` (Mild):** Kernel size $3 \times 3$, $\sigma = 1.0$.
* **`blur_moderate` (Moderate):** Kernel size $5 \times 5$, $\sigma = 2.0$.
* **`blur_severe` (Severe):** Kernel size $7 \times 7$, $\sigma = 3.0$. Attenuates fine facial boundary details.

### 2.3 Spatial Downsampling Suite
Simulates low-resolution transmission or scaled video playback:
* **`downsample_s75` (Mild):** Rescaled to $75\%$ resolution ($96 \times 96$) via `cv2.INTER_AREA`, then upsampled back to $128 \times 128$ via `cv2.INTER_LINEAR`.
* **`downsample_s50` (Moderate):** Rescaled to $50\%$ resolution ($64 \times 64$) and restored.
* **`downsample_s25` (Severe):** Rescaled to $25\%$ resolution ($32 \times 32$) and restored.

---

## 3. PAIRED ROBUSTNESS EVALUATION PIPELINE

Located at `src/robustness/evaluator.py`.

### 3.1 Strict Paired Sample Matching
To prevent statistical bias, all perturbation conditions are evaluated on the exact same subset of successfully preprocessed test videos. If a video fails decoding or face detection, it is omitted from both the clean baseline and all perturbation evaluations, guaranteeing strictly paired comparisons:
$$N_{\text{clean}} = N_{\text{jpeg\_q90}} = N_{\text{jpeg\_q70}} = \dots = N_{\text{downsample\_s25}}$$

### 3.2 Quantitative Metric Deltas
Performance degradation is quantified relative to the unperturbed clean baseline:
$$\Delta\text{Accuracy} = \text{Accuracy}_{\text{perturbed}} - \text{Accuracy}_{\text{clean}}$$
$$\Delta F_1 = F1_{\text{perturbed}} - F1_{\text{clean}}$$
$$\Delta\text{ROC-AUC} = \text{ROC-AUC}_{\text{perturbed}} - \text{ROC-AUC}_{\text{clean}}$$

### 3.3 Prediction Consistency Analysis
* **Prediction Flipping:** Counts videos whose binary classification changes:
  $$\text{Flip Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_{i, \text{perturbed}} \ne \hat{y}_{i, \text{clean}})$$
* **Mean Absolute Probability Change (MAPC):** Quantifies shift in predicted manipulation likelihood:
  $$\text{MAPC} = \frac{1}{N} \sum_{i=1}^N |p_{i, \text{perturbed}} - p_{i, \text{clean}}|$$
* **Error Shift Breakdown:** Tracks changes in False Positives ($\Delta\text{FP}$) and False Negatives ($\Delta\text{FN}$).

---

## 4. CROSS-DATASET GENERALIZATION WORKFLOW

Located at `src/robustness/generalization.py`.

### 4.1 Strict Leakage Safeguards
* **Zero Target Tuning:** The decision threshold $\tau$ is frozen strictly from the source dataset validation split (default $\tau = 0.5$ or validation-optimized threshold). No threshold adjustment or parameter adaptation is permitted on the external target dataset.
* **Label Normalization:** Enforces standard binary forensic labeling across disparate dataset naming conventions (Real $= 0$, Fake $= 1$).

### 4.2 Generalization Gap Formulation
$$\Delta_{\text{generalization}} = \text{Metric}_{\text{target}} - \text{Metric}_{\text{source}}$$
A large negative delta indicates reliance on source-dataset specific artifacts (e.g., blending boundaries unique to FaceForensics++ manipulations) rather than generalizable deepfake signatures.

---

## 5. ARTIFACT SPECIFICATION

The evaluation runner `scripts/run_phase5_robustness.py` generates the following reproducible artifacts:

| Artifact Name | Format | Content Description |
|---|---|---|
| `robustness_report.json` | JSON | Complete multi-condition metrics, parameters, deltas, consistency, and cryptographic hashes |
| `robustness_summary.csv` | CSV | Tabular summary of all conditions, accuracies, F1 scores, MAPC, and flip rates |
| `per_video_predictions.csv` | CSV | Video-level prediction scores and binary decisions across all conditions |
| `probability_deltas.csv` | CSV | Per-video probability differences $(p_{\text{perturbed}} - p_{\text{clean}})$ |
| `robustness_summary.md` | Markdown | Formatted academic evaluation report table |
| `generalization_comparison_report.json` | JSON | Cross-dataset benchmark comparison and domain shift gap analysis |

---

## 6. CLI AND COLAB EXECUTION GUIDE

### 6.1 Running Robustness Pipeline on Google Colab GPU
```bash
# Run comprehensive robustness evaluation
python scripts/run_phase5_robustness.py \
    --checkpoint experiments/mini_baseline/checkpoints/deepguard_mobilenetv2_lstm_baseline.keras \
    --manifest experiments/mini_baseline/mini_manifest.csv \
    --output-dir experiments/phase5_robustness \
    --threshold 0.5
```

### 6.2 Running Optional Cross-Dataset Evaluation
```bash
python scripts/run_phase5_robustness.py \
    --checkpoint experiments/mini_baseline/checkpoints/deepguard_mobilenetv2_lstm_baseline.keras \
    --manifest experiments/mini_baseline/mini_manifest.csv \
    --cross-manifest path/to/celebdf_manifest.csv \
    --output-dir experiments/phase5_robustness
```

### 6.3 Running Full Unit Test Suite Locally
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
# or
pytest tests/ -v
```
*(92 total tests: 88 passed, 4 skipped due to missing local TensorFlow.)*
