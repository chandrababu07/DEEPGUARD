# DEEPGUARD — PHASE 1 SPECIFICATION AND RESEARCH REPORT
**Project Title:** DeepGuard — An Explainable and Robust Spatial–Temporal Framework for Deepfake Video Detection  
**Level:** Academic Undergraduate AI/ML Research Project  
**Author / Student Developer:** CSE STUDENT
**Workspace:** 'DEEPGUARD'  
**Primary Reference Paper:** *A Spatial–Temporal Deep Learning Framework for Deepfake Detection Using MobileNetV2 and LSTM* (Inspected from `file/Batch06_Deepfake.pdf`)  
**Phase Status:** **Phase 1 Complete (Technical Specification Verified & Locked)**

---

## 1. PROJECT TITLE AND PROBLEM STATEMENT

### 1.1 Project Title
**DeepGuard — An Explainable and Robust Spatial–Temporal Framework for Deepfake Video Detection**

### 1.2 Problem Statement in Simple Language
Deepfakes are synthetic videos created using deep generative models (such as Generative Adversarial Networks and autoencoders) that alter a person's facial identity, expressions, or lip movements. 

Detecting these manipulated videos is challenging because:
1. **Spatial-only detectors** (standard 2D CNNs analyzing single frames) often miss temporal flaws such as flickering, unnatural eye-blinking rates, and frame-to-frame boundary jitter.
2. **Temporal-only detectors** (standard recurrent models) lack the fine-grained visual capability to inspect facial blending edges and textures.
3. **Heavy 3D architectures** (such as 3D CNNs or large Vision Transformers) require massive compute resources that are impractical for student environments and real-time deployment.
4. **Compression sensitivity:** High-frequency visual clues are often erased when videos are re-encoded and compressed by social media platforms.
5. **Black-box opacity:** Standard binary detectors output a single probability score without explaining *which frame* or *which region of the face* triggered the detection.

DeepGuard solves these challenges by combining a lightweight spatial feature extractor (**MobileNetV2**) with a temporal sequence model (**LSTM**), reinforced with explainability and rigorous evaluation against real-world video compression.

---

## 2. PROJECT OBJECTIVES AND EXPECTED OUTCOMES

### 2.1 Research Objectives
1. **Faithful Baseline Reproduction:** Implement the core MobileNetV2 + LSTM pipeline as described in the reference paper and establish an independently measured performance baseline.
2. **Temporal Window Evaluation:** Test whether increasing sequence length from 8 frames to 16 frames meaningfully improves detection of subtle blending artifacts or merely increases compute cost.
3. **Compression Robustness Benchmarking:** Systematically measure how detection accuracy degrades when videos undergo heavy H.264 compression (comparing high quality c23 vs. heavy compression c40).
4. **Cross-Dataset Generalization Analysis:** Evaluate zero-shot performance drop when a model trained on one dataset distribution is tested on an unseen dataset.
5. **Spatial Explainability via Grad-CAM:** Generate visual saliency heatmaps highlighting facial manipulation regions to verify whether the model is learning genuine synthetic artifacts rather than background noise.
6. **Student-Feasible Demonstration:** Package the detection model into a lightweight, interactive web application (Streamlit) capable of analyzing an uploaded video and displaying frame-level confidence curves and heatmaps.

### 2.2 Expected Outcomes
- A verified, leakage-free spatial–temporal deepfake detector.
- Complete experimental tables with honest, un-cherry-picked metrics.
- A fully documented, defense-ready codebase that can be explained and justified in an academic viva.

---

## 3. REFERENCE PAPER ANALYSIS (FROM `Batch06_Deepfake.pdf`)

### 3.1 Paper Identification
- **Title:** *A Spatial–Temporal Deep Learning Framework for Deepfake Detection Using MobileNetV2 and LSTM*
- **Authors:** J. Lakshmi Sravya, T. Tejaswini, G. Deva Harshini, T. Anirudh
- **Faculty Guide:** Mr. P. Srinivas Rao, Assistant Professor
- **Institution:** Department of Advanced Computer Science Engineering, VFSTR, Guntur, Andhra Pradesh, India
- **File Inspected:** `file/Batch06_Deepfake.pdf` (6 pages)

### 3.2 The Paper's Baseline Pipeline
The reference paper defines an end-to-end 10-step algorithm (Section IV-H):
1. **Video Ingestion:** Extract raw frames from input video $V$.
2. **Frame Sampling:** Select exactly **8 uniform frames** per video ($T = 8$).
3. **Face Localization:** Detect and crop faces using **OpenCV**.
4. **Resizing:** Resize all cropped face images to **$128 \times 128$ pixels**.
5. **Pixel Normalization:** Normalize pixel intensity values using Equation (1):
   $$x' = \frac{x}{255}$$
6. **Spatial Feature Extraction:** Extract spatial representations from each frame crop using pretrained **MobileNetV2**.
7. **Global Average Pooling:** Apply 2D Global Average Pooling (Equation 5) across spatial feature maps:
   $$G_k = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W f_{i,j}^k$$
   This yields a single **1,280-dimensional feature vector** for each frame.
8. **Temporal Modelling:** Form a sequence of 8 feature vectors $X = \{x_1, x_2, \dots, x_8\}$ where $x_t \in \mathbb{R}^{1280}$, and pass this sequence into an **LSTM** network.
9. **Regularization:** Apply **dropout** to the recurrent output.
10. **Classification:** Classify the sequence using a **sigmoid** activation output:
    $$\hat{y} = \sigma(W \cdot h + b) \in [0.0, 1.0]$$
    Loss function: Binary Cross-Entropy. Optimizer: Adam. Learning rate: $0.0001$. Batch size: $2$. Epochs: $15$.

---

## 4. EVIDENCE, GAPS, AND INCONSISTENCY REGISTER

### 4.1 Significant Inconsistency in the Reference Paper
The reference paper contains an internal numerical contradiction between its reported summary metrics and its page 5 confusion matrix:

- **Paper's Stated Text Claim:** The paper reports **95.2% accuracy**, 94% precision, 95% recall, and 94% F1-score (Abstract, Section IV-B, Section V-C, Fig. 3 caption).
- **Confusion Matrix Counts (Fig. 3(c), Page 5):**
  - True Positive (TP): 282
  - False Negative (FN): 18
  - False Positive (FP): 22
  - True Negative (TN): 278
  - **Total Evaluated Videos:** $282 + 18 + 22 + 278 = 600$ test videos.
- **Independent Recalculation:**
  $$\text{Calculated Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}} = \frac{282 + 278}{600} = \frac{560}{600} = 93.33\%$$
- **Additional Discrepancies:**
  1. In Section IV-C, the paper states the test split contains **624 videos** ($10.6\%$ of 5,856), yet the confusion matrix totals only **600 videos** (leaving 24 videos unaccounted for).
  2. In Section V-E, the authors state: *"validation accuracy remains consistent at 93.3%"*. This strongly suggests the confusion matrix shown in Fig. 3(c) was evaluated on the validation set or at an earlier epoch, while 95.2% was the final training accuracy.
  3. Fig. 3(c) labels "Real" as "True Positive" and "Fake" as "True Negative", which inverts standard media forensics conventions where "Fake/Manipulated" is defined as the positive class ($1$).

**Academic Rule for DeepGuard:** Neither 95.2% nor 93.33% belongs to DeepGuard. Both are recorded as literature reference data with the noted discrepancy. DeepGuard will compute all metrics directly from its own verified test predictions.

### 4.2 Evidence and Specification Table

| Pipeline Component | Stated in Reference Paper | Missing / Unspecified in Paper | DeepGuard Design Decision |
|---|---|---|---|
| **Input Resolution** | $128 \times 128$ RGB (Section IV-D.3) | Aspect ratio handling method | Bilinear resize to $128 \times 128 \times 3$ with square face crop. |
| **Pixel Normalization** | $x' = x / 255$ (Equation 1) | None (explicitly stated) | Divide pixel values by $255.0$ to scale to $[0.0, 1.0]$. |
| **Sequence Length ($T$)** | 8 frames (Section IV-D.1) | Stride calculation formula | Sample 8 uniformly spaced frames across video duration. |
| **Face Detection** | OpenCV methods (Section IV-D.2) | Specific OpenCV algorithm (Haar vs DNN vs YuNet) | OpenCV YuNet (or Haar cascade fallback) with $+20\%$ margin. |
| **Spatial Backbone** | MobileNetV2 pretrained (Section IV-E) | Freezing strategy (frozen vs fine-tuned) | Use frozen MobileNetV2 for feature caching; evaluate fine-tuning in later ablation. |
| **Spatial Pooling** | Global Average Pooling (Eq. 5) | None (GAP to 1,280-d explicit) | Keras `GlobalAveragePooling2D()` yielding shape $(T, 1280)$. |
| **LSTM Architecture** | LSTM network (Section IV-F) | **Number of hidden units not reported** | Set to **64 units** provisionally; test 128 in ablation. |
| **Regularization** | Dropout applied (Section IV-H.9) | **Dropout rate not reported** | Set dropout rate to **$0.3$** provisionally. |
| **Batch Size** | 2 (Section IV-G) | Reason for size 2 (likely memory limit) | 2 for end-to-end training; batch size 16 for cached feature training. |
| **Optimizer & LR** | Adam, LR = 0.0001 (Section IV-G) | Epsilon, weight decay, schedule | Adam, $\text{lr}=10^{-4}$ with `ReduceLROnPlateau`. |
| **Training Epochs** | 15 epochs (Section IV-G) | Early stopping patience | 15 epochs maximum; early stopping patience 4 on validation loss. |
| **Dataset Splitting** | 71.6% train, 17.8% val, 10.6% test | **Disjointness not specified** (risk of actor leakage) | Enforce strict video-level and identity-disjoint splitting. |

---

## 5. DATASET ASSESSMENT AND PRE-SELECTION CHECKS

The reference paper states it combined videos from FaceForensics++, DFDC, and Celeb-DF (totaling 5,856 videos: 1,583 authentic and 4,273 deepfakes). 

For DeepGuard, **we must not assume any dataset is immediately available or downloadable.** Before selecting and downloading data in Phase 2, the following formal checks are required:

### 5.1 Required Dataset Assessment Protocol
1. **Access Terms & Application Requirements:**
   - *FaceForensics++ (FF++):* Requires signing an academic agreement via their official Google Form to receive an authorized download script and token.
   - *Celeb-DF (v2):* Requires submitting an academic request to the authors (SUNY Albany).
   - *DFDC (Preview):* Available on Kaggle under non-commercial competition license.
2. **Storage Footprint vs. Budget:**
   - Raw video archives exceed 500 GB (impossible on student Google Drive).
   - We must verify that a **curated subset** (e.g., 200 real + 200 fake videos of FaceForensics++ c23) can be downloaded within the 15 GB Google Drive storage quota.
3. **Annotation and Split Metadata:**
   - Verify whether actor/subject identity metadata is provided to enforce identity-disjoint splits.
4. **Fallback Strategy:**
   - If official FaceForensics++ access credentials take time to approve, use the publicly accessible DFDC Preview subset or Celeb-DF v2 test partition as an immediate operational fallback.

---

## 6. VIDEO-LEVEL DATA SPLITTING AND LEAKAGE PREVENTION

Data leakage is the most common reason machine learning models achieve artificially high accuracy in testing but fail completely in the real world.

### 6.1 Principles of Leakage Prevention
1. **Video-Level Disjoint Splitting:** All frames extracted from Video $V_i$ must belong to exactly one partition (Train, Validation, or Test). Frames from the same video must never be shared across splits.
2. **Actor/Identity Disjoint Splitting:** If Actor $A$ appears in an authentic video and is manipulated to create Deepfake Video $D_{A \to B}$, both the authentic video and the deepfake video must remain in the **same split**. If the model trains on Actor $A$'s face in training and is evaluated on Actor $A$'s face in testing, it may simply memorize the actor's facial identity rather than learning synthesis artifacts.
3. **Pre-Split Partitioning:** Splits must be generated and locked into permanent manifest files (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`) *before* any preprocessing, cropping, or feature extraction occurs.
4. **Fit-on-Train Only:** Any data transformations, statistics, or normalizations must be fitted strictly on the training partition.

---

## 7. PREPROCESSING WORKFLOW

The verified preprocessing workflow executes sequentially:

```
[ Raw Video File ]
       │
       ▼
1. Video Verification: Open video with cv2.VideoCapture; check frame count ≥ 8.
       │
       ▼
2. Uniform Frame Sampling: Select exactly 8 equidistant frame indices:
   indices = [int(i * total_frames / 8) for i in range(8)]
       │
       ▼
3. Face Detection & Tracking:
   - Detect face in each frame using OpenCV (YuNet / Haar).
   - Apply a +20% bounding box margin around the face.
   - If detection fails on frame k, forward-fill bounding box from frame k-1.
       │
       ▼
4. Crop & Resizing:
   - Crop the facial region.
   - Resize cropped face to exactly 128 × 128 pixels.
       │
       ▼
5. Pixel Normalization:
   - Divide RGB values by 255.0 to yield values in [0.0, 1.0].
       │
       ▼
6. Sequence Assembly:
   - Form tensor of shape (8, 128, 128, 3) ready for spatial feature extraction.
```

---

## 8. BASELINE MODEL ARCHITECTURE & TENSOR DIMENSIONS

### 8.1 Tensor Flow Table

| Stage | Operation / Layer | Input Tensor Shape | Output Tensor Shape | Parameters |
|---|---|---|---|---|
| **Input** | Video Frame Sequence | Batch of videos | $(B, 8, 128, 128, 3)$ | 0 |
| **Backbone** | MobileNetV2 (`include_top=False`) | $(B \times 8, 128, 128, 3)$ | $(B \times 8, 4, 4, 1280)$ | Pretrained weights (~2.2M params) |
| **Spatial Pool** | `GlobalAveragePooling2D()` | $(B \times 8, 4, 4, 1280)$ | $(B \times 8, 1280)$ | 0 |
| **Reshape** | Sequence Grouping | $(B \times 8, 1280)$ | $(B, 8, 1280)$ | 0 |
| **Temporal** | `LSTM(units=64)` | $(B, 8, 1280)$ | $(B, 64)$ | Trainable recurrent weights |
| **Dropout** | `Dropout(rate=0.3)` | $(B, 64)$ | $(B, 64)$ | 0 |
| **Classifier**| `Dense(1, activation='sigmoid')` | $(B, 64)$ | $(B, 1)$ | Trainable classification weights |

### 8.2 Two Execution Modes
1. **End-to-End Mode:** The video frames pass directly through MobileNetV2 and LSTM simultaneously during backpropagation. This matches the paper's batch size of 2 due to GPU VRAM limits.
2. **Feature-Caching Mode (Recommended for Student Compute):** Pass all frames through the frozen MobileNetV2 + GAP layer once, and save the resulting $(8, 1280)$ feature tensors to disk as `.npz` files. Training the LSTM on these compact vectors takes seconds per epoch on Colab, allowing fast iterations, larger batch sizes (e.g., 16 or 32), and zero VRAM exhaustion.

---

## 9. EVALUATION METRICS AND REPORTING PROTOCOL

To ensure an academically rigorous and un-cherry-picked evaluation, DeepGuard uses eight standard metrics:

1. **Accuracy:** Overall proportion of correct predictions (reported alongside class balance).
2. **Precision:** $\frac{TP}{TP + FP}$ — Measures how many flagged videos were truly manipulated.
3. **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — Measures how many actual deepfakes were successfully detected.
4. **F1-Score:** Harmonic mean of Precision and Recall.
5. **ROC-AUC:** Area Under the Receiver Operating Characteristic Curve (threshold-independent).
6. **Confusion Matrix:** Full counts of True Positives, False Positives, True Negatives, and False Negatives.
7. **Model Size:** Total parameter count and disk footprint in megabytes (MB).
8. **Inference Latency:** Average processing time per video (in milliseconds) and throughput in frames per second (FPS).

*Semantics:* Class $1 \triangleq \textbf{Manipulated (Deepfake)}$; Class $0 \triangleq \textbf{Authentic (Real)}$.

---

## 10. CONTROLLED EXPERIMENT MATRIX FOR FUTURE PHASES

| Experiment | Target Phase | Research Question | Independent Variable | Controlled Conditions | Acceptance / Comparison Metric |
|---|---|---|---|---|---|
| **EXP-01: Baseline Reproduction** | Phase 3 & 4 | Can we replicate the paper's MobileNetV2 + LSTM baseline on our dataset? | Baseline implementation | $T=8$, 128x128, Adam $\text{lr}=10^{-4}$ | Attain stable loss convergence and ROC-AUC benchmark. |
| **EXP-02: Temporal Sequence Window** | Phase 6 | Does doubling frames from 8 to 16 improve detection of transient artifacts? | Sequence length $T \in \{8, 16\}$ | Identical data splits, model architecture, training epochs | Compare $\Delta\text{AUC}$, $\Delta\text{F1}$, and inference latency penalty. |
| **EXP-03: Temporal Attention** | Phase 5 | Does self-attention over frame features outperform standard LSTM? | Aggregation mechanism (LSTM vs Attn-LSTM) | $T=8$, identical feature vectors | Compare validation loss variance and test ROC-AUC. |
| **EXP-04: Compression Robustness** | Phase 7 | How severely does heavy H.264 compression degrade detection? | Compression level (c23 vs c40) | Identical trained model weights | Measure drop in accuracy and AUC between c23 and c40. |
| **EXP-05: Cross-Dataset Transfer** | Phase 8 | How well does the model generalize to an unseen manipulation distribution? | Target dataset (in-domain vs out-of-domain) | Identical trained model weights | Measure cross-dataset generalization gap ($\Delta\text{AUC}$). |
| **EXP-06: Grad-CAM Explainability** | Phase 9 | Do spatial attention heatmaps highlight genuine manipulation boundaries? | Frame artifact intensity | MobileNetV2 convolutional layers | Visual correlation between saliency and facial blending artifacts. |

---

## 11. BEGINNER'S GUIDE: HOW OUR TOOLS WORK TOGETHER

As a beginner, it helps to understand clearly why we use each tool and where each piece of the project lives:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   HOW DEEPGUARD'S TOOLS WORK TOGETHER                   │
├───────────────────┬────────────────────────────────────────────────────┤
│ Tool              │ What It Does in DeepGuard                          │
├───────────────────┼────────────────────────────────────────────────────┤
│ 1. Antigravity    │ Your local development studio on your PC.          │
│    (Local PC)     │ Used to write code, inspect files, check git,      │
│                   │ run fast unit tests, and design documentation.     │
├───────────────────┼────────────────────────────────────────────────────┤
│ 2. GitHub         │ The secure cloud backup of your source code.       │
│    (Remote Repo)  │ Tracks project history, code revisions, and phase  │
│                   │ progress so your work is never lost.               │
├───────────────────┼────────────────────────────────────────────────────┤
│ 3. Google Colab   │ Your cloud workstation with a free GPU (Tesla T4). │
│    (Cloud Compute)│ Deep learning is slow on a normal laptop CPU.      │
│                   │ We use Colab's T4 GPU to run feature extraction    │
│                   │ and train the neural network in minutes.           │
├───────────────────┼────────────────────────────────────────────────────┤
│ 4. Google Drive   │ Permanent cloud storage for heavy files.           │
│    (Cloud Storage)│ Colab resets its disk every time you close it.     │
│                   │ Google Drive stores our video datasets, cached     │
│                   │ features (.npz), and trained models (.keras).      │
└───────────────────┴────────────────────────────────────────────────────┘
```

---

## 12. RISKS, LIMITATIONS, AND ETHICAL CONSIDERATIONS

1. **No Absolute Proof of Forgery:** DeepGuard is an educational and experimental machine-learning detector. It outputs statistical likelihoods, not legal or forensic proof.
2. **False Positive Harm:** Flagging an authentic video as a deepfake can damage an innocent person's reputation; missing a malicious deepfake allows misinformation to spread. Both failure modes must be reported honestly.
3. **Data Privacy:** Only public academic research datasets with consenting actors will be utilized.
4. **Academic Integrity:** All metrics presented in DeepGuard reports will be derived from our own code executions. We will never represent the reference paper's figures as DeepGuard's results.

---

## 13. PHASE COMPLETION CHECKLIST AND DECISION REGISTER

### 13.1 Phase 1 Completion Checklist
- [x] Primary reference paper `file/Batch06_Deepfake.pdf` inspected completely.
- [x] Paper baseline settings extracted and documented accurately.
- [x] Page 5 accuracy discrepancy (95.2% vs. 93.33%) recorded in evidence register.
- [x] Missing implementation details identified (LSTM units, dropout rate, split disjointness).
- [x] Video-level and identity-disjoint splitting strategy defined.
- [x] Preprocessing pipeline documented ($x/255$, $128 \times 128$, 8 frames).
- [x] Baseline architecture and tensor shapes derived.
- [x] 8 evaluation metrics and reporting rules defined.
- [x] 6-part controlled experiment matrix specified.
- [x] Beginner-friendly tool explanation created.
- [x] Git repository status and remote URL inspected.

### 13.2 Decision Register

| ID | Decision Item | Selected Option | Status |
|---|---|---|---|
| **DEC-01** | Backbone Architecture | MobileNetV2 pretrained on ImageNet | Decided |
| **DEC-02** | Spatial Feature Representation | Global Average Pooling to 1,280 dimensions | Decided |
| **DEC-03** | Temporal Model | Single-layer LSTM with 64 units (provisional) | Decided |
| **DEC-04** | Frame Sampling | 8 uniform frames across video duration | Decided |
| **DEC-05** | Image Normalization | Divide pixel intensities by 255.0 ($[0, 1]$) | Decided |
| **DEC-06** | Face Resolution | $128 \times 128$ RGB pixels | Decided |
| **DEC-07** | Training Compute Strategy | Feature caching on Google Drive + Colab T4 training | Decided |
| **DEC-08** | Git Remote URL Typo | Update URL from `.gi` to `.git` when user confirms | Pending User Confirmation |

---

## 14. EXACT NEXT ACTIONS FOR THE STUDENT DEVELOPER

As a beginner, here are the exact steps you need to take before Phase 2 starts:

1. **Review this Phase 1 Specification:** Read through this document to ensure the research objectives and baseline settings align with your course goals.
2. **Fix the GitHub Remote URL:** Your current Git remote URL ends in `.gi` instead of `.git`. When you are ready, run this command in your terminal:
   ```bash
   git remote set-url origin https://github.com/chandrababu07/DEEPGUARD.git
   ```
3. **Make Your First Git Commit:** Save this Phase 1 documentation into Git:
   ```bash
   git add .
   git commit -m "docs: complete Phase 1 research specification and baseline audit"
   git push -u origin main
   ```
4. **Approve Transition to Phase 2:** Once the repository is committed and you are satisfied with this specification, confirm to proceed to **Phase 2 (Dataset Engineering and Preprocessing Pipeline)**.
