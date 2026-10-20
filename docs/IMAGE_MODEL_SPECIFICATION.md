# DEEPGUARD — Image Deepfake Model Specification & Training Protocol

## 1. Architectural Contract & Model Interface

The DEEPGUARD framework supports dedicated single-image deepfake detection isolated from the temporal video sequence model.

### 1.1 Input & Output Tensor Shapes

| Dimension | Specification | Notes |
| :--- | :--- | :--- |
| **Input Rank** | 4D `(batch_size, height, width, channels)` or 3D `(height, width, channels)` | Standard batched image input |
| **Spatial Dimensions** | Height: 128 (default) or 224; Width: 128 (default) or 224 | Handled by `ImagePreprocessor` |
| **Color Channels** | 3 (RGB order) | OpenCV BGR converted to RGB |
| **Data Type** | `float32` in `[0.0, 1.0]` | Scaled by `1 / 255.0` |
| **Output Tensor** | `(batch_size, 1)` | Sigmoid probability $\hat{y} \in [0.0, 1.0]$ |
| **Class Semantics** | `0.0` = Authentic / Real, `1.0` = Manipulated / Deepfake | Forensic binary contract |

### 1.2 Checkpoint Format & Storage

- Format: TensorFlow/Keras SavedModel or modern `.keras` archive (`tf.keras.models.save_model`).
- Checkpoint location: Configurable via `DEEPGUARD_IMAGE_CHECKPOINT_PATH`.
- Checksum validation: Verified on initialization via SHA-256 digest calculation.
- Safety: Model files are loaded exclusively from explicit, trusted local filesystem paths. Arbitrary uploaded model weights are strictly prohibited.

---

## 2. Dataset Structure & Leakage Prevention

### 2.1 Supported Datasets
- **FaceForensics++ (FF++)**: High-quality c23 and c40 subsets.
- **Celeb-DF (v2)**: Advanced facial manipulation with minimal visual artifacts.
- **WildDeepfake**: In-the-wild diverse lighting and pose variations.
- **Diffusion Face Datasets**: Artifact-guided synthetic generation (e.g., Stable Diffusion, Midjourney).

### 2.2 Identity-Aware & Source-Aware Splitting
To prevent model over-optimism and evaluation leakage:
1. **Subject Disjoint Splitting**: Images of subject/person $X$ must appear strictly in **one** split (Train, Validation, or Test). Never allow authentic frames of subject $X$ in Train while manipulated frames of subject $X$ are in Test.
2. **Method-Stratified Sampling**: Ensure balanced representation of distinct manipulation types across splits:
   - Deepfakes (autoencoder face swaps)
   - Face2Face (expression reenactment)
   - FaceSwap (classical graphics replacement)
   - NeuralTextures (neural rendering)
   - Latent Diffusion / GAN synthesis

---

## 3. Preprocessing, Augmentation & Robustness

### 3.1 Preprocessing Pipeline
1. **EXIF Normalization**: Orientation tags resolved automatically.
2. **Dimension Boundary Check**: Width and height validated against maximum safe dimensions (8192 px) to mitigate decompression bombs.
3. **Face Crop / Resizing**: Bounding boxes expanded by 20% margin to incorporate subtle blending boundary artifacts along hair and jawline.
4. **Color Normalization**: Converted to RGB float32 in range `[0.0, 1.0]`.

### 3.2 Training Augmentation Strategy
Deepfake classifiers frequently overfit to compression artifacts or resolution signatures. Augmentations must include:
- **JPEG Re-compression**: Quality factors uniformly sampled from $[40, 95]$.
- **Gaussian Blur**: Kernel sizes $\sigma \in [0.5, 2.0]$ simulating camera defocus.
- **Spatial Scaling / Downsampling**: Bilinear downsampling followed by restoration simulating web transcoding.
- **Color Jitter**: Brightness ($\pm 15\%$), contrast ($\pm 15\%$), and hue shifts ($\pm 5\%$).

---

## 4. Evaluation Protocol & Metrics

Any candidate checkpoint must be evaluated against a held-out test split before deployment:

### Required Quantitative Metrics:
- **Accuracy & Balanced Accuracy**: Accounting for class imbalance.
- **Precision & Recall**: Specifically tracking false alarms (False Positives) on authentic media vs missed fakes (False Negatives).
- **F1 Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC & PR-AUC**: Threshold-independent discriminative power.
- **Brier Score & Expected Calibration Error (ECE)**: Assessing posterior probability reliability.
- **Cross-Manipulation Robustness**: Evaluating performance degradation when trained on one manipulation method (e.g. Face2Face) and tested on unseen methods (e.g. Diffusion faces).

---

## 5. Checkpoint Deployment & Configuration

To activate image deepfake inference in DEEPGUARD:

1. Train or export the `.keras` model meeting the input shape `(None, 128, 128, 3)` and output shape `(None, 1)`.
2. Save the checkpoint to disk, for example:
   ```bash
   checkpoints/image_baseline/best_image_model.keras
   ```
3. Set the environment variable:
   ```bash
   set DEEPGUARD_IMAGE_CHECKPOINT_PATH=checkpoints/image_baseline/best_image_model.keras
   ```
4. Start the DEEPGUARD server:
   ```bash
   python scripts/run_server.py --host 127.0.0.1 --port 8000
   ```
5. Check service readiness via `GET /api/v1/ready`. The probe will confirm `image_model_ready: true`.

---

## 6. Single-Image Explainability (Grad-CAM)

When a trained image checkpoint is configured:
1. Target convolutional layer (e.g., `Conv_1` or final bottleneck layer) must be explicitly specified and verified.
2. Heatmap computation calculates gradients of the winning class score with respect to target spatial feature activations.
3. Attributions must always be accompanied by the required epistemic disclaimer:
   > *Grad-CAM heatmaps highlight spatial regions with high model feature correlation. They reflect internal algorithmic attention and do not constitute certified legal or causal proof of image manipulation.*
