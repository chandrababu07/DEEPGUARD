# DEEPGUARD — Spatial-Temporal & Spatial Deepfake Forensics

[![Status](https://img.shields.io/badge/Status-Deployment%20Ready-success)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI%200.110+-blue)]()
[![Forensics](https://img.shields.io/badge/Mode-Dual--Mode%20Forensics-cyan)]()

DEEPGUARD is an explainable deepfake detection application combining spatial-temporal sequence analysis for videos and dedicated spatial forensic analysis for single images.

---

## 1. Analysis Modes

DEEPGUARD provides two isolated forensic workflows accessible directly from the modern web interface or REST API:

### Mode 1: Video Analysis
- **Architecture**: MobileNetV2 spatial feature extractor combined with recurrent LSTM temporal dynamics.
- **Input Contract**: 8 uniform RGB frames of shape `(8, 128, 128, 3)`.
- **Explainability**: Spatial Grad-CAM saliency heatmaps across key early (0), mid (3), and late (7) sequence frames.
- **Supported Formats**: MP4, AVI, MOV, MKV, WebM.
- **Constraints**: Up to 100 MB file size, up to 180 seconds duration, minimum 8 decodable frames.
- **Endpoint**: `POST /api/v1/predict` and `POST /api/v1/explain`.

### Mode 2: Image Analysis
- **Architecture**: Dedicated 2D spatial deepfake classification model (isolated from temporal video models).
- **Input Contract**: Single RGB frame resized to `(128, 128, 3)` with float32 inputs in `[0.0, 1.0]`.
- **Supported Formats**: JPEG, PNG, WebP.
- **Constraints**: Up to 20 MB file size, maximum spatial dimensions 8,192 × 8,192 px (decompression bomb safety).
- **Endpoint**: `POST /api/v1/predict/image`.
- **Safety Policy**: DEEPGUARD strictly prevents substituting video sequence models for single images. If an image checkpoint is not configured, the endpoint returns a clear `HTTP 503 Service Unavailable` response. The application never fabricates scores.

---

## 2. Quick Start

### 2.1 Environment Setup

Activate the project Python environment:
```powershell
.venv-tf\Scripts\activate
```

Install core dependencies (if running in a clean environment):
```bash
pip install -r requirements.txt
```

### 2.2 Starting the Application

Launch the FastAPI application and web interface:
```bash
python scripts/run_server.py --host 127.0.0.1 --port 8000
```

- **Web Application UI**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive OpenAPI Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Service Liveness Probe**: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
- **Model Readiness Probe**: [http://127.0.0.1:8000/api/v1/ready](http://127.0.0.1:8000/api/v1/ready)

---

## 3. Configuration & Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DEEPGUARD_HOST` | `127.0.0.1` | Server binding host |
| `DEEPGUARD_PORT` | `8000` | Server binding port |
| `DEEPGUARD_CHECKPOINT_PATH` | `checkpoints/mini_baseline/best_model.keras` | Video sequence model checkpoint path |
| `DEEPGUARD_DECISION_THRESHOLD` | `0.5` | Video classification threshold |
| `DEEPGUARD_IMAGE_CHECKPOINT_PATH` | *(empty)* | Dedicated image classifier checkpoint path |
| `DEEPGUARD_IMAGE_DECISION_THRESHOLD` | `0.5` | Image classification threshold |
| `DEEPGUARD_MAX_UPLOAD_SIZE` | `104857600` (100 MB) | Max video upload size in bytes |
| `DEEPGUARD_MAX_IMAGE_UPLOAD_SIZE` | `20971520` (20 MB) | Max image upload size in bytes |
| `DEEPGUARD_MAX_IMAGE_DIMENSION` | `8192` | Max width/height in pixels |
| `DEEPGUARD_UPLOAD_TMP_DIR` | `experiments/temp_uploads` | Secure temporary upload storage |

---

## 4. Image Model Status & Checkpoint Configuration

### Current Repository Status
The repository currently contains the trained video baseline checkpoint:
`checkpoints/mini_baseline/best_model.keras` (Input: `(None, 8, 128, 128, 3)`).

No trained single-image deepfake classifier was pre-trained in the repository. As required by DEEPGUARD's forensic integrity policies:
- The image ingestion UI, bounded streaming upload, magic byte verification, OpenCV decoding, and `ImagePredictor` service interface are fully implemented and verified.
- When an image analysis request is made without a configured checkpoint, the backend returns an honest, structured `503 Service Unavailable` status.
- The UI presents an informative status badge explaining that predictions will be activated when a trained checkpoint is provided.

### Configuring a Trained Image Checkpoint
When a trained image model is available:
```powershell
set DEEPGUARD_IMAGE_CHECKPOINT_PATH=checkpoints/image_baseline/image_model.keras
python scripts/run_server.py
```
See [docs/IMAGE_MODEL_SPECIFICATION.md](docs/IMAGE_MODEL_SPECIFICATION.md) for detailed dataset requirements, leakage prevention, training procedures, and verification protocols.

---

## 5. Security & Ingestion Safeguards

- **Magic Byte Verification**: Rejects spoofed media files by checking container signatures before decoding.
- **Bounded Chunk Streaming**: Streams uploads in 64 KB buffers, immediately aborting if size exceeds thresholds (mitigates DoS memory exhaustion).
- **Decompression Bomb Defense**: Enforces dimension limits (8,192 px max) to prevent memory allocation attacks during image decompression.
- **Path Traversal Protection**: Uses secure UUID4 filenames for temporary disk storage and strictly resolves artifact paths within designated directories.
- **Deterministic Temp File Cleanup**: Temporary files are deleted in `finally` blocks during both successful inference and error handling.
- **Sanitized Error Responses**: Internal paths, secrets, and tracebacks are never leaked in API error responses.

---

## 6. Running Tests

Run the test suite using the virtual environment:
```powershell
.venv-tf\Scripts\python -m pytest tests
```

To run focused API and image tests:
```powershell
.venv-tf\Scripts\python -m pytest tests/test_api_predict.py tests/test_api_predict_image.py tests/test_api_security.py
```
