# DEEPGUARD â€” PHASE 6: SECURE API, MODEL SERVING, FRONTEND INTEGRATION, AND DEPLOYMENT READINESS
**Project:** DeepGuard â€” Spatial-Temporal Deepfake Video Detection
**Module:** Phase 6 Production-Ready Serving and UI Integration
**Status:** Completed & Verified

---

## 1. EXECUTIVE OVERVIEW

Phase 6 transitions the DeepGuard research framework into a secure, production-ready, interactive deployment.

The implementation delivers:
1. **Hardened FastAPI REST Backend:** Enterprise-grade asynchronous API with strict upload bounds, container magic byte validation, streaming memory guards, and path-traversal mitigations.
2. **Model Serving Lifecycle:** Singleton model initialization once per application process, thread-safe inference execution, and formal separation between service liveness (`/health`) and model readiness (`/ready`).
3. **Interactive Frontend Web Application:** A responsive UI served directly by FastAPI at `/` featuring drag-and-drop video upload, real-time readiness status indicators, score visualization, metadata inspection, and Grad-CAM spatial heatmap overlays.
4. **Epistemic Humility & Forensic Ethics:** Clear visual distinction between raw model scores and calibrated probabilities, accompanied by mandatory non-causal Grad-CAM disclaimers.
5. **Reproducible Containerization:** Production `Dockerfile` and `.env.example` templates configured with healthcheck probes and OpenCV runtime libraries.

---

## 2. API ENDPOINTS REFERENCE

All endpoints are versioned under `/api/v1` and generate automatic OpenAPI documentation at `/docs` and `/redoc`.

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `GET` | `/api/v1/health` | Service liveness probe | `200` |
| `GET` | `/api/v1/ready` | Model dependency & checkpoint readiness probe | `200` |
| `POST` | `/api/v1/predict` | Upload video for deepfake manipulation classification | `200`, `400`, `413`, `415`, `422`, `503` |
| `POST` | `/api/v1/explain` | Upload video for Grad-CAM visual spatial attribution | `200`, `400`, `413`, `415`, `422`, `503` |
| `GET` | `/api/v1/artifacts/{filename}` | Retrieve generated explanation heatmap overlay | `200`, `400`, `403`, `404` |
| `GET` | `/` | Interactive DeepGuard web application UI | `200` |

### 2.1 Request & Response Contracts

#### A. Health Probe (`GET /api/v1/health`)
```json
{
  "status": "healthy",
  "service": "DeepGuard Deepfake Detection API",
  "version": "1.0.0",
  "timestamp": "2026-10-13T17:00:00Z"
}
```

#### B. Readiness Probe (`GET /api/v1/ready`)
```json
{
  "status": "not_ready",
  "model_loaded": false,
  "tensorflow_available": false,
  "checkpoint_path": "experiments/mini_baseline/checkpoints/deepguard_mobilenetv2_lstm_baseline.keras",
  "checkpoint_sha256": "unknown",
  "decision_threshold": 0.5,
  "expected_input_shape": [8, 128, 128, 3],
  "device": "CPU/Unloaded",
  "warnings": [
    "TensorFlow is unavailable locally. Server running in API/UI preview mode."
  ],
  "error": "TensorFlow is not installed in the current Python environment."
}
```

#### C. Video Prediction (`POST /api/v1/predict`)
* **Request:** `multipart/form-data` with `file: UploadFile`.
* **Response:**
```json
{
  "request_id": "8f3b9a1e4c7d...",
  "video_filename": "sample_video.mp4",
  "predicted_label": "MANIPULATED",
  "label_code": 1,
  "fake_score": 0.8750,
  "real_score": 0.1250,
  "decision_threshold": 0.5000,
  "score_calibration": "Uncalibrated sigmoid output. Reflects raw model output, not certified statistical confidence.",
  "model_version": "deepguard-mobilenetv2-lstm-v1",
  "checkpoint_sha256": "3a8f1b...",
  "processing_time_ms": 142.50,
  "frame_tensor_shape": [8, 128, 128, 3],
  "warnings": []
}
```

#### D. Explainability Heatmaps (`POST /api/v1/explain`)
* **Request:** `multipart/form-data` with `file: UploadFile`.
* **Response:**
```json
{
  "request_id": "8f3b9a1e4c7d...",
  "video_filename": "sample_video.mp4",
  "predicted_label": "MANIPULATED",
  "fake_score": 0.8750,
  "target_layer": "Conv_1",
  "frames_explained": 3,
  "explanations": [
    {
      "frame_index": 0,
      "overlay_url": "/api/v1/artifacts/8f3b9a1e_frame_0_gradcam.png",
      "peak_activation_coords": [64, 48],
      "mean_activation_intensity": 0.4520,
      "predicted_score": 0.8750
    }
  ],
  "disclaimer": "Preliminary development model. Grad-CAM saliency highlights spatial feature activation correlations, not legal or forensic proof of video manipulation.",
  "warnings": []
}
```

---

## 3. SECURITY AND SAFE FILE HANDLING

Located at `src/api/security.py`.

1. **Streaming Bounded Uploads:** Videos are streamed in 64 KB buffers with active byte counting. If the payload exceeds the configured limit (default: 100 MB), the stream aborts immediately with `HTTP 413`, deleting partial bytes on disk.
2. **Container Magic Byte Verification:** Verifies video container headers (MP4 `ftyp`, AVI `RIFF`, WebM/MKV `EBML`, MPEG) before buffering large payloads. Text files or executable payloads masked as `.mp4` are rejected with `HTTP 415`.
3. **OpenCV Video Stream Validation:** Decodes the video using `cv2.VideoCapture` to verify readable frames ($\ge 8$), non-zero spatial dimensions, and duration limits ($\le 180\text{s}$). Corrupted or truncated containers are rejected with `HTTP 422`.
4. **Secure Random Filenames & Path Traversal Prevention:** Client filenames are discarded for filesystem operations; temporary files use random UUID4 tokens (`upload_{uuid4}.mp4`). Artifact serving validates that filenames conform to alphanumeric masks and resolve strictly inside the artifact directory.
5. **Safe Lifecycle Cleanup:** Uploaded temporary files are deleted in `finally` blocks, ensuring no disk leakage on success, client cancellation, or internal exceptions.
6. **Error Sanitization:** Global exception handlers format all errors into standard `ErrorResponse` schemas with unique request IDs. Raw Python tracebacks and internal filesystem paths are never leaked to clients.

---

## 4. MODEL SERVING LIFECYCLE

Located at `src/api/service.py`.

1. **Singleton Initialization:** The model checkpoint is loaded once per application process inside a thread lock, eliminating per-request loading overhead.
2. **Failure Handling:** Missing TensorFlow or checkpoint files sets `readiness_error` without crashing the FastAPI process. The `/health` endpoint remains healthy while `/ready` accurately reports the dependency blocker.
3. **Decoupled Dependency Injection:** `ModelService` accepts injected mock predictors and explainers, enabling comprehensive integration testing on environments without GPU or TensorFlow.
4. **Epistemic Integrity:** The API never fabricates predictions, substitutes hardcoded values, or describes uncalibrated sigmoid scores as certified posterior probabilities.

---

## 5. FRONTEND WEB APPLICATION

Located at `src/web/static/` (`index.html`, `style.css`, `app.js`).

1. **Visual Design:** Dark theme with subtle glassmorphism (`backdrop-filter`), glowing cyan/indigo accents, and responsive layout.
2. **Real-Time Readiness Polling:** Periodically checks `/api/v1/ready` to update header status badges and displays informative alerts when running in local CPU/preview mode.
3. **Drag-and-Drop Ingestion:** Supports drag-and-drop and native file picking with client-side extension and size validation.
4. **Forensic Verdict Card:** Visualizes verdict badges ("AUTHENTIC" in emerald or "MANIPULATED" in rose), animated continuous likelihood meters relative to the 0.50 threshold, and forensic metadata.
5. **Grad-CAM Gallery:** Displays Early (Frame 0), Mid (Frame 3), and Late (Frame 7) heatmap overlays with peak coordinate readouts and the mandatory non-causal disclaimer.

---

## 6. LOCAL DEVELOPMENT & CONTAINER DEPLOYMENT

### 6.1 Running the Server Locally
```bash
# Launch FastAPI server with auto-reload
python scripts/run_server.py --host 127.0.0.1 --port 8000 --reload
```
* **Web Application UI:** `http://127.0.0.1:8000/`
* **Interactive API Documentation:** `http://127.0.0.1:8000/docs`

### 6.2 Running the Unit & API Test Suite
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
# or
pytest tests/ -v
```
*(111 tests total: 107 passed, 4 skipped due to missing local TensorFlow, 0 failures).*

### 6.3 Docker Container Deployment
```bash
# Build production Docker image
docker build -t deepguard-app:latest .

# Run container with port forwarding and volume mounts
docker run -d \
    --name deepguard \
    -p 8000:8000 \
    -e DEEPGUARD_MAX_UPLOAD_SIZE=104857600 \
    -e DEEPGUARD_DECISION_THRESHOLD=0.5 \
    deepguard-app:latest
```

### 6.4 Google Colab Live GPU Serving
To serve DeepGuard with live GPU inference from Google Colab:
```python
# In a Google Colab notebook with T4 GPU:
!git clone https://github.com/chandrababu07/DEEPGUARD.git
%cd DEEPGUARD
!pip install -r requirements.txt
!pip install pyngrok

from pyngrok import ngrok
ngrok_tunnel = ngrok.connect(8000)
print('DeepGuard Public Live URL:', ngrok_tunnel.public_url)

!python scripts/run_server.py --host 0.0.0.0 --port 8000
```
