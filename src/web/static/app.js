/**
 * DEEPGUARD Web Application Client Script.
 * Supports dual-mode deepfake forensic analysis:
 * 1. Video Analysis (8-frame temporal sequence with MobileNetV2 + LSTM and Grad-CAM)
 * 2. Image Analysis (single-frame spatial forensics via dedicated image model)
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation & Readiness Elements
  const readinessBadge = document.getElementById("readiness-badge");
  const readinessText = document.getElementById("readiness-text");
  const modelStatusAlert = document.getElementById("model-status-alert");
  const alertTitle = document.getElementById("alert-title");
  const alertMessage = document.getElementById("alert-message");
  const imageModelAlert = document.getElementById("image-model-alert");
  const imageAlertTitle = document.getElementById("image-alert-title");
  const imageAlertMessage = document.getElementById("image-alert-message");

  // Mode Switcher Elements
  const tabVideoMode = document.getElementById("tab-video-mode");
  const tabImageMode = document.getElementById("tab-image-mode");
  const ingestionTitle = document.getElementById("ingestion-title");
  const ingestionTag = document.getElementById("ingestion-tag");

  // Video Upload & Controls Elements
  const videoDropArea = document.getElementById("video-drop-area");
  const videoFileInput = document.getElementById("video-file-input");
  const browseVideoBtn = document.getElementById("browse-video-btn");
  const videoPreviewCard = document.getElementById("video-preview-card");
  const videoPreviewFilename = document.getElementById("video-preview-filename");
  const videoPreviewFilesize = document.getElementById("video-preview-filesize");
  const clearVideoBtn = document.getElementById("clear-video-btn");

  const videoActions = document.getElementById("video-actions");
  const analyzeVideoBtn = document.getElementById("analyze-video-btn");
  const analyzeVideoSpinner = document.getElementById("analyze-video-spinner");
  const explainBtn = document.getElementById("explain-btn");
  const explainSpinner = document.getElementById("explain-spinner");

  // Image Upload & Controls Elements
  const imageDropArea = document.getElementById("image-drop-area");
  const imageFileInput = document.getElementById("image-file-input");
  const browseImageBtn = document.getElementById("browse-image-btn");
  const imagePreviewCard = document.getElementById("image-preview-card");
  const imagePreviewThumb = document.getElementById("image-preview-thumb");
  const imagePreviewFilename = document.getElementById("image-preview-filename");
  const imagePreviewFilesize = document.getElementById("image-preview-filesize");
  const imagePreviewDimensions = document.getElementById("image-preview-dimensions");
  const clearImageBtn = document.getElementById("clear-image-btn");

  const imageActions = document.getElementById("image-actions");
  const analyzeImageBtn = document.getElementById("analyze-image-btn");
  const analyzeImageSpinner = document.getElementById("analyze-image-spinner");

  // Progress Bar Elements
  const progressContainer = document.getElementById("progress-container");
  const progressFill = document.getElementById("progress-fill");
  const progressPercent = document.getElementById("progress-percent");
  const progressLabel = document.getElementById("progress-label");

  // Results & Verdict Elements
  const emptyState = document.getElementById("empty-state");
  const emptyStateTitle = document.getElementById("empty-state-title");
  const emptyStateDesc = document.getElementById("empty-state-desc");
  const resultsContent = document.getElementById("results-content");
  const verdictBadge = document.getElementById("verdict-badge");
  const verdictLabel = document.getElementById("verdict-label");
  const verdictSublabel = document.getElementById("verdict-sublabel");
  const verdictTime = document.getElementById("verdict-time");
  const scoreValue = document.getElementById("score-value");
  const scoreBarFill = document.getElementById("score-bar-fill");

  const metaRequestId = document.getElementById("meta-request-id");
  const metaTime = document.getElementById("meta-time");
  const metaTensor = document.getElementById("meta-tensor");
  const metaHash = document.getElementById("meta-hash");
  const metaCalibration = document.getElementById("meta-calibration");

  const explainSection = document.getElementById("explain-section");
  const overlaysGrid = document.getElementById("overlays-grid");

  // State Management
  let currentMode = "video"; // "video" | "image"
  let selectedVideoFile = null;
  let selectedImageFile = null;
  let readinessData = null;

  // 1. Initial Readiness Check
  async function checkReadiness() {
    try {
      const res = await fetch("/api/v1/ready");
      if (!res.ok) throw new Error("Service probe returned HTTP " + res.status);
      const data = await res.json();
      readinessData = data;

      if (data.status === "ready") {
        readinessBadge.className = "badge badge-success";
        readinessText.textContent = "Video Model Online";
        modelStatusAlert.style.display = "none";
      } else {
        readinessBadge.className = "badge badge-warning";
        readinessText.textContent = "Model Preview Mode";
        if (currentMode === "video") {
          modelStatusAlert.style.display = "flex";
          alertTitle.textContent = "Video Inference Offline in Current Runtime";
          alertMessage.textContent = data.error || (
            "TensorFlow/GPU runtime is required for live video inference. " +
            (data.warnings ? data.warnings.join(" ") : "")
          );
        }
      }

      // Check image model status
      updateImageModelAlert(data);
    } catch (err) {
      readinessBadge.className = "badge badge-danger";
      readinessText.textContent = "Service Unreachable";
      if (currentMode === "video") {
        modelStatusAlert.style.display = "flex";
        alertTitle.textContent = "API Connection Error";
        alertMessage.textContent = "Could not communicate with the DEEPGUARD backend server.";
      }
    }
  }

  function updateImageModelAlert(data) {
    if (!data) return;
    const isImageReady = data.image_model_ready === true;
    if (currentMode === "image") {
      if (isImageReady) {
        imageModelAlert.style.display = "none";
      } else {
        imageModelAlert.style.display = "flex";
        imageAlertTitle.textContent = "Dedicated Image Model Unconfigured";
        imageAlertMessage.textContent = (
          "The single-image forensic analysis infrastructure is fully operational. " +
          "However, predictions remain disabled until a verified, trained image deepfake checkpoint is configured via DEEPGUARD_IMAGE_CHECKPOINT_PATH. " +
          "DEEPGUARD strictly prevents substituting video sequence models for single images."
        );
      }
    } else {
      imageModelAlert.style.display = "none";
    }
  }

  checkReadiness();

  // 2. Mode Switching Logic
  function switchMode(newMode) {
    if (currentMode === newMode) return;
    currentMode = newMode;

    if (newMode === "video") {
      tabVideoMode.classList.add("active");
      tabVideoMode.setAttribute("aria-selected", "true");
      tabImageMode.classList.remove("active");
      tabImageMode.setAttribute("aria-selected", "false");

      ingestionTitle.textContent = "Video Ingestion";
      ingestionTag.textContent = "8-Frame Uniform Sampling";

      // Show video upload area, hide image
      if (selectedVideoFile) {
        videoPreviewCard.style.display = "flex";
        videoDropArea.style.display = "none";
      } else {
        videoPreviewCard.style.display = "none";
        videoDropArea.style.display = "block";
      }
      imageDropArea.style.display = "none";
      imagePreviewCard.style.display = "none";

      videoActions.style.display = "flex";
      imageActions.style.display = "none";

      imageModelAlert.style.display = "none";
      if (readinessData && readinessData.status !== "ready") {
        modelStatusAlert.style.display = "flex";
      } else {
        modelStatusAlert.style.display = "none";
      }

      emptyStateTitle.textContent = "No Video Analysis Conducted Yet";
      emptyStateDesc.textContent = "Upload a video file on the left and select 'Run Deepfake Analysis' to evaluate manipulation probability.";
    } else {
      tabImageMode.classList.add("active");
      tabImageMode.setAttribute("aria-selected", "true");
      tabVideoMode.classList.remove("active");
      tabVideoMode.setAttribute("aria-selected", "false");

      ingestionTitle.textContent = "Image Ingestion";
      ingestionTag.textContent = "Single-Frame Spatial Forensics";

      // Show image upload area, hide video
      if (selectedImageFile) {
        imagePreviewCard.style.display = "flex";
        imageDropArea.style.display = "none";
      } else {
        imagePreviewCard.style.display = "none";
        imageDropArea.style.display = "block";
      }
      videoDropArea.style.display = "none";
      videoPreviewCard.style.display = "none";

      imageActions.style.display = "flex";
      videoActions.style.display = "none";

      modelStatusAlert.style.display = "none";
      updateImageModelAlert(readinessData);

      // Hide video Grad-CAM section when in image mode
      explainSection.style.display = "none";

      emptyStateTitle.textContent = "No Image Analysis Conducted Yet";
      emptyStateDesc.textContent = "Upload a JPEG, PNG, or WebP image on the left and select 'Analyze Image' to evaluate manipulation probability.";
    }
  }

  tabVideoMode.addEventListener("click", () => switchMode("video"));
  tabImageMode.addEventListener("click", () => switchMode("image"));

  // 3. Video File Selection & Handlers
  browseVideoBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    videoFileInput.click();
  });
  videoDropArea.addEventListener("click", () => videoFileInput.click());

  videoDropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    videoDropArea.classList.add("drag-over");
  });

  videoDropArea.addEventListener("dragleave", () => {
    videoDropArea.classList.remove("drag-over");
  });

  videoDropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    videoDropArea.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleVideoSelected(e.dataTransfer.files[0]);
    }
  });

  videoFileInput.addEventListener("change", () => {
    if (videoFileInput.files && videoFileInput.files.length > 0) {
      handleVideoSelected(videoFileInput.files[0]);
    }
  });

  function handleVideoSelected(file) {
    const validExtensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
    const ext = "." + file.name.split(".").pop().toLowerCase();

    if (!validExtensions.includes(ext)) {
      alert("Unsupported file format for Video Analysis. Supported: MP4, AVI, MOV, MKV, WebM.");
      return;
    }

    const maxSizeBytes = 100 * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      alert("Video file size exceeds maximum allowed limit of 100 MB.");
      return;
    }

    selectedVideoFile = file;
    videoPreviewFilename.textContent = file.name;
    videoPreviewFilesize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";

    videoPreviewCard.style.display = "flex";
    videoDropArea.style.display = "none";
    analyzeVideoBtn.disabled = false;
    explainBtn.disabled = false;
  }

  clearVideoBtn.addEventListener("click", () => {
    selectedVideoFile = null;
    videoFileInput.value = "";
    videoPreviewCard.style.display = "none";
    videoDropArea.style.display = "block";
    analyzeVideoBtn.disabled = true;
    explainBtn.disabled = true;
    if (currentMode === "video") {
      emptyState.style.display = "block";
      resultsContent.style.display = "none";
      explainSection.style.display = "none";
    }
  });

  // 4. Image File Selection & Handlers
  browseImageBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    imageFileInput.click();
  });
  imageDropArea.addEventListener("click", () => imageFileInput.click());

  imageDropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    imageDropArea.classList.add("drag-over");
  });

  imageDropArea.addEventListener("dragleave", () => {
    imageDropArea.classList.remove("drag-over");
  });

  imageDropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    imageDropArea.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleImageSelected(e.dataTransfer.files[0]);
    }
  });

  imageFileInput.addEventListener("change", () => {
    if (imageFileInput.files && imageFileInput.files.length > 0) {
      handleImageSelected(imageFileInput.files[0]);
    }
  });

  function handleImageSelected(file) {
    const validExtensions = [".jpg", ".jpeg", ".png", ".webp"];
    const ext = "." + file.name.split(".").pop().toLowerCase();

    if (!validExtensions.includes(ext)) {
      alert("Unsupported file format for Image Analysis. Supported: JPEG, PNG, WebP.");
      return;
    }

    const maxSizeBytes = 20 * 1024 * 1024; // 20 MB backend config
    if (file.size > maxSizeBytes) {
      alert("Image file size exceeds maximum allowed limit of 20 MB.");
      return;
    }

    selectedImageFile = file;
    imagePreviewFilename.textContent = file.name;
    imagePreviewFilesize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";
    imagePreviewDimensions.textContent = "Loading preview...";

    // Generate local preview thumbnail and extract dimensions
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      imagePreviewThumb.src = dataUrl;

      const img = new Image();
      img.onload = () => {
        imagePreviewDimensions.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);

    imagePreviewCard.style.display = "flex";
    imageDropArea.style.display = "none";
    analyzeImageBtn.disabled = false;
  }

  clearImageBtn.addEventListener("click", () => {
    selectedImageFile = null;
    imageFileInput.value = "";
    imagePreviewThumb.src = "";
    imagePreviewCard.style.display = "none";
    imageDropArea.style.display = "block";
    analyzeImageBtn.disabled = true;
    if (currentMode === "image") {
      emptyState.style.display = "block";
      resultsContent.style.display = "none";
    }
  });

  // 5. Video Deepfake Prediction Submission
  analyzeVideoBtn.addEventListener("click", async () => {
    if (!selectedVideoFile) return;

    analyzeVideoBtn.disabled = true;
    explainBtn.disabled = true;
    analyzeVideoSpinner.style.display = "inline-block";
    progressContainer.style.display = "block";
    progressPercent.textContent = "40%";
    progressFill.style.width = "40%";
    progressLabel.textContent = "Sampling 8 uniform frames & running model inference...";

    const formData = new FormData();
    formData.append("file", selectedVideoFile);

    try {
      const res = await fetch("/api/v1/predict", {
        method: "POST",
        body: formData,
      });

      progressPercent.textContent = "100%";
      progressFill.style.width = "100%";

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Prediction request failed with status " + res.status);
      }

      displayVideoPredictionResult(data);
    } catch (err) {
      alert("Video analysis failed: " + err.message);
    } finally {
      analyzeVideoBtn.disabled = false;
      explainBtn.disabled = false;
      analyzeVideoSpinner.style.display = "none";
      setTimeout(() => {
        progressContainer.style.display = "none";
      }, 500);
    }
  });

  function displayVideoPredictionResult(data) {
    emptyState.style.display = "none";
    resultsContent.style.display = "block";

    verdictTime.textContent = new Date().toLocaleTimeString();

    if (data.predicted_label === "MANIPULATED") {
      verdictBadge.className = "verdict-badge verdict-manipulated";
      verdictLabel.textContent = "MANIPULATED (DEEPFAKE)";
      verdictSublabel.textContent = "High spatial-temporal facial synthesis inconsistency detected";
    } else {
      verdictBadge.className = "verdict-badge verdict-authentic";
      verdictLabel.textContent = "AUTHENTIC (REAL)";
      verdictSublabel.textContent = "No prominent deepfake synthesis signatures detected";
    }

    scoreValue.textContent = Number(data.fake_score).toFixed(4);
    const pct = Math.max(0, Math.min(100, data.fake_score * 100));
    scoreBarFill.style.width = pct + "%";

    metaRequestId.textContent = data.request_id || "-";
    metaTime.textContent = (data.processing_time_ms ? data.processing_time_ms + " ms" : "-");
    metaTensor.textContent = JSON.stringify(data.frame_tensor_shape || [8, 128, 128, 3]);
    metaHash.textContent = (data.checkpoint_sha256 ? data.checkpoint_sha256.substring(0, 16) + "..." : "-");
    metaCalibration.textContent = data.score_calibration || "Uncalibrated sigmoid output.";
  }

  // 6. Image Deepfake Prediction Submission
  analyzeImageBtn.addEventListener("click", async () => {
    if (!selectedImageFile) return;

    analyzeImageBtn.disabled = true;
    analyzeImageSpinner.style.display = "inline-block";
    progressContainer.style.display = "block";
    progressPercent.textContent = "50%";
    progressFill.style.width = "50%";
    progressLabel.textContent = "Decoding image & running spatial deepfake inference...";

    const formData = new FormData();
    formData.append("file", selectedImageFile);

    try {
      const res = await fetch("/api/v1/predict/image", {
        method: "POST",
        body: formData,
      });

      progressPercent.textContent = "100%";
      progressFill.style.width = "100%";

      const data = await res.json();
      if (!res.ok) {
        if (res.status === 503) {
          // Model unconfigured/unavailable - display informative notice
          displayImageModelUnavailable(data.detail || "Image model checkpoint is not configured.");
          return;
        }
        throw new Error(data.detail || "Image prediction request failed with status " + res.status);
      }

      displayImagePredictionResult(data);
    } catch (err) {
      alert("Image analysis failed: " + err.message);
    } finally {
      analyzeImageBtn.disabled = false;
      analyzeImageSpinner.style.display = "none";
      setTimeout(() => {
        progressContainer.style.display = "none";
      }, 500);
    }
  });

  function displayImageModelUnavailable(detailMessage) {
    emptyState.style.display = "none";
    resultsContent.style.display = "block";

    verdictTime.textContent = new Date().toLocaleTimeString();
    verdictBadge.className = "verdict-badge verdict-warning";
    verdictLabel.textContent = "INFERENCE UNAVAILABLE (HTTP 503)";
    verdictSublabel.textContent = "Trained image classifier checkpoint required";

    scoreValue.textContent = "N/A";
    scoreBarFill.style.width = "0%";

    metaRequestId.textContent = "image-probe-" + Date.now();
    metaTime.textContent = "-";
    metaTensor.textContent = selectedImageFile ? selectedImageFile.name : "[128, 128, 3]";
    metaHash.textContent = "Checkpoint Unconfigured";
    metaCalibration.textContent = (
      "DEEPGUARD strictly prevents substituting the video temporal sequence model for single images. " +
      detailMessage
    );
  }

  function displayImagePredictionResult(data) {
    emptyState.style.display = "none";
    resultsContent.style.display = "block";

    verdictTime.textContent = new Date().toLocaleTimeString();

    if (data.predicted_label === "MANIPULATED") {
      verdictBadge.className = "verdict-badge verdict-manipulated";
      verdictLabel.textContent = "MANIPULATED (DEEPFAKE)";
      verdictSublabel.textContent = "Spatial synthesis artifacts detected in image";
    } else {
      verdictBadge.className = "verdict-badge verdict-authentic";
      verdictLabel.textContent = "AUTHENTIC (REAL)";
      verdictSublabel.textContent = "No prominent deepfake manipulation signatures detected";
    }

    scoreValue.textContent = Number(data.fake_score).toFixed(4);
    const pct = Math.max(0, Math.min(100, data.fake_score * 100));
    scoreBarFill.style.width = pct + "%";

    metaRequestId.textContent = data.request_id || "-";
    metaTime.textContent = (data.processing_time_ms ? data.processing_time_ms + " ms" : "-");
    metaTensor.textContent = JSON.stringify(data.image_dimensions || [128, 128, 3]);
    metaHash.textContent = (data.checkpoint_sha256 ? data.checkpoint_sha256.substring(0, 16) + "..." : "unverified");
    metaCalibration.textContent = data.score_calibration || "Uncalibrated raw sigmoid score.";
  }

  // 7. Video Grad-CAM Explainability Submission
  explainBtn.addEventListener("click", async () => {
    if (!selectedVideoFile) return;

    explainBtn.disabled = true;
    analyzeVideoBtn.disabled = true;
    explainSpinner.style.display = "inline-block";
    progressContainer.style.display = "block";
    progressPercent.textContent = "50%";
    progressFill.style.width = "50%";
    progressLabel.textContent = "Computing Grad-CAM gradients on Conv_1 layer...";

    const formData = new FormData();
    formData.append("file", selectedVideoFile);

    try {
      const res = await fetch("/api/v1/explain", {
        method: "POST",
        body: formData,
      });

      progressPercent.textContent = "100%";
      progressFill.style.width = "100%";

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Explainability request failed with status " + res.status);
      }

      displayExplanationResult(data);
    } catch (err) {
      alert("Explainability generation failed: " + err.message);
    } finally {
      explainBtn.disabled = false;
      analyzeVideoBtn.disabled = false;
      explainSpinner.style.display = "none";
      setTimeout(() => {
        progressContainer.style.display = "none";
      }, 500);
    }
  });

  function displayExplanationResult(data) {
    explainSection.style.display = "block";
    overlaysGrid.innerHTML = "";

    if (!data.explanations || data.explanations.length === 0) {
      overlaysGrid.innerHTML = "<p>No explanation heatmaps generated.</p>";
      return;
    }

    data.explanations.forEach((item) => {
      const card = document.createElement("div");
      card.className = "overlay-card";

      const titleMap = { 0: "Early Frame (Index 0)", 3: "Mid Frame (Index 3)", 7: "Late Frame (Index 7)" };
      const frameTitle = titleMap[item.frame_index] || `Frame Index ${item.frame_index}`;

      card.innerHTML = `
        <div class="overlay-image-wrapper">
          <img src="${item.overlay_url}" alt="Grad-CAM Frame ${item.frame_index}" class="overlay-image">
        </div>
        <div class="overlay-details">
          <div class="overlay-frame-title">
            <span>${frameTitle}</span>
            <span class="tag">Activation: ${(item.mean_activation_intensity * 100).toFixed(1)}%</span>
          </div>
          <div class="overlay-stats">
            <span>Peak Focal Coord: (${item.peak_activation_coords.join(", ")})</span>
            <span>Score: ${Number(item.predicted_score).toFixed(4)}</span>
          </div>
        </div>
      `;
      overlaysGrid.appendChild(card);
    });

    explainSection.scrollIntoView({ behavior: "smooth" });
  }
});
