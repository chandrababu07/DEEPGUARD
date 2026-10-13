/**
 * DeepGuard Web Application Client Script.
 * Handles drag-and-drop, API communication, readiness checks, and result rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const readinessBadge = document.getElementById("readiness-badge");
  const readinessText = document.getElementById("readiness-text");
  const modelStatusAlert = document.getElementById("model-status-alert");
  const alertTitle = document.getElementById("alert-title");
  const alertMessage = document.getElementById("alert-message");

  const dropArea = document.getElementById("drop-area");
  const fileInput = document.getElementById("video-file-input");
  const browseBtn = document.getElementById("browse-btn");
  const filePreviewCard = document.getElementById("file-preview-card");
  const previewFilename = document.getElementById("preview-filename");
  const previewFilesize = document.getElementById("preview-filesize");
  const clearFileBtn = document.getElementById("clear-file-btn");

  const analyzeBtn = document.getElementById("analyze-btn");
  const explainBtn = document.getElementById("explain-btn");
  const analyzeSpinner = document.getElementById("analyze-spinner");
  const explainSpinner = document.getElementById("explain-spinner");

  const progressContainer = document.getElementById("progress-container");
  const progressFill = document.getElementById("progress-fill");
  const progressPercent = document.getElementById("progress-percent");
  const progressLabel = document.getElementById("progress-label");

  const emptyState = document.getElementById("empty-state");
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

  let selectedFile = null;

  // 1. Initial Readiness Check
  async function checkReadiness() {
    try {
      const res = await fetch("/api/v1/ready");
      if (!res.ok) throw new Error("Service probe returned HTTP " + res.status);
      const data = await res.json();

      if (data.status === "ready") {
        readinessBadge.className = "badge badge-success";
        readinessText.textContent = "Model Online (Ready)";
        modelStatusAlert.style.display = "none";
      } else {
        readinessBadge.className = "badge badge-warning";
        readinessText.textContent = "Model Preview Mode";
        modelStatusAlert.style.display = "flex";
        alertTitle.textContent = "Inference Offline in Current Runtime";
        alertMessage.textContent = data.error || (
          "TensorFlow/GPU runtime is required for live inference. " +
          (data.warnings ? data.warnings.join(" ") : "")
        );
      }
    } catch (err) {
      readinessBadge.className = "badge badge-danger";
      readinessText.textContent = "Service Unreachable";
      modelStatusAlert.style.display = "flex";
      alertTitle.textContent = "API Connection Error";
      alertMessage.textContent = "Could not communicate with the DeepGuard backend server.";
    }
  }
  checkReadiness();

  // 2. File Selection & Drag-and-Drop
  browseBtn.addEventListener("click", () => fileInput.click());
  dropArea.addEventListener("click", () => fileInput.click());

  dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("drag-over");
  });

  dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("drag-over");
  });

  dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  function handleFileSelected(file) {
    const validExtensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
    const ext = "." + file.name.split(".").pop().toLowerCase();

    if (!validExtensions.includes(ext)) {
      alert("Unsupported file format. Please upload an MP4, AVI, MOV, MKV, or WebM video.");
      return;
    }

    const maxSizeBytes = 100 * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      alert("File size exceeds 100 MB limit.");
      return;
    }

    selectedFile = file;
    previewFilename.textContent = file.name;
    previewFilesize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";

    filePreviewCard.style.display = "flex";
    dropArea.style.display = "none";
    analyzeBtn.disabled = false;
    explainBtn.disabled = false;
  }

  clearFileBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    filePreviewCard.style.display = "none";
    dropArea.style.display = "block";
    analyzeBtn.disabled = true;
    explainBtn.disabled = true;
    emptyState.style.display = "block";
    resultsContent.style.display = "none";
    explainSection.style.display = "none";
  });

  // 3. Prediction Submission
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    explainBtn.disabled = true;
    analyzeSpinner.style.display = "inline-block";
    progressContainer.style.display = "block";
    progressPercent.textContent = "40%";
    progressFill.style.width = "40%";
    progressLabel.textContent = "Sampling 8 uniform frames & running model inference...";

    const formData = new FormData();
    formData.append("file", selectedFile);

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

      displayPredictionResult(data);

    } catch (err) {
      alert("Analysis failed: " + err.message);
    } finally {
      analyzeBtn.disabled = false;
      explainBtn.disabled = false;
      analyzeSpinner.style.display = "none";
      setTimeout(() => {
        progressContainer.style.display = "none";
      }, 500);
    }
  });

  function displayPredictionResult(data) {
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

  // 4. Grad-CAM Explainability Submission
  explainBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    explainBtn.disabled = true;
    analyzeBtn.disabled = true;
    explainSpinner.style.display = "inline-block";
    progressContainer.style.display = "block";
    progressPercent.textContent = "50%";
    progressFill.style.width = "50%";
    progressLabel.textContent = "Computing Grad-CAM gradients on Conv_1 layer...";

    const formData = new FormData();
    formData.append("file", selectedFile);

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
      analyzeBtn.disabled = false;
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
