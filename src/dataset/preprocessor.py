"""
Baseline video preprocessing pipeline for DeepGuard.

Specifications derived from Phase 1 Specification (PHASE1_SPECIFICATION.md):
- Sequence length: Exactly 8 uniform frames (T = 8)
- Video reader: OpenCV cv2.VideoCapture
- Face detection: Bounding box with +20% margin & forward-fill tracking
- Resolution: Resized to 128 x 128 RGB
- Normalization: Pixel intensities divided by 255.0 to [0.0, 1.0]
- Explicit error handling: Corrupted files and missing faces flagged with explicit status codes.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .face_detector import BaseFaceDetector, HaarCascadeFaceDetector, SyntheticFaceDetector


TARGET_SEQUENCE_LENGTH = 8
TARGET_RESOLUTION = (128, 128)  # (width, height)
TARGET_CHANNELS = 3


@dataclass
class PreprocessingResult:
    """Outcome of preprocessing a single video file."""

    sample_id: str
    video_path: str
    status: str  # 'SUCCESS', 'FAILED_UNREADABLE', 'FAILED_INSUFFICIENT_FRAMES', 'FAILED_NO_FACE_DETECTED'
    frames: Optional[np.ndarray] = None  # Shape (8, 128, 128, 3) float32 in [0.0, 1.0] if SUCCESS
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """True if preprocessing succeeded and frames tensor is valid."""
        return self.status == "SUCCESS" and self.frames is not None


class VideoPreprocessor:
    """
    Executes the baseline 6-step video preprocessing pipeline defined in Phase 1.
    """

    def __init__(
        self,
        face_detector: Optional[BaseFaceDetector] = None,
        sequence_length: int = TARGET_SEQUENCE_LENGTH,
        resolution: Tuple[int, int] = TARGET_RESOLUTION,
        margin_ratio: float = 0.20,
    ):
        """
        Initialize the preprocessor.

        Args:
            face_detector: Detector instance implementing BaseFaceDetector.
                           If None, defaults to HaarCascadeFaceDetector.
            sequence_length: Number of uniformly sampled frames (default 8).
            resolution: Target frame crop (width, height), default (128, 128).
            margin_ratio: Context expansion around face bounding box (default 0.20 = +20%).
        """
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.margin_ratio = margin_ratio
        self.face_detector = face_detector or HaarCascadeFaceDetector()

    @staticmethod
    def calculate_uniform_indices(total_frames: int, num_samples: int = 8) -> List[int]:
        """
        Compute equidistant frame indices across video duration.
        Formula from Phase 1 Spec: indices = [int(i * total_frames / num_samples) for i in range(num_samples)]
        """
        if total_frames < num_samples:
            return []
        indices = [int(i * total_frames / num_samples) for i in range(num_samples)]
        # Clamp to ensure strict boundary validity
        return [min(idx, total_frames - 1) for idx in indices]

    def _apply_margin(
        self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Expand bounding box by margin_ratio (+20%) on all sides, clamped to image borders.

        Args:
            bbox: (x, y, w, h)
            frame_shape: (height, width)

        Returns:
            (x1, y1, x2, y2) pixel coordinates
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]

        margin_w = int(self.margin_ratio * w)
        margin_h = int(self.margin_ratio * h)

        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(frame_w, x + w + margin_w)
        y2 = min(frame_h, y + h + margin_h)

        return x1, y1, x2, y2

    def process_video(self, video_path: str, sample_id: str = "") -> PreprocessingResult:
        """
        Process a single video into a normalized (8, 128, 128, 3) tensor.

        Execution steps:
        1. Open video with cv2.VideoCapture; check file readability and frame count.
        2. Uniform frame sampling: compute 8 equidistant frame indices.
        3. Face detection with +20% margin and forward-fill tracking.
        4. Crop and resize to 128x128.
        5. Pixel normalization to [0.0, 1.0] float32 RGB.
        """
        if not os.path.exists(video_path):
            return PreprocessingResult(
                sample_id=sample_id,
                video_path=video_path,
                status="FAILED_UNREADABLE",
                error_message=f"Video file does not exist on disk: {video_path}",
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return PreprocessingResult(
                sample_id=sample_id,
                video_path=video_path,
                status="FAILED_UNREADABLE",
                error_message="OpenCV could not open video stream (corrupted or unreadable format).",
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if total_frames < self.sequence_length:
            cap.release()
            return PreprocessingResult(
                sample_id=sample_id,
                video_path=video_path,
                status="FAILED_INSUFFICIENT_FRAMES",
                error_message=(
                    f"Video has only {total_frames} frames; minimum required is {self.sequence_length}."
                ),
                metadata={"total_frames": total_frames, "fps": fps},
            )

        indices = self.calculate_uniform_indices(total_frames, self.sequence_length)

        # Read the sampled frames
        sampled_bgr_frames: List[np.ndarray] = []
        for target_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                # Fallback: sequential read if seek failed
                break
            sampled_bgr_frames.append(frame)

        cap.release()

        # If seeking failed or yielded fewer frames, attempt full sequential stream read
        if len(sampled_bgr_frames) != self.sequence_length:
            cap_seq = cv2.VideoCapture(video_path)
            all_frames: List[np.ndarray] = []
            while True:
                ret, frame = cap_seq.read()
                if not ret or frame is None:
                    break
                all_frames.append(frame)
            cap_seq.release()

            if len(all_frames) < self.sequence_length:
                return PreprocessingResult(
                    sample_id=sample_id,
                    video_path=video_path,
                    status="FAILED_INSUFFICIENT_FRAMES",
                    error_message=(
                        f"Read only {len(all_frames)} readable frames from video (expected at least {self.sequence_length})."
                    ),
                    metadata={"total_frames": len(all_frames)},
                )
            indices = self.calculate_uniform_indices(len(all_frames), self.sequence_length)
            sampled_bgr_frames = [all_frames[idx] for idx in indices]

        # Step 3 & 4: Face detection, forward-fill tracking, cropping, and resizing
        cropped_frames_rgb: List[np.ndarray] = []
        last_bbox: Optional[Tuple[int, int, int, int]] = None
        bounding_boxes: List[Optional[Tuple[int, int, int, int]]] = []

        # First pass: check if any frame has a detectable face
        detected_boxes: List[Optional[Tuple[int, int, int, int]]] = [
            self.face_detector.detect(f) for f in sampled_bgr_frames
        ]

        # Check if entire video has zero face detections
        if all(b is None for b in detected_boxes):
            return PreprocessingResult(
                sample_id=sample_id,
                video_path=video_path,
                status="FAILED_NO_FACE_DETECTED",
                error_message="Face detector did not find any face in the sampled frames.",
                metadata={"sampled_indices": indices, "total_frames": total_frames},
            )

        # Find the first available detection to initialize tracking if frame 0 missed
        first_valid_box = next((b for b in detected_boxes if b is not None), None)

        for frame_bgr, detected_box in zip(sampled_bgr_frames, detected_boxes):
            if detected_box is not None:
                current_box = detected_box
                last_bbox = detected_box
            elif last_bbox is not None:
                # Forward-fill tracking from previous frame
                current_box = last_bbox
            else:
                # Back-fill from first valid future frame
                current_box = first_valid_box
                last_bbox = first_valid_box

            bounding_boxes.append(current_box)

            # Apply +20% margin
            x1, y1, x2, y2 = self._apply_margin(current_box, frame_bgr.shape[:2])
            face_crop_bgr = frame_bgr[y1:y2, x1:x2]

            # In case of degraded box or empty crop, fallback to full frame
            if face_crop_bgr.size == 0:
                face_crop_bgr = frame_bgr

            # Resize to target resolution (128, 128)
            face_crop_resized = cv2.resize(
                face_crop_bgr, self.resolution, interpolation=cv2.INTER_LINEAR
            )

            # Convert BGR to RGB
            face_crop_rgb = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB)
            cropped_frames_rgb.append(face_crop_rgb)

        # Step 5: Pixel normalization: x' = x / 255.0 to [0.0, 1.0] float32
        tensor = np.stack(cropped_frames_rgb, axis=0).astype(np.float32) / 255.0

        return PreprocessingResult(
            sample_id=sample_id,
            video_path=video_path,
            status="SUCCESS",
            frames=tensor,
            metadata={
                "total_frames": total_frames,
                "sampled_indices": indices,
                "bounding_boxes": bounding_boxes,
                "tensor_shape": list(tensor.shape),
                "tensor_dtype": str(tensor.dtype),
                "min_pixel_value": float(np.min(tensor)),
                "max_pixel_value": float(np.max(tensor)),
            },
        )
