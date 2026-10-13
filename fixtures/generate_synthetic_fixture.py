"""
Synthetic test fixture generator for DeepGuard.

Generates a tiny, self-contained synthetic video dataset and CSV manifest for
testing manifest validation, dataset splitting, and the preprocessing pipeline
without requiring external dataset downloads or real deepfake videos.
"""

import os
import sys
from typing import Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from src.dataset.manifest import ManifestManager, ManifestRecord


def create_synthetic_face_video(
    output_path: str,
    num_frames: int = 12,
    resolution: Tuple[int, int] = (160, 160),
    draw_face: bool = True,
    fps: int = 10,
) -> None:
    """
    Generate a tiny synthetic video file.

    Args:
        output_path: Path to write the video file.
        num_frames: Number of frames to generate.
        resolution: (width, height) in pixels.
        draw_face: If True, draws a distinct face-like pattern; if False, draws uniform noise.
        fps: Frames per second.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = resolution

    # Using 'mp4v' codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        # Fallback to MJPG or XVID if mp4v fails
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        # Base background: dark gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (30, 30, 35)

        if draw_face:
            # Face center with slight motion across frames
            center_x = width // 2 + int(np.sin(frame_idx / 2.0) * 8)
            center_y = height // 2 + int(np.cos(frame_idx / 2.0) * 4)

            # Head (skin-tone oval)
            cv2.ellipse(frame, (center_x, center_y), (35, 45), 0, 0, 360, (160, 190, 225), -1)
            # Eyes
            cv2.circle(frame, (center_x - 12, center_y - 10), 4, (50, 40, 30), -1)
            cv2.circle(frame, (center_x + 12, center_y - 10), 4, (50, 40, 30), -1)
            # Nose
            cv2.line(frame, (center_x, center_y - 4), (center_x, center_y + 8), (110, 130, 160), 2)
            # Mouth
            cv2.ellipse(frame, (center_x, center_y + 20), (14, 6), 0, 0, 180, (60, 60, 180), 2)

        writer.write(frame)

    writer.release()


def generate_synthetic_fixture(target_dir: str) -> str:
    """
    Generate all fixture videos and the corresponding synthetic_manifest.csv.

    Returns:
        Path to the generated synthetic_manifest.csv.
    """
    os.makedirs(target_dir, exist_ok=True)

    v1 = os.path.join(target_dir, "synth_real_01.mp4")
    v2 = os.path.join(target_dir, "synth_fake_01.mp4")
    v3 = os.path.join(target_dir, "synth_real_02.mp4")
    v4 = os.path.join(target_dir, "synth_noface.mp4")
    v5 = os.path.join(target_dir, "synth_corrupt.mp4")

    # 1. Authentic synthetic video (12 frames, with face)
    create_synthetic_face_video(v1, num_frames=12, draw_face=True)

    # 2. Manipulated synthetic video linked to v1 (12 frames, with face)
    create_synthetic_face_video(v2, num_frames=12, draw_face=True)

    # 3. Another authentic video (12 frames, with face)
    create_synthetic_face_video(v3, num_frames=12, draw_face=True)

    # 4. Video without any face (uniform background)
    create_synthetic_face_video(v4, num_frames=12, draw_face=False)

    # 5. Corrupted / 0-byte video
    with open(v5, "wb") as f:
        f.write(b"CORRUPTED_VIDEO_HEADER_NON_OPENCV_DATA")

    # Build synthetic manifest records with cross-platform forward-slash paths
    def to_rel(p: str) -> str:
        return os.path.relpath(p, start=os.path.dirname(target_dir)).replace("\\", "/")

    records = [
        ManifestRecord(
            sample_id="SYNTH_REAL_001",
            video_path=to_rel(v1),
            label=0,
            dataset_name="synthetic_fixture",
            split="train",
            manipulation_type="original",
            original_video_id="ACTOR_01",
            preprocessing_status="PENDING",
        ),
        ManifestRecord(
            sample_id="SYNTH_FAKE_001",
            video_path=to_rel(v2),
            label=1,
            dataset_name="synthetic_fixture",
            split="train",
            manipulation_type="synthetic_blend",
            original_video_id="ACTOR_01",  # Linked to ACTOR_01 to test leakage prevention
            preprocessing_status="PENDING",
        ),
        ManifestRecord(
            sample_id="SYNTH_REAL_002",
            video_path=to_rel(v3),
            label=0,
            dataset_name="synthetic_fixture",
            split="validation",
            manipulation_type="original",
            original_video_id="ACTOR_02",
            preprocessing_status="PENDING",
        ),
        ManifestRecord(
            sample_id="SYNTH_NOFACE_001",
            video_path=to_rel(v4),
            label=0,
            dataset_name="synthetic_fixture",
            split="test",
            manipulation_type="original",
            original_video_id="ACTOR_03",
            preprocessing_status="PENDING",
        ),
        ManifestRecord(
            sample_id="SYNTH_CORRUPT_001",
            video_path=to_rel(v5),
            label=1,
            dataset_name="synthetic_fixture",
            split="test",
            manipulation_type="synthetic_corrupt",
            original_video_id="ACTOR_04",
            preprocessing_status="PENDING",
        ),
    ]

    manifest_path = os.path.join(target_dir, "synthetic_manifest.csv")
    ManifestManager.write_manifest(records, manifest_path)
    return manifest_path


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_fixture_dir = os.path.join(current_dir, "synthetic")
    csv_file = generate_synthetic_fixture(target_fixture_dir)
    print(f"Generated synthetic fixture manifest at: {csv_file}")
