"""
File security, upload validation, magic byte checking, and sanitization for DeepGuard.

Implements:
1. Bounded chunk streaming to prevent memory exhaustion (DoS mitigation).
2. Video container magic byte verification (rejects masked/corrupted files).
3. OpenCV video stream decoding validation (checks frame count, duration, spatial dimensions).
4. Secure random filenames (UUID4) to eliminate path traversal risks.
5. Safe temporary file cleanup on success, exceptions, and timeouts.
6. Safe artifact directory resolution preventing directory traversal.
"""

import os
from pathlib import Path
import re
from typing import Optional, Set, Tuple
import uuid
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from .config import settings


# Known video container magic byte signatures
VIDEO_SIGNATURES = [
    # MP4 / QuickTime / ISO Base Media (bytes 4-8 'ftyp', 'moov', 'mdat')
    lambda b: len(b) >= 12 and b[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"skip"),
    # AVI / RIFF (bytes 0-4 'RIFF' and bytes 8-12 'AVI ')
    lambda b: len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"AVI ",
    # WebM / Matroska MKV (EBML signature \x1a\x45\xdf\xa3)
    lambda b: len(b) >= 4 and b[:4] == b"\x1a\x45\xdf\xa3",
    # MPEG Video (\x00\x00\x01\xba or \x00\x00\x01\xb3)
    lambda b: len(b) >= 4 and b[:4] in (b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"),
    # FLV
    lambda b: len(b) >= 3 and b[:3] == b"FLV",
]

# Known image container magic byte signatures
IMAGE_SIGNATURES = [
    # JPEG (SOI marker \xff\xd8\xff)
    lambda b: len(b) >= 3 and b[:3] == b"\xff\xd8\xff",
    # PNG (\x89PNG\r\n\x1a\n)
    lambda b: len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n",
    # WebP (RIFF....WEBP)
    lambda b: len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP",
]


def is_valid_video_header(header_bytes: bytes) -> bool:
    """Check if initial bytes match a recognized video container signature."""
    if not header_bytes or len(header_bytes) < 4:
        return False
    return any(check(header_bytes) for check in VIDEO_SIGNATURES)


def is_valid_image_header(header_bytes: bytes) -> bool:
    """Check if initial bytes match a recognized image container signature (JPEG, PNG, WebP)."""
    if not header_bytes or len(header_bytes) < 3:
        return False
    return any(check(header_bytes) for check in IMAGE_SIGNATURES)


def sanitize_extension(filename: Optional[str]) -> str:
    """Extract and validate file extension."""
    if not filename:
        return ".mp4"
    ext = Path(filename).suffix.lower()
    return ext if ext in settings.allowed_extensions else ".mp4"


def sanitize_image_extension(filename: Optional[str]) -> str:
    """Extract and validate image file extension."""
    if not filename:
        return ".jpg"
    ext = Path(filename).suffix.lower()
    return ext if ext in settings.allowed_image_extensions else ".jpg"


def safe_cleanup(filepath: Optional[str]) -> None:
    """Safely remove a file if it exists, suppressing exceptions."""
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass


async def save_bounded_upload(
    upload_file: UploadFile,
    dest_dir: str,
    max_bytes: int,
) -> Tuple[str, str]:
    """
    Stream upload into destination directory with strict size and magic byte bounding.

    Args:
        upload_file: FastAPI UploadFile object.
        dest_dir: Directory where the safe temporary file is created.
        max_bytes: Maximum allowed bytes before aborting upload.

    Returns:
        Tuple of (saved_file_path, original_filename)

    Raises:
        HTTPException: If file is empty, exceeds max_bytes, or fails header check.
    """
    os.makedirs(dest_dir, exist_ok=True)
    original_name = upload_file.filename or "unknown.mp4"
    ext = sanitize_extension(original_name)

    # Secure random UUID4 filename to prevent path traversal
    safe_name = f"upload_{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(dest_dir, safe_name)

    total_bytes = 0
    header_checked = False
    chunk_size = 64 * 1024  # 64 KB streaming buffer

    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break

                # Verify container magic bytes on first chunk
                if not header_checked:
                    if not is_valid_video_header(chunk[:32]):
                        raise HTTPException(
                            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail=(
                                f"Uploaded file '{original_name}' is not a recognized or supported video container. "
                                "Supported formats: MP4, AVI, MOV, MKV, WebM."
                            ),
                        )
                    header_checked = True

                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File size exceeds maximum allowed limit of "
                            f"{max_bytes // (1024 * 1024)} MB."
                        ),
                    )

                f.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

    except Exception:
        safe_cleanup(dest_path)
        raise

    return dest_path, original_name


async def save_bounded_image_upload(
    upload_file: UploadFile,
    dest_dir: str,
    max_bytes: int,
) -> Tuple[str, str]:
    """
    Stream image upload into destination directory with strict size and magic byte bounding.

    Args:
        upload_file: FastAPI UploadFile object.
        dest_dir: Directory where the safe temporary file is created.
        max_bytes: Maximum allowed bytes before aborting upload.

    Returns:
        Tuple of (saved_file_path, original_filename)

    Raises:
        HTTPException: If file is empty, exceeds max_bytes, or fails header check.
    """
    os.makedirs(dest_dir, exist_ok=True)
    original_name = upload_file.filename or "unknown.jpg"
    ext = sanitize_image_extension(original_name)

    # Secure random UUID4 filename to prevent path traversal
    safe_name = f"upload_img_{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(dest_dir, safe_name)

    total_bytes = 0
    header_checked = False
    chunk_size = 64 * 1024  # 64 KB streaming buffer

    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break

                # Verify container magic bytes on first chunk
                if not header_checked:
                    if not is_valid_image_header(chunk[:32]):
                        raise HTTPException(
                            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail=(
                                f"Uploaded file '{original_name}' is not a recognized or supported image format. "
                                "Supported formats: JPEG, PNG, WebP."
                            ),
                        )
                    header_checked = True

                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Image file size exceeds maximum allowed limit of "
                            f"{max_bytes // (1024 * 1024)} MB."
                        ),
                    )

                f.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file is empty (0 bytes).",
            )

    except Exception:
        safe_cleanup(dest_path)
        raise

    return dest_path, original_name


def validate_decoded_image(
    image_path: str,
    max_dimension: int = 8192,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Validate that OpenCV can decode the image file and that properties meet constraints.
    Applies EXIF orientation handling and validates dimensions against decompression bomb limits.

    Returns:
        Tuple of (rgb_image_array, (height, width, channels))

    Raises:
        HTTPException: If image is unreadable, corrupted, or has invalid/excessive dimensions.
    """
    # cv2.imread with IMREAD_COLOR respects EXIF orientation by default in OpenCV 4.x
    bgr_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr_img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to decode image. The file appears corrupted, incomplete, or unreadable.",
        )

    if bgr_img.ndim != 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image has invalid tensor rank: {bgr_img.ndim}. Expected 3 channels.",
        )

    height, width, channels = bgr_img.shape

    if height <= 0 or width <= 0 or channels != 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image has invalid spatial dimensions ({width}x{height}, {channels} channels).",
        )

    if width > max_dimension or height > max_dimension:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image dimensions ({width}x{height}) exceed maximum allowed limit of "
                f"{max_dimension}x{max_dimension} pixels (decompression safety guard)."
            ),
        )

    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img, (height, width, channels)


def validate_decoded_video(
    video_path: str,
    max_duration_seconds: float = 180.0,
    min_required_frames: int = 8,
) -> Tuple[int, float, Tuple[int, int]]:
    """
    Validate that OpenCV can decode the video stream and that properties meet constraints.

    Returns:
        Tuple of (total_frames, duration_seconds, (width, height))

    Raises:
        HTTPException: If video is unreadable, corrupted, too short, or too long.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to decode video stream. The video file appears corrupted or unreadable.",
        )

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Check minimal dimensions
        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Video has invalid or zero spatial dimensions.",
            )

        # Check frame count
        if total_frames < min_required_frames:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Video has only {total_frames} frames. "
                    f"DeepGuard requires at least {min_required_frames} frames for temporal sequence analysis."
                ),
            )

        # Estimate duration
        duration = (total_frames / fps) if fps > 0 else 0.0
        if duration > max_duration_seconds:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Video duration ({duration:.1f}s) exceeds maximum allowed limit of "
                    f"{max_duration_seconds:.0f} seconds."
                ),
            )

        return total_frames, duration, (width, height)

    finally:
        cap.release()


def resolve_safe_artifact_path(artifact_filename: str, base_dir: str) -> str:
    """
    Verify that an artifact filename resolves safely within the designated artifact directory.

    Guards against path traversal attacks (e.g. '../', '..\\').
    """
    # Strict regex check on artifact filename (only alphanumeric, underscores, hyphens, and safe extension)
    if not re.match(r"^[a-zA-Z0-9_\-]+\.(png|jpg|jpeg|json|csv)$", artifact_filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact identifier format.",
        )

    base_path = Path(base_dir).resolve()
    resolved = (base_path / artifact_filename).resolve()

    # Ensure resolved path is strictly within base directory
    try:
        resolved.relative_to(base_path)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to artifact path is forbidden.",
        )

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requested explanation artifact was not found.",
        )

    return str(resolved)
