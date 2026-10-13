"""
Controlled spatial and compression perturbation library for DeepGuard.

Implements deterministic, reproducible image and frame perturbations:
1. JPEG Compression: Q90, Q70, Q50, Q30
2. Gaussian Blur: Mild (k=3, s=1.0), Moderate (k=5, s=2.0), Severe (k=7, s=3.0)
3. Downsampling (Spatial Rescaling): Scale 0.75, 0.50, 0.25 (rescaled back to 128x128)

Design Principles:
- Pure functions: Zero in-place mutation of input arrays.
- Preserves input tensor rank (8, 128, 128, 3), RGB channel ordering, and [0.0, 1.0] normalization.
- Handles decoding or encoding edge cases with explicit validation and informative errors.
- Never mutates ground truth labels or underlying video sources.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass(frozen=True)
class PerturbationConfig:
    """Specification of a deterministic perturbation condition."""

    name: str
    perturbation_type: str  # "jpeg", "gaussian_blur", "downsample"
    parameters: Dict[str, Any]
    severity_level: str  # "baseline", "mild", "moderate", "severe"
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_frame_tensor(frames: np.ndarray) -> np.ndarray:
    """
    Validates input video frame sequence tensor.

    Expects:
    - 4D shape: (8, 128, 128, 3)
    - Or 5D single-video batch: (1, 8, 128, 128, 3)
    - Normalized float in range [0.0, 1.0]

    Returns:
        Clean float32 numpy copy of shape (8, 128, 128, 3).
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 4:
        raise ValueError(
            f"Expected 4D sequence tensor (num_frames, H, W, C), received {arr.ndim}D tensor with shape {arr.shape}"
        )

    if arr.shape[1:] != (128, 128, 3):
        raise ValueError(
            f"Expected spatial resolution (128, 128, 3), but received {arr.shape[1:]}"
        )

    if arr.shape[0] != 8:
        raise ValueError(
            f"Expected sequence length of 8 frames, but received {arr.shape[0]} frames"
        )

    # Check for NaN / Inf
    if not np.all(np.isfinite(arr)):
        raise ValueError("Frame tensor contains NaN or Infinite values.")

    # Safe clipping to [0.0, 1.0] if float
    arr = np.clip(arr, 0.0, 1.0)
    return arr.copy()


def apply_jpeg_compression(
    frames: np.ndarray,
    quality: int = 70,
) -> np.ndarray:
    """
    Apply JPEG compression and decompression artifacts to each frame.

    Args:
        frames: Normalized RGB frame sequence of shape (8, 128, 128, 3) in [0.0, 1.0].
        quality: JPEG compression quality factor in range [1, 100].

    Returns:
        Perturbed float32 array of shape (8, 128, 128, 3) in [0.0, 1.0].
    """
    if not isinstance(quality, int) or not (1 <= quality <= 100):
        raise ValueError(f"JPEG quality must be an integer between 1 and 100, got {quality}")

    arr = validate_frame_tensor(frames)
    perturbed_frames = np.empty_like(arr)

    for i in range(arr.shape[0]):
        # Convert RGB float [0, 1] -> BGR uint8 [0, 255]
        frame_rgb_uint8 = np.clip(arr[i] * 255.0, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_rgb_uint8, cv2.COLOR_RGB2BGR)

        # Encode to JPEG in memory buffer
        success, encoded = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not success or encoded is None:
            raise RuntimeError(f"Failed to encode frame {i} with JPEG quality {quality}")

        # Decode JPEG buffer back to BGR image
        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            raise RuntimeError(f"Failed to decode JPEG frame {i} buffer")

        # Convert back to RGB float [0.0, 1.0]
        decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        perturbed_frames[i] = decoded_rgb.astype(np.float32) / 255.0

    return np.clip(perturbed_frames, 0.0, 1.0)


def apply_gaussian_blur(
    frames: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Apply 2D Gaussian blur filter across all frames.

    Args:
        frames: Normalized RGB frame sequence of shape (8, 128, 128, 3) in [0.0, 1.0].
        kernel_size: Odd positive integer representing filter kernel width and height.
        sigma: Standard deviation of Gaussian distribution along X and Y axes.

    Returns:
        Perturbed float32 array of shape (8, 128, 128, 3) in [0.0, 1.0].
    """
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"Gaussian blur kernel_size must be a positive odd integer (e.g. 3, 5, 7), got {kernel_size}"
        )
    if not isinstance(sigma, (int, float)) or sigma <= 0.0:
        raise ValueError(f"Gaussian blur sigma must be a positive number, got {sigma}")

    arr = validate_frame_tensor(frames)
    perturbed_frames = np.empty_like(arr)

    for i in range(arr.shape[0]):
        # Apply blur on float32 frame
        blurred = cv2.GaussianBlur(
            arr[i],
            (kernel_size, kernel_size),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT_101,
        )
        perturbed_frames[i] = blurred

    return np.clip(perturbed_frames, 0.0, 1.0)


def apply_downsampling(
    frames: np.ndarray,
    scale_factor: float = 0.5,
    interpolation_down: int = cv2.INTER_AREA,
    interpolation_up: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Apply spatial downsampling followed by restoration upsampling.
    Simulates transmission over bandwidth-constrained channels or low-resolution capture.

    Args:
        frames: Normalized RGB frame sequence of shape (8, 128, 128, 3) in [0.0, 1.0].
        scale_factor: Rescaling ratio in range (0.0, 1.0). E.g., 0.5 downsamples 128x128 to 64x64.
        interpolation_down: OpenCV interpolation method for downsampling (default: INTER_AREA).
        interpolation_up: OpenCV interpolation method for upsampling (default: INTER_LINEAR).

    Returns:
        Perturbed float32 array of shape (8, 128, 128, 3) in [0.0, 1.0].
    """
    if not isinstance(scale_factor, (int, float)) or not (0.0 < scale_factor < 1.0):
        raise ValueError(
            f"Downsampling scale_factor must be in range (0.0, 1.0), got {scale_factor}"
        )

    arr = validate_frame_tensor(frames)
    orig_h, orig_w = arr.shape[1], arr.shape[2]  # 128, 128
    target_w = max(4, int(round(orig_w * scale_factor)))
    target_h = max(4, int(round(orig_h * scale_factor)))

    perturbed_frames = np.empty_like(arr)

    for i in range(arr.shape[0]):
        frame = arr[i]
        # Downscale
        downscaled = cv2.resize(
            frame,
            (target_w, target_h),
            interpolation=interpolation_down,
        )
        # Upscale back to original input shape (128, 128)
        upscaled = cv2.resize(
            downscaled,
            (orig_w, orig_h),
            interpolation=interpolation_up,
        )
        perturbed_frames[i] = upscaled

    return np.clip(perturbed_frames, 0.0, 1.0)


def get_standard_phase5_perturbations() -> List[PerturbationConfig]:
    """
    Returns the standard suite of Phase 5 perturbation conditions.

    Conditions included:
    - JPEG: Q90 (mild), Q70 (moderate), Q50 (aggressive), Q30 (severe)
    - Gaussian Blur: k=3/s=1.0 (mild), k=5/s=2.0 (moderate), k=7/s=3.0 (severe)
    - Downsampling: 0.75 (mild, 96x96), 0.50 (moderate, 64x64), 0.25 (severe, 32x32)
    """
    return [
        # JPEG Compression Suite
        PerturbationConfig(
            name="jpeg_q90",
            perturbation_type="jpeg",
            parameters={"quality": 90},
            severity_level="mild",
            description="High-quality JPEG compression (Q=90)",
        ),
        PerturbationConfig(
            name="jpeg_q70",
            perturbation_type="jpeg",
            parameters={"quality": 70},
            severity_level="moderate",
            description="Standard web-quality JPEG compression (Q=70)",
        ),
        PerturbationConfig(
            name="jpeg_q50",
            perturbation_type="jpeg",
            parameters={"quality": 50},
            severity_level="moderate",
            description="Medium compression with visible block artifacts (Q=50)",
        ),
        PerturbationConfig(
            name="jpeg_q30",
            perturbation_type="jpeg",
            parameters={"quality": 30},
            severity_level="severe",
            description="Aggressive low-bitrate compression with heavy blocking (Q=30)",
        ),
        # Gaussian Blur Suite
        PerturbationConfig(
            name="blur_mild",
            perturbation_type="gaussian_blur",
            parameters={"kernel_size": 3, "sigma": 1.0},
            severity_level="mild",
            description="Mild optical defocus blur (kernel=3x3, sigma=1.0)",
        ),
        PerturbationConfig(
            name="blur_moderate",
            perturbation_type="gaussian_blur",
            parameters={"kernel_size": 5, "sigma": 2.0},
            severity_level="moderate",
            description="Moderate optical or motion blur (kernel=5x5, sigma=2.0)",
        ),
        PerturbationConfig(
            name="blur_severe",
            perturbation_type="gaussian_blur",
            parameters={"kernel_size": 7, "sigma": 3.0},
            severity_level="severe",
            description="Severe heavy blur attenuating high spatial frequencies (kernel=7x7, sigma=3.0)",
        ),
        # Spatial Downsampling Suite
        PerturbationConfig(
            name="downsample_s75",
            perturbation_type="downsample",
            parameters={"scale_factor": 0.75},
            severity_level="mild",
            description="Spatial downscaling to 75% resolution (96x96) and restored",
        ),
        PerturbationConfig(
            name="downsample_s50",
            perturbation_type="downsample",
            parameters={"scale_factor": 0.50},
            severity_level="moderate",
            description="Spatial downscaling to 50% resolution (64x64) and restored",
        ),
        PerturbationConfig(
            name="downsample_s25",
            perturbation_type="downsample",
            parameters={"scale_factor": 0.25},
            severity_level="severe",
            description="Severe downscaling to 25% resolution (32x32) and restored",
        ),
    ]


def apply_perturbation(
    frames: np.ndarray,
    config: PerturbationConfig,
) -> np.ndarray:
    """
    Applies perturbation according to a PerturbationConfig object.
    """
    p_type = config.perturbation_type.lower()
    params = config.parameters

    if p_type == "jpeg":
        return apply_jpeg_compression(frames, quality=params.get("quality", 70))
    elif p_type in ("gaussian_blur", "blur"):
        return apply_gaussian_blur(
            frames,
            kernel_size=params.get("kernel_size", 5),
            sigma=params.get("sigma", 2.0),
        )
    elif p_type in ("downsample", "downsampling"):
        return apply_downsampling(
            frames,
            scale_factor=params.get("scale_factor", 0.5),
        )
    else:
        raise ValueError(f"Unknown perturbation type: '{config.perturbation_type}'")
