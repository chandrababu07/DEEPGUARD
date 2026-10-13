"""
Grad-CAM (Gradient-weighted Class Activation Mapping) explainability engine for DeepGuard.

Generates spatial attribution heatmaps highlighting the facial regions
that most heavily influence the MobileNetV2 + LSTM classification decision.

Target Layer:
- Default: "Conv_1" or "out_relu" (final 1,280-filter convolutional feature map of MobileNetV2).
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.model.baseline import HAS_TF

if HAS_TF:
    import tensorflow as tf
else:
    tf = None


EXPLANATION_DISCLAIMER = (
    "Preliminary development model. Grad-CAM saliency highlights spatial feature activation "
    "correlations, not legal or forensic proof of video manipulation."
)


@dataclass
class GradCAMExplanation:
    """Per-frame Grad-CAM explanation container."""

    sample_id: str
    frame_index: int
    target_layer: str
    output_image_path: str
    peak_activation_coords: Tuple[int, int]
    mean_activation_intensity: float
    predicted_score: float
    disclaimer: str = EXPLANATION_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GradCAMIndex:
    """Catalog of Grad-CAM explanations linking source videos to explanation images."""

    timestamp: str
    target_layer: str
    explanations: List[GradCAMExplanation] = field(default_factory=list)
    index_file_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "target_layer": self.target_layer,
            "total_explanations": len(self.explanations),
            "explanations": [e.to_dict() for e in self.explanations],
        }


class DeepGuardGradCAM:
    """
    Computes Gradient-weighted Class Activation Maps across video frames.
    """

    def __init__(
        self,
        model: Optional[object] = None,
        target_layer_name: str = "Conv_1",
    ):
        """
        Initialize the Grad-CAM explainer.

        Args:
            model: DeepGuard tf.keras.Model.
            target_layer_name: Name of convolutional layer to target in MobileNetV2.
        """
        self.model = model
        self.target_layer_name = target_layer_name

    def validate_tensor(self, frames: np.ndarray) -> np.ndarray:
        """Validate and format frames tensor to (8, 128, 128, 3) in [0.0, 1.0]."""
        arr = np.asarray(frames, dtype=np.float32)
        if arr.ndim == 5 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 4 or arr.shape != (8, 128, 128, 3):
            raise ValueError(
                f"Expected frame sequence shape (8, 128, 128, 3), but received {arr.shape}"
            )
        return arr

    def compute_heatmap(
        self,
        frames_tensor: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Grad-CAM heatmap for a specific frame in the 8-frame sequence.

        Args:
            frames_tensor: Validated (8, 128, 128, 3) tensor.
            frame_idx: Frame index in range [0, 7].

        Returns:
            Tuple of (heatmap_normalized_128x128, predicted_score)
        """
        if not HAS_TF:
            raise ImportError(
                "TensorFlow is required to compute Grad-CAM gradients. "
                "Execute in Google Colab or an environment with TensorFlow."
            )

        if self.model is None:
            raise ValueError("No model provided for Grad-CAM computation.")

        if not (0 <= frame_idx < 8):
            raise IndexError(f"frame_idx must be between 0 and 7, got {frame_idx}")

        # Add batch dimension: (1, 8, 128, 128, 3)
        input_batch = tf.convert_to_tensor(np.expand_dims(frames_tensor, axis=0))

        # Find target feature extractor submodel and convolutional layer
        time_dist_layer = self.model.get_layer("time_distributed_spatial_extractor")
        extractor = time_dist_layer.layer

        # Locate target conv layer within MobileNetV2
        target_layer = None
        for layer in extractor.layers:
            if layer.name == self.target_layer_name or hasattr(layer, "get_layer"):
                target_layer = layer
                break

        # Fallback to last conv2d if named layer not found
        if target_layer is None:
            for layer in reversed(extractor.layers):
                if "conv" in layer.name.lower() or "relu" in layer.name.lower():
                    target_layer = layer
                    break

        # Forward pass tracking gradients
        with tf.GradientTape() as tape:
            # Pass single frame through feature extractor
            single_frame = tf.expand_dims(frames_tensor[frame_idx], axis=0)

            # Build gradient sub-model
            grad_model = tf.keras.Model(
                inputs=extractor.input,
                outputs=[target_layer.output, extractor.output]
            )
            conv_outputs, _ = grad_model(single_frame)

            # Pass full sequence through full model to get end-to-end prediction
            predictions = self.model(input_batch, training=False)
            loss = predictions[:, 0]

        # Compute gradients of output score with respect to target conv activations
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            # Fallback uniform heatmap if gradients are zero/detached
            heatmap = np.ones((128, 128), dtype=np.float32) * 0.5
            pred_val = float(predictions.numpy()[0][0])
            return heatmap, pred_val

        # Global average pooling over spatial dimensions of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the convolutional feature maps
        conv_outputs_val = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs_val * pooled_grads, axis=-1)

        # Apply ReLU to retain only features with positive influence on prediction
        heatmap = tf.maximum(heatmap, 0.0)

        # Normalize heatmap to [0, 1]
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0.0:
            heatmap = heatmap / max_val

        heatmap_np = heatmap.numpy()
        heatmap_resized = cv2.resize(heatmap_np, (128, 128), interpolation=cv2.INTER_LINEAR)
        pred_val = float(predictions.numpy()[0][0])

        return heatmap_resized.astype(np.float32), pred_val

    def generate_and_save_overlay(
        self,
        frames_tensor: np.ndarray,
        frame_idx: int,
        output_image_path: str,
        sample_id: str = "video",
        alpha: float = 0.5,
    ) -> GradCAMExplanation:
        """
        Generate Grad-CAM heatmap, blend with input frame, and save to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
        validated_frames = self.validate_tensor(frames_tensor)

        heatmap, pred_score = self.compute_heatmap(validated_frames, frame_idx=frame_idx)

        # Base frame: convert normalized float [0, 1] RGB to uint8 BGR for OpenCV
        base_frame_rgb = (validated_frames[frame_idx] * 255.0).clip(0, 255).astype(np.uint8)
        base_frame_bgr = cv2.cvtColor(base_frame_rgb, cv2.COLOR_RGB2BGR)

        # Colorize heatmap (JET colormap)
        heatmap_uint8 = (heatmap * 255.0).clip(0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend base frame and heatmap
        overlay = cv2.addWeighted(base_frame_bgr, 1.0 - alpha, colored_heatmap, alpha, 0)

        # Write overlay image
        cv2.imwrite(output_image_path, overlay)

        # Peak activation coordinate (y, x)
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        return GradCAMExplanation(
            sample_id=sample_id,
            frame_index=frame_idx,
            target_layer=self.target_layer_name,
            output_image_path=output_image_path,
            peak_activation_coords=(int(peak_x), int(peak_y)),
            mean_activation_intensity=float(np.mean(heatmap)),
            predicted_score=round(pred_score, 4),
        )

    def explain_video(
        self,
        frames_tensor: np.ndarray,
        output_dir: str,
        sample_id: str,
        frame_indices: Optional[List[int]] = None,
    ) -> List[GradCAMExplanation]:
        """
        Explain selected or all 8 frames of a video sequence and save overlays.
        """
        if frame_indices is None:
            # Explain frame 0, 3, and 7 by default (early, mid, late)
            frame_indices = [0, 3, 7]

        os.makedirs(output_dir, exist_ok=True)
        explanations: List[GradCAMExplanation] = []

        for f_idx in frame_indices:
            out_img = os.path.join(output_dir, f"{sample_id}_frame_{f_idx}_gradcam.png")
            exp = self.generate_and_save_overlay(
                frames_tensor=frames_tensor,
                frame_idx=f_idx,
                output_image_path=out_img,
                sample_id=sample_id,
            )
            explanations.append(exp)

        return explanations
