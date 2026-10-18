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
        decision_threshold: float = 0.5,
    ):
        """
        Initialize the Grad-CAM explainer.

        Args:
            model: DeepGuard tf.keras.Model.
            target_layer_name: Name of convolutional layer to target in MobileNetV2.
            decision_threshold: Binary decision threshold for fake class (default 0.5).
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.decision_threshold = float(decision_threshold)

        self._resolved_model = None
        self._time_dist_layer = None
        self._pre_target_model = None
        self._post_target_layers: List[Any] = []
        self._temporal_layers: List[Any] = []
        self._target_layer = None
        self._effective_target_layer_name = target_layer_name
        self.last_diagnostics: Dict[str, Any] = {}

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

    def _determine_target(
        self,
        p_tensor: Any,
        pred_val: float,
        target_class: Optional[str] = None,
    ) -> Tuple[Any, str, float]:
        """
        Determine target loss tensor, label, and score scalar based on model output semantics.

        Model Output Semantics:
        - Output p is sigmoid probability in [0.0, 1.0] for FAKE / MANIPULATED.
        - Complement 1.0 - p is probability for REAL / AUTHENTIC.
        - If p >= decision_threshold: predicted class is FAKE, target is p.
        - If p < decision_threshold: predicted class is REAL, target is 1.0 - p.

        Returns:
            Tuple of (loss_tensor, target_label, target_score_scalar)
        """
        if target_class is not None:
            tc_upper = str(target_class).strip().upper()
            if tc_upper in ("FAKE", "1", "MANIPULATED"):
                return p_tensor, "FAKE", pred_val
            elif tc_upper in ("REAL", "0", "AUTHENTIC"):
                loss_t = (1.0 - p_tensor) if not isinstance(p_tensor, str) else f"1.0 - {p_tensor}"
                return loss_t, "REAL", float(1.0 - pred_val)
            else:
                raise ValueError(
                    f"Invalid target_class '{target_class}'. Must be 'FAKE', 'REAL', or None."
                )

        if pred_val >= self.decision_threshold:
            return p_tensor, "FAKE", pred_val
        else:
            loss_t = (1.0 - p_tensor) if not isinstance(p_tensor, str) else f"1.0 - {p_tensor}"
            return loss_t, "REAL", float(1.0 - pred_val)

    def _find_target_layer(
        self,
        extractor: Any,
        target_name: str,
    ) -> Tuple[Any, Optional[Any], List[Any]]:
        """
        Locate target convolutional layer within the spatial extractor.

        Searches inside nested backbone models (e.g. MobileNetV2) and top-level layers.
        Enforces that a nested Model container is NEVER selected as the target layer.

        Returns:
            Tuple of (target_layer, backbone_model_or_None, post_target_layers)
        """
        # 1. Identify nested backbone model if present
        backbone = None
        for layer in getattr(extractor, "layers", []):
            if hasattr(layer, "layers") and len(layer.layers) > 0:
                backbone = layer
                break

        # Candidate layers to search
        candidate_layers = backbone.layers if backbone is not None else getattr(extractor, "layers", [])

        # 2. Exact match by name (excluding Model containers)
        target_layer = None
        for layer in candidate_layers:
            if layer.name == target_name and not hasattr(layer, "layers"):
                target_layer = layer
                break

        # Also check top-level extractor layers if not found in backbone
        if target_layer is None and backbone is not None:
            for layer in getattr(extractor, "layers", []):
                if layer.name == target_name and not hasattr(layer, "layers"):
                    target_layer = layer
                    backbone = None  # target is top-level in extractor
                    candidate_layers = extractor.layers
                    break

        # 3. Fallback: search backwards for last conv / relu layer (excluding Model containers)
        if target_layer is None:
            for layer in reversed(candidate_layers):
                if hasattr(layer, "layers"):
                    continue
                layer_cls = layer.__class__.__name__.lower()
                layer_name = layer.name.lower()
                is_conv_or_relu = "conv" in layer_cls or "conv" in layer_name or "relu" in layer_name
                if HAS_TF and tf is not None and hasattr(tf, "keras"):
                    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
                    is_conv_or_relu = is_conv_or_relu or isinstance(layer, conv_types)
                if is_conv_or_relu:
                    target_layer = layer
                    break

        if target_layer is None:
            raise ValueError(
                f"Could not locate target layer '{target_name}' or any convolutional "
                f"fallback within feature extractor '{getattr(extractor, 'name', 'unknown')}'."
            )

        # 4. Determine post-target layers leading up to extractor output
        target_idx = candidate_layers.index(target_layer)
        post_target_layers: List[Any] = list(candidate_layers[target_idx + 1:])

        if backbone is not None:
            bb_idx = extractor.layers.index(backbone)
            post_target_layers.extend(extractor.layers[bb_idx + 1:])

        return target_layer, backbone, post_target_layers

    def _resolve_graph_components(self) -> None:
        """
        Inspect and wire the loaded model's graph components for connected Grad-CAM differentiation.
        """
        if self._resolved_model is self.model and self._pre_target_model is not None:
            return

        if self.model is None:
            raise ValueError("No model provided for Grad-CAM computation.")

        if not HAS_TF:
            raise ImportError(
                "TensorFlow is required to compute Grad-CAM gradients. "
                "Execute in Google Colab or an environment with TensorFlow."
            )

        # 1. Locate TimeDistributed spatial extractor
        time_dist_layer = None
        try:
            time_dist_layer = self.model.get_layer("time_distributed_spatial_extractor")
        except (ValueError, KeyError, AttributeError):
            for layer in getattr(self.model, "layers", []):
                if isinstance(layer, tf.keras.layers.TimeDistributed) or hasattr(layer, "layer"):
                    time_dist_layer = layer
                    break

        if time_dist_layer is None or not hasattr(time_dist_layer, "layer"):
            raise ValueError(
                "Model does not contain a TimeDistributed spatial extractor layer."
            )

        self._time_dist_layer = time_dist_layer
        extractor = time_dist_layer.layer

        # 2. Locate temporal / classifier layers following TimeDistributed
        model_layers = list(self.model.layers)
        time_dist_idx = model_layers.index(time_dist_layer)
        self._temporal_layers = model_layers[time_dist_idx + 1:]
        if not self._temporal_layers:
            raise ValueError(
                "Model does not contain temporal/classifier layers after the spatial extractor."
            )

        # 3. Locate target layer, backbone, and post-target layers
        target_layer, backbone, post_layers = self._find_target_layer(
            extractor, self.target_layer_name
        )
        self._target_layer = target_layer
        self._effective_target_layer_name = target_layer.name
        self._post_target_layers = post_layers

        # 4. Construct pre-target sub-model: frame_input -> target_layer.output
        submodel_input = backbone.input if backbone is not None else extractor.input
        self._pre_target_model = tf.keras.Model(
            inputs=submodel_input,
            outputs=target_layer.output,
            name="gradcam_pre_target_model",
        )
        self._resolved_model = self.model

    def compute_heatmap(
        self,
        frames_tensor: np.ndarray,
        frame_idx: int = 0,
        target_class: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Grad-CAM heatmap for a specific frame in the 8-frame sequence.

        Args:
            frames_tensor: Validated (8, 128, 128, 3) tensor.
            frame_idx: Frame index in range [0, 7].
            target_class: Optional target class to explain ('FAKE', 'REAL', or None for predicted).

        Returns:
            Tuple of (heatmap_normalized_128x128, predicted_score)
        """
        if (
            isinstance(frame_idx, bool)
            or not isinstance(frame_idx, (int, np.integer))
            or not (0 <= frame_idx < 8)
        ):
            raise IndexError(f"frame_idx must be in [0, 7], got {frame_idx}")

        if not HAS_TF:
            raise ImportError(
                "TensorFlow is required to compute Grad-CAM gradients. "
                "Execute in Google Colab or an environment with TensorFlow."
            )

        if self.model is None:
            raise ValueError("No model provided for Grad-CAM computation.")

        self._resolve_graph_components()

        # Format input batch: (1, 8, 128, 128, 3)
        validated_frames = self.validate_tensor(frames_tensor)
        input_batch = tf.convert_to_tensor(
            np.expand_dims(validated_frames, axis=0), dtype=tf.float32
        )
        frame_k = input_batch[:, frame_idx, :, :, :]

        # Compute spatial features across all frames using model's TimeDistributed layer
        time_dist_features = self._time_dist_layer(input_batch, training=False)

        # GradientTape tracks forward pass from conv_activations through temporal model
        with tf.GradientTape() as tape:
            # Pass target frame through pre-target model
            conv_activations = self._pre_target_model(frame_k, training=False)
            tape.watch(conv_activations)

            # Pass conv_activations through remaining spatial layers to obtain v_k
            x = conv_activations
            for layer in self._post_target_layers:
                try:
                    x = layer(x, training=False)
                except TypeError:
                    x = layer(x)
            v_k = x

            # Splice v_k into the sequence at frame_idx
            frame_vecs = [time_dist_features[:, i, :] for i in range(8)]
            frame_vecs[frame_idx] = v_k
            feature_seq = tf.stack(frame_vecs, axis=1)

            # Pass through temporal model and classifier
            x = feature_seq
            for layer in self._temporal_layers:
                try:
                    x = layer(x, training=False)
                except TypeError:
                    x = layer(x)
            predictions = x
            p_tensor = predictions[:, 0]

            # Compute prediction scalar
            pred_val = float(predictions.numpy()[0][0])
            pred_val = float(np.clip(pred_val, 0.0, 1.0))

            # Determine target score tensor and class label based on model output semantics
            loss, target_label, target_score_val = self._determine_target(
                p_tensor, pred_val, target_class=target_class
            )

        # Compute gradients of target loss with respect to conv_activations
        grads = tape.gradient(loss, conv_activations)

        if grads is None:
            raise RuntimeError(
                "Grad-CAM gradient computation failed: gradients are disconnected or None. "
                "Ensure the model provides a differentiable path from the target layer to the prediction."
            )

        grads_finite = bool(tf.reduce_all(tf.math.is_finite(grads)).numpy())
        if not grads_finite:
            raise RuntimeError(
                "Grad-CAM gradient computation produced non-finite (NaN or Inf) gradients."
            )

        grads_np = grads.numpy()
        grads_min = float(np.min(grads_np))
        grads_max = float(np.max(grads_np))
        grads_mean = float(np.mean(grads_np))

        # Check for completely zero/unusable gradients
        if max(abs(grads_min), abs(grads_max)) < 1e-12:
            raise RuntimeError(
                "Grad-CAM gradient computation failed: gradients are identically zero or unusable."
            )

        # Global average pooling over spatial dimensions of gradients
        # grads shape: (1, H, W, C) -> pooled_grads shape: (1, 1, 1, C)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        pooled_grads_np = pooled_grads.numpy()

        act_np = conv_activations.numpy()
        act_min = float(np.min(act_np))
        act_max = float(np.max(act_np))
        act_mean = float(np.mean(act_np))

        # Weight the convolutional feature maps by pooled gradients
        # raw_cam shape: (H, W) before ReLU
        raw_cam_tf = tf.reduce_sum(conv_activations * pooled_grads, axis=-1)[0]
        raw_cam_np = raw_cam_tf.numpy()
        raw_cam_min = float(np.min(raw_cam_np))
        raw_cam_max = float(np.max(raw_cam_np))
        raw_cam_mean = float(np.mean(raw_cam_np))

        # Capture complete numerical diagnostics for inspection & verification
        self.last_diagnostics = {
            "frame_idx": frame_idx,
            "target_label": target_label,
            "target_score": target_score_val,
            "predicted_fake_score": pred_val,
            "decision_threshold": self.decision_threshold,
            "grads_none": False,
            "grads_finite": grads_finite,
            "grads_min": grads_min,
            "grads_max": grads_max,
            "grads_mean": grads_mean,
            "activations_min": act_min,
            "activations_max": act_max,
            "activations_mean": act_mean,
            "pooled_grads_min": float(np.min(pooled_grads_np)),
            "pooled_grads_max": float(np.max(pooled_grads_np)),
            "pooled_grads_mean": float(np.mean(pooled_grads_np)),
            "raw_cam_min": raw_cam_min,
            "raw_cam_max": raw_cam_max,
            "raw_cam_mean": raw_cam_mean,
        }

        # Apply ReLU to retain only positive evidence for the targeted class
        cam = tf.maximum(raw_cam_tf, 0.0)

        # Normalize heatmap to [0.0, 1.0]
        max_val = float(tf.math.reduce_max(cam).numpy())
        if max_val <= 1e-10:
            raise RuntimeError(
                f"Grad-CAM produced an all-zero attribution map for frame {frame_idx} "
                f"(target_class='{target_label}', score={target_score_val:.4f}). "
                f"No positive gradient activations were detected in target layer '{self._effective_target_layer_name}'."
            )

        cam_normalized = (cam / max_val).numpy().astype(np.float32)
        heatmap_resized = cv2.resize(cam_normalized, (128, 128), interpolation=cv2.INTER_LINEAR)
        heatmap_normalized = np.clip(heatmap_resized, 0.0, 1.0).astype(np.float32)

        return heatmap_normalized, pred_val

    def generate_and_save_overlay(
        self,
        frames_tensor: np.ndarray,
        frame_idx: int,
        output_image_path: str,
        sample_id: str = "video",
        alpha: float = 0.5,
        target_class: Optional[str] = None,
    ) -> GradCAMExplanation:
        """
        Generate Grad-CAM heatmap, blend with input frame, and save to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
        validated_frames = self.validate_tensor(frames_tensor)

        heatmap, pred_score = self.compute_heatmap(
            validated_frames, frame_idx=frame_idx, target_class=target_class
        )

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
        target_class: Optional[str] = None,
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
                target_class=target_class,
            )
            explanations.append(exp)

        return explanations
