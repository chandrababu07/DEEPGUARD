"""
MobileNetV2 + LSTM baseline architecture for DeepGuard.

Specifications derived from Phase 1 Specification (PHASE1_SPECIFICATION.md):
- Input shape: (batch_size, 8, 128, 128, 3) [5D tensor]
- Sequence length: Exactly 8 uniform frames (T = 8)
- Spatial backbone: MobileNetV2 (include_top=False) + GlobalAveragePooling2D -> 1,280 features/frame
- Temporal model: LSTM with 64 units
- Regularization: Dropout(0.3)
- Classifier: Dense(1, activation='sigmoid') for binary classification (0 = Real, 1 = Fake)
- Loss & Optimizer: Binary Cross-Entropy with Adam (learning rate = 0.0001)
- Input normalization: Directly expects [0.0, 1.0] float32 inputs matching VideoPreprocessor.
  Does NOT silently apply Keras's mobilenet_v2.preprocess_input [-1, 1] scaling on top of inputs.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:
    import tensorflow as tf
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    tf = None
    layers = None
    HAS_TF = False


@dataclass
class DeepGuardModelConfig:
    """Configuration contract for DeepGuard MobileNetV2 + LSTM model."""

    sequence_length: int = 8
    frame_height: int = 128
    frame_width: int = 128
    channels: int = 3
    feature_dim: int = 1280
    lstm_units: int = 64
    dropout_rate: float = 0.3
    learning_rate: float = 0.0001
    trainable_backbone: bool = False
    weights: Optional[str] = "imagenet"
    input_normalization: str = "unit_interval"  # Expected range: [0.0, 1.0]

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """Returns (height, width, channels) tuple."""
        return (self.frame_height, self.frame_width, self.channels)

    @property
    def video_input_shape(self) -> Tuple[int, int, int, int]:
        """Returns 4D shape without batch dimension: (sequence_length, height, width, channels)."""
        return (self.sequence_length, self.frame_height, self.frame_width, self.channels)

    def validate_tensor_shape(self, shape: Tuple[int, ...]) -> None:
        """
        Validates that a candidate tensor shape matches the required 5D video input shape.
        Raises ValueError with an informative message if mismatched.
        """
        if len(shape) != 5:
            raise ValueError(
                f"Invalid input rank: Expected 5D tensor (batch_size, {self.sequence_length}, "
                f"{self.frame_height}, {self.frame_width}, {self.channels}), but received shape {shape} with rank {len(shape)}."
            )
        _, t, h, w, c = shape
        if t != self.sequence_length:
            raise ValueError(
                f"Invalid sequence length: Expected {self.sequence_length} frames, but received {t} frames in shape {shape}."
            )
        if (h, w, c) != self.frame_shape:
            raise ValueError(
                f"Invalid frame dimensions: Expected {self.frame_shape}, but received {(h, w, c)} in shape {shape}."
            )


def build_frame_feature_extractor(config: DeepGuardModelConfig) -> Any:
    """
    Constructs the per-frame 2D spatial feature extractor:
    MobileNetV2 (pretrained on ImageNet, include_top=False) + GlobalAveragePooling2D.

    Produces a 1,280-dimensional feature vector per frame.

    Args:
        config: DeepGuardModelConfig instance.

    Returns:
        tf.keras.Model: Frame feature extractor model.
    """
    if not HAS_TF:
        raise ImportError(
            "TensorFlow is required to build the feature extractor, but it is not installed "
            "in the local Python environment. Please execute in an environment with TensorFlow (such as Google Colab)."
        )

    frame_input = layers.Input(shape=config.frame_shape, name="single_frame_input")

    # MobileNetV2 convolutional base
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=config.frame_shape,
        include_top=False,
        weights=config.weights,
    )
    mobilenet.trainable = config.trainable_backbone

    # Pass frame through MobileNetV2
    features_map = mobilenet(frame_input)

    # 2D Global Average Pooling across spatial dimensions (Eq. 5 in reference paper)
    # Reduces spatial tensor from (B, 4, 4, 1280) to (B, 1280)
    features_vec = layers.GlobalAveragePooling2D(name="spatial_global_avg_pooling")(features_map)

    extractor_model = tf.keras.Model(
        inputs=frame_input,
        outputs=features_vec,
        name="mobilenetv2_feature_extractor",
    )
    return extractor_model


def build_deepguard_model(config: Optional[DeepGuardModelConfig] = None) -> Any:
    """
    Constructs and compiles the complete DeepGuard spatial-temporal model:
    Input (B, 8, 128, 128, 3)
      -> TimeDistributed(MobileNetV2 + GAP) -> (B, 8, 1280)
      -> LSTM(64 units)                     -> (B, 64)
      -> Dropout(0.3)                       -> (B, 64)
      -> Dense(1, sigmoid)                  -> (B, 1)

    Input Normalization Convention:
      The model expects input values directly in the range [0.0, 1.0], matching the
      output of DeepGuard's VideoPreprocessor. It does NOT silently apply Keras's
      mobilenet_v2.preprocess_input [-1, 1] scaling on top of [0, 1] inputs.

    Backbone Trainability:
      config.trainable_backbone = False (frozen) by default for architecture validation
      and feature caching. Can be set to True for end-to-end fine-tuning evaluations.

    Args:
        config: Optional DeepGuardModelConfig instance (uses defaults if None).

    Returns:
        tf.keras.Model: Compiled DeepGuard model.
    """
    if not HAS_TF:
        raise ImportError(
            "TensorFlow is required to build the DeepGuard model, but it is not installed "
            "in the local Python environment. Please execute in an environment with TensorFlow (such as Google Colab)."
        )

    if config is None:
        config = DeepGuardModelConfig()

    # 5D Video frame input tensor: (batch_size, sequence_length, height, width, channels)
    video_input = layers.Input(shape=config.video_input_shape, name="video_frames_input")

    # Step 1: Per-frame spatial feature extractor
    frame_extractor = build_frame_feature_extractor(config)

    # Step 2: TimeDistributed application across all 8 frames -> output shape (B, 8, 1280)
    time_distributed_features = layers.TimeDistributed(
        frame_extractor,
        name="time_distributed_spatial_extractor",
    )(video_input)

    # Step 3: Recurrent temporal modeling -> output shape (B, 64)
    lstm_output = layers.LSTM(
        units=config.lstm_units,
        return_sequences=False,
        name="temporal_lstm",
    )(time_distributed_features)

    # Step 4: Dropout regularization (0.3) -> output shape (B, 64)
    dropout_output = layers.Dropout(
        rate=config.dropout_rate,
        name="regularization_dropout",
    )(lstm_output)

    # Step 5: Binary classification with sigmoid activation -> output shape (B, 1)
    # Output semantics: 0.0 = Real / Authentic, 1.0 = Fake / Manipulated
    prediction = layers.Dense(
        units=1,
        activation="sigmoid",
        name="binary_classifier",
    )(dropout_output)

    # Assemble Model
    model = tf.keras.Model(
        inputs=video_input,
        outputs=prediction,
        name="deepguard_mobilenetv2_lstm",
    )

    # Compile with Binary Cross-Entropy and Adam (lr = 0.0001) as specified in paper
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
