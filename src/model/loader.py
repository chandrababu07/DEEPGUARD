"""
Model checkpoint loader and validation for DeepGuard.

Loads trained .keras model checkpoints outside of notebook environments,
verifying file integrity, tensor input shapes, and architectural compatibility.
"""

import hashlib
import os
from typing import Optional, Tuple

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    tf = None
    HAS_TF = False


EXPECTED_INPUT_SHAPE: Tuple[Optional[int], int, int, int, int] = (None, 8, 128, 128, 3)
EXPECTED_OUTPUT_SHAPE: Tuple[Optional[int], int] = (None, 1)


def compute_file_sha256(file_path: str, buffer_size: int = 65536) -> str:
    """Computes the SHA256 checksum of a file on disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found for hash calculation: {file_path}")

    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(buffer_size):
            sha.update(chunk)
    return sha.hexdigest()


def load_deepguard_checkpoint(
    checkpoint_path: str,
    compile_model: bool = True,
    verify_shapes: bool = True,
) -> Tuple[object, str]:
    """
    Load a trained DeepGuard model checkpoint (.keras format).

    Args:
        checkpoint_path: Path to the .keras model file.
        compile_model: Whether to compile the model upon loading.
        verify_shapes: Whether to assert input and output tensor shapes.

    Returns:
        Tuple of (loaded_keras_model, checkpoint_sha256_hash)

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ImportError: If TensorFlow is unavailable.
        ValueError: If model input/output tensor shapes do not match DeepGuard contract.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DeepGuard model checkpoint not found at: '{checkpoint_path}'")

    if not HAS_TF:
        raise ImportError(
            "TensorFlow is required to load the model checkpoint, but is not installed in "
            "the current Python environment. Execute in Google Colab or an environment with TensorFlow."
        )

    file_hash = compute_file_sha256(checkpoint_path)

    # Load model using tf.keras
    model = tf.keras.models.load_model(checkpoint_path, compile=compile_model)

    if verify_shapes:
        actual_input_shape = tuple(model.input_shape)
        actual_output_shape = tuple(model.output_shape)

        # Allow None or batch size in first dimension
        input_rank_match = (
            len(actual_input_shape) == 5
            and actual_input_shape[1:] == EXPECTED_INPUT_SHAPE[1:]
        )
        if not input_rank_match:
            raise ValueError(
                f"Checkpoint input shape mismatch: Expected {EXPECTED_INPUT_SHAPE}, "
                f"but loaded model has input shape {actual_input_shape}."
            )

        output_rank_match = (
            len(actual_output_shape) == 2
            and actual_output_shape[1:] == EXPECTED_OUTPUT_SHAPE[1:]
        )
        if not output_rank_match:
            raise ValueError(
                f"Checkpoint output shape mismatch: Expected {EXPECTED_OUTPUT_SHAPE}, "
                f"but loaded model has output shape {actual_output_shape}."
            )

    return model, file_hash
