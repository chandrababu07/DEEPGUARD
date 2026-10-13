"""
DeepGuard Phase 2.4: Model Architecture Verification Script.

Designed for Google Colab or environments with TensorFlow installed.
Verifies:
1. TensorFlow and GPU detection.
2. Construction of frame feature extractor (MobileNetV2 + GAP -> 1,280-d).
3. Assembly of full spatial-temporal model (Input -> TimeDistributed -> LSTM -> Dropout -> Sigmoid).
4. Forward-pass smoke test on dummy 5D tensor (B=2, T=8, H=128, W=128, C=3) in [0.0, 1.0].
5. Parameter breakdown (Trainable vs. Frozen Backbone weights).
"""

import os
import sys
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.baseline import (
    DeepGuardModelConfig,
    build_deepguard_model,
    build_frame_feature_extractor,
    HAS_TF,
)


def run_model_verification():
    print("=" * 75)
    print(" DEEPGUARD - PHASE 2.4 MODEL ARCHITECTURE VERIFICATION")
    print("=" * 75)

    if not HAS_TF:
        print("\n[!] TensorFlow is not installed in this environment.")
        print("    This local environment is Python 3.13 without TensorFlow.")
        print("\n    To run this architecture verification with GPU support on Google Colab:")
        print("    1. Open your Google Colab notebook.")
        print("    2. Run:")
        print("       !git clone https://github.com/chandrababu07/DEEPGUARD.git")
        print("       %cd DEEPGUARD")
        print("       !python scripts/verify_phase2_model.py")
        print("=" * 75)
        return False

    import tensorflow as tf

    print(f"\n[Environment] TensorFlow Version : {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[Environment] GPU Available        : Yes ({len(gpus)} device(s): {[g.name for g in gpus]})")
    else:
        print("[Environment] GPU Available        : No (Running on CPU)")

    # 1. Configuration Contract
    config = DeepGuardModelConfig(weights="imagenet", trainable_backbone=False)
    print("\n[Step 1] Configuration Contract:")
    print(f"  - Sequence Length (T)      : {config.sequence_length} frames")
    print(f"  - Frame Resolution (H x W) : {config.frame_height} x {config.frame_width} x {config.channels}")
    print(f"  - Spatial Feature Vector   : {config.feature_dim} dimensions (from GAP)")
    print(f"  - Temporal Model           : LSTM ({config.lstm_units} units)")
    print(f"  - Regularization           : Dropout ({config.dropout_rate})")
    print(f"  - Classifier Output        : Sigmoid (1 unit: 0=Real, 1=Fake)")
    print(f"  - Optimizer / Loss         : Adam (lr={config.learning_rate}) / Binary Cross-Entropy")
    print(f"  - Backbone Status          : Frozen (trainable_backbone={config.trainable_backbone})")
    print(f"  - Input Normalization      : [0.0, 1.0] (matching VideoPreprocessor)")

    # 2. Build and verify frame feature extractor
    print("\n[Step 2] Building Frame Feature Extractor (MobileNetV2 + GAP)...")
    extractor = build_frame_feature_extractor(config)
    print(f"  - Extractor Input Shape    : {extractor.input_shape}")
    print(f"  - Extractor Output Shape   : {extractor.output_shape} (Expected: (None, 1280))")
    assert extractor.output_shape == (None, 1280), "Extractor output dimension mismatch!"

    # 3. Build full model
    print("\n[Step 3] Assembling Complete Spatial-Temporal Model...")
    model = build_deepguard_model(config)
    print(f"  - Model Input Shape        : {model.input_shape} (Expected: (None, 8, 128, 128, 3))")
    print(f"  - Model Output Shape       : {model.output_shape} (Expected: (None, 1))")
    assert model.input_shape == (None, 8, 128, 128, 3), "Model input shape mismatch!"
    assert model.output_shape == (None, 1), "Model output shape mismatch!"

    # 4. Model parameter breakdown
    total_params = model.count_params()
    trainable_params = sum(np.prod(v.get_shape().as_list()) for v in model.trainable_weights)
    non_trainable_params = sum(np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights)
    print("\n[Step 4] Parameter Counts:")
    print(f"  - Total Parameters         : {total_params:,}")
    print(f"  - Trainable Parameters     : {trainable_params:,} (LSTM + Classifier head)")
    print(f"  - Non-Trainable Parameters : {non_trainable_params:,} (Frozen MobileNetV2 backbone)")

    # 5. Forward-pass smoke test
    print("\n[Step 5] Running Forward-Pass Smoke Test (Batch Size = 2)...")
    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 8, 128, 128, 3)).astype(np.float32)
    predictions = model(dummy_input, training=False).numpy()

    print(f"  - Input Tensor Shape       : {dummy_input.shape}")
    print(f"  - Input Value Range        : [{dummy_input.min():.4f}, {dummy_input.max():.4f}]")
    print(f"  - Prediction Shape         : {predictions.shape}")
    print(f"  - Video 1 Predicted Score  : {predictions[0][0]:.4f}")
    print(f"  - Video 2 Predicted Score  : {predictions[1][0]:.4f}")

    assert predictions.shape == (2, 1), "Prediction tensor shape mismatch!"
    assert np.all(np.isfinite(predictions)), "Predictions contain non-finite values (NaN / Inf)!"
    assert np.all((predictions >= 0.0) & (predictions <= 1.0)), "Predictions out of [0, 1] range!"

    print("\n[SUCCESS] Phase 2.4 Model Architecture verified and fully operational!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    run_model_verification()
