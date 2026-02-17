"""
Test script to verify multi-GPU training setup and utilities.

This script tests:
1. Distribution strategy creation
2. Mixed precision setup
3. Data augmentation
4. Model compilation in distributed scope
5. Small training run (if GPUs available)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras

from train_utils import (
    create_distribution_strategy,
    DistributedStrategyConfig,
    setup_mixed_precision,
    create_augmented_dataset,
    print_training_summary,
)
from shift_equivariant_unet import CircularConv2D
from train import build_model


def test_strategy_creation():
    """Test 1: Distribution strategy creation."""
    print("\n" + "="*60)
    print("TEST 1: Distribution Strategy Creation")
    print("="*60)

    try:
        # Test auto strategy
        config = DistributedStrategyConfig(strategy_type='auto')
        strategy, num_replicas = create_distribution_strategy(config, verbose=True)

        print(f"\n✓ Strategy created successfully")
        print(f"  Type: {strategy.__class__.__name__}")
        print(f"  Replicas: {num_replicas}")

        return True, strategy, num_replicas
    except Exception as e:
        print(f"\n✗ Strategy creation failed: {e}")
        return False, None, None


def test_nccl_fallback():
    """Test 2: NCCL fallback mechanism."""
    print("\n" + "="*60)
    print("TEST 2: NCCL Fallback Mechanism")
    print("="*60)

    gpus = tf.config.list_physical_devices('GPU')

    if len(gpus) < 2:
        print("⚠ Skipping NCCL test (need 2+ GPUs)")
        print(f"  Available GPUs: {len(gpus)}")
        return True

    try:
        # Test with NCCL workarounds
        config = DistributedStrategyConfig(
            strategy_type='mirrored',
            enable_nccl_workarounds=True
        )
        strategy, num_replicas = create_distribution_strategy(config, verbose=True)

        print(f"\n✓ NCCL fallback mechanism working")
        return True
    except Exception as e:
        print(f"\n✗ NCCL fallback test failed: {e}")
        return False


def test_mixed_precision():
    """Test 3: Mixed precision setup."""
    print("\n" + "="*60)
    print("TEST 3: Mixed Precision Setup")
    print("="*60)

    try:
        policy = setup_mixed_precision(enabled=True, verbose=True)
        print(f"\n✓ Mixed precision configured")
        print(f"  Compute dtype: {policy.compute_dtype}")
        print(f"  Variable dtype: {policy.variable_dtype}")

        return True
    except Exception as e:
        print(f"\n✗ Mixed precision test failed: {e}")
        return False


def test_data_augmentation():
    """Test 4: Data augmentation."""
    print("\n" + "="*60)
    print("TEST 4: Data Augmentation")
    print("="*60)

    try:
        # Create dummy data
        masks = np.random.rand(10, 64, 64, 1).astype(np.float32)
        aerials = np.random.rand(10, 64, 64, 1).astype(np.float32)

        # Create augmented dataset
        ds = create_augmented_dataset(masks, aerials, batch_size=2, shuffle=True)

        # Get one batch
        for batch_masks, batch_aerials in ds.take(1):
            print(f"\n✓ Augmented dataset created")
            print(f"  Batch mask shape: {batch_masks.shape}")
            print(f"  Batch aerial shape: {batch_aerials.shape}")
            print(f"  Dtype: {batch_masks.dtype}")

        return True
    except Exception as e:
        print(f"\n✗ Data augmentation test failed: {e}")
        return False


def test_model_compilation(strategy):
    """Test 5: Model compilation in distributed scope."""
    print("\n" + "="*60)
    print("TEST 5: Model Compilation in Distributed Scope")
    print("="*60)

    try:
        with strategy.scope():
            model = build_model(input_shape=(64, 64, 1), num_filters_base=16)
            model.compile(
                optimizer=keras.optimizers.Adam(1e-3),
                loss='mse',
                metrics=['mae']
            )

        print(f"\n✓ Model compiled successfully")
        print(f"  Parameters: {model.count_params():,}")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")

        return True, model
    except Exception as e:
        print(f"\n✗ Model compilation failed: {e}")
        return False, None


def test_small_training(strategy, model):
    """Test 6: Small training run."""
    print("\n" + "="*60)
    print("TEST 6: Small Training Run")
    print("="*60)

    try:
        # Create tiny dataset
        masks = np.random.rand(16, 64, 64, 1).astype(np.float32)
        aerials = np.random.rand(16, 64, 64, 1).astype(np.float32)

        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices((masks, aerials))
        ds = ds.batch(4).prefetch(tf.data.AUTOTUNE)

        # Distribute dataset
        ds = strategy.experimental_distribute_dataset(ds)

        # Train for 2 epochs
        print("\nTraining for 2 epochs on dummy data...")
        history = model.fit(ds, epochs=2, verbose=1)

        print(f"\n✓ Training completed successfully")
        print(f"  Final loss: {history.history['loss'][-1]:.6f}")

        return True
    except Exception as e:
        print(f"\n✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Multi-GPU Training Setup Verification")
    print("="*60)

    # System info
    print("\nSystem Information:")
    print(f"  TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  GPUs available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"    GPU {i}: {gpu.name}")

    results = {}

    # Test 1: Strategy creation
    success, strategy, num_replicas = test_strategy_creation()
    results['strategy_creation'] = success

    if not success:
        print("\n✗ Critical failure in strategy creation, skipping remaining tests")
        return 1

    # Test 2: NCCL fallback
    results['nccl_fallback'] = test_nccl_fallback()

    # Test 3: Mixed precision
    results['mixed_precision'] = test_mixed_precision()

    # Test 4: Data augmentation
    results['data_augmentation'] = test_data_augmentation()

    # Test 5: Model compilation
    success, model = test_model_compilation(strategy)
    results['model_compilation'] = success

    # Test 6: Small training (only if model compilation succeeded)
    if success and model is not None:
        results['small_training'] = test_small_training(strategy, model)
    else:
        results['small_training'] = False
        print("\n⚠ Skipping training test due to model compilation failure")

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:.<40} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n🎉 All tests passed! Multi-GPU training setup is ready.")
        return 0
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        return 1


if __name__ == '__main__':
    exit(main())
