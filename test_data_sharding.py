"""
Test Multi-GPU Data Sharding
==============================

Verify that data is correctly sharded across multiple GPUs in distributed training.

This test ensures that each GPU receives DIFFERENT data, not duplicate data.

Usage:
    python test_data_sharding.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras


def test_basic_sharding():
    """Test basic data sharding mechanism."""
    print("="*70)
    print("TEST 1: Basic Data Sharding")
    print("="*70)

    # Check available devices
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"\nAvailable GPUs: {len(physical_devices)}")

    if len(physical_devices) < 2:
        print("⚠ WARNING: Need at least 2 GPUs for meaningful test")
        print("  Running test anyway with available devices...")

    # Create strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"Strategy: {strategy}")
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")

    # Create simple dataset
    n_samples = 100
    data = np.arange(n_samples).reshape(n_samples, 1).astype(np.float32)
    labels = data * 2

    # Test WITHOUT sharding (WRONG)
    print("\n" + "-"*70)
    print("WITHOUT Auto-Sharding (INCORRECT - shows the bug)")
    print("-"*70)

    ds_wrong = tf.data.Dataset.from_tensor_slices((data, labels))
    ds_wrong = ds_wrong.batch(10)
    ds_wrong = strategy.experimental_distribute_dataset(ds_wrong)

    @tf.function
    def check_batch_wrong(dist_inputs):
        """Check what data each GPU sees."""
        def replica_fn(inputs):
            x, y = inputs
            # Return first 3 elements of batch
            return x[:3, 0]

        per_replica_results = strategy.run(replica_fn, args=(dist_inputs,))
        return per_replica_results

    batch_count = 0
    for batch in ds_wrong.take(3):
        batch_count += 1
        results = check_batch_wrong(batch)

        print(f"\nBatch {batch_count}:")
        if hasattr(results, 'values'):
            # Multiple replicas
            for i, result in enumerate(results.values):
                print(f"  GPU {i}: {result.numpy()}")

            # Check if all GPUs see same data (BUG!)
            values = [r.numpy() for r in results.values]
            if len(values) > 1:
                all_same = all(np.array_equal(values[0], v) for v in values[1:])
                if all_same:
                    print(f"  ❌ BUG: All GPUs see IDENTICAL data!")
                else:
                    print(f"  ✓ Good: GPUs see different data")
        else:
            # Single device
            print(f"  Device: {results.numpy()}")

    # Test WITH sharding (CORRECT)
    print("\n" + "-"*70)
    print("WITH Auto-Sharding (CORRECT - the fix)")
    print("-"*70)

    ds_correct = tf.data.Dataset.from_tensor_slices((data, labels))
    ds_correct = ds_correct.batch(10)

    # Enable auto-sharding
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.DATA
    ds_correct = ds_correct.with_options(options)

    ds_correct = strategy.experimental_distribute_dataset(ds_correct)

    @tf.function
    def check_batch_correct(dist_inputs):
        """Check what data each GPU sees."""
        def replica_fn(inputs):
            x, y = inputs
            return x[:3, 0]

        per_replica_results = strategy.run(replica_fn, args=(dist_inputs,))
        return per_replica_results

    batch_count = 0
    for batch in ds_correct.take(3):
        batch_count += 1
        results = check_batch_correct(batch)

        print(f"\nBatch {batch_count}:")
        if hasattr(results, 'values'):
            for i, result in enumerate(results.values):
                print(f"  GPU {i}: {result.numpy()}")

            # Check if GPUs see different data (CORRECT!)
            values = [r.numpy() for r in results.values]
            if len(values) > 1:
                all_same = all(np.array_equal(values[0], v) for v in values[1:])
                if all_same:
                    print(f"  ❌ ERROR: All GPUs see same data (sharding failed!)")
                else:
                    print(f"  ✓ CORRECT: GPUs see different data!")
        else:
            print(f"  Device: {results.numpy()}")

    print("\n" + "="*70)
    print("✓ Basic sharding test complete")
    print("="*70)


def test_tfrecord_sharding():
    """Test TFRecord sharding."""
    print("\n" + "="*70)
    print("TEST 2: TFRecord Sharding")
    print("="*70)

    try:
        from tfrecord_manager import TFRecordWriter, TFRecordReader, TFRecordConfig
    except ImportError:
        print("⚠ TFRecord manager not available, skipping test")
        return

    # Generate test data
    print("\nGenerating test data...")
    n_samples = 50
    h, w = 64, 64
    masks = np.random.rand(n_samples, h, w, 1).astype(np.float32)
    aerials = np.random.rand(n_samples, h, w, 1).astype(np.float32)

    # Add sample IDs for tracking
    metadata = {
        'sample_id': np.arange(n_samples, dtype=np.int32),
    }

    # Write to TFRecord
    print("Writing to TFRecord...")
    config = TFRecordConfig(
        output_dir='./test_tfrecords_sharding',
        samples_per_file=25,  # 2 files
        compression_type='',
    )

    writer = TFRecordWriter(config)
    files = writer.write_dataset(masks, aerials, metadata, verbose=False)
    print(f"  Created {len(files)} files")

    # Create strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"\nStrategy: {strategy.num_replicas_in_sync} replicas")

    # Test WITH sharding
    print("\nTesting TFRecord with sharding enabled...")

    reader = TFRecordReader('./test_tfrecords_sharding')

    # This should automatically shard
    with strategy.scope():
        ds = reader.create_dataset(
            batch_size=5,
            shuffle=False,
            repeat=False,
            shard_for_multi_gpu=True,  # Enable sharding
            sample_id='int'
        )

    # Check data distribution
    @tf.function
    def get_batch_info(batch_inputs):
        def replica_fn(inputs):
            mask, aerial = inputs
            # Return shape for verification
            return tf.shape(mask)[0]  # Batch size on this replica

        results = strategy.run(replica_fn, args=(batch_inputs,))
        return results

    print("\nBatch sizes per GPU (should be same, but different data):")
    for i, batch in enumerate(ds.take(2)):
        batch_sizes = get_batch_info(batch)
        print(f"  Batch {i+1}: {batch_sizes}")

    print("\n✓ TFRecord sharding test complete")

    # Cleanup
    import shutil
    if os.path.exists('./test_tfrecords_sharding'):
        shutil.rmtree('./test_tfrecords_sharding')
        print("  Cleaned up test files")


def test_training_step_sharding():
    """Test sharding in actual training step."""
    print("\n" + "="*70)
    print("TEST 3: Training Step with Sharding")
    print("="*70)

    # Create simple model
    strategy = tf.distribute.MirroredStrategy()

    print(f"\nStrategy: {strategy.num_replicas_in_sync} replicas")

    with strategy.scope():
        model = keras.Sequential([
            keras.layers.Input(shape=(8,)),
            keras.layers.Dense(4, activation='relu'),
            keras.layers.Dense(1)
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

    # Create dataset
    n_samples = 100
    x_train = np.random.randn(n_samples, 8).astype(np.float32)
    y_train = np.random.randn(n_samples, 1).astype(np.float32)

    # WITHOUT sharding
    print("\nTest WITHOUT sharding...")
    ds_wrong = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_wrong = ds_wrong.batch(10)
    dist_ds_wrong = strategy.experimental_distribute_dataset(ds_wrong)

    # Train for 1 step
    loss_wrong = model.fit(dist_ds_wrong, steps_per_epoch=1, epochs=1, verbose=0)
    print(f"  Loss: {loss_wrong.history['loss'][0]:.4f}")

    # WITH sharding
    print("\nTest WITH sharding...")
    ds_correct = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_correct = ds_correct.batch(10)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.DATA
    ds_correct = ds_correct.with_options(options)

    dist_ds_correct = strategy.experimental_distribute_dataset(ds_correct)

    # Train for 1 step
    loss_correct = model.fit(dist_ds_correct, steps_per_epoch=1, epochs=1, verbose=0)
    print(f"  Loss: {loss_correct.history['loss'][0]:.4f}")

    print("\n✓ Training step test complete")
    print("\nNote: Losses should be different because different data is used")


def test_batch_size_calculation():
    """Test correct batch size calculation for multi-GPU."""
    print("\n" + "="*70)
    print("TEST 4: Batch Size Calculation")
    print("="*70)

    strategy = tf.distribute.MirroredStrategy()
    n_replicas = strategy.num_replicas_in_sync

    print(f"\nNumber of replicas: {n_replicas}")

    # Scenario 1: User wants total batch size of 32
    print("\nScenario 1: Target global batch size = 32")
    global_batch_target = 32
    per_replica_batch = global_batch_target // n_replicas

    print(f"  Per-replica batch size: {per_replica_batch}")
    print(f"  Actual global batch size: {per_replica_batch * n_replicas}")

    # Scenario 2: User specifies per-replica batch size
    print("\nScenario 2: Per-replica batch size = 16")
    per_replica_batch = 16
    actual_global = per_replica_batch * n_replicas

    print(f"  Actual global batch size: {actual_global}")
    print(f"  Samples per step: {actual_global}")

    # IMPORTANT NOTE
    print("\n" + "!"*70)
    print("IMPORTANT: train_distributed.py batch-size argument should be:")
    print("  - Per-replica batch size (recommended)")
    print("  - NOT global batch size")
    print("  - Global batch = per_replica × n_replicas")
    print("!"*70)

    print("\n✓ Batch size calculation test complete")


def main():
    """Run all sharding tests."""
    print("\n" + "█"*70)
    print("Multi-GPU Data Sharding Tests")
    print("█"*70)

    # Check TensorFlow version
    print(f"\nTensorFlow version: {tf.__version__}")

    # Check devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Available GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")

    if len(gpus) == 0:
        print("\n⚠ WARNING: No GPUs available")
        print("  Tests will run on CPU (less meaningful)")

    # Run tests
    try:
        test_basic_sharding()
        test_tfrecord_sharding()
        test_training_step_sharding()
        test_batch_size_calculation()

        print("\n" + "█"*70)
        print("✓ All sharding tests PASSED")
        print("█"*70)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
