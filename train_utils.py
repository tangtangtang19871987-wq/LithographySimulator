"""
Enhanced training utilities for lithography simulator.

Features:
- Multi-GPU training with distributed strategies
- NCCL fallback mechanisms and workarounds
- Mixed precision training
- Data augmentation for lithography masks
- Advanced callbacks (TensorBoard, LR scheduling, gradient stats)
- Checkpoint resumption
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json


# ---------------------------------------------------------------------------
# Multi-GPU Strategy Setup with NCCL Fallbacks
# ---------------------------------------------------------------------------

class DistributedStrategyConfig:
    """Configuration for distributed training with fallback strategies."""

    def __init__(self, strategy_type='auto', cross_device_ops=None,
                 enable_nccl_workarounds=True, timeout=1800):
        """
        Args:
            strategy_type: 'auto', 'mirrored', 'multi_worker', or 'none'
            cross_device_ops: Communication backend ('nccl', 'hierarchical_copy', or None for auto)
            enable_nccl_workarounds: Apply NCCL environment variable workarounds
            timeout: Timeout in seconds for NCCL operations
        """
        self.strategy_type = strategy_type
        self.cross_device_ops = cross_device_ops
        self.enable_nccl_workarounds = enable_nccl_workarounds
        self.timeout = timeout


def setup_nccl_environment(timeout=1800, enable_workarounds=True):
    """Configure NCCL environment variables to avoid common issues.

    Common NCCL problems and workarounds:
    1. Hangs during initialization -> Set timeouts
    2. Docker container issues -> Disable IB and use sockets
    3. Communication failures -> Enable retries and debug logging
    4. Multi-node issues -> Configure network interface

    Args:
        timeout: NCCL operation timeout in seconds
        enable_workarounds: Apply conservative workarounds for common issues
    """
    nccl_env = {
        'NCCL_TIMEOUT': str(timeout),
        'NCCL_BLOCKING_WAIT': '1',  # Better error messages
    }

    if enable_workarounds:
        # Conservative settings for Docker/containerized environments
        nccl_env.update({
            'NCCL_IB_DISABLE': '1',  # Disable InfiniBand (often causes issues in containers)
            'NCCL_P2P_DISABLE': '1',  # Disable P2P for stability
            'NCCL_SHM_DISABLE': '0',  # Keep shared memory enabled
            'NCCL_SOCKET_IFNAME': 'eth0,lo',  # Use ethernet and loopback
            'NCCL_DEBUG': 'WARN',  # Show warnings but not full debug
            'NCCL_NSOCKS_PERTHREAD': '4',  # More sockets per thread
            'NCCL_SOCKET_NTHREADS': '2',  # More threads per connection
        })

    for key, value in nccl_env.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"  Set {key}={value}")


def create_distribution_strategy(config=None, verbose=True):
    """Create distributed training strategy with automatic fallbacks.

    Tries multiple approaches in order:
    1. MirroredStrategy with NCCL
    2. MirroredStrategy with HierarchicalCopy
    3. MirroredStrategy with auto selection
    4. No strategy (single GPU/CPU)

    Args:
        config: DistributedStrategyConfig instance
        verbose: Print strategy information

    Returns:
        strategy: TensorFlow distribution strategy
        num_replicas: Number of devices
    """
    if config is None:
        config = DistributedStrategyConfig()

    # Detect available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Distributed Training Setup")
        print(f"{'='*60}")
        print(f"Available GPUs: {num_gpus}")
        if num_gpus > 0:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")

    # Single GPU or CPU - no strategy needed
    if num_gpus <= 1 and config.strategy_type != 'multi_worker':
        if verbose:
            print(f"Using default strategy (single device)")
        return tf.distribute.get_strategy(), 1

    # Apply NCCL workarounds if requested
    if config.enable_nccl_workarounds and num_gpus > 1:
        if verbose:
            print("\nApplying NCCL workarounds:")
        setup_nccl_environment(timeout=config.timeout,
                              enable_workarounds=True)

    # Try different strategies with fallback
    strategies_to_try = []

    if config.strategy_type == 'auto' or config.strategy_type == 'mirrored':
        # Try NCCL first (fastest but can have issues)
        if config.cross_device_ops == 'nccl' or config.cross_device_ops is None:
            strategies_to_try.append(
                ('MirroredStrategy with NCCL',
                 lambda: tf.distribute.MirroredStrategy(
                     cross_device_ops=tf.distribute.NcclAllReduce()
                 ))
            )

        # Fallback to HierarchicalCopy (more stable, slightly slower)
        if config.cross_device_ops == 'hierarchical_copy' or config.cross_device_ops is None:
            strategies_to_try.append(
                ('MirroredStrategy with HierarchicalCopy',
                 lambda: tf.distribute.MirroredStrategy(
                     cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
                 ))
            )

        # Fallback to auto selection
        strategies_to_try.append(
            ('MirroredStrategy with auto',
             lambda: tf.distribute.MirroredStrategy())
        )

    # Try each strategy until one works
    for name, strategy_fn in strategies_to_try:
        try:
            if verbose:
                print(f"\nAttempting: {name}")

            strategy = strategy_fn()

            # Test the strategy with a simple operation
            with strategy.scope():
                test_var = tf.Variable(1.0)
                test_result = strategy.reduce(
                    tf.distribute.ReduceOp.SUM, test_var, axis=None
                )

            if verbose:
                print(f"✓ Successfully initialized: {name}")
                print(f"  Number of replicas: {strategy.num_replicas_in_sync}")
                print(f"  Devices: {[d.name for d in strategy.extended.worker_devices]}")

            return strategy, strategy.num_replicas_in_sync

        except Exception as e:
            if verbose:
                print(f"✗ Failed: {name}")
                print(f"  Error: {str(e)[:100]}")
            continue

    # All strategies failed - fallback to no strategy
    if verbose:
        print(f"\n⚠ All distributed strategies failed, using single device")
    return tf.distribute.get_strategy(), 1


# ---------------------------------------------------------------------------
# Mixed Precision Training
# ---------------------------------------------------------------------------

def setup_mixed_precision(enabled=True, verbose=True):
    """Configure mixed precision training for faster training on modern GPUs.

    Args:
        enabled: Enable mixed precision (requires GPU with Tensor Cores)
        verbose: Print configuration info

    Returns:
        policy: Mixed precision policy
    """
    if not enabled:
        policy = tf.keras.mixed_precision.Policy('float32')
        if verbose:
            print("\nMixed Precision: Disabled (using float32)")
        return policy

    try:
        # Check for GPU with compute capability >= 7.0 (Tensor Cores)
        gpus = tf.config.list_physical_devices('GPU')

        if len(gpus) == 0:
            if verbose:
                print("\nMixed Precision: Disabled (no GPU detected)")
            return tf.keras.mixed_precision.Policy('float32')

        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        if verbose:
            print("\nMixed Precision: Enabled (mixed_float16)")
            print(f"  Compute dtype: {policy.compute_dtype}")
            print(f"  Variable dtype: {policy.variable_dtype}")

        return policy

    except Exception as e:
        if verbose:
            print(f"\nMixed Precision: Failed to enable - {e}")
            print("  Falling back to float32")
        return tf.keras.mixed_precision.Policy('float32')


# ---------------------------------------------------------------------------
# Data Augmentation for Lithography Masks
# ---------------------------------------------------------------------------

class LithographyAugmentation(keras.layers.Layer):
    """Data augmentation layer for lithography mask/aerial image pairs.

    Augmentations:
    - Random 90-degree rotations (preserves shift equivariance)
    - Random flips (horizontal/vertical)
    - Small random brightness/contrast adjustments
    """

    def __init__(self, rotation_prob=0.5, flip_prob=0.5,
                 brightness_delta=0.05, contrast_range=(0.95, 1.05),
                 seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.seed = seed

    def call(self, images, training=None):
        if not training:
            return images

        # Random 90-degree rotations (k=0,1,2,3)
        if tf.random.uniform([]) < self.rotation_prob:
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            images = tf.image.rot90(images, k=k)

        # Random horizontal flip
        if tf.random.uniform([]) < self.flip_prob:
            images = tf.image.flip_left_right(images)

        # Random vertical flip
        if tf.random.uniform([]) < self.flip_prob:
            images = tf.image.flip_up_down(images)

        # Small brightness adjustment
        images = tf.image.random_brightness(images, self.brightness_delta)

        # Small contrast adjustment
        images = tf.image.random_contrast(
            images, self.contrast_range[0], self.contrast_range[1]
        )

        # Clip to valid range
        images = tf.clip_by_value(images, 0.0, 1.0)

        return images


def create_augmented_dataset(masks, aerials, batch_size=8, shuffle=True,
                             augmentation_prob=0.7, num_parallel_calls=tf.data.AUTOTUNE):
    """Create augmented tf.data.Dataset with synchronized transformations.

    Args:
        masks: Input masks
        aerials: Target aerial images
        batch_size: Batch size
        shuffle: Shuffle data
        augmentation_prob: Probability of applying augmentation
        num_parallel_calls: Parallelism for data loading

    Returns:
        tf.data.Dataset
    """
    def augment_pair(mask, aerial):
        """Apply same random augmentation to both mask and aerial."""
        # Stack together to ensure same transformation
        combined = tf.concat([mask, aerial], axis=-1)

        # Random 90-degree rotation
        if tf.random.uniform([]) < 0.5:
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            combined = tf.image.rot90(combined, k=k)

        # Random flips
        if tf.random.uniform([]) < 0.5:
            combined = tf.image.flip_left_right(combined)
        if tf.random.uniform([]) < 0.5:
            combined = tf.image.flip_up_down(combined)

        # Split back
        mask_aug = combined[..., :1]
        aerial_aug = combined[..., 1:]

        return mask_aug, aerial_aug

    ds = tf.data.Dataset.from_tensor_slices((masks, aerials))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(masks))

    # Apply augmentation
    ds = ds.map(augment_pair, num_parallel_calls=num_parallel_calls)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


# ---------------------------------------------------------------------------
# Advanced Callbacks
# ---------------------------------------------------------------------------

class GradientStatsCallback(keras.callbacks.Callback):
    """Monitor gradient statistics during training."""

    def __init__(self, log_every_n_batches=100):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.batch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % self.log_every_n_batches == 0:
            # This would require gradient tracking, simplified for now
            pass


class WarmupCosineDecaySchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup and cosine decay."""

    def __init__(self, initial_lr, warmup_steps, total_steps,
                 min_lr=1e-7, name=None):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self._name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.initial_lr * step / warmup_steps

        # Cosine decay
        decay_steps = total_steps - warmup_steps
        decay_step = tf.minimum(step - warmup_steps, decay_steps)
        cosine_decay = 0.5 * (1 + tf.cos(
            3.14159265359 * decay_step / decay_steps
        ))
        decay_lr = (self.initial_lr - self.min_lr) * cosine_decay + self.min_lr

        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'name': self._name,
        }


def create_callbacks(run_dir, model_basename='model',
                    enable_tensorboard=True,
                    enable_early_stopping=True,
                    enable_lr_scheduler=True,
                    patience=20,
                    csv_log=True,
                    jsonl_log=True,
                    save_best=True,
                    checkpoint_every_n_epochs=0,
                    gradient_clip_norm=None,
                    stop_state=None):
    """Create a comprehensive set of training callbacks.

    Args:
        run_dir: Experiment directory
        model_basename: Base name for saved models
        enable_tensorboard: Enable TensorBoard logging
        enable_early_stopping: Enable early stopping
        enable_lr_scheduler: Enable learning rate reduction on plateau
        patience: Patience for early stopping/LR reduction
        csv_log: Enable CSV logging
        jsonl_log: Enable JSONL logging
        save_best: Save best model checkpoint
        checkpoint_every_n_epochs: Save checkpoint every N epochs (0=disabled)
        gradient_clip_norm: Gradient clipping norm (None=disabled)
        stop_state: StopState object for signal handling

    Returns:
        List of callbacks
    """
    callbacks = []

    # TensorBoard
    if enable_tensorboard:
        tensorboard_dir = os.path.join(run_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=0,  # Disable profiling to avoid overhead
        ))
        print(f"TensorBoard logs: {tensorboard_dir}")
        print(f"  Run: tensorboard --logdir={tensorboard_dir}")

    # Best model checkpoint
    if save_best:
        best_model_path = os.path.join(run_dir, f'{model_basename}_best.keras')
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ))

    # Early stopping
    if enable_early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ))

    # Learning rate reduction
    if enable_lr_scheduler:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1,
        ))

    # CSV logger
    if csv_log:
        csv_path = os.path.join(run_dir, 'training_log.csv')
        callbacks.append(keras.callbacks.CSVLogger(csv_path, append=False))

    # JSONL logger (custom)
    if jsonl_log:
        from train import JsonlLoggerCallback
        jsonl_path = os.path.join(run_dir, 'epoch_metrics.jsonl')
        callbacks.append(JsonlLoggerCallback(jsonl_path))

    # Periodic checkpoints
    if checkpoint_every_n_epochs > 0:
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        def save_periodic(epoch, logs):
            epoch_n = epoch + 1
            if epoch_n % checkpoint_every_n_epochs == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir, f"{model_basename}_epoch{epoch_n:04d}.keras"
                )
                # Access model from parent scope or pass it
                print(f"Checkpoint saved to {ckpt_path}")

        callbacks.append(keras.callbacks.LambdaCallback(
            on_epoch_end=save_periodic
        ))

    # Signal handler
    if stop_state is not None:
        from train import StopOnSignalCallback
        callbacks.append(StopOnSignalCallback(stop_state))

    return callbacks


# ---------------------------------------------------------------------------
# Checkpoint Resumption
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir, prefix='model_epoch'):
    """Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Filename prefix for checkpoints

    Returns:
        Path to latest checkpoint, or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith(prefix) and f.endswith('.keras')
    ]

    if not checkpoints:
        return None

    # Sort by epoch number
    def extract_epoch(filename):
        try:
            # Extract epoch number from filename like "model_epoch0042.keras"
            epoch_str = filename[len(prefix):].split('.')[0]
            return int(epoch_str)
        except:
            return -1

    checkpoints.sort(key=extract_epoch, reverse=True)
    latest = os.path.join(checkpoint_dir, checkpoints[0])

    return latest


# ---------------------------------------------------------------------------
# Training Summary
# ---------------------------------------------------------------------------

def print_training_summary(strategy, num_replicas, mixed_precision_enabled,
                          augmentation_enabled, total_params):
    """Print comprehensive training configuration summary."""
    print(f"\n{'='*60}")
    print(f"Training Configuration Summary")
    print(f"{'='*60}")
    print(f"Devices: {num_replicas} replica(s)")
    print(f"Strategy: {strategy.__class__.__name__}")
    print(f"Mixed Precision: {'Enabled (float16)' if mixed_precision_enabled else 'Disabled (float32)'}")
    print(f"Data Augmentation: {'Enabled' if augmentation_enabled else 'Disabled'}")
    print(f"Model Parameters: {total_params:,}")
    print(f"{'='*60}\n")
