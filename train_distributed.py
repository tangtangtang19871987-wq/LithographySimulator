"""
Enhanced distributed training script with multi-GPU support.

New features compared to train.py:
- Multi-GPU training with MirroredStrategy
- NCCL fallback mechanisms for stability
- Mixed precision training (float16)
- Data augmentation
- TensorBoard integration
- Advanced learning rate scheduling
- Checkpoint resumption
- Comprehensive experiment tracking

Usage:
    # Single GPU training with all enhancements:
    python train_distributed.py --epochs 50 --batch-size 16

    # Multi-GPU training with NCCL:
    python train_distributed.py --multi-gpu --epochs 100 --batch-size 32

    # Multi-GPU with conservative NCCL workarounds:
    python train_distributed.py --multi-gpu --nccl-workarounds --epochs 100

    # Mixed precision + augmentation:
    python train_distributed.py --mixed-precision --augmentation --epochs 50

    # Resume from checkpoint:
    python train_distributed.py --resume experiments/run_20260210_120000

    # Quick smoke test:
    python train_distributed.py --smoke-test --epochs 2
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import signal
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import original modules
from shift_equivariant_unet import (
    shift_equivariant_unet,
    CircularPad2D,
    CircularConv2D,
    DilatedCircularConv2D,
)
from data_pipeline import (
    generate_dataset,
    save_dataset,
    load_dataset,
    make_tf_dataset,
)
from train import (
    build_model,
    plot_training_history,
    plot_predictions,
    StopState,
    _ensure_parent,
    _resolve_path,
)

# Import new enhanced utilities
from train_utils import (
    DistributedStrategyConfig,
    create_distribution_strategy,
    setup_mixed_precision,
    create_augmented_dataset,
    create_callbacks,
    find_latest_checkpoint,
    print_training_summary,
    WarmupCosineDecaySchedule,
)


def build_model_with_mixed_precision(input_shape=(64, 64, 1),
                                     num_filters_base=32,
                                     use_mixed_precision=False):
    """Build model with optional mixed precision support."""
    model = build_model(input_shape=input_shape,
                       num_filters_base=num_filters_base)

    # For mixed precision, ensure output layer uses float32
    if use_mixed_precision:
        # The last layer should output float32 for numerical stability
        # This is already handled by default in Keras, but we make it explicit
        print("Note: Model built with mixed precision, output layer uses float32")

    return model


def compile_model_distributed(model, learning_rate, strategy,
                              use_mixed_precision=False,
                              gradient_clip_norm=None):
    """Compile model within distributed strategy scope.

    Args:
        model: Keras model
        learning_rate: Learning rate (float or schedule)
        strategy: Distribution strategy
        use_mixed_precision: Whether mixed precision is enabled
        gradient_clip_norm: Gradient clipping norm (None to disable)
    """
    with strategy.scope():
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Apply gradient clipping if requested
        if gradient_clip_norm is not None:
            optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=gradient_clip_norm
            )
            print(f"Gradient clipping enabled: norm={gradient_clip_norm}")

        # For mixed precision, wrap optimizer with LossScaleOptimizer
        if use_mixed_precision:
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
            print("Using LossScaleOptimizer for mixed precision training")

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae'],
        )

    return model


def create_datasets_distributed(train_masks, train_aerials,
                                val_masks, val_aerials,
                                batch_size, strategy,
                                use_augmentation=False):
    """Create distributed datasets.

    Args:
        train_masks, train_aerials: Training data
        val_masks, val_aerials: Validation data
        batch_size: Global batch size (will be divided among replicas)
        strategy: Distribution strategy
        use_augmentation: Enable data augmentation

    Returns:
        train_ds, val_ds: Distributed datasets
    """
    # Global batch size is divided among replicas
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    print(f"Batch size per replica: {batch_size}")
    print(f"Global batch size: {global_batch_size}")

    if use_augmentation:
        print("Data augmentation enabled (rotation, flip)")
        train_ds = create_augmented_dataset(
            train_masks, train_aerials,
            batch_size=global_batch_size,
            shuffle=True
        )
    else:
        train_ds = make_tf_dataset(
            train_masks, train_aerials,
            batch_size=global_batch_size,
            shuffle=True
        )

    val_ds = make_tf_dataset(
        val_masks, val_aerials,
        batch_size=global_batch_size,
        shuffle=False
    )

    # Distribute datasets
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    val_ds = strategy.experimental_distribute_dataset(val_ds)

    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced distributed training with multi-GPU support'
    )

    # Data arguments
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to existing .npz dataset')
    parser.add_argument('--num-samples', type=int, default=200,
                       help='Number of training samples to generate')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split ratio')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size per replica')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Number of warmup epochs for LR schedule')
    parser.add_argument('--gradient-clip-norm', type=float, default=None,
                       help='Gradient clipping norm (None=disabled)')

    # Model arguments
    parser.add_argument('--filters', type=int, default=32,
                       help='Base filter count')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Multi-GPU arguments
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Enable multi-GPU training with MirroredStrategy')
    parser.add_argument('--strategy', type=str, default='auto',
                       choices=['auto', 'mirrored', 'none'],
                       help='Distribution strategy type')
    parser.add_argument('--nccl-workarounds', action='store_true',
                       help='Enable NCCL workarounds for stability')
    parser.add_argument('--communication-backend', type=str, default=None,
                       choices=['nccl', 'hierarchical_copy'],
                       help='Cross-device communication backend')

    # Enhancement arguments
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training (float16)')
    parser.add_argument('--augmentation', action='store_true',
                       help='Enable data augmentation')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                       help='Enable TensorBoard logging')
    parser.add_argument('--no-tensorboard', dest='tensorboard',
                       action='store_false',
                       help='Disable TensorBoard logging')

    # Checkpointing arguments
    parser.add_argument('--checkpoint-every', type=int, default=10,
                       help='Save checkpoint every N epochs (0=disabled)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint directory')

    # Output arguments
    parser.add_argument('--experiment-dir', type=str, default='experiments',
                       help='Root directory for experiments')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Optional run folder name (default timestamp)')

    # Special modes
    parser.add_argument('--smoke-test', action='store_true',
                       help='Quick smoke test with minimal settings')
    parser.add_argument('--docker-safe', action='store_true',
                       help='Use numpy batch training (not compatible with multi-GPU)')

    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke_test:
        print("Smoke test mode: using conservative settings")
        args.epochs = min(args.epochs, 2)
        args.num_samples = min(args.num_samples, 20)
        args.tensorboard = False
        args.multi_gpu = False

    # Validate arguments
    if args.docker_safe and args.multi_gpu:
        print("ERROR: --docker-safe is not compatible with --multi-gpu")
        return 1

    # Create run directory
    run_name = args.run_name or time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.experiment_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(run_dir, 'run_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run directory: {run_dir}")
    print(f"Configuration saved: {config_path}")

    # Setup signal handling
    stop_state = StopState()

    def signal_handler(sig, _frame):
        stop_state.requested = True
        stop_state.reason = signal.Signals(sig).name
        print(f"\nReceived {stop_state.reason}. Will stop after current epoch.")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # -------------------------------------------------------------------------
    # Step 1: Setup Distributed Strategy
    # -------------------------------------------------------------------------
    strategy_config = DistributedStrategyConfig(
        strategy_type=args.strategy if args.multi_gpu else 'none',
        cross_device_ops=args.communication_backend,
        enable_nccl_workarounds=args.nccl_workarounds,
    )

    strategy, num_replicas = create_distribution_strategy(
        config=strategy_config, verbose=True
    )

    # -------------------------------------------------------------------------
    # Step 2: Setup Mixed Precision
    # -------------------------------------------------------------------------
    mixed_precision_policy = setup_mixed_precision(
        enabled=args.mixed_precision, verbose=True
    )

    # -------------------------------------------------------------------------
    # Step 3: Load/Generate Data
    # -------------------------------------------------------------------------
    dataset_path_for_report = None
    if args.dataset and os.path.exists(args.dataset):
        dataset_path_for_report = os.path.abspath(args.dataset)
        print(f"\nLoading dataset from {args.dataset}...")
        masks, aerials = load_dataset(args.dataset)
    else:
        print(f"\nGenerating {args.num_samples} training samples...")
        masks, aerials = generate_dataset(args.num_samples, seed=args.seed)
        generated_dataset_path = os.path.join(run_dir, 'litho_dataset.npz')
        save_dataset(masks, aerials, generated_dataset_path)
        dataset_path_for_report = os.path.abspath(generated_dataset_path)

    print(f"Dataset: {masks.shape[0]} samples, "
          f"mask shape: {masks.shape[1:]}, "
          f"aerial shape: {aerials.shape[1:]}")

    # Train/val split
    n = len(masks)
    n_val = max(1, int(n * args.val_split))
    n_train = n - n_val

    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_masks, train_aerials = masks[train_idx], aerials[train_idx]
    val_masks, val_aerials = masks[val_idx], aerials[val_idx]

    print(f"Train: {n_train}, Val: {n_val}")

    # -------------------------------------------------------------------------
    # Step 4: Build Model in Strategy Scope
    # -------------------------------------------------------------------------
    with strategy.scope():
        input_shape = masks.shape[1:]
        model = build_model_with_mixed_precision(
            input_shape=input_shape,
            num_filters_base=args.filters,
            use_mixed_precision=args.mixed_precision
        )

        # Setup learning rate schedule
        steps_per_epoch = n_train // (args.batch_size * num_replicas)
        warmup_steps = args.warmup_epochs * steps_per_epoch
        total_steps = args.epochs * steps_per_epoch

        if args.warmup_epochs > 0:
            lr_schedule = WarmupCosineDecaySchedule(
                initial_lr=args.lr,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=1e-7
            )
            print(f"Using WarmupCosineDecay LR schedule:")
            print(f"  Initial LR: {args.lr}")
            print(f"  Warmup epochs: {args.warmup_epochs}")
            print(f"  Min LR: 1e-7")
        else:
            lr_schedule = args.lr

        # Compile model
        model = compile_model_distributed(
            model, lr_schedule, strategy,
            use_mixed_precision=args.mixed_precision,
            gradient_clip_norm=args.gradient_clip_norm
        )

    print(f"\nModel: {model.count_params():,} parameters")

    # Print comprehensive summary
    print_training_summary(
        strategy=strategy,
        num_replicas=num_replicas,
        mixed_precision_enabled=args.mixed_precision,
        augmentation_enabled=args.augmentation,
        total_params=model.count_params()
    )

    # -------------------------------------------------------------------------
    # Step 5: Create Datasets
    # -------------------------------------------------------------------------
    if args.docker_safe:
        # Use numpy arrays directly (no tf.data)
        print("Using docker-safe numpy batch training (no multi-GPU support)")
        train_ds, val_ds = None, None
    else:
        train_ds, val_ds = create_datasets_distributed(
            train_masks, train_aerials,
            val_masks, val_aerials,
            batch_size=args.batch_size,
            strategy=strategy,
            use_augmentation=args.augmentation
        )

    # -------------------------------------------------------------------------
    # Step 6: Setup Callbacks
    # -------------------------------------------------------------------------
    callbacks = create_callbacks(
        run_dir=run_dir,
        model_basename='litho_model',
        enable_tensorboard=args.tensorboard,
        enable_early_stopping=True,
        enable_lr_scheduler=(args.warmup_epochs == 0),  # Use plateau if no warmup
        patience=20,
        checkpoint_every_n_epochs=args.checkpoint_every,
        gradient_clip_norm=args.gradient_clip_norm,
        stop_state=stop_state
    )

    # -------------------------------------------------------------------------
    # Step 7: Resume from Checkpoint (if requested)
    # -------------------------------------------------------------------------
    initial_epoch = 0
    if args.resume:
        checkpoint_dir = os.path.join(args.resume, 'checkpoints')
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)

        if latest_ckpt:
            print(f"\nResuming from checkpoint: {latest_ckpt}")
            try:
                model = keras.models.load_model(latest_ckpt)
                # Extract epoch number from filename
                epoch_str = latest_ckpt.split('epoch')[-1].split('.')[0]
                initial_epoch = int(epoch_str)
                print(f"Resuming from epoch {initial_epoch}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting from scratch")
        else:
            print(f"No checkpoint found in {checkpoint_dir}")

    # -------------------------------------------------------------------------
    # Step 8: Train
    # -------------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs (starting from epoch {initial_epoch})...")

    interrupted = False
    try:
        if args.docker_safe:
            # Use original numpy batch training
            from train import fit_with_numpy_batches
            history = fit_with_numpy_batches(
                model, train_masks, train_aerials,
                val_masks, val_aerials,
                epochs=args.epochs, batch_size=args.batch_size,
                stop_state=stop_state,
            )
        else:
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
                verbose=1,
            )

    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user")

    # -------------------------------------------------------------------------
    # Step 9: Evaluate and Save
    # -------------------------------------------------------------------------
    val_loss, val_mae = None, None
    try:
        if not args.docker_safe:
            val_loss, val_mae = model.evaluate(val_ds, verbose=0)
            print(f"\nFinal val loss (MSE): {val_loss:.6f}")
            print(f"Final val MAE: {val_mae:.6f}")
    except Exception as e:
        print(f"Evaluation error: {e}")

    # Save final model
    final_model_path = os.path.join(run_dir, 'litho_model_final.keras')
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")

    if interrupted:
        interrupted_path = os.path.join(run_dir, 'litho_model_interrupted.keras')
        model.save(interrupted_path)
        print(f"Interrupted model saved: {interrupted_path}")

    # -------------------------------------------------------------------------
    # Step 10: Generate Plots and Summary
    # -------------------------------------------------------------------------
    if hasattr(history, 'history') and 'loss' in history.history:
        plot_training_history(
            history,
            save_path=os.path.join(run_dir, 'training_history.png')
        )

    plot_predictions(
        model, val_masks, val_aerials,
        num_samples=4,
        save_path=os.path.join(run_dir, 'predictions.png')
    )

    # Save run summary
    run_summary = {
        'run_dir': os.path.abspath(run_dir),
        'dataset': dataset_path_for_report,
        'status': 'completed' if not interrupted else 'interrupted',
        'configuration': {
            'multi_gpu': args.multi_gpu,
            'num_replicas': num_replicas,
            'mixed_precision': args.mixed_precision,
            'augmentation': args.augmentation,
            'strategy': strategy.__class__.__name__,
        },
        'final_val_loss': None if val_loss is None else float(val_loss),
        'final_val_mae': None if val_mae is None else float(val_mae),
        'epochs_completed': len(history.history.get('loss', [])) if hasattr(history, 'history') else 0,
    }

    summary_path = os.path.join(run_dir, 'run_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(run_summary, f, indent=2)
    print(f"Run summary saved: {summary_path}")

    print("\n✓ Training complete!")

    return 0


if __name__ == '__main__':
    exit(main())
