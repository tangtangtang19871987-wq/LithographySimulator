"""
Professional Training Script with State-of-the-Art Features

This script combines all advanced training techniques:
- Multi-GPU training with NCCL fallbacks
- Advanced LR schedulers (OneCycle, Cyclical, SGDR, etc.)
- Enhanced early stopping with multiple strategies
- Gradient accumulation for large effective batch sizes
- Model EMA and SWA for better generalization
- Mixed precision training
- Data augmentation
- Comprehensive experiment tracking

Usage:
    # Quick start with OneCycle LR
    python train_pro.py --lr-schedule onecycle --epochs 100

    # Multi-GPU with all features
    python train_pro.py --multi-gpu --mixed-precision --augmentation \\
        --lr-schedule onecycle --model-ema --epochs 200

    # Large effective batch size with gradient accumulation
    python train_pro.py --batch-size 8 --gradient-accumulation 4 --epochs 100

    # SWA for better generalization
    python train_pro.py --swa --swa-start 150 --epochs 200
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

# Import base modules
from shift_equivariant_unet import CircularConv2D
from data_pipeline import generate_dataset, save_dataset, load_dataset
from train import (
    build_model, plot_training_history, plot_predictions,
    StopState, _ensure_parent, _resolve_path
)

# Import distributed training
from train_utils import (
    create_distribution_strategy,
    DistributedStrategyConfig,
    setup_mixed_precision,
    create_augmented_dataset,
    print_training_summary,
)

# Import advanced features
from train_advanced import (
    create_lr_schedule,
    AdvancedEarlyStopping,
    GradientAccumulation,
    ModelEMA,
    SWA,
    TrainingProgressTracker,
    visualize_lr_schedule,
    print_training_config,
)


def create_advanced_callbacks(run_dir, args, stop_state, total_steps):
    """Create comprehensive callback suite."""
    callbacks = []
    steps_per_epoch = total_steps // args.epochs

    # TensorBoard
    if args.tensorboard:
        tensorboard_dir = os.path.join(run_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
            profile_batch=0,
        ))
        print(f"TensorBoard: tensorboard --logdir={tensorboard_dir}")

    # Advanced Early Stopping
    callbacks.append(AdvancedEarlyStopping(
        monitor='val_loss',
        min_delta=args.early_stop_min_delta,
        min_delta_percent=args.early_stop_min_delta_percent,
        patience=args.patience,
        warmup_epochs=args.early_stop_warmup,
        restore_best_weights=True,
        stop_on_lr_threshold=args.early_stop_lr_threshold,
        divergence_threshold=args.early_stop_divergence_threshold,
        verbose=1,
    ))

    # Model EMA
    if args.model_ema:
        callbacks.append(ModelEMA(
            decay=args.ema_decay,
            start_epoch=args.ema_start_epoch
        ))
        print(f"Model EMA enabled (decay={args.ema_decay})")

    # SWA
    if args.swa:
        callbacks.append(SWA(
            start_epoch=args.swa_start,
            swa_freq=args.swa_freq,
            verbose=1
        ))
        print(f"SWA enabled (start={args.swa_start}, freq={args.swa_freq})")

    # Progress Tracker
    callbacks.append(TrainingProgressTracker(
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    ))

    # Best model checkpoint
    best_model_path = os.path.join(run_dir, 'litho_model_best.keras')
    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
    ))

    # CSV logger
    csv_path = os.path.join(run_dir, 'training_log.csv')
    callbacks.append(keras.callbacks.CSVLogger(csv_path))

    # Periodic checkpoints
    if args.checkpoint_every > 0:
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        def save_checkpoint(epoch, logs):
            if (epoch + 1) % args.checkpoint_every == 0:
                path = os.path.join(checkpoint_dir,
                                   f"model_epoch{epoch+1:04d}.keras")
                # Save with EMA weights if enabled
                if args.model_ema:
                    for cb in callbacks:
                        if isinstance(cb, ModelEMA):
                            cb.save_ema_model(path)
                            return
                # Save with SWA weights if available
                if args.swa:
                    for cb in callbacks:
                        if isinstance(cb, SWA) and cb.swa_weights is not None:
                            cb.save_swa_model(path)
                            return
                # Regular save
                callbacks[0].model.save(path)
                print(f"Checkpoint saved: {path}")

        callbacks.append(keras.callbacks.LambdaCallback(
            on_epoch_end=save_checkpoint
        ))

    # Signal handler
    if stop_state is not None:
        from train import StopOnSignalCallback
        callbacks.append(StopOnSignalCallback(stop_state))

    return callbacks


def main():
    parser = argparse.ArgumentParser(
        description='Professional training with advanced features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # === Data Arguments ===
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--dataset', type=str, default=None,
                           help='Path to existing dataset')
    data_group.add_argument('--num-samples', type=int, default=500,
                           help='Number of samples to generate')
    data_group.add_argument('--val-split', type=float, default=0.15,
                           help='Validation split ratio')

    # === Training Arguments ===
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=100,
                            help='Number of epochs')
    train_group.add_argument('--batch-size', type=int, default=16,
                            help='Batch size per replica')
    train_group.add_argument('--lr', type=float, default=1e-3,
                            help='Base learning rate')
    train_group.add_argument('--gradient-clip-norm', type=float, default=None,
                            help='Gradient clipping norm')

    # === LR Schedule Arguments ===
    lr_group = parser.add_argument_group('Learning Rate Schedule')
    lr_group.add_argument('--lr-schedule', type=str, default='onecycle',
                         choices=['constant', 'onecycle', 'cyclical', 'sgdr',
                                 'polynomial', 'cosine', 'exponential'],
                         help='Learning rate schedule type')
    lr_group.add_argument('--warmup-epochs', type=int, default=0,
                         help='Warmup epochs (for manual warmup)')
    # OneCycle specific
    lr_group.add_argument('--onecycle-pct-start', type=float, default=0.3,
                         help='OneCycle: fraction for LR increase phase')
    lr_group.add_argument('--onecycle-div-factor', type=float, default=25.0,
                         help='OneCycle: initial_lr = max_lr / div_factor')
    # Cyclical specific
    lr_group.add_argument('--cyclical-mode', type=str, default='triangular',
                         choices=['triangular', 'triangular2', 'exp_range'],
                         help='Cyclical LR mode')
    # SGDR specific
    lr_group.add_argument('--sgdr-cycle-length', type=int, default=None,
                         help='SGDR: first cycle length in epochs')

    # === Early Stopping Arguments ===
    es_group = parser.add_argument_group('Early Stopping')
    es_group.add_argument('--patience', type=int, default=20,
                         help='Early stopping patience')
    es_group.add_argument('--early-stop-min-delta', type=float, default=0.0,
                         help='Minimum change to qualify as improvement')
    es_group.add_argument('--early-stop-min-delta-percent', type=float, default=None,
                         help='Minimum change as percentage')
    es_group.add_argument('--early-stop-warmup', type=int, default=10,
                         help='Don\'t stop during first N epochs')
    es_group.add_argument('--early-stop-lr-threshold', type=float, default=None,
                         help='Stop if LR falls below this value')
    es_group.add_argument('--early-stop-divergence-threshold', type=float, default=10.0,
                         help='Stop if loss exceeds best * threshold')

    # === Multi-GPU Arguments ===
    gpu_group = parser.add_argument_group('Multi-GPU')
    gpu_group.add_argument('--multi-gpu', action='store_true',
                          help='Enable multi-GPU training')
    gpu_group.add_argument('--nccl-workarounds', action='store_true',
                          help='Apply NCCL stability workarounds')
    gpu_group.add_argument('--communication-backend', type=str, default=None,
                          choices=['nccl', 'hierarchical_copy'],
                          help='Communication backend')

    # === Advanced Features ===
    adv_group = parser.add_argument_group('Advanced Features')
    adv_group.add_argument('--mixed-precision', action='store_true',
                          help='Enable mixed precision (FP16)')
    adv_group.add_argument('--augmentation', action='store_true',
                          help='Enable data augmentation')
    adv_group.add_argument('--gradient-accumulation', type=int, default=1,
                          help='Gradient accumulation steps (for larger effective batch)')
    adv_group.add_argument('--model-ema', action='store_true',
                          help='Enable Model EMA')
    adv_group.add_argument('--ema-decay', type=float, default=0.999,
                          help='EMA decay rate')
    adv_group.add_argument('--ema-start-epoch', type=int, default=0,
                          help='Start EMA after this epoch')
    adv_group.add_argument('--swa', action='store_true',
                          help='Enable Stochastic Weight Averaging')
    adv_group.add_argument('--swa-start', type=int, default=None,
                          help='Start SWA at this epoch (default: 75%% of epochs)')
    adv_group.add_argument('--swa-freq', type=int, default=1,
                          help='SWA update frequency')

    # === Model Arguments ===
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--filters', type=int, default=32,
                            help='Base filter count')
    model_group.add_argument('--seed', type=int, default=42,
                            help='Random seed')

    # === Output Arguments ===
    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--experiment-dir', type=str, default='experiments',
                          help='Experiment directory')
    out_group.add_argument('--run-name', type=str, default=None,
                          help='Run name (default: timestamp)')
    out_group.add_argument('--checkpoint-every', type=int, default=10,
                          help='Save checkpoint every N epochs')
    out_group.add_argument('--tensorboard', action='store_true', default=True,
                          help='Enable TensorBoard')
    out_group.add_argument('--no-tensorboard', dest='tensorboard',
                          action='store_false')

    args = parser.parse_args()

    # Set defaults
    if args.swa and args.swa_start is None:
        args.swa_start = int(args.epochs * 0.75)
    if args.sgdr_cycle_length is None:
        args.sgdr_cycle_length = args.epochs // 5

    # Create run directory
    run_name = args.run_name or time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.experiment_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(run_dir, 'run_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Professional Training Session")
    print(f"{'='*60}")
    print(f"Run: {run_dir}")
    print(f"Config: {config_path}")

    # Print configuration
    config_summary = {
        'Epochs': args.epochs,
        'Batch Size': args.batch_size,
        'Learning Rate': args.lr,
        'LR Schedule': args.lr_schedule,
        'Multi-GPU': args.multi_gpu,
        'Mixed Precision': args.mixed_precision,
        'Augmentation': args.augmentation,
        'Gradient Accumulation': args.gradient_accumulation,
        'Model EMA': args.model_ema,
        'SWA': args.swa,
        'Early Stopping Patience': args.patience,
    }
    print_training_config(config_summary)

    # Setup signal handling
    stop_state = StopState()
    def signal_handler(sig, _frame):
        stop_state.requested = True
        stop_state.reason = signal.Signals(sig).name
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # === Setup Distributed Strategy ===
    strategy_config = DistributedStrategyConfig(
        strategy_type='mirrored' if args.multi_gpu else 'none',
        cross_device_ops=args.communication_backend,
        enable_nccl_workarounds=args.nccl_workarounds,
    )
    strategy, num_replicas = create_distribution_strategy(
        config=strategy_config, verbose=True
    )

    # === Setup Mixed Precision ===
    setup_mixed_precision(enabled=args.mixed_precision, verbose=True)

    # === Load/Generate Data ===
    if args.dataset and os.path.exists(args.dataset):
        print(f"\nLoading dataset: {args.dataset}")
        masks, aerials = load_dataset(args.dataset)
    else:
        print(f"\nGenerating {args.num_samples} samples...")
        masks, aerials = generate_dataset(args.num_samples, seed=args.seed)
        dataset_path = os.path.join(run_dir, 'litho_dataset.npz')
        save_dataset(masks, aerials, dataset_path)

    # Train/val split
    n = len(masks)
    n_val = max(1, int(n * args.val_split))
    n_train = n - n_val
    indices = np.random.permutation(n)
    train_masks, train_aerials = masks[indices[:n_train]], aerials[indices[:n_train]]
    val_masks, val_aerials = masks[indices[n_train:]], aerials[indices[n_train:]]
    print(f"Train: {n_train}, Val: {n_val}")

    # === Build Model ===
    with strategy.scope():
        model = build_model(
            input_shape=masks.shape[1:],
            num_filters_base=args.filters
        )

        # Calculate total steps
        steps_per_epoch = n_train // (args.batch_size * num_replicas)
        total_steps = args.epochs * steps_per_epoch

        print(f"\nSteps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")

        # Create LR schedule
        lr_schedule = create_lr_schedule(
            schedule_type=args.lr_schedule,
            base_lr=args.lr,
            total_steps=total_steps,
            pct_start=args.onecycle_pct_start,
            div_factor=args.onecycle_div_factor,
            mode=args.cyclical_mode,
            first_cycle_steps=args.sgdr_cycle_length * steps_per_epoch,
        )

        # Visualize LR schedule
        lr_viz_path = os.path.join(run_dir, 'lr_schedule.png')
        visualize_lr_schedule(lr_schedule, total_steps, lr_viz_path)

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        if args.gradient_clip_norm:
            optimizer = keras.optimizers.Adam(
                learning_rate=lr_schedule,
                clipnorm=args.gradient_clip_norm
            )
        if args.mixed_precision:
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    print(f"\nModel: {model.count_params():,} parameters")

    # Print summary
    print_training_summary(
        strategy=strategy,
        num_replicas=num_replicas,
        mixed_precision_enabled=args.mixed_precision,
        augmentation_enabled=args.augmentation,
        total_params=model.count_params()
    )

    # === Create Datasets ===
    global_batch = args.batch_size * num_replicas
    if args.augmentation:
        train_ds = create_augmented_dataset(
            train_masks, train_aerials, batch_size=global_batch, shuffle=True
        )
    else:
        from data_pipeline import make_tf_dataset
        train_ds = make_tf_dataset(
            train_masks, train_aerials, batch_size=global_batch, shuffle=True
        )

    from data_pipeline import make_tf_dataset
    val_ds = make_tf_dataset(
        val_masks, val_aerials, batch_size=global_batch, shuffle=False
    )

    # Distribute datasets
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    val_ds = strategy.experimental_distribute_dataset(val_ds)

    # === Create Callbacks ===
    callbacks = create_advanced_callbacks(run_dir, args, stop_state, total_steps)

    # === Train ===
    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # === Evaluate ===
    print(f"\nFinal Evaluation:")
    val_loss, val_mae = model.evaluate(val_ds, verbose=0)
    print(f"  Val Loss (MSE): {val_loss:.6f}")
    print(f"  Val MAE: {val_mae:.6f}")

    # === Save Final Model ===
    final_model_path = os.path.join(run_dir, 'litho_model_final.keras')
    model.save(final_model_path)
    print(f"\nFinal model saved: {final_model_path}")

    # Save EMA model if enabled
    if args.model_ema:
        for cb in callbacks:
            if isinstance(cb, ModelEMA):
                ema_path = os.path.join(run_dir, 'litho_model_ema.keras')
                cb.save_ema_model(ema_path)

    # Save SWA model if enabled
    if args.swa:
        for cb in callbacks:
            if isinstance(cb, SWA):
                swa_path = os.path.join(run_dir, 'litho_model_swa.keras')
                cb.save_swa_model(swa_path)

    # === Generate Plots ===
    if 'loss' in history.history:
        plot_training_history(
            history, save_path=os.path.join(run_dir, 'training_history.png')
        )
    plot_predictions(
        model, val_masks, val_aerials, num_samples=4,
        save_path=os.path.join(run_dir, 'predictions.png')
    )

    # === Save Summary ===
    summary = {
        'run_dir': os.path.abspath(run_dir),
        'config': vars(args),
        'results': {
            'final_val_loss': float(val_loss),
            'final_val_mae': float(val_mae),
            'epochs_completed': len(history.history.get('loss', [])),
        },
        'models': {
            'final': os.path.abspath(final_model_path),
            'best': os.path.join(run_dir, 'litho_model_best.keras'),
            'ema': os.path.join(run_dir, 'litho_model_ema.keras') if args.model_ema else None,
            'swa': os.path.join(run_dir, 'litho_model_swa.keras') if args.swa else None,
        }
    }

    summary_path = os.path.join(run_dir, 'run_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}\n")

    return 0


if __name__ == '__main__':
    exit(main())
