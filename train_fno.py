"""
HalfUNetFNO Training Script — Lithography
==========================================

Trains the TF 2.10 HalfUNetFNO port on lithography mask→aerial-image data.

Integrates with:
  • data_pipeline.py      — numpy dataset generation / loading
  • tfrecord_manager.py   — TFRecord I/O with chunking + sharding
  • train_utils.py        — multi-GPU strategy, mixed-precision, callbacks
  • half_unet_fno.py      — model definition

Usage examples
--------------
# Generate data on-the-fly, train single GPU:
    python train_fno.py --epochs 50 --batch-size 8

# Load existing .npz dataset:
    python train_fno.py --dataset data/litho_1k.npz --epochs 100

# Load pre-built TFRecord:
    python train_fno.py --tfrecord-dir ./tfrecords --epochs 100

# Multi-GPU:
    python train_fno.py --multi-gpu --epochs 100 --batch-size 16

# Write TFRecord from .npz, then train from it:
    python train_fno.py --dataset data.npz --write-tfrecord --tfrecord-dir ./tfrecords

# Smoke test:
    python train_fno.py --smoke-test
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_pipeline import generate_dataset, save_dataset, load_dataset
from half_unet_fno import build_and_compile, HalfUNetFNO
from tfrecord_manager import TFRecordWriter, TFRecordReader, TFRecordConfig


# ---------------------------------------------------------------------------
# Learning-rate schedule helpers
# ---------------------------------------------------------------------------

def make_step_lr(initial_lr: float, step_size: int, gamma: float,
                 steps_per_epoch: int):
    """Approximate PyTorch StepLR: multiply LR by gamma every step_size epochs.

    Returns a tf.keras.optimizers.schedules object.
    """
    boundaries = [step_size * steps_per_epoch * (i + 1)
                  for i in range(20)]           # first 20 decay steps
    values = [initial_lr * (gamma ** i) for i in range(len(boundaries) + 1)]
    return keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_or_generate(args) -> tuple:
    """Return (masks, aerials) as float32 numpy arrays shaped (N, H, W, 1)."""
    if args.dataset:
        print(f"Loading dataset: {args.dataset}")
        masks, aerials = load_dataset(args.dataset)
    else:
        print(f"Generating {args.num_samples} samples …")
        masks, aerials = generate_dataset(
            n=args.num_samples, seed=args.seed, verbose=True)

    # Ensure float32
    masks   = masks.astype(np.float32)
    aerials = aerials.astype(np.float32)

    # Ensure channel dimension exists
    if masks.ndim == 3:
        masks   = masks[..., np.newaxis]
    if aerials.ndim == 3:
        aerials = aerials[..., np.newaxis]

    # Min-max normalise to [-1, 1] per channel (matches original repo)
    def norm_minmax(arr):
        mn = arr.min(axis=(0, 1, 2), keepdims=True)
        mx = arr.max(axis=(0, 1, 2), keepdims=True)
        return 2.0 * (arr - mn) / (mx - mn + 1e-8) - 1.0

    masks   = norm_minmax(masks)
    aerials = norm_minmax(aerials)

    print(f"  masks   {masks.shape}  [{masks.min():.2f}, {masks.max():.2f}]")
    print(f"  aerials {aerials.shape}  [{aerials.min():.2f}, {aerials.max():.2f}]")
    return masks, aerials


def split_dataset(masks: np.ndarray, aerials: np.ndarray,
                  val_split: float = 0.15, seed: int = 42):
    """Shuffle and split into train / validation."""
    n = len(masks)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (masks[train_idx], aerials[train_idx],
            masks[val_idx],   aerials[val_idx])


def make_numpy_dataset(masks, aerials, batch_size: int,
                       shuffle: bool = True,
                       shard_for_multi_gpu: bool = False) -> tf.data.Dataset:
    """Create a tf.data.Dataset from numpy arrays with optional auto-sharding."""
    ds = tf.data.Dataset.from_tensor_slices((masks, aerials))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(masks), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)

    if shard_for_multi_gpu:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)

    return ds.prefetch(tf.data.AUTOTUNE)


def make_tfrecord_dataset(tfrecord_dir: str, batch_size: int,
                           shuffle: bool = True,
                           shard_for_multi_gpu: bool = False) -> tf.data.Dataset:
    """Create a tf.data.Dataset from TFRecord files."""
    reader = TFRecordReader(tfrecord_dir)
    return reader.create_dataset(
        batch_size=batch_size,
        shuffle=shuffle,
        repeat=True,
        drop_remainder=True,
        shard_for_multi_gpu=shard_for_multi_gpu,
    )


def write_tfrecord(masks, aerials, tfrecord_dir: str,
                   samples_per_file: int = 500) -> None:
    """Write numpy arrays to chunked TFRecord files."""
    config = TFRecordConfig(
        output_dir=tfrecord_dir,
        samples_per_file=samples_per_file,
        image_height=masks.shape[1],
        image_width=masks.shape[2],
        mask_channels=masks.shape[3],
        aerial_channels=aerials.shape[3],
        compression_type="GZIP",
    )
    writer = TFRecordWriter(config)
    writer.write_dataset(masks, aerials, verbose=True)


# ---------------------------------------------------------------------------
# Multi-GPU strategy helper
# ---------------------------------------------------------------------------

def get_strategy(multi_gpu: bool):
    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        n = strategy.num_replicas_in_sync
        print(f"Multi-GPU: {n} replica(s) detected")
    else:
        strategy = tf.distribute.get_strategy()
        n = 1
        print("Single device training")
    return strategy, n


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    # ---- Reproducibility ----
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    run_dir = os.path.join(args.experiment_dir,
                           args.run_name or time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ---- Strategy ----
    strategy, n_replicas = get_strategy(args.multi_gpu)
    per_replica_bs = args.batch_size
    global_bs = per_replica_bs * n_replicas
    print(f"Batch size: {per_replica_bs}/replica × {n_replicas} = {global_bs} global")

    # ---- Data ----
    use_tfrecord = bool(args.tfrecord_dir)

    if use_tfrecord and not args.write_tfrecord:
        # Load directly from TFRecord
        print(f"\nLoading TFRecord dataset from: {args.tfrecord_dir}")
        train_ds = make_tfrecord_dataset(args.tfrecord_dir, per_replica_bs,
                                          shuffle=True,
                                          shard_for_multi_gpu=args.multi_gpu)
        val_ds   = make_tfrecord_dataset(args.tfrecord_dir, per_replica_bs,
                                          shuffle=False,
                                          shard_for_multi_gpu=args.multi_gpu)
        # Approximate sample count from info
        reader = TFRecordReader(args.tfrecord_dir)
        info = reader.info()
        n_train = int(info.get("total_samples", 1000) * 0.85)
        n_val   = info.get("total_samples", 1000) - n_train
        h = info.get("config", {}).get("image_height", args.image_size)
        w = info.get("config", {}).get("image_width",  args.image_size)
        c_in  = info.get("config", {}).get("mask_channels",   1)
        c_out = info.get("config", {}).get("aerial_channels", 1)
    else:
        # Numpy path
        masks, aerials = load_or_generate(args)
        if args.write_tfrecord and args.tfrecord_dir:
            print(f"\nWriting TFRecord to: {args.tfrecord_dir}")
            write_tfrecord(masks, aerials, args.tfrecord_dir)

        tr_masks, tr_aerials, va_masks, va_aerials = split_dataset(
            masks, aerials, val_split=args.val_split, seed=args.seed)

        train_ds = make_numpy_dataset(tr_masks, tr_aerials, per_replica_bs,
                                       shuffle=True,
                                       shard_for_multi_gpu=args.multi_gpu)
        val_ds   = make_numpy_dataset(va_masks, va_aerials, per_replica_bs,
                                       shuffle=False,
                                       shard_for_multi_gpu=args.multi_gpu)
        n_train = len(tr_masks)
        n_val   = len(va_masks)
        _, h, w, c_in  = masks.shape
        c_out = aerials.shape[-1]

    steps_per_epoch = max(1, n_train // global_bs)
    val_steps       = max(1, n_val   // global_bs)
    print(f"\nTrain samples: {n_train}  ({steps_per_epoch} steps/epoch)")
    print(f"Val   samples: {n_val}    ({val_steps} steps)")

    # Distribute datasets for multi-GPU
    if args.multi_gpu:
        train_ds = strategy.experimental_distribute_dataset(train_ds)
        val_ds   = strategy.experimental_distribute_dataset(val_ds)

    # ---- Model ----
    print(f"\nBuilding HalfUNetFNO  "
          f"(modes={args.modes}, width={args.width}, levels={args.levels})")

    lr_schedule = make_step_lr(
        args.lr, args.step_size, args.gamma,
        steps_per_epoch=steps_per_epoch)

    with strategy.scope():
        model = build_and_compile(
            in_channels=c_in,
            out_channels=c_out,
            input_shape_hw=(h, w),
            modes=args.modes,
            width=args.width,
            levels=args.levels,
            num_final_blocks=args.num_final_blocks,
            activation=args.activation,
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay,
            loss="mse",
        )

    # ---- Callbacks ----
    callbacks = []

    # Best-model checkpoint
    ckpt_path = os.path.join(run_dir, "best_model.keras")
    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ))

    # Early stopping
    if args.patience > 0:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ))

    # CSV logger
    callbacks.append(keras.callbacks.CSVLogger(
        os.path.join(run_dir, "history.csv")))

    # TensorBoard
    if args.tensorboard:
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=os.path.join(run_dir, "tb_logs"),
            histogram_freq=0,
            write_graph=False,
        ))

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Training ----
    print(f"\nStarting training for {args.epochs} epochs …")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=2,
    )

    # ---- Save final model ----
    final_path = os.path.join(run_dir, "final_model.keras")
    model.save(final_path)
    print(f"\n✓ Final model saved: {final_path}")
    print(f"✓ Best  model saved: {ckpt_path}")

    best_val = min(history.history.get("val_loss", [float("inf")]))
    print(f"✓ Best val_loss: {best_val:.6f}")

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train HalfUNetFNO on lithography data")

    # Data
    p.add_argument("--dataset",      type=str,  default=None,
                   help=".npz file with masks/aerials arrays")
    p.add_argument("--tfrecord-dir", type=str,  default=None,
                   help="Directory with TFRecord files")
    p.add_argument("--write-tfrecord", action="store_true",
                   help="Convert loaded numpy data to TFRecord before training")
    p.add_argument("--num-samples",  type=int,  default=500,
                   help="Samples to generate when --dataset is absent")
    p.add_argument("--image-size",   type=int,  default=64,
                   help="Spatial size (H=W) for on-the-fly generation")
    p.add_argument("--val-split",    type=float, default=0.15)

    # Architecture
    p.add_argument("--modes",            type=int,   default=20)
    p.add_argument("--width",            type=int,   default=48)
    p.add_argument("--levels",           type=int,   default=3)
    p.add_argument("--num-final-blocks", type=int,   default=2)
    p.add_argument("--activation",       type=str,   default="gelu",
                   choices=["gelu", "relu"])

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=8,
                   help="Batch size PER REPLICA")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--step-size",    type=int,   default=20,
                   help="LR decay every N epochs (StepLR)")
    p.add_argument("--gamma",        type=float, default=0.5,
                   help="LR decay factor")
    p.add_argument("--patience",     type=int,   default=10,
                   help="Early-stopping patience (0 = disabled)")

    # Distributed
    p.add_argument("--multi-gpu", action="store_true")

    # Output
    p.add_argument("--experiment-dir", type=str, default="experiments_fno")
    p.add_argument("--run-name",       type=str, default=None)
    p.add_argument("--tensorboard",    action="store_true", default=True)
    p.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    p.add_argument("--seed",           type=int,  default=42)

    # Smoke test
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny run to verify the pipeline end-to-end")

    args = p.parse_args()

    if args.smoke_test:
        args.epochs      = 2
        args.num_samples = 20
        args.image_size  = 64
        args.modes       = 8
        args.width       = 16
        args.levels      = 2
        args.num_final_blocks = 1
        args.batch_size  = 4
        args.patience    = 0
        args.tensorboard = False
        print("Smoke-test mode: using minimal settings")

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
