#!/usr/bin/env python3
"""
Integration test: Simulator → HalfUNetFNO training → evaluation
================================================================

End-to-end pipeline using deliberately small compute scale:
  1. Generate mask / aerial-image dataset via SOCC simulator
  2. Split into train / validation sets
  3. Train a compact HalfUNetFNO model
  4. Evaluate predictions and report metrics
  5. Save model weights to models/saved/

Scale choices (fast on CPU, completes in ~2-5 min):
  • frequency_samples = 32  → 32×32 spatial grid
  • source_samples    = 16  → fast TCC source integration
  • n_socc_modes      = 8   → 8 coherent imaging modes
  • dataset           = 200 samples (160 train, 40 val)
  • model             : modes=6, width=16, levels=2
  • training          : 20 epochs, batch=8
"""

import os
import sys
import time

import numpy as np
import tensorflow as tf

# ── Project root on path ────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from simulator import OpticalSettings, ImageSimulator
from models.half_unet_fno import build_and_compile

# ── Hyper-parameters (small scale) ──────────────────────────────────────────
N_GRID       = 32    # spatial / frequency grid size
SOURCE_PTS   = 16    # source integration points (default 64 → reduced)
N_SOCC       = 8     # SOCC coherent modes
N_TOTAL      = 200   # total dataset size
N_TRAIN      = 160   # training samples  (40 val)
BATCH        = 8
EPOCHS       = 20
LR           = 1e-3
MODES_FNO    = 6     # Fourier modes per spatial dim
WIDTH        = 16    # channel width
LEVELS       = 2     # encoder levels
NRMSE_PASS   = 0.30  # NRMSE threshold for PASS verdict (30%)

SEP = "=" * 65


def banner(msg: str) -> None:
    print(f"\n{SEP}\n  {msg}\n{SEP}")


# ── Mask generation ──────────────────────────────────────────────────────────

def _random_mask(rng: np.random.Generator, N: int) -> np.ndarray:
    """Return a random binary lithography mask (N×N float32)."""
    kind = rng.integers(0, 4)
    mask = np.zeros((N, N), dtype=np.float32)

    if kind == 0:                           # vertical lines
        pitch = int(rng.integers(4, max(5, N // 3)))
        w     = int(rng.integers(2, max(3, pitch // 2 + 1)))
        for x in range(0, N, pitch):
            mask[:, x : x + w] = 1.0

    elif kind == 1:                         # horizontal lines
        pitch = int(rng.integers(4, max(5, N // 3)))
        w     = int(rng.integers(2, max(3, pitch // 2 + 1)))
        for y in range(0, N, pitch):
            mask[y : y + w, :] = 1.0

    elif kind == 2:                         # contact holes
        pitch = int(rng.integers(5, max(6, N // 3)))
        w     = max(2, pitch // 3)
        for y in range(0, N, pitch):
            for x in range(0, N, pitch):
                mask[y : y + w, x : x + w] = 1.0

    else:                                   # random binary
        thresh = rng.uniform(0.3, 0.7)
        mask   = (rng.random((N, N)) > thresh).astype(np.float32)

    return mask.clip(0.0, 1.0)


def generate_dataset(
    sim: ImageSimulator,
    n_total: int,
    n_grid: int,
    seed: int = 42,
) -> tuple:
    """Simulate n_total mask→aerial-image pairs.

    Returns:
        masks:   np.ndarray (n_total, n_grid, n_grid)
        aerials: np.ndarray (n_total, n_grid, n_grid)
    """
    rng   = np.random.default_rng(seed)
    masks = np.stack([_random_mask(rng, n_grid) for _ in range(n_total)])

    print(f"Simulating {n_total} aerial images  "
          f"(SOCC, {n_grid}×{n_grid} grid, {N_SOCC} modes) ...")
    t0      = time.time()
    aerials = sim.batch_simulate(masks, verbose=True)   # (N, H, W) tf.Tensor
    dt      = time.time() - t0
    rate    = n_total / dt if dt > 0 else float("inf")
    print(f"Simulation complete: {dt:.1f}s  ({rate:.1f} samples/s)")

    return masks, aerials.numpy()


# ── tf.data pipeline ─────────────────────────────────────────────────────────

def make_tf_dataset(
    masks:   np.ndarray,
    aerials: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> tf.data.Dataset:
    """Wrap numpy arrays into a batched tf.data.Dataset.

    Adds a channel dimension:  (N, H, W) → (N, H, W, 1).
    """
    x = masks[..., np.newaxis].astype(np.float32)    # (N, H, W, 1)
    y = aerials[..., np.newaxis].astype(np.float32)  # (N, H, W, 1)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(masks), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> bool:
    tf.random.set_seed(42)
    np.random.seed(42)

    # ── Step 1: Simulator ──────────────────────────────────────────────────
    banner("Step 1 — Initialise SOCC simulator")

    settings = OpticalSettings(
        wavelength        = 193.0,   # nm  ArF
        na                = 1.35,    # immersion
        sigma_inner       = 0.7,
        sigma_outer       = 0.9,
        pixel_size        = 8.0,     # nm/pixel
        frequency_samples = N_GRID,  # 32 × 32 grid
        source_samples    = SOURCE_PTS,
    )

    sim = ImageSimulator(settings, method="socc", n_modes=N_SOCC)
    print(f"Simulator ready  —  grid {N_GRID}×{N_GRID}, "
          f"SOCC modes={N_SOCC}, source pts={SOURCE_PTS}")

    # ── Step 2: Dataset ────────────────────────────────────────────────────
    banner("Step 2 — Generate mask / aerial-image dataset")

    masks, aerials = generate_dataset(sim, N_TOTAL, N_GRID, seed=42)

    print(f"Masks   shape: {masks.shape}  "
          f"(min={masks.min():.2f}  max={masks.max():.2f})")
    print(f"Aerials shape: {aerials.shape}  "
          f"(min={aerials.min():.4f}  max={aerials.max():.4f})")

    # Normalise aerials to [0, 1]
    aerial_max   = aerials.max() + 1e-8
    aerials_norm = aerials / aerial_max

    train_masks  = masks[:N_TRAIN]
    train_aer    = aerials_norm[:N_TRAIN]
    val_masks    = masks[N_TRAIN:]
    val_aer      = aerials_norm[N_TRAIN:]

    print(f"Train: {N_TRAIN} samples   Val: {N_TOTAL - N_TRAIN} samples")

    # ── Step 3: Build model ────────────────────────────────────────────────
    banner("Step 3 — Build HalfUNetFNO  (small scale)")

    model = build_and_compile(
        in_channels      = 1,
        out_channels     = 1,
        input_shape_hw   = (N_GRID, N_GRID),
        modes            = MODES_FNO,
        width            = WIDTH,
        levels           = LEVELS,
        num_final_blocks = 1,
        learning_rate    = LR,
        loss             = "mse",
    )
    model.summary(line_length=72, expand_nested=False)

    # ── Step 4: Train ──────────────────────────────────────────────────────
    banner(f"Step 4 — Training  ({EPOCHS} epochs · {N_TRAIN} train / "
           f"{N_TOTAL - N_TRAIN} val · batch={BATCH})")

    train_ds = make_tf_dataset(train_masks, train_aer, BATCH, shuffle=True)
    val_ds   = make_tf_dataset(val_masks,   val_aer,   BATCH, shuffle=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-5, verbose=1),
    ]

    t0      = time.time()
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS,
        callbacks       = callbacks,
        verbose         = 1,
    )
    train_time = time.time() - t0
    print(f"\nTraining finished in {train_time:.1f}s")

    # ── Step 5: Evaluate ───────────────────────────────────────────────────
    banner("Step 5 — Evaluation")

    x_val = val_masks[..., np.newaxis].astype(np.float32)   # (N_val, H, W, 1)
    y_val = val_aer[..., np.newaxis].astype(np.float32)     # (N_val, H, W, 1)

    y_pred = model.predict(x_val, verbose=0)                 # (N_val, H, W, 1)

    mse   = float(np.mean((y_pred - y_val) ** 2))
    mae   = float(np.mean(np.abs(y_pred - y_val)))
    rmse  = float(np.sqrt(mse))
    nrmse = rmse / (float(y_val.max()) + 1e-8)

    # Per-sample SSIM approximation (mean structural similarity)
    def _ssim_approx(a: np.ndarray, b: np.ndarray) -> float:
        mu_a, mu_b = a.mean(), b.mean()
        sig_a  = a.std() + 1e-8
        sig_b  = b.std() + 1e-8
        sig_ab = float(np.mean((a - mu_a) * (b - mu_b)))
        c1, c2 = 0.01**2, 0.03**2
        return float(
            (2*mu_a*mu_b + c1) * (2*sig_ab + c2)
            / ((mu_a**2 + mu_b**2 + c1) * (sig_a**2 + sig_b**2 + c2))
        )

    ssim_scores = [
        _ssim_approx(y_pred[i, :, :, 0], y_val[i, :, :, 0])
        for i in range(len(y_val))
    ]
    mean_ssim = float(np.mean(ssim_scores))

    train_losses = history.history.get("loss", [])
    val_losses   = history.history.get("val_loss", [])

    print(f"  MSE            : {mse:.6f}")
    print(f"  MAE            : {mae:.6f}")
    print(f"  RMSE           : {rmse:.6f}")
    print(f"  NRMSE          : {nrmse:.4f}  ({nrmse * 100:.2f}%)")
    print(f"  Mean SSIM      : {mean_ssim:.4f}")
    print(f"  Train loss     : {train_losses[0]:.6f} → {train_losses[-1]:.6f}")
    if val_losses:
        print(f"  Val   loss     : {val_losses[0]:.6f} → {val_losses[-1]:.6f}")
    print(f"  Epochs trained : {len(train_losses)}")

    # ── Step 6: Save weights ───────────────────────────────────────────────
    save_dir = os.path.join(ROOT, "models", "saved")
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "half_unet_fno_litho.weights.h5")
    model.save_weights(weights_path)
    print(f"\n  Model weights → {weights_path}")

    # ── Verdict ────────────────────────────────────────────────────────────
    banner("Result")
    passed = nrmse < NRMSE_PASS
    verdict = "PASS ✓" if passed else "FAIL ✗"
    print(f"  {verdict}  (NRMSE={nrmse * 100:.2f}%  threshold={NRMSE_PASS * 100:.0f}%)")
    print(f"  Mean SSIM: {mean_ssim:.4f}")
    if not passed:
        print("  Tip: increase EPOCHS or dataset size for better accuracy.")
    print()
    return passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
