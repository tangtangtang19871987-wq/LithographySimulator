"""
Create shift-equivariance verification data from an existing lithography dataset.

This script:
1. Randomly selects a validation subset from (masks, aerials).
2. Applies the same random circular translation to input and output per sample.
3. Saves original and shifted tensors plus metadata for later inference checks.
"""

import argparse
import numpy as np


def circular_shift_2d(image, shift_h, shift_w):
    """Apply circular (wrap-around) shift on H/W axes."""
    shifted = np.roll(image, shift=shift_h, axis=0)
    shifted = np.roll(shifted, shift=shift_w, axis=1)
    return shifted


def main():
    parser = argparse.ArgumentParser(
        description="Build shifted verification subset for translation equivariance tests."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="litho_dataset_800.npz",
        help="Path to source .npz with keys: masks, aerials",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="shift_verification_20.npz",
        help="Output .npz path",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=20,
        help="Number of random validation samples to select",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=32,
        help="Random shift range per axis: [-max_shift, +max_shift]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    data = np.load(args.dataset)
    if "masks" not in data or "aerials" not in data:
        raise KeyError("Dataset must contain keys 'masks' and 'aerials'.")

    masks = data["masks"]
    aerials = data["aerials"]
    n = len(masks)
    if n != len(aerials):
        raise ValueError("masks and aerials length mismatch.")
    if args.num_val <= 0 or args.num_val > n:
        raise ValueError(f"--num-val must be in [1, {n}]")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n, size=args.num_val, replace=False)
    indices.sort()

    val_masks = masks[indices].copy()
    val_aerials = aerials[indices].copy()

    shift_h = rng.integers(-args.max_shift, args.max_shift + 1, size=args.num_val)
    shift_w = rng.integers(-args.max_shift, args.max_shift + 1, size=args.num_val)

    shifted_masks = np.empty_like(val_masks)
    shifted_aerials = np.empty_like(val_aerials)

    for i in range(args.num_val):
        m = val_masks[i]
        a = val_aerials[i]

        # Handle (H, W, 1) and (H, W) safely.
        if m.ndim == 3 and m.shape[-1] == 1:
            shifted_masks[i, :, :, 0] = circular_shift_2d(m[:, :, 0], shift_h[i], shift_w[i])
        else:
            shifted_masks[i] = circular_shift_2d(m, shift_h[i], shift_w[i])

        if a.ndim == 3 and a.shape[-1] == 1:
            shifted_aerials[i, :, :, 0] = circular_shift_2d(a[:, :, 0], shift_h[i], shift_w[i])
        else:
            shifted_aerials[i] = circular_shift_2d(a, shift_h[i], shift_w[i])

    np.savez_compressed(
        args.output,
        source_dataset=np.array(args.dataset),
        seed=np.array(args.seed, dtype=np.int64),
        max_shift=np.array(args.max_shift, dtype=np.int64),
        selected_indices=indices.astype(np.int64),
        shift_h=shift_h.astype(np.int64),
        shift_w=shift_w.astype(np.int64),
        val_masks=val_masks,
        val_aerials=val_aerials,
        shifted_masks=shifted_masks,
        shifted_aerials=shifted_aerials,
    )

    print(f"Saved: {args.output}")
    print(f"Samples: {args.num_val}")
    print(f"Index range: {indices.min()}..{indices.max()}")
    print(f"Shift range h: {shift_h.min()}..{shift_h.max()}")
    print(f"Shift range w: {shift_w.min()}..{shift_w.max()}")


if __name__ == "__main__":
    main()
