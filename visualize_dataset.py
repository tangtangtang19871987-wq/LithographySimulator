"""
Visualize lithography dataset pairs from .npz in 10x10 chunked pages.

Layout per page:
  - 10 rows x 10 cols = 100 images total
  - 50 input/output pairs
  - odd row (1-based): input mask
  - even row directly below: output aerial image
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _to_2d(x):
    """Convert (H,W,1) to (H,W) while keeping (H,W) unchanged."""
    if x.ndim == 3 and x.shape[-1] == 1:
        return x[:, :, 0]
    return x


def plot_chunk(masks, aerials, start_idx, end_idx, out_path):
    """Plot one chunk as a 10x10 page (50 pairs)."""
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    axes = np.asarray(axes)

    # Hide everything first, then fill active slots.
    for ax in axes.ravel():
        ax.axis("off")

    pair_count = end_idx - start_idx
    for p in range(pair_count):
        idx = start_idx + p
        block_row = p // 10
        col = p % 10
        row_in = 2 * block_row
        row_out = row_in + 1

        mask_img = _to_2d(masks[idx])
        aerial_img = _to_2d(aerials[idx])

        ax_in = axes[row_in, col]
        ax_out = axes[row_out, col]

        ax_in.imshow(mask_img, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax_out.imshow(aerial_img, cmap="inferno", interpolation="nearest", vmin=0, vmax=1)

        ax_in.set_title(f"idx {idx} in", fontsize=8)
        ax_out.set_title(f"idx {idx} out", fontsize=8)
        ax_in.axis("off")
        ax_out.axis("off")

    fig.suptitle(f"Samples {start_idx} to {end_idx - 1}", fontsize=16)
    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize litho_dataset.npz in 10x10 pages (50 pairs/page)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="litho_dataset.npz",
        help="Path to .npz dataset with keys: masks, aerials",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset_viz",
        help="Directory to save chunk images",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Images per page (default 100 => 50 pairs). Must be even.",
    )
    args = parser.parse_args()

    if args.chunk_size != 100:
        raise ValueError("This script is designed for 10x10 layout, so --chunk-size must be 100.")

    if args.chunk_size % 2 != 0:
        raise ValueError("--chunk-size must be even.")

    data = np.load(args.dataset)
    if "masks" not in data or "aerials" not in data:
        raise KeyError("Dataset must contain keys 'masks' and 'aerials'.")

    masks = data["masks"]
    aerials = data["aerials"]

    if len(masks) != len(aerials):
        raise ValueError("masks and aerials lengths do not match.")

    n = len(masks)
    pairs_per_page = args.chunk_size // 2
    num_pages = math.ceil(n / pairs_per_page)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loaded dataset: {n} pairs")
    print(f"Saving {num_pages} page(s) to: {args.output_dir}")

    for page in range(num_pages):
        start_idx = page * pairs_per_page
        end_idx = min((page + 1) * pairs_per_page, n)
        out_name = f"pairs_{start_idx:05d}_{end_idx - 1:05d}.png"
        out_path = os.path.join(args.output_dir, out_name)
        plot_chunk(masks, aerials, start_idx, end_idx, out_path)
        print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
