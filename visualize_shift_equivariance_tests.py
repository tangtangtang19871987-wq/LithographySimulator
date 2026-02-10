"""
Visualize shift-equivariance inference checks.

Produces:
1) A summary plot of MAE / max-diff by sample.
2) A panel of worst-k samples with predicted shift consistency views.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from shift_equivariant_unet import CircularPad2D, CircularConv2D, DilatedCircularConv2D


def roll_hw(x, shift_h, shift_w):
    y = np.roll(x, shift=int(shift_h), axis=0)
    y = np.roll(y, shift=int(shift_w), axis=1)
    return y


def infer_in_batches(model, x, batch_size=8):
    """Run inference without model.predict to avoid tf.data threadpool usage."""
    outs = []
    n = len(x)
    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size]
        yb = model(xb, training=False).numpy()
        outs.append(yb)
    return np.concatenate(outs, axis=0) if outs else np.empty((0,))


def save_summary_plot(results, out_path):
    sample_ids = np.arange(len(results))
    maes = np.array([r["mae"] for r in results], dtype=np.float64)
    max_diffs = np.array([r["max_diff"] for r in results], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].bar(sample_ids, maes, color="#1f77b4")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Shift Equivariance Error by Sample")
    axes[0].grid(alpha=0.25)

    axes[1].bar(sample_ids, max_diffs, color="#ff7f0e")
    axes[1].set_ylabel("Max Abs Diff")
    axes[1].set_xlabel("Sample (verification order)")
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_examples_plot(model, data, results, top_k, out_path):
    val_masks = data["val_masks"]
    shifted_masks = data["shifted_masks"]
    shift_h = data["shift_h"]
    shift_w = data["shift_w"]

    pred_orig = infer_in_batches(model, val_masks, batch_size=8)
    pred_shifted = infer_in_batches(model, shifted_masks, batch_size=8)

    order = np.argsort([-r["mae"] for r in results])
    picks = order[:top_k]
    top_k = len(picks)

    fig, axes = plt.subplots(top_k, 6, figsize=(18, 3.2 * top_k))
    if top_k == 1:
        axes = axes[np.newaxis, :]

    for row, p in enumerate(picks):
        idx = int(p)
        h = int(shift_h[idx])
        w = int(shift_w[idx])
        r = results[idx]

        in0 = val_masks[idx, :, :, 0]
        in1 = shifted_masks[idx, :, :, 0]
        out0 = pred_orig[idx, :, :, 0]
        expected = roll_hw(out0, h, w)
        actual = pred_shifted[idx, :, :, 0]
        err = np.abs(actual - expected)

        axes[row, 0].imshow(in0, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Input idx={r['dataset_index']}")
        axes[row, 1].imshow(in1, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Shifted Input ({h},{w})")
        axes[row, 2].imshow(out0, cmap="inferno", vmin=0, vmax=1)
        axes[row, 2].set_title("Pred f(x)")
        axes[row, 3].imshow(expected, cmap="inferno", vmin=0, vmax=1)
        axes[row, 3].set_title("Expected shift(f(x))")
        axes[row, 4].imshow(actual, cmap="inferno", vmin=0, vmax=1)
        axes[row, 4].set_title("Actual f(shift(x))")
        im = axes[row, 5].imshow(err, cmap="magma")
        axes[row, 5].set_title(f"|diff| MAE={r['mae']:.2e}")

        for c in range(6):
            axes[row, c].axis("off")
        plt.colorbar(im, ax=axes[row, 5], fraction=0.046, pad=0.04)

    fig.suptitle("Worst-Case Shift Equivariance Samples", fontsize=14)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize shift-equivariance test outputs.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .keras model")
    parser.add_argument(
        "--verification-data",
        type=str,
        default="shift_verification_20.npz",
        help="Path to verification .npz",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="shift_equivariance_inference_report.json",
        help="Path to inference report json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="shift_equivariance_viz",
        help="Directory for output png files",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Number of worst samples to visualize")
    args = parser.parse_args()

    with open(args.report_json, "r", encoding="utf-8") as f:
        report = json.load(f)
    results = report["results"]
    data = np.load(args.verification_data)

    custom_objects = {
        "CircularPad2D": CircularPad2D,
        "CircularConv2D": CircularConv2D,
        "DilatedCircularConv2D": DilatedCircularConv2D,
    }
    model = keras.models.load_model(args.model, custom_objects=custom_objects)

    os.makedirs(args.output_dir, exist_ok=True)
    summary_png = os.path.join(args.output_dir, "summary_errors.png")
    examples_png = os.path.join(args.output_dir, "worst_cases.png")

    save_summary_plot(results, summary_png)
    save_examples_plot(model, data, results, args.top_k, examples_png)

    print(f"Saved: {summary_png}")
    print(f"Saved: {examples_png}")


if __name__ == "__main__":
    main()
