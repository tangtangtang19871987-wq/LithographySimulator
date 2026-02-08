"""
Inference-time shift-equivariance checker.

Given:
  - a trained model (.keras)
  - a verification dataset created by create_shift_verification_data.py

This script measures whether:
  f(shift(x)) ~= shift(f(x))
for each validation sample.
"""

import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from shift_equivariant_unet import CircularPad2D, CircularConv2D, DilatedCircularConv2D


def roll_hw(batch, shift_h, shift_w):
    """Circularly shift a single sample tensor on H/W axes."""
    out = np.roll(batch, shift=int(shift_h), axis=0)
    out = np.roll(out, shift=int(shift_w), axis=1)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Check translation equivariance on shifted verification data."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained .keras model",
    )
    parser.add_argument(
        "--verification-data",
        type=str,
        default="shift_verification_20.npz",
        help="Path to shift verification .npz",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="MAE threshold for pass/fail decision",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for model.predict",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="shift_equivariance_inference_report.json",
        help="Path to save detailed JSON report",
    )
    args = parser.parse_args()

    data = np.load(args.verification_data)
    needed = [
        "selected_indices",
        "shift_h",
        "shift_w",
        "val_masks",
        "shifted_masks",
    ]
    for k in needed:
        if k not in data:
            raise KeyError(f"Missing key in verification data: {k}")

    selected_indices = data["selected_indices"]
    shift_h = data["shift_h"]
    shift_w = data["shift_w"]
    val_masks = data["val_masks"]
    shifted_masks = data["shifted_masks"]

    custom_objects = {
        "CircularPad2D": CircularPad2D,
        "CircularConv2D": CircularConv2D,
        "DilatedCircularConv2D": DilatedCircularConv2D,
    }
    model = keras.models.load_model(args.model, custom_objects=custom_objects)

    pred_original = model.predict(val_masks, batch_size=args.batch_size, verbose=0)
    pred_shifted = model.predict(shifted_masks, batch_size=args.batch_size, verbose=0)

    results = []
    maes = []
    mses = []
    rels = []
    max_diffs = []

    for i in range(len(val_masks)):
        expected = roll_hw(pred_original[i], shift_h[i], shift_w[i])
        actual = pred_shifted[i]
        diff = actual - expected

        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(np.square(diff)))
        max_diff = float(np.max(np.abs(diff)))
        denom = float(np.mean(np.abs(expected))) + 1e-8
        rel = float(mae / denom)

        maes.append(mae)
        mses.append(mse)
        rels.append(rel)
        max_diffs.append(max_diff)

        results.append(
            {
                "sample_i": int(i),
                "dataset_index": int(selected_indices[i]),
                "shift_h": int(shift_h[i]),
                "shift_w": int(shift_w[i]),
                "mae": mae,
                "mse": mse,
                "max_diff": max_diff,
                "relative_error": rel,
            }
        )

    summary = {
        "num_samples": int(len(val_masks)),
        "avg_mae": float(np.mean(maes)),
        "avg_mse": float(np.mean(mses)),
        "avg_relative_error": float(np.mean(rels)),
        "max_mae": float(np.max(maes)),
        "max_diff": float(np.max(max_diffs)),
        "threshold": float(args.threshold),
        "is_equivariant": bool(np.mean(maes) < args.threshold),
    }

    report = {
        "model": args.model,
        "verification_data": args.verification_data,
        "summary": summary,
        "results": results,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Shift-equivariance inference check complete.")
    print(f"Model: {args.model}")
    print(f"Verification data: {args.verification_data}")
    print(f"Samples: {summary['num_samples']}")
    print(f"Average MAE: {summary['avg_mae']:.8f}")
    print(f"Average MSE: {summary['avg_mse']:.8f}")
    print(f"Average Relative Error: {summary['avg_relative_error']:.8f}")
    print(f"Max MAE: {summary['max_mae']:.8f}")
    print(f"Max Diff: {summary['max_diff']:.8f}")
    print(
        f"Result: {'PASS' if summary['is_equivariant'] else 'FAIL'} "
        f"(threshold={summary['threshold']})"
    )
    print(f"Report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
