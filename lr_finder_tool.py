"""
Learning Rate Finder Tool

Standalone tool to find optimal learning rate before training.

Usage:
    python lr_finder_tool.py --num-samples 100 --min-lr 1e-7 --max-lr 10
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from train import build_model
from data_pipeline import generate_dataset, make_tf_dataset
from train_advanced import LRFinder


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal learning rate for training'
    )
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--min-lr', type=float, default=1e-7,
                       help='Minimum learning rate to test')
    parser.add_argument('--max-lr', type=float, default=10.0,
                       help='Maximum learning rate to test')
    parser.add_argument('--num-steps', type=int, default=100,
                       help='Number of steps to test')
    parser.add_argument('--filters', type=int, default=32,
                       help='Base filter count for model')
    parser.add_argument('--output', type=str, default='lr_finder.png',
                       help='Output plot path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Learning Rate Finder")
    print(f"{'='*60}")

    # Generate data
    print(f"\nGenerating {args.num_samples} samples...")
    masks, aerials = generate_dataset(args.num_samples, seed=args.seed,
                                      verbose=False)

    # Create dataset
    dataset = make_tf_dataset(masks, aerials, batch_size=args.batch_size,
                             shuffle=True)

    # Build model
    print(f"Building model...")
    input_shape = masks.shape[1:]
    model = build_model(input_shape=input_shape,
                       num_filters_base=args.filters)

    # Compile model with dummy LR (will be changed during search)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='mse',
        metrics=['mae']
    )

    print(f"Model: {model.count_params():,} parameters")

    # Run LR finder
    print(f"\nRunning LR finder...")
    lr_finder = LRFinder(model, dataset)
    lr_finder.find(
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_steps=args.num_steps
    )

    # Get optimal LR
    optimal_lr = lr_finder.get_optimal_lr()

    # Plot results
    lr_finder.plot(save_path=args.output)

    # Print recommendations
    print(f"\n{'='*60}")
    print(f"Recommendations")
    print(f"{'='*60}")
    print(f"  Suggested LR: {optimal_lr:.2e}")
    print(f"  Conservative range: {optimal_lr/10:.2e} to {optimal_lr:.2e}")
    print(f"  Aggressive range: {optimal_lr:.2e} to {optimal_lr*2:.2e}")
    print(f"\nExample commands:")
    print(f"  # Conservative")
    print(f"  python train_distributed.py --lr {optimal_lr/10:.2e}")
    print(f"\n  # Recommended")
    print(f"  python train_distributed.py --lr {optimal_lr:.2e}")
    print(f"\n  # With OneCycle")
    print(f"  python train_pro.py --lr {optimal_lr:.2e} --lr-schedule onecycle")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
