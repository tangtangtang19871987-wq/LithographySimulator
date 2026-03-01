"""Smoke tests for SAOT TensorFlow model."""

import sys
import numpy as np
import tensorflow as tf

from model_saot import build_saot_model


def test_build_and_forward(input_shape, batch_size=2):
    model = build_saot_model(
        input_shape=input_shape,
        output_channels=1,
        embed_dim=128,
        num_blocks=2,
        wavelet_dim=32,
        attn_dim=32,
    )
    x = tf.random.normal([batch_size, *input_shape])
    y = model(x, training=False)

    expected = (batch_size, input_shape[0], input_shape[1], 1)
    shape_ok = tuple(y.shape) == expected
    finite_ok = bool(np.all(np.isfinite(y.numpy())))
    return shape_ok and finite_ok, y.shape


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    configs = [
        (85, 85, 1),  # odd spatial size (Darcy-like)
        (64, 64, 1),
        (96, 80, 2),
    ]

    all_passed = True
    print('=' * 60)
    print('SAOT model smoke tests')
    print('=' * 60)

    for cfg in configs:
        passed, shape = test_build_and_forward(cfg)
        status = 'PASS' if passed else 'FAIL'
        print(f'input={cfg}, output={shape} [{status}]')
        if not passed:
            all_passed = False

    print('=' * 60)
    print('ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED')
    print('=' * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
