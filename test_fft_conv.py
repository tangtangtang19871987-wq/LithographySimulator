"""Smoke tests for FFT-based circular depthwise convolution.

Verifies that the FFT path (fft_conv.py / fft_layers.py) produces
numerically close results to the spatial (CircularPad + DepthwiseConv2D)
path for various kernel sizes, spatial dimensions, and channel counts.

Run:
    python test_fft_conv.py
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Ensure reproducible results
tf.random.set_seed(42)
np.random.seed(42)


def spatial_circular_depthwise_conv1d(x, kernel_1d_per_channel, axis_name):
    """Reference implementation using CircularPad + DepthwiseConv2D.

    Args:
        x: (B, H, W, C) input tensor.
        kernel_1d_per_channel: (K, C) depthwise kernel.
        axis_name: 'width' (axis=2) or 'height' (axis=1).

    Returns:
        (B, H, W, C) output tensor.
    """
    K = kernel_1d_per_channel.shape[0]
    C = kernel_1d_per_channel.shape[1]
    half = K // 2

    from shift_equivariant_unet import CircularPad1D

    pad_layer = CircularPad1D(half, axis=axis_name)
    x_padded = pad_layer(x)

    dw_conv = layers.DepthwiseConv2D(
        kernel_size=(1, K) if axis_name == 'width' else (K, 1),
        padding='valid',
        use_bias=False,
    )
    # Build the layer so we can set weights
    dw_conv.build(x_padded.shape)

    # Reshape kernel_1d_per_channel (K, C) → depthwise kernel shape
    if axis_name == 'width':
        dw_weights = tf.reshape(kernel_1d_per_channel, [1, K, C, 1])
    else:
        dw_weights = tf.reshape(kernel_1d_per_channel, [K, 1, C, 1])

    dw_conv.set_weights([dw_weights.numpy()])
    return dw_conv(x_padded)


def test_fft_vs_spatial(kernel_size, height, width, channels, batch=2,
                        axis_name='width', atol=1e-4):
    """Compare FFT and spatial depthwise circular conv1d.

    Returns (pass, max_abs_error, mean_abs_error).
    """
    from fft_conv import fft_circular_depthwise_conv1d

    axis = 2 if axis_name == 'width' else 1

    x = tf.random.normal([batch, height, width, channels])
    kernel = tf.Variable(
        tf.random.normal([kernel_size, channels]) * 0.1,
        trainable=False,
    )

    out_fft = fft_circular_depthwise_conv1d(x, kernel, axis=axis)
    out_spatial = spatial_circular_depthwise_conv1d(x, kernel, axis_name)

    diff = tf.abs(out_fft - out_spatial)
    max_err = float(tf.reduce_max(diff))
    mean_err = float(tf.reduce_mean(diff))
    passed = max_err < atol

    return passed, max_err, mean_err


def test_fft_conv_circular_property(kernel_size=31, size=64, channels=8):
    """Verify that FFT conv produces the same result when input is circularly
    shifted — i.e. the output shifts by the same amount (shift equivariance)."""
    from fft_conv import fft_circular_depthwise_conv1d

    x = tf.random.normal([1, size, size, channels])
    kernel = tf.Variable(tf.random.normal([kernel_size, channels]) * 0.1)

    # Width axis
    out_orig = fft_circular_depthwise_conv1d(x, kernel, axis=2)
    shift = 13
    x_shifted = tf.roll(x, shift=shift, axis=2)
    out_shifted = fft_circular_depthwise_conv1d(x_shifted, kernel, axis=2)
    out_expected = tf.roll(out_orig, shift=shift, axis=2)

    diff = tf.abs(out_shifted - out_expected)
    max_err = float(tf.reduce_max(diff))
    return max_err < 1e-5, max_err


def test_fft_layer_vs_spatial_layer():
    """End-to-end: FFTAxisCircularConv vs original AxisCircularConv."""
    from shift_equivariant_unet import AxisCircularConv
    from fft_layers import FFTAxisCircularConv

    K = 31
    C_in = 16
    C_out = 16
    H, W = 32, 32
    B = 2
    x = tf.random.normal([B, H, W, C_in])

    # Build FFT layer
    layer_fft = FFTAxisCircularConv(C_out, axis='width', kernel_size=K)
    _ = layer_fft(x)  # build

    # Build spatial layer
    layer_spatial = AxisCircularConv(C_out, axis='width', kernel_size=K)
    _ = layer_spatial(x)  # build

    # Copy depthwise kernel from FFT layer to spatial layer
    fft_dw = layer_fft._dw_kernel.numpy()  # (K, C_in)
    # Spatial layer stores kernel in DepthwiseConv2D: shape (1, K, C_in, 1)
    spatial_dw = np.reshape(fft_dw, [1, K, C_in, 1])
    layer_spatial.dw_conv.set_weights([spatial_dw])

    # Copy pointwise conv weights
    layer_spatial.pw_conv.set_weights(layer_fft.pw_conv.get_weights())

    out_fft = layer_fft(x)
    out_spatial = layer_spatial(x)

    diff = tf.abs(out_fft - out_spatial)
    max_err = float(tf.reduce_max(diff))
    mean_err = float(tf.reduce_mean(diff))
    passed = max_err < 1e-3
    return passed, max_err, mean_err


def test_fft_layer_in_model():
    """Verify FFTAxisCircularConv works inside a Keras Functional model."""
    from fft_layers import FFTAxisCircularConv
    from shift_equivariant_unet import GELUApprox

    inp = tf.keras.Input(shape=(32, 32, 16))
    x = FFTAxisCircularConv(16, axis='width', kernel_size=31)(inp)
    x = GELUApprox()(x)
    x = FFTAxisCircularConv(16, axis='height', kernel_size=31)(x)
    x = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    model = tf.keras.Model(inp, x)

    x_in = tf.random.normal([1, 32, 32, 16])
    out = model(x_in, training=False)
    shape_ok = out.shape == (1, 32, 32, 1)
    range_ok = float(tf.reduce_min(out)) >= 0.0 and float(tf.reduce_max(out)) <= 1.0
    return shape_ok and range_ok, out.shape


def test_original_model_unchanged():
    """Verify original TokenizedMLP model still builds with spatial conv."""
    from model_tokenized_mlp import build_model
    model = build_model(input_shape=(32, 32, 1), num_filters_base=16,
                        num_mlp_blocks=1, axis_kernel=31)
    x = tf.random.normal([1, 32, 32, 1])
    out = model(x, training=False)
    shape_ok = out.shape == (1, 32, 32, 1)
    range_ok = float(tf.reduce_min(out)) >= 0.0 and float(tf.reduce_max(out)) <= 1.0
    return shape_ok and range_ok, out.shape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_passed = True
    print("=" * 65)
    print("FFT Circular Convolution — Smoke Tests")
    print("=" * 65)

    # Test 1: FFT vs Spatial for various configs
    configs = [
        # (kernel_size, H, W, C, axis)
        (31, 64, 64, 8, 'width'),
        (31, 64, 64, 8, 'height'),
        (15, 32, 32, 16, 'width'),
        (15, 32, 32, 16, 'height'),
        (11, 48, 48, 4, 'width'),
        (21, 64, 64, 32, 'height'),
        (31, 128, 128, 8, 'width'),
        # Odd spatial sizes
        (31, 63, 63, 8, 'width'),
        (31, 33, 65, 8, 'height'),
    ]

    print("\n[Test 1] FFT vs Spatial depthwise conv1d equivalence")
    print("-" * 65)
    for ks, h, w, c, axis in configs:
        passed, max_err, mean_err = test_fft_vs_spatial(ks, h, w, c,
                                                         axis_name=axis)
        status = "PASS" if passed else "FAIL"
        print(f"  K={ks:3d}  {h:3d}x{w:3d}x{c:2d}  axis={axis:6s}  "
              f"max_err={max_err:.2e}  mean_err={mean_err:.2e}  [{status}]")
        if not passed:
            all_passed = False

    # Test 2: Circular shift equivariance of FFT conv
    print("\n[Test 2] Shift equivariance of FFT conv")
    print("-" * 65)
    passed, max_err = test_fft_conv_circular_property()
    status = "PASS" if passed else "FAIL"
    print(f"  Circular shift test: max_err={max_err:.2e}  [{status}]")
    if not passed:
        all_passed = False

    # Test 3: FFTAxisCircularConv vs AxisCircularConv (end-to-end layer test)
    print("\n[Test 3] FFTAxisCircularConv vs AxisCircularConv (same weights)")
    print("-" * 65)
    passed, max_err, mean_err = test_fft_layer_vs_spatial_layer()
    status = "PASS" if passed else "FAIL"
    print(f"  max_err={max_err:.2e}  mean_err={mean_err:.2e}  [{status}]")
    if not passed:
        all_passed = False

    # Test 4: FFTAxisCircularConv inside a Keras Functional model
    print("\n[Test 4] FFTAxisCircularConv in Keras Functional model")
    print("-" * 65)
    passed, shape = test_fft_layer_in_model()
    status = "PASS" if passed else "FAIL"
    print(f"  Output shape: {shape}  [{status}]")
    if not passed:
        all_passed = False

    # Test 5: Original model unchanged (still uses spatial conv)
    print("\n[Test 5] Original TokenizedMLP model unchanged")
    print("-" * 65)
    passed, shape = test_original_model_unchanged()
    status = "PASS" if passed else "FAIL"
    print(f"  Output shape: {shape}  [{status}]")
    if not passed:
        all_passed = False

    # Summary
    print("\n" + "=" * 65)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 65)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
