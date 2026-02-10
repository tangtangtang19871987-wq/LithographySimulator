"""Visual comparison of spatial conv2d vs FFT circular convolution.

Generates a figure showing the outputs side-by-side and the pixel-wise
absolute difference map.

Run:
    python compare_conv_fft.py
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

from shift_equivariant_unet import CircularPad1D
from fft_conv import fft_circular_depthwise_conv1d

tf.random.set_seed(0)
np.random.seed(0)


def spatial_conv(x, kernel_1d, axis_name):
    """Reference spatial circular depthwise conv."""
    K, C = kernel_1d.shape
    half = K // 2
    pad_layer = CircularPad1D(half, axis=axis_name)
    x_padded = pad_layer(x)
    ks = (1, K) if axis_name == 'width' else (K, 1)
    dw = layers.DepthwiseConv2D(kernel_size=ks, padding='valid', use_bias=False)
    dw.build(x_padded.shape)
    w = tf.reshape(kernel_1d, [ks[0], ks[1], C, 1])
    dw.set_weights([w.numpy()])
    return dw(x_padded)


def run_comparison(kernel_size, H, W, C, axis_name):
    """Run one spatial-vs-FFT comparison, return outputs and diff."""
    axis = 2 if axis_name == 'width' else 1
    x = tf.random.normal([1, H, W, C])
    kernel = tf.Variable(tf.random.normal([kernel_size, C]) * 0.1)

    out_spatial = spatial_conv(x, kernel, axis_name)
    out_fft = fft_circular_depthwise_conv1d(x, kernel, axis=axis)
    diff = tf.abs(out_fft - out_spatial)

    return (x[0, :, :, 0].numpy(),
            out_spatial[0, :, :, 0].numpy(),
            out_fft[0, :, :, 0].numpy(),
            diff[0, :, :, 0].numpy())


# ── Configs to compare ──────────────────────────────────────────
configs = [
    {'kernel_size': 31, 'H': 64, 'W': 64, 'C': 8,  'axis': 'width'},
    {'kernel_size': 31, 'H': 64, 'W': 64, 'C': 8,  'axis': 'height'},
    {'kernel_size': 15, 'H': 64, 'W': 64, 'C': 16, 'axis': 'width'},
]

fig, axes = plt.subplots(len(configs), 4, figsize=(18, 4 * len(configs)))

for row, cfg in enumerate(configs):
    inp, sp, fft_out, diff = run_comparison(
        cfg['kernel_size'], cfg['H'], cfg['W'], cfg['C'], cfg['axis'])

    label = f"K={cfg['kernel_size']}  {cfg['H']}x{cfg['W']}x{cfg['C']}  {cfg['axis']}"

    ax = axes[row]
    im0 = ax[0].imshow(inp, cmap='gray')
    ax[0].set_title(f'Input (ch 0)\n{label}', fontsize=9)
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    vmin = min(sp.min(), fft_out.min())
    vmax = max(sp.max(), fft_out.max())

    im1 = ax[1].imshow(sp, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title('Spatial conv2d', fontsize=9)
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(fft_out, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[2].set_title('FFT conv', fontsize=9)
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    im3 = ax[3].imshow(diff, cmap='hot')
    ax[3].set_title(f'|diff|  max={diff.max():.2e}  mean={diff.mean():.2e}',
                    fontsize=9)
    plt.colorbar(im3, ax=ax[3], fraction=0.046)

    for a in ax:
        a.axis('off')

plt.tight_layout()
plt.savefig('compare_conv_fft.png', dpi=150)
plt.close()
print("Saved compare_conv_fft.png")
