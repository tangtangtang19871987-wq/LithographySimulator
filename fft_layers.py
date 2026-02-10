"""FFT-accelerated drop-in replacements for large-kernel circular conv layers.

Provides ``FFTAxisCircularConv`` — a drop-in replacement for
``AxisCircularConv`` from ``shift_equivariant_unet.py`` that uses
FFT-based circular cross-correlation instead of spatial
``DepthwiseConv2D`` for the depthwise step.

Usage
-----
Swap ``AxisCircularConv`` for ``FFTAxisCircularConv`` wherever the
kernel size is large (recommended threshold: >= 11).  Small kernels
should keep using the spatial ``AxisCircularConv``.

    from fft_layers import FFTAxisCircularConv

    # Instead of:  AxisCircularConv(dim, axis='width', kernel_size=31)
    layer = FFTAxisCircularConv(dim, axis='width', kernel_size=31)

The two layers produce numerically close results (max ~1e-6 in float32).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from fft_conv import fft_circular_depthwise_conv1d

# Serialization decorator for Keras 3; no-op on Keras 2
try:
    _register = keras.saving.register_keras_serializable(package='litho_fft')
except AttributeError:
    def _register(cls):
        return cls


@_register
class FFTAxisCircularConv(layers.Layer):
    """FFT-accelerated large-kernel depthwise circular conv along one axis.

    Drop-in replacement for ``AxisCircularConv``.  Performs circular
    cross-correlation via FFT along a single spatial axis (width or
    height), followed by a pointwise (1x1) conv for channel mixing.

    For kernel sizes >= ~11 this is faster than the spatial
    ``CircularPad1D`` + ``DepthwiseConv2D`` path while producing
    numerically equivalent results (float32 precision).

    Args:
        filters: Number of output channels (set by pointwise conv).
        axis: ``'width'`` or ``'height'`` — the spatial axis to convolve along.
        kernel_size: Depthwise kernel length along the chosen axis.
    """

    def __init__(self, filters, axis='width', kernel_size=31, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.axis = axis
        self.ks = kernel_size
        self.pw_conv = layers.Conv2D(filters, 1, use_bias=True)

    def build(self, input_shape):
        C = input_shape[-1]
        # Depthwise kernel: one 1-D kernel of length K per channel → (K, C)
        self._dw_kernel = self.add_weight(
            name='fft_depthwise_kernel',
            shape=(self.ks, C),
            initializer='glorot_uniform',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        conv_axis = 2 if self.axis == 'width' else 1
        x = fft_circular_depthwise_conv1d(x, self._dw_kernel, axis=conv_axis)
        x = self.pw_conv(x)
        return x

    def compute_output_shape(self, input_shape):
        # Spatial dims preserved (circular conv); only channels change.
        return (*input_shape[:-1], self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'axis': self.axis,
            'kernel_size': self.ks,
        })
        return config
