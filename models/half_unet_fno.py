"""
Half-UNet Fourier Neural Operator — TensorFlow 2.10 Port
==========================================================

Faithful TF 2.10 port of https://github.com/Robh96/UNet-FNO

Architecture (unchanged from original):
  • SpectralConv2d  — 2-D spectral convolution with learnable complex weights
                      (positive + negative frequency modes kept)
  • FNOBlock        — spectral path + pointwise-spatial path + residual
  • HalfUNetFNO     — encoder of FNOBlocks, full-scale feature fusion, decoder

Key TF 2.10 adaptations:
  • Data format: NHWC (B, H, W, C) — PyTorch used NCHW
  • Complex weights stored as real + imaginary float32 variable pairs
    (tf.Variable(dtype=complex64) gradient flow is unstable in TF 2.10)
  • rfft2d / irfft2d via tf.signal; ortho normalisation applied manually
  • Bilinear upsample: tf.image.resize (align_corners=False)
  • AdamW: tf.keras.optimizers.experimental.AdamW
  • Einsum adapted to NHWC:
      PyTorch "bixy,ioxy->boxy"  →  TF "bhwi,hwio->bhwo"

Comparison with our existing fft_conv.py / fft_layers.py
  (see FNO_IMPLEMENTATION_COMPARISON.md for full details)
  • SpectralConv2d here: 2-D rfft, truncated modes, cross-channel mixing
  • fft_circular_depthwise_conv1d: 1-D rfft, full spectrum, depthwise only
  Both use rfft + irfft but serve different purposes.

Author: Claude Code  (ported 2026-02-11)
Original: https://github.com/Robh96/UNet-FNO (MIT License)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Complex multiplication helper
# ---------------------------------------------------------------------------

def _compl_mul2d(x_c: tf.Tensor, w_re: tf.Tensor, w_im: tf.Tensor) -> tf.Tensor:
    """Complex matmul in frequency space.

    Computes:  out = x_c · (w_re + j·w_im)

    Args:
        x_c:  Complex64 tensor (B, h, w, C_in)  — truncated frequency slice.
        w_re: Float32 weight real part (h, w, C_in, C_out).
        w_im: Float32 weight imaginary part (h, w, C_in, C_out).

    Returns:
        Complex64 (B, h, w, C_out).

    Equivalent to PyTorch:
        torch.einsum("bixy,ioxy->boxy", x, w)   [NCHW ordering]
    mapped to NHWC einsum "bhwi,hwio->bhwo".
    """
    x_re = tf.math.real(x_c)   # (B, h, w, C_in)
    x_im = tf.math.imag(x_c)

    out_re = (tf.einsum("bhwi,hwio->bhwo", x_re, w_re)
              - tf.einsum("bhwi,hwio->bhwo", x_im, w_im))
    out_im = (tf.einsum("bhwi,hwio->bhwo", x_re, w_im)
              + tf.einsum("bhwi,hwio->bhwo", x_im, w_re))

    return tf.complex(out_re, out_im)   # (B, h, w, C_out)


# ---------------------------------------------------------------------------
# SpectralConv2d
# ---------------------------------------------------------------------------

class SpectralConv2d(layers.Layer):
    """2-D Fourier layer: rfft2 → truncated spectral mixing → irfft2.

    Keeps the lowest ``modes1`` H-frequencies (both positive and negative)
    and the lowest ``modes2//2+1`` positive W-frequencies.  All other
    frequency coefficients are zeroed out before the inverse transform,
    acting as a learned low-pass filter.

    Weight storage:
        Two sets of complex weights (for positive / negative H-modes):
            w1, w2  each of shape (modes1, modes2//2+1, C_in, C_out)
        Each stored as a real-part + imag-part float32 variable pair for
        stable gradient flow in TF 2.10.

    Args:
        in_channels:  Number of input channels  (C_in).
        out_channels: Number of output channels (C_out).
        modes1:       Fourier modes to keep along H dimension.
        modes2:       Fourier modes to keep along W dimension.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

    def build(self, input_shape):
        scale = 1.0 / (self.in_channels * self.out_channels)
        # Weight shape in NHWC-spectral order: (modes1, modes2//2+1, C_in, C_out)
        m1, m2h = self.modes1, self.modes2 // 2 + 1
        w_shape = (m1, m2h, self.in_channels, self.out_channels)

        for name in ("w1_re", "w1_im", "w2_re", "w2_im"):
            setattr(self, name, self.add_weight(
                name=name, shape=w_shape,
                initializer=keras.initializers.RandomUniform(0.0, scale),
                trainable=True, dtype=tf.float32))

        super().build(input_shape)

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
        """Forward pass.

        Args:
            x: (B, H, W, C_in)  float32

        Returns:
            (B, H, W, C_out)  float32
        """
        # Static spatial sizes — needed for irfft2d fft_length
        H_s = x.shape[1]   # int or None
        W_s = x.shape[2]
        # Dynamic sizes for runtime shapes
        H_d = tf.shape(x)[1]
        W_d = tf.shape(x)[2]

        # ---- 2-D rfft (ortho normalised) ----
        # tf.signal.rfft2d operates on the two innermost dims → transpose to NCHW
        x_nchw = tf.transpose(x, [0, 3, 1, 2])            # (B, C_in, H, W)
        x_ft_nchw = tf.signal.rfft2d(x_nchw)              # (B, C_in, H, W//2+1) complex64
        x_ft = tf.transpose(x_ft_nchw, [0, 2, 3, 1])      # (B, H, W//2+1, C_in)

        # Ortho normalisation: divide by sqrt(H*W)
        norm = tf.cast(tf.math.sqrt(tf.cast(H_d * W_d, tf.float32)), tf.complex64)
        x_ft = x_ft / norm

        # ---- Effective mode counts (clip to grid) ----
        m1_eff = min(self.modes1, H_s) if H_s else self.modes1
        m2_eff = min(self.modes2 // 2 + 1, W_s // 2 + 1) if W_s else self.modes2 // 2 + 1

        # ---- Truncated frequency slices ----
        x_top    = x_ft[:,  :m1_eff,  :m2_eff, :]   # (B, m1_eff, m2_eff, C_in)
        x_bot    = x_ft[:, -m1_eff:,  :m2_eff, :]   # (B, m1_eff, m2_eff, C_in)

        # ---- Spectral mixing via complex multiplication ----
        w1_re = self.w1_re[:m1_eff, :m2_eff]   # (m1_eff, m2_eff, C_in, C_out)
        w1_im = self.w1_im[:m1_eff, :m2_eff]
        w2_re = self.w2_re[:m1_eff, :m2_eff]
        w2_im = self.w2_im[:m1_eff, :m2_eff]

        out_top = _compl_mul2d(x_top, w1_re, w1_im)   # (B, m1_eff, m2_eff, C_out)
        out_bot = _compl_mul2d(x_bot, w2_re, w2_im)   # (B, m1_eff, m2_eff, C_out)

        # ---- Reconstruct full frequency tensor with zeros elsewhere ----
        # We need shape (B, H, W//2+1, C_out).
        # Use tf.pad on the mode blocks, then concat top+mid+bot vertically.
        C_out = self.out_channels

        # Pad W dimension to full W//2+1
        W_half_s = W_s // 2 + 1 if W_s else None
        W_half_d = W_d // 2 + 1

        pad_w = W_half_d - m2_eff   # padding on right of W-half axis
        zeros_w_top = tf.zeros(
            tf.stack([tf.shape(x)[0], m1_eff, pad_w, C_out]),
            dtype=tf.complex64)
        zeros_w_bot = tf.zeros_like(zeros_w_top)

        out_top_full = tf.concat([out_top, zeros_w_top], axis=2)  # (B, m1_eff, W//2+1, C_out)
        out_bot_full = tf.concat([out_bot, zeros_w_bot], axis=2)

        # Middle (zero) rows
        mid_h = H_d - 2 * m1_eff
        zeros_mid = tf.zeros(
            tf.stack([tf.shape(x)[0], mid_h, W_half_d, C_out]),
            dtype=tf.complex64)

        out_ft = tf.concat([out_top_full, zeros_mid, out_bot_full], axis=1)  # (B,H,W//2+1,C_out)

        # ---- irfft2d (undo ortho normalisation first) ----
        out_ft = out_ft * norm

        out_nchw_ft = tf.transpose(out_ft, [0, 3, 1, 2])   # (B, C_out, H, W//2+1)

        if H_s and W_s:
            out_nchw = tf.signal.irfft2d(out_nchw_ft, fft_length=[H_s, W_s])
        else:
            out_nchw = tf.signal.irfft2d(out_nchw_ft, fft_length=[H_d, W_d])

        out = tf.transpose(out_nchw, [0, 2, 3, 1])          # (B, H, W, C_out)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        modes1=self.modes1, modes2=self.modes2))
        return cfg


# ---------------------------------------------------------------------------
# FNOBlock
# ---------------------------------------------------------------------------

class FNOBlock(layers.Layer):
    """Spectral + pointwise-spatial path with residual connection.

    out = activation( SpectralConv2d(x) + Conv2D_1x1(x) ) + residual(x)

    Residual is identity when C_in == C_out, else 1×1 Conv2D.

    Args:
        in_channels:   C_in.
        out_channels:  C_out.
        modes1, modes2: Fourier mode counts.
        activation:    Keras activation name string.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int,
                 activation: str = "gelu", **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.activation_name = activation

        self.spec_conv = SpectralConv2d(
            in_channels, out_channels, modes1, modes2, name="spec_conv")
        self.spatial_conv = layers.Conv2D(
            out_channels, kernel_size=1, padding="same",
            use_bias=True, name="spatial_conv")

        if in_channels != out_channels:
            self.residual_conv = layers.Conv2D(
                out_channels, kernel_size=1, padding="same",
                use_bias=False, name="res_conv")
        else:
            self.residual_conv = None

        self.act = keras.activations.get(activation)

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
        x_res  = self.residual_conv(x) if self.residual_conv is not None else x
        x_spec = self.spec_conv(x, training=training)
        x_spat = self.spatial_conv(x)
        return self.act(x_spec + x_spat) + x_res

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        modes1=self.modes1, modes2=self.modes2,
                        activation=self.activation_name))
        return cfg


# ---------------------------------------------------------------------------
# HalfUNetFNO
# ---------------------------------------------------------------------------

class HalfUNetFNO(keras.Model):
    """Half-UNet architecture with Fourier Neural Operator blocks.

    Encoder:  (levels+1) FNOBlocks at decreasing spatial resolutions.
    Decoder:  All encoder feature maps bilinearly upsampled to the original
              spatial size, element-wise summed (full-scale feature fusion).
    Head:     num_final_blocks FNOBlocks + two-stage projection Conv2D head.

    Channel width is kept constant (= width) throughout.

    Args:
        in_channels:      Input channel count.
        out_channels:     Output channel count.
        modes:            Fourier modes per spatial dimension (modes1 = modes2).
        width:            Constant internal channel width.
        levels:           Downsampling steps; total encoder blocks = levels+1.
        activation:       Keras activation name.
        num_final_blocks: FNOBlocks applied after feature fusion.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes: int = 20, width: int = 48, levels: int = 3,
                 activation: str = "gelu", num_final_blocks: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.cfg = dict(in_channels=in_channels, out_channels=out_channels,
                        modes=modes, width=width, levels=levels,
                        activation=activation,
                        num_final_blocks=num_final_blocks)

        self.lifting = layers.Conv2D(
            width, kernel_size=1, use_bias=True, name="lifting")

        self.encoder_blocks = [
            FNOBlock(width, width, modes, modes,
                     activation=activation, name=f"enc_{i}")
            for i in range(levels + 1)
        ]
        self.pool_layers = [
            layers.MaxPool2D(2, name=f"pool_{i}")
            for i in range(levels)
        ]
        self.final_blocks = [
            FNOBlock(width, width, modes, modes,
                     activation=activation, name=f"final_{i}")
            for i in range(num_final_blocks)
        ]
        self.proj_expand = layers.Conv2D(
            width * 4, kernel_size=1, use_bias=True, name="proj_expand")
        self.proj_act = keras.activations.get(activation)
        self.proj_out = layers.Conv2D(
            out_channels, kernel_size=1, use_bias=True, name="proj_out")

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
        # Lifting
        x = self.lifting(x)
        target_h = tf.shape(x)[1]
        target_w = tf.shape(x)[2]
        target_size = (target_h, target_w)

        # Encoder
        enc_outputs = []
        h = x
        for i, blk in enumerate(self.encoder_blocks):
            h = blk(h, training=training)
            enc_outputs.append(h)
            if i < len(self.pool_layers):
                h = self.pool_layers[i](h)

        # Full-scale feature fusion
        fused = None
        for feat in enc_outputs:
            up = tf.image.resize(feat, target_size, method="bilinear")
            fused = up if fused is None else fused + up

        # Final FNO blocks
        out = fused
        for blk in self.final_blocks:
            out = blk(out, training=training)

        # Projection head
        out = self.proj_expand(out)
        out = self.proj_act(out)
        out = self.proj_out(out)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(self.cfg)
        return cfg


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

ACTIVATION_MAP = {"relu": "relu", "gelu": "gelu"}


def build_half_unet_fno(
    in_channels: int, out_channels: int,
    modes: int = 20, width: int = 48,
    levels: int = 3, num_final_blocks: int = 2,
    activation: str = "gelu",
) -> HalfUNetFNO:
    """Return an un-compiled HalfUNetFNO."""
    if activation.lower() not in ACTIVATION_MAP:
        raise ValueError(f"Unsupported activation '{activation}'. "
                         f"Choose from {list(ACTIVATION_MAP)}")
    return HalfUNetFNO(in_channels=in_channels, out_channels=out_channels,
                       modes=modes, width=width, levels=levels,
                       activation=ACTIVATION_MAP[activation.lower()],
                       num_final_blocks=num_final_blocks)


def build_and_compile(
    in_channels: int, out_channels: int,
    input_shape_hw: tuple,
    modes: int = 20, width: int = 48,
    levels: int = 3, num_final_blocks: int = 2,
    activation: str = "gelu",
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    loss: str = "mse",
) -> HalfUNetFNO:
    """Build, warm-up (variable creation), compile, and return the model.

    Args:
        in_channels, out_channels: Channel counts.
        input_shape_hw: Spatial (H, W) used for warm-up forward pass.
        modes, width, levels, num_final_blocks, activation: Architecture HP.
        learning_rate, weight_decay: Optimiser HP.
        loss: Keras loss string.

    Returns:
        Compiled HalfUNetFNO model.
    """
    model = build_half_unet_fno(in_channels, out_channels, modes, width,
                                 levels, num_final_blocks, activation)

    # Warm-up: materialise all tf.Variable weights
    H, W = input_shape_hw
    _ = model(tf.zeros((1, H, W, in_channels)), training=False)

    # Optimiser — TF 2.10 ships AdamW as experimental
    try:
        opt = tf.keras.optimizers.experimental.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay)
    except AttributeError:
        # Older TF builds: fall back to Adam
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss=loss, metrics=["mae"])

    n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"HalfUNetFNO | params: {n_params:,} | "
          f"modes={modes} width={width} levels={levels}")

    return model


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("HalfUNetFNO — TF 2.10 port — self-test")
    print("=" * 70)
    tf.random.set_seed(42)

    H, W = 128, 128
    B, C_in, C_out = 2, 3, 1

    model = build_and_compile(
        in_channels=C_in, out_channels=C_out,
        input_shape_hw=(H, W),
        modes=12, width=32, levels=3,
        num_final_blocks=2,
    )

    x = tf.random.normal((B, H, W, C_in))
    y = model(x, training=False)

    assert y.shape == (B, H, W, C_out), f"Shape mismatch: {y.shape}"
    assert not tf.reduce_any(tf.math.is_nan(y)), "NaN in output!"
    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("✓ Forward pass OK")

    # Quick training step
    ds = tf.data.Dataset.from_tensors(
        (tf.random.normal((4, H, W, C_in)),
         tf.random.normal((4, H, W, C_out)))
    ).batch(2)

    hist = model.fit(ds, epochs=3, verbose=0)
    losses = [f"{v:.4f}" for v in hist.history["loss"]]
    print(f"✓ Training OK — losses: {losses}")
    print("\nAll checks passed.")
