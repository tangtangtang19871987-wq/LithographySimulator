"""Physics-guided feature layers for lithography U-Net.

This module builds a fixed coherent-dipole aerial estimate from the mask and
injects it into encoder/decoder blocks so the network always sees a strong
physics prior.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


class CoherentDipolePhysicsLayer(keras.layers.Layer):
    """Non-trainable coherent dipole simulation layer.

    Input: mask tensor [B, H, W, 1] in [0, 1].
    Output: normalized physics intensity [B, H, W, 1].
    """

    def __init__(self, dx_nm=8.0, lambda0_nm=193.0, n_imm=1.44,
                 na_obj=1.35, pitch_nm=76.0, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.dx_nm = float(dx_nm)
        self.lambda0_nm = float(lambda0_nm)
        self.n_imm = float(n_imm)
        self.na_obj = float(na_obj)
        self.pitch_nm = float(pitch_nm)

        sin_theta = self.lambda0_nm / (2.0 * self.n_imm * self.pitch_nm)
        na_illum = self.n_imm * sin_theta
        if na_illum > self.na_obj:
            raise ValueError("NA_illum > NA_obj; adjust pitch_nm or na_obj.")

        s = np.float32(na_illum / self.lambda0_nm)
        self.sfx = tf.constant(s, dtype=tf.float32)
        self.sfy = tf.constant(0.0, dtype=tf.float32)
        self.fc = tf.constant(self.na_obj / self.lambda0_nm, dtype=tf.float32)

    def build(self, input_shape):
        _, h, w, _ = input_shape
        if h is None or w is None:
            raise ValueError("CoherentDipolePhysicsLayer requires static H/W.")

        fx = np.fft.fftfreq(int(w), d=self.dx_nm).astype(np.float32)
        fy = np.fft.fftfreq(int(h), d=self.dx_nm).astype(np.float32)
        fx_t = tf.constant(fx)[tf.newaxis, :]
        fy_t = tf.constant(fy)[:, tf.newaxis]
        self.FX = tf.tile(fx_t, [int(h), 1])
        self.FY = tf.tile(fy_t, [1, int(w)])
        super().build(input_shape)

    def call(self, mask):
        mask_real = tf.cast(mask[..., 0], tf.float32)
        m = tf.signal.fft2d(tf.cast(mask_real, tf.complex64))

        pp = tf.cast((self.FX - self.sfx) ** 2 + (self.FY - self.sfy) ** 2 <= self.fc ** 2, tf.float32)
        pm = tf.cast((self.FX + self.sfx) ** 2 + (self.FY + self.sfy) ** 2 <= self.fc ** 2, tf.float32)

        e = tf.signal.ifft2d(m * tf.cast(pp, tf.complex64))
        e += tf.signal.ifft2d(m * tf.cast(pm, tf.complex64))
        intensity = tf.math.real(e * tf.math.conj(e))

        intensity /= (tf.reduce_max(intensity, axis=[1, 2], keepdims=True) + 1e-12)
        return intensity[..., tf.newaxis]


class PhysicsFuse(keras.layers.Layer):
    """Fuse physics map into feature tensor with guaranteed non-zero influence.

    y = concat([x + alpha * proj(phys), proj(phys)]).
    alpha has a floor to avoid forgetting/being drowned.
    """

    def __init__(self, min_alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.min_alpha = float(min_alpha)

    def build(self, input_shape):
        feat_shape, _ = input_shape
        channels = int(feat_shape[-1])
        self.proj = keras.layers.Conv2D(
            channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer='ones',
            trainable=False,
            name=f"{self.name}_physics_proj",
        )
        self.alpha_raw = self.add_weight(
            name='alpha_raw',
            shape=(),
            initializer=keras.initializers.Constant(0.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        x, phys = inputs
        p = self.proj(phys)
        alpha = self.min_alpha + tf.nn.softplus(self.alpha_raw)
        merged = x + alpha * p
        return tf.concat([merged, p], axis=-1)


class PhysicsOutputBlend(keras.layers.Layer):
    """Blend learned output with physics prior with non-zero prior floor."""

    def __init__(self, min_beta=0.15, **kwargs):
        super().__init__(**kwargs)
        self.min_beta = float(min_beta)

    def build(self, input_shape):
        self.beta_raw = self.add_weight(
            name='beta_raw',
            shape=(),
            initializer=keras.initializers.Constant(-1.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        pred, phys = inputs
        beta = self.min_beta + tf.nn.sigmoid(self.beta_raw)
        return tf.clip_by_value((1.0 - beta) * pred + beta * phys, 0.0, 1.0)
