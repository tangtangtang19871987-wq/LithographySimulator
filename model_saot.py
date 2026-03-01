"""TensorFlow implementation of a Spectral Attention Operator Transformer (SAOT).

SAOT mixes two complementary spectral paths for PDE operator learning:
1) Fourier Attention (FA): global dependency modeling in frequency domain.
2) Wavelet Attention (WA): local/high-frequency modeling via Haar wavelet bands.

The block fuses FA and WA outputs with a learnable gate, followed by an MLP.
All layers are standard Keras layers and support dynamic input sizes.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LinearAttention2D(layers.Layer):
    """Linear-complexity attention over flattened 2D tokens.

    Uses feature map phi(x)=elu(x)+1 and computes:
      out_i = (phi(q_i) @ (phi(K)^T V)) / (phi(q_i) @ sum_j phi(k_j) + eps)
    Complexity is O(N * d^2), avoiding O(N^2) pairwise attention.
    """

    def __init__(self, dim, attn_dim=None, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.attn_dim = int(attn_dim or dim)
        self.epsilon = float(epsilon)

        self.q_proj = layers.Dense(self.attn_dim, use_bias=False)
        self.k_proj = layers.Dense(self.attn_dim, use_bias=False)
        self.v_proj = layers.Dense(self.attn_dim, use_bias=False)
        self.out_proj = layers.Dense(self.dim)

    def call(self, x):
        # x: [B, H, W, C]
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]

        tokens = tf.reshape(x, [b, h * w, self.dim])
        q = tf.nn.elu(self.q_proj(tokens)) + 1.0
        k = tf.nn.elu(self.k_proj(tokens)) + 1.0
        v = self.v_proj(tokens)

        # kv: [B, D, D]
        kv = tf.einsum('bnd,bne->bde', k, v)
        # k_sum: [B, D]
        k_sum = tf.reduce_sum(k, axis=1)

        num = tf.einsum('bnd,bde->bne', q, kv)
        den = tf.einsum('bnd,bd->bn', q, k_sum)
        den = tf.expand_dims(den, axis=-1)

        out = num / (den + self.epsilon)
        out = self.out_proj(out)
        return tf.reshape(out, [b, h, w, self.dim])


class FourierAttention(layers.Layer):
    """Global spectral mixing using FFT -> channel MLP -> IFFT."""

    def __init__(self, dim, hidden_ratio=2.0, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        hidden = max(1, int(dim * hidden_ratio))
        self.dim = int(dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.real_mlp = keras.Sequential([
            layers.Dense(hidden, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(dim),
        ])
        self.imag_mlp = keras.Sequential([
            layers.Dense(hidden, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(dim),
        ])

    def call(self, x, training=False):
        # x: [B, H, W, C]
        residual = x
        x = self.norm(x)

        # FFT over spatial dims
        x_chw = tf.transpose(x, [0, 3, 1, 2])  # [B, C, H, W]
        x_freq = tf.signal.rfft2d(tf.cast(x_chw, tf.complex64))  # [B, C, H, Wf]

        real = tf.transpose(tf.math.real(x_freq), [0, 2, 3, 1])
        imag = tf.transpose(tf.math.imag(x_freq), [0, 2, 3, 1])

        real = self.real_mlp(real, training=training)
        imag = self.imag_mlp(imag, training=training)

        x_freq = tf.complex(
            tf.transpose(real, [0, 3, 1, 2]),
            tf.transpose(imag, [0, 3, 1, 2]),
        )

        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        x_out = tf.signal.irfft2d(x_freq, fft_length=[h, w])
        x_out = tf.transpose(x_out, [0, 2, 3, 1])
        return x_out + residual


class WaveletAttention(layers.Layer):
    """Local/high-frequency branch with Haar wavelet + linear attention."""

    def __init__(self, dim, wavelet_dim=64, attn_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.wavelet_dim = int(wavelet_dim)

        self.pre = layers.Conv2D(self.wavelet_dim, kernel_size=1, padding='same')
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.attn = LinearAttention2D(dim=self.wavelet_dim * 4, attn_dim=attn_dim)
        self.post = layers.Conv2D(self.dim, kernel_size=1, padding='same')

    @staticmethod
    def _haar_fwt(x):
        # x: [B, H, W, C], H/W even
        x00 = x[:, 0::2, 0::2, :]
        x01 = x[:, 0::2, 1::2, :]
        x10 = x[:, 1::2, 0::2, :]
        x11 = x[:, 1::2, 1::2, :]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    @staticmethod
    def _haar_ifwt(ll, lh, hl, hh):
        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5

        b = tf.shape(ll)[0]
        h2 = tf.shape(ll)[1]
        w2 = tf.shape(ll)[2]
        c = tf.shape(ll)[3]

        top = tf.reshape(tf.stack([x00, x01], axis=3), [b, h2, w2 * 2, c])
        bot = tf.reshape(tf.stack([x10, x11], axis=3), [b, h2, w2 * 2, c])
        rec = tf.reshape(tf.stack([top, bot], axis=2), [b, h2 * 2, w2 * 2, c])
        return rec

    def call(self, x):
        # x: [B, H, W, C]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]

        x = self.pre(x)

        # Pad to even size if needed (for 2x2 Haar splits)
        pad_h = h % 2
        pad_w = w % 2
        x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        ll, lh, hl, hh = self._haar_fwt(x)
        bands = tf.concat([ll, lh, hl, hh], axis=-1)

        bands = self.norm(bands)
        bands = self.attn(bands)

        c = self.wavelet_dim
        ll, lh, hl, hh = tf.split(bands, [c, c, c, c], axis=-1)
        rec = self._haar_ifwt(ll, lh, hl, hh)

        # Crop back to original shape
        rec = rec[:, :h, :w, :]
        return self.post(rec)


class GatedFusion(layers.Layer):
    """Adaptive fusion: gate * global + (1-gate) * local."""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.gate = layers.Dense(dim, activation='sigmoid')

    def call(self, global_feat, local_feat):
        x = tf.concat([global_feat, local_feat], axis=-1)
        g = self.gate(x)
        return g * global_feat + (1.0 - g) * local_feat


class SAOTBlock(layers.Layer):
    """One SAOT block with parallel FA/WA, gated fusion, and MLP."""

    def __init__(self, dim, wavelet_dim=64, attn_dim=64,
                 mlp_ratio=4.0, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.fa = FourierAttention(dim=dim, hidden_ratio=2.0, dropout=dropout)
        self.wa = WaveletAttention(dim=dim, wavelet_dim=wavelet_dim,
                                   attn_dim=attn_dim)
        self.fuse = GatedFusion(dim=dim)

        hidden = max(1, int(dim * mlp_ratio))
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([
            layers.Dense(hidden, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        y = self.norm(x)
        global_feat = self.fa(y, training=training)
        local_feat = self.wa(y)
        x = x + self.fuse(global_feat, local_feat)
        x = x + self.mlp(self.norm2(x), training=training)
        return x


def build_saot_model(
    input_shape=(85, 85, 1),
    output_channels=1,
    embed_dim=256,
    num_blocks=4,
    wavelet_dim=64,
    attn_dim=64,
    mlp_ratio=4.0,
    dropout=0.0,
):
    """Build SAOT-style hybrid spectral network for PDE operator learning.

    Args:
        input_shape: Spatial input shape (H, W, C_in), e.g. Darcy coefficient map.
        output_channels: Number of output channels, e.g. pressure=1.
        embed_dim: Feature dimension after input lifting.
        num_blocks: Number of SAOT transformer blocks.
        wavelet_dim: Compressed channels inside WA path.
        attn_dim: Attention latent dim used in linear attention.
        mlp_ratio: Expansion ratio for feed-forward MLP.
        dropout: Dropout probability used in FA/MLP.

    Returns:
        keras.Model mapping (H, W, C_in) -> (H, W, output_channels).
    """
    inputs = keras.Input(shape=input_shape)

    # Input lifting: low-dimensional physics field -> high-dimensional embedding
    x = layers.Dense(embed_dim)(inputs)

    for _ in range(num_blocks):
        x = SAOTBlock(
            dim=embed_dim,
            wavelet_dim=wavelet_dim,
            attn_dim=attn_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )(x)

    outputs = layers.Dense(output_channels)(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='saot_pde_operator')
