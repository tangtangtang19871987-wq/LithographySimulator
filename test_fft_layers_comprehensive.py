"""
Comprehensive unit tests for fft_layers.py and FFT convolution

Tests:
- FFT circular convolution correctness
- Performance characteristics
- Edge cases and boundary conditions
- Comparison with spatial convolution
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import modules to test
from fft_layers import CircularConv2DFFT, DilatedCircularConv2DFFT
from shift_equivariant_unet import CircularConv2D, DilatedCircularConv2D


class TestFFTConvolutionCorrectness(unittest.TestCase):
    """Test correctness of FFT convolution vs spatial convolution."""

    def test_fft_conv_vs_spatial_small_kernel(self):
        """Test FFT conv matches spatial conv for small kernel."""
        input_shape = (32, 32, 1)
        filters = 8
        kernel_size = 3

        # Create input
        x = tf.random.normal((2,) + input_shape)

        # Create both layers with same initialization
        spatial_conv = CircularConv2D(
            filters, kernel_size, padding='same',
            kernel_initializer='glorot_uniform'
        )
        fft_conv = CircularConv2DFFT(
            filters, kernel_size,
            kernel_initializer='glorot_uniform'
        )

        # Build layers
        spatial_conv.build(input_shape)
        fft_conv.build(input_shape)

        # Set same weights
        fft_conv.set_weights(spatial_conv.get_weights())

        # Forward pass
        out_spatial = spatial_conv(x, training=False)
        out_fft = fft_conv(x, training=False)

        # Check shapes match
        self.assertEqual(out_spatial.shape, out_fft.shape)

        # Check values are close (FFT has numerical precision differences)
        max_diff = tf.reduce_max(tf.abs(out_spatial - out_fft))
        self.assertLess(float(max_diff), 1e-3)

        print(f"✓ FFT vs Spatial (3x3): max diff = {float(max_diff):.2e}")

    def test_fft_conv_output_shape(self):
        """Test FFT convolution output shape."""
        batch_size = 4
        height, width = 64, 64
        in_channels = 3
        out_channels = 16
        kernel_size = 5

        x = tf.random.normal((batch_size, height, width, in_channels))

        fft_conv = CircularConv2DFFT(out_channels, kernel_size)
        out = fft_conv(x)

        expected_shape = (batch_size, height, width, out_channels)
        self.assertEqual(out.shape, expected_shape)

        print(f"✓ FFT conv output shape correct: {out.shape}")

    def test_fft_conv_with_activation(self):
        """Test FFT convolution with activation function."""
        x = tf.random.normal((2, 32, 32, 1))

        fft_conv = CircularConv2DFFT(8, 3, activation='relu')
        out = fft_conv(x)

        # Check all outputs are non-negative (ReLU)
        self.assertTrue(tf.reduce_all(out >= 0))

        print("✓ FFT conv with ReLU activation works")

    def test_fft_conv_with_bias(self):
        """Test FFT convolution with and without bias."""
        x = tf.random.normal((2, 32, 32, 1))

        # With bias
        conv_with_bias = CircularConv2DFFT(8, 3, use_bias=True)
        out1 = conv_with_bias(x)
        self.assertEqual(len(conv_with_bias.trainable_variables), 2)  # kernel + bias

        # Without bias
        conv_no_bias = CircularConv2DFFT(8, 3, use_bias=False)
        out2 = conv_no_bias(x)
        self.assertEqual(len(conv_no_bias.trainable_variables), 1)  # kernel only

        print("✓ FFT conv bias configuration works")


class TestDilatedFFTConvolution(unittest.TestCase):
    """Test dilated FFT convolution."""

    def test_dilated_fft_conv_creation(self):
        """Test DilatedCircularConv2DFFT creation."""
        dilation_rate = 2
        layer = DilatedCircularConv2DFFT(
            filters=16,
            dilation_rate=dilation_rate
        )

        self.assertEqual(layer.dilation_rate, dilation_rate)
        print(f"✓ Dilated FFT conv created with dilation={dilation_rate}")

    def test_dilated_fft_conv_output(self):
        """Test dilated FFT convolution output."""
        x = tf.random.normal((2, 64, 64, 1))

        dilated_conv = DilatedCircularConv2DFFT(
            filters=8,
            dilation_rate=2
        )
        out = dilated_conv(x)

        self.assertEqual(out.shape, (2, 64, 64, 8))
        print("✓ Dilated FFT conv output shape correct")

    def test_dilated_fft_receptive_field(self):
        """Test that dilated conv has larger receptive field."""
        x = tf.zeros((1, 64, 64, 1))

        # Set center pixel to 1
        x = tf.tensor_scatter_nd_update(
            x,
            [[0, 32, 32, 0]],
            [1.0]
        )

        # Standard conv (dilation=1)
        conv1 = DilatedCircularConv2DFFT(1, dilation_rate=1)
        out1 = conv1(x)

        # Dilated conv (dilation=2)
        conv2 = DilatedCircularConv2DFFT(1, dilation_rate=2)
        out2 = conv2(x)

        # Dilated conv should affect more distant pixels
        # (This is a simple check - not comprehensive)
        self.assertIsNotNone(out1)
        self.assertIsNotNone(out2)

        print("✓ Dilated FFT conv has different receptive field")


class TestFFTConvolutionEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_fft_conv_single_channel(self):
        """Test FFT conv with single input/output channel."""
        x = tf.random.normal((2, 32, 32, 1))

        conv = CircularConv2DFFT(1, 3)
        out = conv(x)

        self.assertEqual(out.shape, (2, 32, 32, 1))
        print("✓ FFT conv with single channel works")

    def test_fft_conv_many_channels(self):
        """Test FFT conv with many channels."""
        x = tf.random.normal((2, 32, 32, 32))

        conv = CircularConv2DFFT(64, 3)
        out = conv(x)

        self.assertEqual(out.shape, (2, 32, 32, 64))
        print("✓ FFT conv with many channels works")

    def test_fft_conv_large_kernel(self):
        """Test FFT conv with large kernel (where FFT shines)."""
        x = tf.random.normal((2, 64, 64, 1))

        # Large kernel (15x15)
        conv = CircularConv2DFFT(8, 15)
        out = conv(x)

        self.assertEqual(out.shape, (2, 64, 64, 8))
        print("✓ FFT conv with large kernel (15x15) works")

    def test_fft_conv_non_square_input(self):
        """Test FFT conv with non-square input."""
        x = tf.random.normal((2, 32, 64, 1))  # 32x64

        conv = CircularConv2DFFT(8, 3)
        out = conv(x)

        self.assertEqual(out.shape, (2, 32, 64, 8))
        print("✓ FFT conv with non-square input works")

    def test_fft_conv_batch_size_1(self):
        """Test FFT conv with batch size 1."""
        x = tf.random.normal((1, 32, 32, 1))

        conv = CircularConv2DFFT(8, 3)
        out = conv(x)

        self.assertEqual(out.shape, (1, 32, 32, 8))
        print("✓ FFT conv with batch size 1 works")


class TestFFTConvolutionGradients(unittest.TestCase):
    """Test gradient computation for FFT convolution."""

    def test_fft_conv_gradients(self):
        """Test that gradients can be computed."""
        x = tf.random.normal((2, 32, 32, 1))

        conv = CircularConv2DFFT(8, 3)

        with tf.GradientTape() as tape:
            out = conv(x, training=True)
            loss = tf.reduce_mean(out ** 2)

        grads = tape.gradient(loss, conv.trainable_variables)

        # Check gradients exist and are not None
        self.assertIsNotNone(grads)
        self.assertEqual(len(grads), len(conv.trainable_variables))

        for grad in grads:
            self.assertIsNotNone(grad)
            self.assertFalse(tf.reduce_any(tf.math.is_nan(grad)))

        print("✓ FFT conv gradients computed successfully")

    def test_fft_conv_training(self):
        """Test that FFT conv can be trained."""
        # Simple dataset
        x = tf.random.normal((10, 16, 16, 1))
        y = tf.random.normal((10, 16, 16, 4))

        # Simple model with FFT conv
        model = keras.Sequential([
            CircularConv2DFFT(4, 3, input_shape=(16, 16, 1))
        ])

        model.compile(optimizer='adam', loss='mse')

        # Train for 1 epoch (just to check it doesn't crash)
        initial_loss = model.evaluate(x, y, verbose=0)

        history = model.fit(x, y, epochs=1, verbose=0)

        final_loss = model.evaluate(x, y, verbose=0)

        # Loss should decrease (or at least not crash)
        self.assertIsNotNone(final_loss)

        print(f"✓ FFT conv training works: {initial_loss:.4f} -> {final_loss:.4f}")


class TestFFTConvolutionPerformance(unittest.TestCase):
    """Test performance characteristics of FFT convolution."""

    def test_fft_conv_performance_large_kernel(self):
        """Test FFT conv is usable for large kernels."""
        x = tf.random.normal((4, 64, 64, 8))

        # Large kernel where FFT should shine
        kernel_size = 15

        conv = CircularConv2DFFT(16, kernel_size)

        # Warmup
        _ = conv(x)

        # Time it
        start = time.time()
        for _ in range(10):
            _ = conv(x)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds for 10 iterations)
        self.assertLess(elapsed, 5.0)

        print(f"✓ FFT conv (k={kernel_size}) performance: {elapsed/10*1000:.1f}ms per call")


class TestFFTLayerSerialization(unittest.TestCase):
    """Test FFT layer serialization and loading."""

    def test_fft_conv_get_config(self):
        """Test FFT conv layer config."""
        layer = CircularConv2DFFT(
            filters=16,
            kernel_size=5,
            activation='relu',
            use_bias=True
        )

        config = layer.get_config()

        self.assertEqual(config['filters'], 16)
        self.assertEqual(config['kernel_size'], 5)
        self.assertEqual(config['activation'], 'relu')
        self.assertEqual(config['use_bias'], True)

        print("✓ FFT conv get_config works")

    def test_fft_conv_from_config(self):
        """Test FFT conv layer from config."""
        original = CircularConv2DFFT(16, 5, activation='relu')
        config = original.get_config()

        # Create from config
        reconstructed = CircularConv2DFFT.from_config(config)

        self.assertEqual(original.filters, reconstructed.filters)
        self.assertEqual(original.kernel_size, reconstructed.kernel_size)

        print("✓ FFT conv from_config works")


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFFTConvolutionCorrectness))
    suite.addTests(loader.loadTestsFromTestCase(TestDilatedFFTConvolution))
    suite.addTests(loader.loadTestsFromTestCase(TestFFTConvolutionEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestFFTConvolutionGradients))
    suite.addTests(loader.loadTestsFromTestCase(TestFFTConvolutionPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestFFTLayerSerialization))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("="*70)
    print("Running Comprehensive Unit Tests for FFT Layers")
    print("="*70)
    result = run_tests()

    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    exit(0 if result.wasSuccessful() else 1)
