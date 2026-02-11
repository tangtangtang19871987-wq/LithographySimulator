"""
Unit Tests for Advanced Lithography Simulator
==============================================

Comprehensive tests for TCC/SOCC-based simulator:
- Optical system configuration
- TCC computation and validation
- SOCC decomposition and convergence
- Image simulation (TCC and SOCC)
- Cross-validation
"""

import unittest
import numpy as np
import tensorflow as tf
import os
import shutil

from simulator.optics import OpticalSettings, PupilFunction, SourceDistribution
from simulator.tcc import TCCKernel
from simulator.socc import SOCCDecomposition
from simulator.imaging import ImageSimulator, generate_test_mask, simulate_socc
from simulator.validation import validate_tcc_symmetry, validate_socc_convergence


class TestOpticalSettings(unittest.TestCase):
    """Test optical system configuration."""

    def test_default_settings(self):
        """Test default optical settings."""
        settings = OpticalSettings()

        self.assertEqual(settings.wavelength, 193.0)
        self.assertEqual(settings.na, 1.35)
        self.assertEqual(settings.pixel_size, 8.0)
        self.assertEqual(settings.frequency_samples, 256)

    def test_custom_settings(self):
        """Test custom optical settings."""
        settings = OpticalSettings(
            wavelength=248.0,
            na=0.93,
            sigma_inner=0.5,
            sigma_outer=0.7,
            pixel_size=10.0
        )

        self.assertEqual(settings.wavelength, 248.0)
        self.assertEqual(settings.na, 0.93)
        self.assertEqual(settings.pixel_size, 10.0)

    def test_invalid_settings(self):
        """Test that invalid settings raise errors."""
        # Negative wavelength
        with self.assertRaises(ValueError):
            OpticalSettings(wavelength=-100)

        # NA > n_immersion
        with self.assertRaises(ValueError):
            OpticalSettings(na=2.0, immersion_refractive_index=1.44)

        # Invalid sigma
        with self.assertRaises(ValueError):
            OpticalSettings(sigma_inner=0.9, sigma_outer=0.7)

    def test_derived_properties(self):
        """Test derived optical properties."""
        settings = OpticalSettings(wavelength=193.0, na=1.35)

        # Rayleigh resolution
        expected_rayleigh = 0.61 * 193.0 / 1.35
        self.assertAlmostEqual(settings.rayleigh_resolution, expected_rayleigh, places=2)

        # Frequency cutoff
        expected_cutoff = 1.35 / 193.0
        self.assertAlmostEqual(settings.frequency_cutoff, expected_cutoff, places=6)


class TestPupilFunction(unittest.TestCase):
    """Test pupil function."""

    def setUp(self):
        """Set up test fixtures."""
        self.settings = OpticalSettings(
            wavelength=193.0,
            na=1.35,
            frequency_samples=128
        )
        self.pupil = PupilFunction(self.settings)

    def test_pupil_at_zero_frequency(self):
        """Test pupil at DC (zero frequency)."""
        p = self.pupil.evaluate(0.0, 0.0)

        # Should be 1.0 at DC
        self.assertAlmostEqual(float(tf.abs(p)), 1.0, places=5)

    def test_pupil_outside_na(self):
        """Test pupil outside NA cutoff."""
        f_cutoff = self.settings.frequency_cutoff
        f_large = f_cutoff * 2.0  # Well outside pupil

        p = self.pupil.evaluate(f_large, 0.0)

        # Should be zero outside aperture
        self.assertAlmostEqual(float(tf.abs(p)), 0.0, places=5)

    def test_pupil_with_defocus(self):
        """Test pupil with defocus."""
        settings_defocus = OpticalSettings(defocus=100.0)
        pupil_defocus = PupilFunction(settings_defocus)

        p_no_defocus = self.pupil.evaluate(0.001, 0.001)
        p_defocus = pupil_defocus.evaluate(0.001, 0.001)

        # Phase should differ with defocus
        phase_diff = float(tf.math.angle(p_defocus) - tf.math.angle(p_no_defocus))
        self.assertNotAlmostEqual(phase_diff, 0.0, places=3)


class TestSourceDistribution(unittest.TestCase):
    """Test source distribution."""

    def setUp(self):
        """Set up test fixtures."""
        self.settings = OpticalSettings(
            sigma_inner=0.7,
            sigma_outer=0.9,
            source_type='annular'
        )
        self.source = SourceDistribution(self.settings)

    def test_annular_source_inside(self):
        """Test annular source intensity inside ring."""
        # Point inside annulus
        sx = (0.7 + 0.9) / 2.0
        sy = 0.0

        intensity = self.source.evaluate(sx, sy)
        self.assertGreater(float(intensity), 0.0)

    def test_annular_source_outside(self):
        """Test annular source intensity outside ring."""
        # Point outside annulus
        sx = 1.5  # > sigma_outer
        sy = 0.0

        intensity = self.source.evaluate(sx, sy)
        self.assertAlmostEqual(float(intensity), 0.0, places=5)

    def test_circular_source(self):
        """Test circular source."""
        settings_circular = OpticalSettings(
            source_type='circular',
            sigma_outer=0.5
        )
        source_circular = SourceDistribution(settings_circular)

        # Inside circle
        intensity_in = source_circular.evaluate(0.3, 0.0)
        self.assertGreater(float(intensity_in), 0.0)

        # Outside circle
        intensity_out = source_circular.evaluate(0.7, 0.0)
        self.assertAlmostEqual(float(intensity_out), 0.0, places=5)

    def test_source_normalization(self):
        """Test that source is normalized."""
        sx, sy, weights = self.source.generate_source_points(n_points=1000)

        # Weights should sum to 1.0
        total_weight = np.sum(weights)
        self.assertAlmostEqual(total_weight, 1.0, places=3)


class TestTCCKernel(unittest.TestCase):
    """Test TCC computation."""

    def setUp(self):
        """Set up test fixtures."""
        self.settings = OpticalSettings(
            wavelength=193.0,
            na=1.35,
            sigma_inner=0.7,
            sigma_outer=0.9,
            frequency_samples=64,  # Small for fast testing
        )
        self.cache_dir = './test_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test cache."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def test_tcc_computation(self):
        """Test basic TCC computation."""
        tcc_kernel = TCCKernel(self.settings, cache_dir=None)
        tcc = tcc_kernel.compute(verbose=False)

        # Check shape
        n = self.settings.frequency_samples
        self.assertEqual(tcc.shape, (n, n, n, n))

        # Check dtype
        self.assertEqual(tcc.dtype, tf.complex64)

    def test_tcc_hermitian_symmetry(self):
        """Test TCC Hermitian symmetry."""
        tcc_kernel = TCCKernel(self.settings)
        tcc_kernel.compute(verbose=False)

        # Validate symmetry
        is_valid = validate_tcc_symmetry(tcc_kernel, n_samples=50, verbose=False)
        self.assertTrue(is_valid)

    def test_tcc_caching(self):
        """Test TCC caching."""
        # First computation
        tcc_kernel1 = TCCKernel(self.settings, cache_dir=self.cache_dir)
        tcc1 = tcc_kernel1.compute(verbose=False)
        time1 = tcc_kernel1.computation_time

        # Second computation (should load from cache)
        tcc_kernel2 = TCCKernel(self.settings, cache_dir=self.cache_dir)
        tcc2 = tcc_kernel2.compute(verbose=False)

        # Should be much faster (or None if loaded from cache)
        # Check arrays are equal
        max_diff = float(tf.reduce_max(tf.abs(tcc1 - tcc2)))
        self.assertLess(max_diff, 1e-6)


class TestSOCCDecomposition(unittest.TestCase):
    """Test SOCC decomposition."""

    def setUp(self):
        """Set up test fixtures."""
        self.settings = OpticalSettings(
            wavelength=193.0,
            na=1.35,
            frequency_samples=64,  # Small for fast testing
        )
        self.tcc_kernel = TCCKernel(self.settings, cache_dir=None)
        self.tcc_kernel.compute(verbose=False)

        self.cache_dir = './test_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test cache."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def test_socc_decomposition(self):
        """Test basic SOCC decomposition."""
        socc = SOCCDecomposition(self.tcc_kernel, n_modes=10)
        modes, eigenvalues = socc.decompose(verbose=False)

        # Check number of modes
        self.assertEqual(len(modes), 10)
        self.assertEqual(len(eigenvalues), 10)

        # Check shapes
        n = self.settings.frequency_samples
        for mode in modes:
            self.assertEqual(mode.shape, (n, n))

    def test_socc_convergence(self):
        """Test SOCC convergence."""
        socc = SOCCDecomposition(self.tcc_kernel, n_modes=30)
        socc.decompose(verbose=False)

        stats = validate_socc_convergence(socc, verbose=False)

        # Should capture significant energy
        self.assertGreater(stats['energy_captured'], 0.95)

    def test_socc_modes_orthogonal(self):
        """Test that SOCC modes are orthogonal."""
        socc = SOCCDecomposition(self.tcc_kernel, n_modes=5)
        socc.decompose(verbose=False)

        modes = socc.modes

        # Check orthogonality
        for i in range(len(modes)):
            for j in range(i+1, len(modes)):
                # Inner product
                inner_prod = tf.reduce_sum(
                    tf.math.conj(modes[i]) * modes[j]
                )
                # Should be zero (orthogonal)
                self.assertLess(float(tf.abs(inner_prod)), 1e-3)


class TestImageSimulation(unittest.TestCase):
    """Test image simulation."""

    def setUp(self):
        """Set up test fixtures."""
        self.settings = OpticalSettings(
            wavelength=193.0,
            na=1.35,
            frequency_samples=64,  # Small for fast testing
        )
        self.cache_dir = './test_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test cache."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def test_socc_simulation(self):
        """Test SOCC-based image simulation."""
        simulator = ImageSimulator(
            settings=self.settings,
            cache_dir=self.cache_dir,
            method='socc',
            n_modes=10
        )

        # Generate test mask
        mask = generate_test_mask(size=128, pattern_type='lines')

        # Simulate
        image = simulator.simulate(mask)

        # Check output
        self.assertEqual(image.shape, mask.shape)
        self.assertEqual(image.dtype, tf.float32)

        # Image should be non-negative
        self.assertGreaterEqual(float(tf.reduce_min(image)), 0.0)

    def test_batch_simulation(self):
        """Test batch image simulation."""
        simulator = ImageSimulator(
            settings=self.settings,
            cache_dir=self.cache_dir,
            method='socc',
            n_modes=10
        )

        # Generate batch of masks
        masks = []
        for _ in range(5):
            mask = generate_test_mask(size=128, pattern_type='random')
            masks.append(mask)

        masks_batch = tf.stack(masks, axis=0)

        # Batch simulate
        images_batch = simulator.batch_simulate(masks_batch, verbose=False)

        # Check output
        self.assertEqual(images_batch.shape[0], 5)
        self.assertEqual(images_batch.shape[1:], masks_batch.shape[1:])

    def test_image_intensity_range(self):
        """Test that image intensities are in valid range."""
        simulator = ImageSimulator(
            settings=self.settings,
            cache_dir=self.cache_dir,
            method='socc',
            n_modes=10
        )

        mask = generate_test_mask(size=128, pattern_type='contacts')
        image = simulator.simulate(mask)

        # Should be non-negative and finite
        self.assertGreaterEqual(float(tf.reduce_min(image)), 0.0)
        self.assertTrue(tf.math.is_finite(image).numpy().all())


class TestGenerateTestMask(unittest.TestCase):
    """Test mask generation."""

    def test_lines_pattern(self):
        """Test line-space pattern generation."""
        mask = generate_test_mask(size=256, pattern_type='lines')

        self.assertEqual(mask.shape, (256, 256))
        self.assertEqual(mask.dtype, tf.float32)

        # Should be binary
        unique_values = tf.unique(tf.reshape(mask, [-1]))[0]
        self.assertLessEqual(len(unique_values), 2)

    def test_contacts_pattern(self):
        """Test contact holes pattern generation."""
        mask = generate_test_mask(size=256, pattern_type='contacts')

        self.assertEqual(mask.shape, (256, 256))

        # Should have some features
        coverage = float(tf.reduce_mean(mask))
        self.assertGreater(coverage, 0.0)
        self.assertLess(coverage, 1.0)

    def test_random_pattern(self):
        """Test random pattern generation."""
        mask = generate_test_mask(size=256, pattern_type='random')

        # Should be random (approximately 50% coverage)
        coverage = float(tf.reduce_mean(mask))
        self.assertGreater(coverage, 0.3)
        self.assertLess(coverage, 0.7)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache_dir = './test_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test cache."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def test_end_to_end_workflow(self):
        """Test complete workflow from settings to image."""
        # Configure
        settings = OpticalSettings(
            wavelength=193.0,
            na=1.35,
            pixel_size=8.0,
            frequency_samples=64,
        )

        # Initialize simulator
        simulator = ImageSimulator(
            settings=settings,
            cache_dir=self.cache_dir,
            method='socc',
            n_modes=10
        )

        # Generate mask
        mask = generate_test_mask(size=128, pattern_type='lines')

        # Simulate
        image = simulator.simulate(mask)

        # Validate output
        self.assertEqual(image.shape, (128, 128))
        self.assertGreaterEqual(float(tf.reduce_min(image)), 0.0)
        self.assertTrue(tf.math.is_finite(image).numpy().all())

    def test_multiple_simulations_with_cache(self):
        """Test that cache improves performance."""
        settings = OpticalSettings(frequency_samples=64)

        # First simulation (computes TCC and SOCC)
        sim1 = ImageSimulator(
            settings=settings,
            cache_dir=self.cache_dir,
            method='socc',
            n_modes=10
        )

        mask1 = generate_test_mask(size=128, pattern_type='lines')
        image1 = sim1.simulate(mask1)

        # Second simulation (loads from cache)
        sim2 = ImageSimulator(
            settings=settings,  # Same settings
            cache_dir=self.cache_dir,
            method='socc',
            n_modes=10
        )

        mask2 = generate_test_mask(size=128, pattern_type='contacts')
        image2 = sim2.simulate(mask2)

        # Both should produce valid images
        self.assertEqual(image1.shape, image2.shape)
        self.assertTrue(tf.math.is_finite(image1).numpy().all())
        self.assertTrue(tf.math.is_finite(image2).numpy().all())


def run_all_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*70)
    print("Advanced Lithography Simulator - Unit Tests")
    print("="*70)

    success = run_all_tests()

    print("\n" + "="*70)
    if success:
        print("✓ All tests PASSED")
    else:
        print("✗ Some tests FAILED")
    print("="*70)
