"""
Comprehensive unit tests for data_pipeline.py

Tests:
- Dataset generation
- Mask generators
- Simulation context
- Dataset saving/loading
- TF dataset creation
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import tempfile
import shutil
import numpy as np
import tensorflow as tf

# Import modules to test
from data_pipeline import (
    make_vertical_lines,
    make_horizontal_lines,
    make_contact_holes,
    make_l_shape,
    make_random_rectangles,
    generate_random_mask,
    SimulationContext,
    generate_dataset,
    save_dataset,
    load_dataset,
    make_tf_dataset,
)


class TestMaskGenerators(unittest.TestCase):
    """Test individual mask generation functions."""

    def test_vertical_lines(self):
        """Test vertical lines mask generation."""
        mask = make_vertical_lines(n=64, num_lines=3, line_width=2)

        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(mask.dtype, np.float32)
        self.assertGreaterEqual(mask.min(), 0.0)
        self.assertLessEqual(mask.max(), 1.0)

        # Should have some 1s (lines) and some 0s (background)
        self.assertGreater(np.sum(mask), 0)
        self.assertLess(np.sum(mask), mask.size)

        print(f"✓ Vertical lines: shape={mask.shape}, coverage={np.mean(mask):.2%}")

    def test_horizontal_lines(self):
        """Test horizontal lines mask generation."""
        mask = make_horizontal_lines(n=64, num_lines=4, line_width=3)

        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(mask.dtype, np.float32)
        self.assertIn(1.0, mask)
        self.assertIn(0.0, mask)

        print(f"✓ Horizontal lines: coverage={np.mean(mask):.2%}")

    def test_contact_holes(self):
        """Test contact holes mask generation."""
        mask = make_contact_holes(n=64, num_holes=5, hole_size=4)

        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(mask.dtype, np.float32)

        # Should have isolated regions of 1s
        self.assertGreater(np.sum(mask), 0)

        print(f"✓ Contact holes: coverage={np.mean(mask):.2%}")

    def test_l_shape(self):
        """Test L-shape mask generation."""
        mask = make_l_shape(n=64)

        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(mask.dtype, np.float32)

        # Should have L-shaped pattern
        self.assertGreater(np.sum(mask), 0)
        self.assertLess(np.sum(mask), mask.size * 0.5)

        print(f"✓ L-shape: coverage={np.mean(mask):.2%}")

    def test_random_rectangles(self):
        """Test random rectangles mask generation."""
        mask = make_random_rectangles(n=64, num_rects=6)

        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(mask.dtype, np.float32)

        # Should have some filled regions
        self.assertGreater(np.sum(mask), 0)

        print(f"✓ Random rectangles: coverage={np.mean(mask):.2%}")

    def test_generate_random_mask(self):
        """Test random mask generation (picks random generator)."""
        # Generate multiple masks to test randomness
        masks = [generate_random_mask(n=64) for _ in range(10)]

        for mask in masks:
            self.assertEqual(mask.shape, (64, 64))
            self.assertEqual(mask.dtype, np.float32)

        # Check they're not all identical (randomness)
        unique_coverages = set(np.mean(mask) for mask in masks)
        self.assertGreater(len(unique_coverages), 1)

        print(f"✓ Random mask: generated {len(masks)} unique masks")

    def test_mask_generator_reproducibility(self):
        """Test mask generation is reproducible with same seed."""
        np.random.seed(42)
        mask1 = make_vertical_lines(n=64, num_lines=3)

        np.random.seed(42)
        mask2 = make_vertical_lines(n=64, num_lines=3)

        np.testing.assert_array_equal(mask1, mask2)

        print("✓ Mask generation is reproducible")


class TestSimulationContext(unittest.TestCase):
    """Test simulation context and lithography simulation."""

    def test_simulation_context_creation(self):
        """Test SimulationContext creation."""
        ctx = SimulationContext(
            pixel_number=32,
            pixel_size=25,
            wavelength=193.0,
            sigma_in=0.4,
            sigma_out=0.8,
            na=0.7
        )

        self.assertEqual(ctx.pixel_number, 32)
        self.assertEqual(ctx.pixel_size, 25)
        self.assertIsNotNone(ctx.light_source)
        self.assertIsNotNone(ctx.pupil_function)

        print("✓ SimulationContext created")

    def test_simulation_single_mask(self):
        """Test simulating a single mask."""
        ctx = SimulationContext(pixel_number=32, pixel_size=25)

        # Create simple mask
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[12:20, 12:20] = 1.0  # Square in center

        # Simulate
        aerial = ctx.simulate(mask)

        self.assertEqual(aerial.shape, (32, 32))
        self.assertEqual(aerial.dtype, np.float32)

        # Aerial should be normalized to [0, 1]
        self.assertGreaterEqual(aerial.min(), 0.0)
        self.assertLessEqual(aerial.max(), 1.0)

        # Aerial image should have some variation
        self.assertGreater(aerial.std(), 0.01)

        print(f"✓ Simulation: aerial range=[{aerial.min():.3f}, {aerial.max():.3f}]")

    def test_simulation_annular_vs_quasar(self):
        """Test different illumination sources."""
        # Annular
        ctx_annular = SimulationContext(
            pixel_number=32,
            source_type='annular'
        )

        # Quasar
        ctx_quasar = SimulationContext(
            pixel_number=32,
            source_type='quasar'
        )

        mask = np.ones((32, 32), dtype=np.float32)

        aerial_annular = ctx_annular.simulate(mask)
        aerial_quasar = ctx_quasar.simulate(mask)

        # Both should produce valid results
        self.assertEqual(aerial_annular.shape, (32, 32))
        self.assertEqual(aerial_quasar.shape, (32, 32))

        # Results should be different
        self.assertFalse(np.allclose(aerial_annular, aerial_quasar))

        print("✓ Different illumination sources work")


class TestDatasetGeneration(unittest.TestCase):
    """Test dataset generation."""

    def test_generate_dataset_small(self):
        """Test generating small dataset."""
        num_samples = 5
        masks, aerials = generate_dataset(
            num_samples=num_samples,
            pixel_number=32,
            seed=42,
            verbose=False
        )

        # Check shapes
        self.assertEqual(masks.shape, (num_samples, 32, 32, 1))
        self.assertEqual(aerials.shape, (num_samples, 32, 32, 1))

        # Check dtypes
        self.assertEqual(masks.dtype, np.float32)
        self.assertEqual(aerials.dtype, np.float32)

        # Check value ranges
        self.assertTrue(np.all((masks >= 0) & (masks <= 1)))
        self.assertTrue(np.all((aerials >= 0) & (aerials <= 1)))

        print(f"✓ Generated dataset: {num_samples} samples of shape {masks.shape[1:]}")

    def test_generate_dataset_reproducibility(self):
        """Test dataset generation is reproducible."""
        masks1, aerials1 = generate_dataset(10, seed=42, verbose=False)
        masks2, aerials2 = generate_dataset(10, seed=42, verbose=False)

        np.testing.assert_array_equal(masks1, masks2)
        np.testing.assert_array_equal(aerials1, aerials2)

        print("✓ Dataset generation is reproducible")

    def test_generate_dataset_variety(self):
        """Test generated dataset has variety."""
        masks, aerials = generate_dataset(20, seed=42, verbose=False)

        # Check masks are not all identical
        unique_masks = []
        for i in range(len(masks)):
            is_unique = True
            for j in range(len(unique_masks)):
                if np.array_equal(masks[i], unique_masks[j]):
                    is_unique = False
                    break
            if is_unique:
                unique_masks.append(masks[i])

        self.assertGreater(len(unique_masks), 10)

        print(f"✓ Dataset has variety: {len(unique_masks)} unique patterns")

    def test_generate_dataset_custom_params(self):
        """Test dataset generation with custom parameters."""
        masks, aerials = generate_dataset(
            num_samples=3,
            pixel_number=16,
            pixel_size=20,
            wavelength=200.0,
            sigma_in=0.5,
            sigma_out=0.9,
            na=0.8,
            seed=42,
            verbose=False
        )

        self.assertEqual(masks.shape, (3, 16, 16, 1))
        self.assertEqual(aerials.shape, (3, 16, 16, 1))

        print("✓ Dataset generation with custom parameters works")


class TestDatasetSaveLoad(unittest.TestCase):
    """Test dataset saving and loading."""

    def setUp(self):
        """Create temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_and_load_dataset(self):
        """Test saving and loading dataset."""
        # Generate dataset
        masks_orig, aerials_orig = generate_dataset(5, seed=42, verbose=False)

        # Save
        save_path = os.path.join(self.test_dir, 'test_dataset.npz')
        save_dataset(masks_orig, aerials_orig, save_path)

        # Check file exists
        self.assertTrue(os.path.exists(save_path))

        # Load
        masks_loaded, aerials_loaded = load_dataset(save_path)

        # Check loaded data matches original
        np.testing.assert_array_equal(masks_orig, masks_loaded)
        np.testing.assert_array_equal(aerials_orig, aerials_loaded)

        print("✓ Dataset save/load works correctly")

    def test_save_dataset_file_size(self):
        """Test saved dataset file size is reasonable."""
        masks, aerials = generate_dataset(10, pixel_number=32, verbose=False)

        save_path = os.path.join(self.test_dir, 'test_dataset.npz')
        save_dataset(masks, aerials, save_path)

        file_size = os.path.getsize(save_path)

        # Should be compressed (less than uncompressed size)
        uncompressed_size = masks.nbytes + aerials.nbytes
        self.assertLess(file_size, uncompressed_size)

        print(f"✓ Compressed: {uncompressed_size/1024:.1f}KB -> {file_size/1024:.1f}KB")


class TestTFDataset(unittest.TestCase):
    """Test TensorFlow dataset creation."""

    def test_make_tf_dataset(self):
        """Test creating TF dataset."""
        masks = np.random.rand(20, 32, 32, 1).astype(np.float32)
        aerials = np.random.rand(20, 32, 32, 1).astype(np.float32)

        dataset = make_tf_dataset(masks, aerials, batch_size=4, shuffle=True)

        # Check dataset is iterable
        for batch_masks, batch_aerials in dataset.take(1):
            self.assertEqual(batch_masks.shape[0], 4)
            self.assertEqual(batch_masks.shape[1:], (32, 32, 1))
            self.assertEqual(batch_aerials.shape[1:], (32, 32, 1))

        print("✓ TF dataset created successfully")

    def test_tf_dataset_no_shuffle(self):
        """Test TF dataset without shuffling."""
        masks = np.arange(10 * 8 * 8 * 1).reshape(10, 8, 8, 1).astype(np.float32)
        aerials = masks * 2

        dataset = make_tf_dataset(masks, aerials, batch_size=2, shuffle=False)

        # Get first batch
        for batch_masks, batch_aerials in dataset.take(1):
            # Should be first 2 samples in order
            np.testing.assert_array_almost_equal(
                batch_masks[0].numpy(),
                masks[0]
            )

        print("✓ TF dataset without shuffle preserves order")

    def test_tf_dataset_batching(self):
        """Test TF dataset batching."""
        masks = np.random.rand(17, 16, 16, 1).astype(np.float32)
        aerials = np.random.rand(17, 16, 16, 1).astype(np.float32)

        batch_size = 5
        dataset = make_tf_dataset(masks, aerials, batch_size=batch_size)

        batch_sizes = []
        for batch_masks, _ in dataset:
            batch_sizes.append(int(batch_masks.shape[0]))

        # Should have 3 batches of 5, and 1 batch of 2
        self.assertEqual(sum(batch_sizes), 17)
        self.assertEqual(batch_sizes[:-1], [5, 5, 5])
        self.assertEqual(batch_sizes[-1], 2)

        print(f"✓ TF dataset batching: {batch_sizes}")


class TestDataPipelineIntegration(unittest.TestCase):
    """Test integration of full data pipeline."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline: generate -> save -> load -> tf.data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate
            masks, aerials = generate_dataset(10, seed=42, verbose=False)

            # Save
            save_path = os.path.join(tmpdir, 'dataset.npz')
            save_dataset(masks, aerials, save_path)

            # Load
            masks_loaded, aerials_loaded = load_dataset(save_path)

            # Create TF dataset
            dataset = make_tf_dataset(
                masks_loaded, aerials_loaded,
                batch_size=2, shuffle=True
            )

            # Iterate
            count = 0
            for batch_x, batch_y in dataset:
                count += 1
                self.assertEqual(batch_x.shape[1:], (64, 64, 1))

            self.assertEqual(count, 5)  # 10 samples / 2 batch_size

            print("✓ End-to-end pipeline works")


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMaskGenerators))
    suite.addTests(loader.loadTestsFromTestCase(TestSimulationContext))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetSaveLoad))
    suite.addTests(loader.loadTestsFromTestCase(TestTFDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPipelineIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("="*70)
    print("Running Comprehensive Unit Tests for data_pipeline.py")
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
