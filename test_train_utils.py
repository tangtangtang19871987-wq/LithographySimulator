"""
Comprehensive unit tests for train_utils.py

Tests:
- Distribution strategy creation and fallbacks
- NCCL environment setup
- Mixed precision configuration
- Data augmentation
- Callback creation
- Checkpoint utilities
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import tempfile
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import modules to test
from train_utils import (
    DistributedStrategyConfig,
    create_distribution_strategy,
    setup_nccl_environment,
    setup_mixed_precision,
    LithographyAugmentation,
    create_augmented_dataset,
    create_callbacks,
    find_latest_checkpoint,
    print_training_summary,
)


class TestDistributedStrategy(unittest.TestCase):
    """Test distribution strategy creation and configuration."""

    def test_config_creation(self):
        """Test DistributedStrategyConfig creation."""
        config = DistributedStrategyConfig(
            strategy_type='auto',
            cross_device_ops='nccl',
            enable_nccl_workarounds=True,
            timeout=1800
        )
        self.assertEqual(config.strategy_type, 'auto')
        self.assertEqual(config.cross_device_ops, 'nccl')
        self.assertTrue(config.enable_nccl_workarounds)
        self.assertEqual(config.timeout, 1800)

    def test_strategy_creation_default(self):
        """Test default strategy creation (should work on CPU)."""
        config = DistributedStrategyConfig(strategy_type='none')
        strategy, num_replicas = create_distribution_strategy(config, verbose=False)

        self.assertIsNotNone(strategy)
        self.assertEqual(num_replicas, 1)
        print(f"✓ Default strategy created: {strategy.__class__.__name__}")

    def test_strategy_creation_auto(self):
        """Test auto strategy creation."""
        config = DistributedStrategyConfig(strategy_type='auto')
        strategy, num_replicas = create_distribution_strategy(config, verbose=False)

        self.assertIsNotNone(strategy)
        self.assertGreaterEqual(num_replicas, 1)
        print(f"✓ Auto strategy created with {num_replicas} replica(s)")

    def test_nccl_environment_setup(self):
        """Test NCCL environment variable configuration."""
        # Save original env vars
        original_env = os.environ.copy()

        # Clear NCCL vars
        for key in list(os.environ.keys()):
            if key.startswith('NCCL_'):
                del os.environ[key]

        # Setup NCCL
        setup_nccl_environment(timeout=3600, enable_workarounds=True)

        # Check environment variables
        self.assertIn('NCCL_TIMEOUT', os.environ)
        self.assertEqual(os.environ['NCCL_TIMEOUT'], '3600')
        self.assertIn('NCCL_BLOCKING_WAIT', os.environ)

        # Restore original env
        os.environ.clear()
        os.environ.update(original_env)
        print("✓ NCCL environment setup works correctly")


class TestMixedPrecision(unittest.TestCase):
    """Test mixed precision setup."""

    def test_mixed_precision_disabled(self):
        """Test mixed precision when disabled."""
        policy = setup_mixed_precision(enabled=False, verbose=False)

        self.assertEqual(policy.compute_dtype, tf.float32)
        self.assertEqual(policy.variable_dtype, tf.float32)
        print("✓ Mixed precision disabled correctly")

    def test_mixed_precision_enabled_no_gpu(self):
        """Test mixed precision on CPU (should fallback to float32)."""
        policy = setup_mixed_precision(enabled=True, verbose=False)

        # On CPU, should fallback to float32
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) == 0:
            self.assertEqual(policy.compute_dtype, tf.float32)
            print("✓ Mixed precision fallback to float32 on CPU")
        else:
            # On GPU, should enable float16
            print(f"✓ Mixed precision enabled on {len(gpus)} GPU(s)")


class TestDataAugmentation(unittest.TestCase):
    """Test data augmentation layers."""

    def setUp(self):
        """Create test data."""
        self.batch_size = 4
        self.image_size = 64
        self.test_images = tf.random.uniform(
            (self.batch_size, self.image_size, self.image_size, 1),
            minval=0, maxval=1, dtype=tf.float32
        )

    def test_augmentation_layer_creation(self):
        """Test LithographyAugmentation layer creation."""
        aug_layer = LithographyAugmentation(
            rotation_prob=0.5,
            flip_prob=0.5,
            brightness_delta=0.05,
            seed=42
        )
        self.assertIsInstance(aug_layer, keras.layers.Layer)
        print("✓ Augmentation layer created")

    def test_augmentation_training_mode(self):
        """Test augmentation in training mode."""
        aug_layer = LithographyAugmentation(seed=42)

        # Apply augmentation
        augmented = aug_layer(self.test_images, training=True)

        # Check shape preserved
        self.assertEqual(augmented.shape, self.test_images.shape)

        # Check values in valid range
        self.assertTrue(tf.reduce_all(augmented >= 0.0))
        self.assertTrue(tf.reduce_all(augmented <= 1.0))
        print("✓ Augmentation works in training mode")

    def test_augmentation_inference_mode(self):
        """Test augmentation in inference mode (should be identity)."""
        aug_layer = LithographyAugmentation(seed=42)

        # Apply augmentation in inference mode
        result = aug_layer(self.test_images, training=False)

        # Should be identical to input
        tf.debugging.assert_near(result, self.test_images, rtol=1e-5)
        print("✓ Augmentation is identity in inference mode")

    def test_create_augmented_dataset(self):
        """Test augmented dataset creation."""
        masks = np.random.rand(10, 64, 64, 1).astype(np.float32)
        aerials = np.random.rand(10, 64, 64, 1).astype(np.float32)

        dataset = create_augmented_dataset(
            masks, aerials,
            batch_size=2,
            shuffle=True,
            augmentation_prob=0.7
        )

        # Get one batch
        for batch_x, batch_y in dataset.take(1):
            self.assertEqual(batch_x.shape[0], 2)  # batch size
            self.assertEqual(batch_x.shape[1:], (64, 64, 1))
            self.assertEqual(batch_y.shape[1:], (64, 64, 1))

        print("✓ Augmented dataset created successfully")


class TestCallbacks(unittest.TestCase):
    """Test callback creation utilities."""

    def setUp(self):
        """Create temporary directory for testing."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_create_callbacks_basic(self):
        """Test basic callback creation."""
        from train import StopState
        stop_state = StopState()

        callbacks = create_callbacks(
            run_dir=self.test_dir,
            model_basename='test_model',
            enable_tensorboard=False,
            enable_early_stopping=True,
            patience=10,
            stop_state=stop_state
        )

        self.assertIsInstance(callbacks, list)
        self.assertGreater(len(callbacks), 0)
        print(f"✓ Created {len(callbacks)} callbacks")

    def test_tensorboard_callback(self):
        """Test TensorBoard callback creation."""
        callbacks = create_callbacks(
            run_dir=self.test_dir,
            enable_tensorboard=True,
            enable_early_stopping=False,
        )

        # Check TensorBoard directory created
        tensorboard_dir = os.path.join(self.test_dir, 'tensorboard')
        self.assertTrue(os.path.exists(tensorboard_dir))
        print("✓ TensorBoard callback and directory created")


class TestCheckpointUtilities(unittest.TestCase):
    """Test checkpoint management utilities."""

    def setUp(self):
        """Create temporary directory with test checkpoints."""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.test_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir)

        # Create dummy checkpoint files
        for epoch in [10, 20, 30, 25]:
            filename = f"model_epoch{epoch:04d}.keras"
            path = os.path.join(self.checkpoint_dir, filename)
            with open(path, 'w') as f:
                f.write('dummy')

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_find_latest_checkpoint(self):
        """Test finding latest checkpoint."""
        latest = find_latest_checkpoint(
            self.checkpoint_dir,
            prefix='model_epoch'
        )

        self.assertIsNotNone(latest)
        self.assertIn('epoch0030', latest)  # Should find epoch 30
        print(f"✓ Found latest checkpoint: {os.path.basename(latest)}")

    def test_find_latest_checkpoint_empty_dir(self):
        """Test finding checkpoint in empty directory."""
        empty_dir = os.path.join(self.test_dir, 'empty')
        os.makedirs(empty_dir)

        latest = find_latest_checkpoint(empty_dir)
        self.assertIsNone(latest)
        print("✓ Correctly returns None for empty directory")

    def test_find_latest_checkpoint_nonexistent_dir(self):
        """Test finding checkpoint in non-existent directory."""
        nonexistent = os.path.join(self.test_dir, 'nonexistent')

        latest = find_latest_checkpoint(nonexistent)
        self.assertIsNone(latest)
        print("✓ Correctly returns None for non-existent directory")


class TestTrainingSummary(unittest.TestCase):
    """Test training summary utilities."""

    def test_print_training_summary(self):
        """Test printing training summary (should not crash)."""
        strategy = tf.distribute.get_strategy()

        try:
            print_training_summary(
                strategy=strategy,
                num_replicas=1,
                mixed_precision_enabled=False,
                augmentation_enabled=True,
                total_params=1000000
            )
            success = True
        except Exception as e:
            success = False
            print(f"Error: {e}")

        self.assertTrue(success)
        print("✓ Training summary printed successfully")


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDistributedStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestMixedPrecision))
    suite.addTests(loader.loadTestsFromTestCase(TestDataAugmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestCallbacks))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckpointUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingSummary))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("="*70)
    print("Running Unit Tests for train_utils.py")
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
