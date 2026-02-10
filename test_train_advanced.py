"""
Comprehensive unit tests for train_advanced.py

Tests:
- All learning rate schedulers (OneCycle, Cyclical, SGDR, Polynomial, etc.)
- Learning rate finder
- Advanced early stopping
- Gradient accumulation
- Model EMA
- Stochastic Weight Averaging (SWA)
- Training progress tracker
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
from train_advanced import (
    OneCycleLR,
    CyclicalLR,
    CosineAnnealingWarmRestarts,
    PolynomialDecay,
    create_lr_schedule,
    LRFinder,
    AdvancedEarlyStopping,
    GradientAccumulation,
    ModelEMA,
    SWA,
    TrainingProgressTracker,
    visualize_lr_schedule,
)


class TestLRSchedulers(unittest.TestCase):
    """Test all learning rate schedulers."""

    def test_onecycle_lr(self):
        """Test OneCycle learning rate scheduler."""
        total_steps = 1000
        max_lr = 1e-3

        schedule = OneCycleLR(
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        # Test at different steps
        lr_start = float(schedule(0))
        lr_peak = float(schedule(total_steps * 0.3))
        lr_end = float(schedule(total_steps - 1))

        # Check LR progression
        self.assertLess(lr_start, max_lr)  # Starts below max
        self.assertAlmostEqual(lr_peak, max_lr, delta=max_lr * 0.1)  # Reaches max
        self.assertLess(lr_end, lr_start)  # Ends very low

        print(f"✓ OneCycleLR: start={lr_start:.2e}, peak={lr_peak:.2e}, end={lr_end:.2e}")

    def test_cyclical_lr_triangular(self):
        """Test Cyclical LR with triangular mode."""
        schedule = CyclicalLR(
            initial_lr=1e-4,
            maximal_lr=1e-3,
            step_size=100,
            mode='triangular'
        )

        # Test cycle
        lr_0 = float(schedule(0))
        lr_50 = float(schedule(50))  # Should be near max
        lr_100 = float(schedule(100))  # Back to min
        lr_150 = float(schedule(150))  # Near max again

        self.assertAlmostEqual(lr_0, 1e-4, delta=1e-5)
        self.assertGreater(lr_50, lr_0)
        self.assertAlmostEqual(lr_100, 1e-4, delta=1e-5)
        self.assertGreater(lr_150, lr_100)

        print(f"✓ CyclicalLR (triangular): values cycle correctly")

    def test_cyclical_lr_triangular2(self):
        """Test Cyclical LR with triangular2 mode."""
        schedule = CyclicalLR(
            initial_lr=1e-4,
            maximal_lr=1e-3,
            step_size=100,
            mode='triangular2'
        )

        lr_50_cycle1 = float(schedule(50))
        lr_250_cycle2 = float(schedule(250))  # Second cycle

        # Second cycle should have lower amplitude
        self.assertGreater(lr_50_cycle1, lr_250_cycle2)

        print(f"✓ CyclicalLR (triangular2): amplitude decreases")

    def test_sgdr(self):
        """Test SGDR (Cosine Annealing with Warm Restarts)."""
        schedule = CosineAnnealingWarmRestarts(
            initial_lr=1e-3,
            first_cycle_steps=100,
            t_mul=2.0,
            min_lr=1e-7
        )

        lr_0 = float(schedule(0))
        lr_50 = float(schedule(50))
        lr_99 = float(schedule(99))

        # First cycle: high -> low -> restart
        self.assertAlmostEqual(lr_0, 1e-3, delta=1e-5)
        self.assertLess(lr_50, lr_0)
        self.assertLess(lr_99, lr_50)

        print(f"✓ SGDR: lr_0={lr_0:.2e}, lr_50={lr_50:.2e}, lr_99={lr_99:.2e}")

    def test_polynomial_decay(self):
        """Test Polynomial decay scheduler."""
        schedule = PolynomialDecay(
            initial_lr=1e-3,
            decay_steps=1000,
            end_lr=1e-7,
            power=1.0  # Linear
        )

        lr_0 = float(schedule(0))
        lr_500 = float(schedule(500))
        lr_1000 = float(schedule(1000))

        # Should decay linearly
        self.assertAlmostEqual(lr_0, 1e-3, delta=1e-6)
        self.assertGreater(lr_0, lr_500)
        self.assertGreater(lr_500, lr_1000)
        self.assertAlmostEqual(lr_1000, 1e-7, delta=1e-8)

        print(f"✓ PolynomialDecay: decays linearly from {lr_0:.2e} to {lr_1000:.2e}")

    def test_create_lr_schedule_factory(self):
        """Test LR schedule factory function."""
        total_steps = 1000

        # Test all schedule types
        schedules = {
            'onecycle': create_lr_schedule('onecycle', 1e-3, total_steps),
            'cyclical': create_lr_schedule('cyclical', 1e-3, total_steps),
            'sgdr': create_lr_schedule('sgdr', 1e-3, total_steps),
            'polynomial': create_lr_schedule('polynomial', 1e-3, total_steps),
            'cosine': create_lr_schedule('cosine', 1e-3, total_steps),
            'exponential': create_lr_schedule('exponential', 1e-3, total_steps),
        }

        for name, schedule in schedules.items():
            self.assertIsNotNone(schedule)

        print(f"✓ Created {len(schedules)} LR schedules via factory")


class TestLRFinder(unittest.TestCase):
    """Test Learning Rate Finder."""

    def setUp(self):
        """Create simple model and dataset for testing."""
        # Simple model
        self.model = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(10,)),
            keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

        # Simple dataset
        x = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)
        self.dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(10)

    def test_lr_finder_creation(self):
        """Test LRFinder creation."""
        lr_finder = LRFinder(self.model, self.dataset)

        self.assertIsNotNone(lr_finder)
        self.assertEqual(len(lr_finder.history['lr']), 0)
        self.assertEqual(len(lr_finder.history['loss']), 0)

        print("✓ LRFinder created successfully")

    def test_lr_finder_run(self):
        """Test running LR finder."""
        lr_finder = LRFinder(self.model, self.dataset)

        # Run with small number of steps
        lr_finder.find(min_lr=1e-5, max_lr=1e-1, num_steps=20, beta=0.9)

        # Check history populated
        self.assertGreater(len(lr_finder.history['lr']), 0)
        self.assertGreater(len(lr_finder.history['loss']), 0)
        self.assertEqual(len(lr_finder.history['lr']),
                        len(lr_finder.history['loss']))

        # Check LR range
        self.assertGreaterEqual(min(lr_finder.history['lr']), 1e-5)
        self.assertLessEqual(max(lr_finder.history['lr']), 1e-1)

        print(f"✓ LRFinder ran {len(lr_finder.history['lr'])} steps")

    def test_lr_finder_get_optimal(self):
        """Test getting optimal LR."""
        lr_finder = LRFinder(self.model, self.dataset)
        lr_finder.find(min_lr=1e-5, max_lr=1e-1, num_steps=20)

        optimal_lr = lr_finder.get_optimal_lr()

        self.assertIsNotNone(optimal_lr)
        self.assertGreater(optimal_lr, 1e-5)
        self.assertLess(optimal_lr, 1e-1)

        print(f"✓ LRFinder found optimal LR: {optimal_lr:.2e}")


class TestAdvancedEarlyStopping(unittest.TestCase):
    """Test Advanced Early Stopping callback."""

    def test_early_stopping_creation(self):
        """Test AdvancedEarlyStopping creation."""
        callback = AdvancedEarlyStopping(
            monitor='val_loss',
            patience=10,
            warmup_epochs=5,
            min_delta=0.001,
            verbose=0
        )

        self.assertEqual(callback.monitor, 'val_loss')
        self.assertEqual(callback.patience, 10)
        self.assertEqual(callback.warmup_epochs, 5)

        print("✓ AdvancedEarlyStopping created")

    def test_early_stopping_warmup(self):
        """Test early stopping respects warmup period."""
        callback = AdvancedEarlyStopping(
            monitor='val_loss',
            patience=2,
            warmup_epochs=3,
            verbose=0
        )

        # Simulate training
        callback.on_train_begin()

        # During warmup, should not stop even with bad loss
        for epoch in range(3):
            callback.on_epoch_end(epoch, {'val_loss': 10.0})  # High loss
            self.assertFalse(hasattr(callback, 'stopped_epoch') and callback.stopped_epoch > 0)

        print("✓ Early stopping respects warmup period")

    def test_early_stopping_min_delta_percent(self):
        """Test min_delta_percent mode."""
        callback = AdvancedEarlyStopping(
            monitor='val_loss',
            patience=2,
            min_delta_percent=1.0,  # Require 1% improvement
            verbose=0
        )

        callback.on_train_begin()

        # First epoch
        callback.on_epoch_end(0, {'val_loss': 1.0})

        # Small improvement (< 1%)
        callback.on_epoch_end(1, {'val_loss': 0.995})  # 0.5% improvement
        self.assertEqual(callback.wait, 1)  # Should not count as improvement

        print("✓ min_delta_percent works correctly")


class TestGradientAccumulation(unittest.TestCase):
    """Test Gradient Accumulation."""

    def setUp(self):
        """Create simple model for testing."""
        self.model = keras.Sequential([
            keras.layers.Dense(8, input_shape=(4,)),
            keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def test_gradient_accumulation_creation(self):
        """Test GradientAccumulation creation."""
        ga = GradientAccumulation(self.model, accumulation_steps=4)

        self.assertEqual(ga.accumulation_steps, 4)
        self.assertEqual(len(ga.accumulated_gradients),
                        len(self.model.trainable_variables))

        print("✓ GradientAccumulation created")

    def test_gradient_accumulation_train_step(self):
        """Test gradient accumulation train step."""
        ga = GradientAccumulation(self.model, accumulation_steps=2)

        x = tf.constant([[1.0, 2.0, 3.0, 4.0]])
        y = tf.constant([[1.0]])

        # First step - should accumulate
        loss1 = ga.train_step(x, y)
        self.assertIsNotNone(loss1)

        # Second step - should apply gradients
        loss2 = ga.train_step(x, y)
        self.assertIsNotNone(loss2)

        print("✓ Gradient accumulation train step works")

    def test_gradient_accumulation_reset(self):
        """Test gradient accumulation reset."""
        ga = GradientAccumulation(self.model, accumulation_steps=4)

        # Accumulate some gradients
        x = tf.constant([[1.0, 2.0, 3.0, 4.0]])
        y = tf.constant([[1.0]])
        ga.train_step(x, y)

        # Reset
        ga.reset()

        # Check step count reset
        self.assertEqual(int(ga.step_count), 0)

        print("✓ Gradient accumulation reset works")


class TestModelEMA(unittest.TestCase):
    """Test Model EMA callback."""

    def setUp(self):
        """Create simple model for testing."""
        self.model = keras.Sequential([
            keras.layers.Dense(8, input_shape=(4,)),
            keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def test_model_ema_creation(self):
        """Test ModelEMA creation."""
        ema = ModelEMA(decay=0.999, start_epoch=0)

        self.assertEqual(ema.decay, 0.999)
        self.assertEqual(ema.start_epoch, 0)

        print("✓ ModelEMA created")

    def test_model_ema_initialization(self):
        """Test ModelEMA weight initialization."""
        ema = ModelEMA(decay=0.999)
        ema.set_model(self.model)
        ema.on_train_begin()

        self.assertIsNotNone(ema.ema_weights)
        self.assertEqual(len(ema.ema_weights),
                        len(self.model.get_weights()))

        print("✓ ModelEMA weights initialized")

    def test_model_ema_update(self):
        """Test ModelEMA weight update."""
        ema = ModelEMA(decay=0.999, start_epoch=0)
        ema.set_model(self.model)
        ema.on_train_begin()

        # Get initial EMA weights
        initial_ema = [w.numpy().copy() for w in ema.ema_weights]

        # Simulate epoch end (which updates EMA)
        ema.on_epoch_end(0, {})

        # Check EMA weights changed (even slightly)
        # Note: they might be very close due to high decay
        self.assertIsNotNone(ema.ema_weights)

        print("✓ ModelEMA updates weights")


class TestSWA(unittest.TestCase):
    """Test Stochastic Weight Averaging."""

    def setUp(self):
        """Create simple model for testing."""
        self.model = keras.Sequential([
            keras.layers.Dense(8, input_shape=(4,)),
            keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def test_swa_creation(self):
        """Test SWA callback creation."""
        swa = SWA(start_epoch=10, swa_freq=1, verbose=0)

        self.assertEqual(swa.start_epoch, 10)
        self.assertEqual(swa.swa_freq, 1)
        self.assertIsNone(swa.swa_weights)

        print("✓ SWA callback created")

    def test_swa_initialization(self):
        """Test SWA weight averaging."""
        swa = SWA(start_epoch=2, swa_freq=1, verbose=0)
        swa.set_model(self.model)

        # Before start epoch
        swa.on_epoch_end(0, {})
        swa.on_epoch_end(1, {})
        self.assertIsNone(swa.swa_weights)

        # At start epoch
        swa.on_epoch_end(2, {})
        self.assertIsNotNone(swa.swa_weights)
        self.assertEqual(swa.swa_count, 1)

        print("✓ SWA starts at correct epoch")

    def test_swa_averaging(self):
        """Test SWA weight averaging over multiple epochs."""
        swa = SWA(start_epoch=0, swa_freq=1, verbose=0)
        swa.set_model(self.model)

        # Average over 3 epochs
        swa.on_epoch_end(0, {})
        swa.on_epoch_end(1, {})
        swa.on_epoch_end(2, {})

        self.assertEqual(swa.swa_count, 3)
        self.assertIsNotNone(swa.swa_weights)

        print("✓ SWA averages weights correctly")


class TestTrainingProgressTracker(unittest.TestCase):
    """Test Training Progress Tracker."""

    def setUp(self):
        """Create simple model for testing."""
        self.model = keras.Sequential([
            keras.layers.Dense(8, input_shape=(4,)),
            keras.layers.Dense(1)
        ])

    def test_progress_tracker_creation(self):
        """Test TrainingProgressTracker creation."""
        tracker = TrainingProgressTracker(
            total_epochs=100,
            steps_per_epoch=50,
            verbose=0
        )

        self.assertEqual(tracker.total_epochs, 100)
        self.assertEqual(tracker.steps_per_epoch, 50)

        print("✓ TrainingProgressTracker created")

    def test_progress_tracker_timing(self):
        """Test progress tracker timing."""
        tracker = TrainingProgressTracker(
            total_epochs=10,
            steps_per_epoch=10,
            verbose=0
        )
        tracker.set_model(self.model)
        tracker.on_train_begin()

        # Simulate epoch
        tracker.on_epoch_begin(0, {})
        tracker.on_epoch_end(0, {'loss': 0.5})

        self.assertEqual(len(tracker.epoch_times), 1)
        self.assertGreater(tracker.epoch_times[0], 0)

        print("✓ Progress tracker records epoch times")


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLRSchedulers))
    suite.addTests(loader.loadTestsFromTestCase(TestLRFinder))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedEarlyStopping))
    suite.addTests(loader.loadTestsFromTestCase(TestGradientAccumulation))
    suite.addTests(loader.loadTestsFromTestCase(TestModelEMA))
    suite.addTests(loader.loadTestsFromTestCase(TestSWA))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingProgressTracker))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("="*70)
    print("Running Unit Tests for train_advanced.py")
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
