"""
Example Usage of Advanced Lithography Simulator
================================================

This script demonstrates how to use the TCC/SOCC-based simulator
for generating lithography training data.

Run this script to:
1. Configure optical settings
2. Precompute TCC and SOCC kernels
3. Generate aerial images from masks
4. Compare TCC vs SOCC performance
5. Validate accuracy
"""

import numpy as np
import tensorflow as tf
import os

# Import simulator modules
from optics import OpticalSettings
from tcc import TCCKernel
from socc import SOCCDecomposition
from imaging import ImageSimulator, generate_test_mask, compare_tcc_socc
from validation import run_all_validations


def example_basic_usage():
    """
    Example 1: Basic usage with SOCC (fast method)
    """
    print("="*70)
    print("EXAMPLE 1: Basic Usage with SOCC")
    print("="*70)

    # Step 1: Configure optical settings
    settings = OpticalSettings(
        wavelength=193.0,      # nm (ArF laser)
        na=1.35,               # Numerical aperture (immersion)
        sigma_inner=0.7,       # Annular illumination
        sigma_outer=0.9,
        pixel_size=8.0,        # nm (as requested)
        frequency_samples=128,  # Reduced for faster demo
    )

    print(settings.summary())

    # Step 2: Initialize simulator with SOCC
    cache_dir = './simulator_cache'
    os.makedirs(cache_dir, exist_ok=True)

    simulator = ImageSimulator(
        settings=settings,
        cache_dir=cache_dir,
        method='socc',         # Fast SOCC method
        n_modes=30             # 30 modes for ~99% accuracy
    )

    # Step 3: Generate test mask
    mask = generate_test_mask(
        size=512,
        pattern_type='lines',
        feature_size=40,
        pitch=80
    )

    print(f"\nMask shape: {mask.shape}")
    print(f"Mask range: [{tf.reduce_min(mask):.2f}, {tf.reduce_max(mask):.2f}]")

    # Step 4: Simulate aerial image
    print("\nSimulating aerial image...")
    aerial_image = simulator.simulate(mask)

    print(f"Aerial image shape: {aerial_image.shape}")
    print(f"Aerial image range: [{tf.reduce_min(aerial_image):.4f}, {tf.reduce_max(aerial_image):.4f}]")

    print("\n✓ Basic simulation complete!")

    return simulator, mask, aerial_image


def example_batch_generation():
    """
    Example 2: Batch generation for training dataset
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Dataset Generation")
    print("="*70)

    # Configure settings
    settings = OpticalSettings(
        wavelength=193.0,
        na=1.35,
        sigma_inner=0.7,
        sigma_outer=0.9,
        pixel_size=8.0,
        frequency_samples=128,
    )

    # Initialize simulator
    simulator = ImageSimulator(
        settings=settings,
        cache_dir='./simulator_cache',
        method='socc',
        n_modes=30
    )

    # Generate batch of random masks
    print("\nGenerating batch of 10 random masks...")
    n_masks = 10
    masks = []

    for i in range(n_masks):
        mask = generate_test_mask(
            size=512,
            pattern_type='random',
            feature_size=30
        )
        masks.append(mask)

    masks_batch = tf.stack(masks, axis=0)
    print(f"Masks batch shape: {masks_batch.shape}")

    # Batch simulation
    print("\nSimulating batch...")
    images_batch = simulator.batch_simulate(masks_batch, verbose=True)

    print(f"Images batch shape: {images_batch.shape}")
    print("\n✓ Batch generation complete!")

    return masks_batch, images_batch


def example_tcc_vs_socc():
    """
    Example 3: Compare TCC and SOCC methods
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: TCC vs SOCC Comparison")
    print("="*70)

    # Configure settings
    settings = OpticalSettings(
        wavelength=193.0,
        na=1.35,
        sigma_inner=0.7,
        sigma_outer=0.9,
        pixel_size=8.0,
        frequency_samples=128,
    )

    # Precompute TCC
    print("\nPrecomputing TCC...")
    tcc_kernel = TCCKernel(settings, cache_dir='./simulator_cache')
    tcc_kernel.compute(verbose=True)

    # Decompose SOCC
    print("\nDecomposing SOCC...")
    socc_decomp = SOCCDecomposition(tcc_kernel, n_modes=30, cache_dir='./simulator_cache')
    socc_decomp.decompose(verbose=True)

    # Generate test mask
    mask = generate_test_mask(size=512, pattern_type='contacts')

    # Compare methods
    print("\nComparing TCC and SOCC...")
    comparison = compare_tcc_socc(mask, tcc_kernel, socc_decomp, verbose=True)

    print("\n✓ Comparison complete!")

    return comparison


def example_custom_settings():
    """
    Example 4: Custom optical settings
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Optical Settings")
    print("="*70)

    # Customize optical parameters
    custom_settings = OpticalSettings(
        # Different wavelength (KrF laser)
        wavelength=248.0,      # nm
        na=0.93,               # Dry lithography
        immersion_refractive_index=1.0,

        # Circular source instead of annular
        source_type='circular',
        sigma_inner=0.0,
        sigma_outer=0.5,

        # Higher resolution sampling
        pixel_size=5.0,        # nm
        frequency_samples=256,  # More samples

        # Add some defocus
        defocus=100.0,         # nm

        # Enable polarization effects
        enable_polarization=True,
        polarization_type='TE',
    )

    print(custom_settings.summary())

    # Simulate with custom settings
    simulator = ImageSimulator(
        settings=custom_settings,
        cache_dir='./simulator_cache',
        method='socc',
        n_modes=30
    )

    mask = generate_test_mask(size=256, pattern_type='lines')
    image = simulator.simulate(mask)

    print(f"\nSimulated with custom settings:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image range: [{tf.reduce_min(image):.4f}, {tf.reduce_max(image):.4f}]")

    print("\n✓ Custom settings simulation complete!")

    return custom_settings, simulator


def example_validation():
    """
    Example 5: Run validation suite
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Validation Suite")
    print("="*70)

    # Default settings
    settings = OpticalSettings(
        wavelength=193.0,
        na=1.35,
        sigma_inner=0.7,
        sigma_outer=0.9,
        pixel_size=8.0,
        frequency_samples=128,
    )

    # Run comprehensive validation
    results = run_all_validations(
        settings=settings,
        cache_dir='./simulator_cache',
        verbose=True
    )

    return results


def example_dataset_generation():
    """
    Example 6: Generate large training dataset
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Large Dataset Generation")
    print("="*70)

    # Configure for production
    settings = OpticalSettings(
        wavelength=193.0,
        na=1.35,
        sigma_inner=0.7,
        sigma_outer=0.9,
        pixel_size=8.0,
        frequency_samples=256,  # Full resolution
    )

    # Initialize simulator
    simulator = ImageSimulator(
        settings=settings,
        cache_dir='./simulator_cache',
        method='socc',
        n_modes=50  # Higher accuracy for dataset
    )

    # Generate large dataset
    n_samples = 100  # Reduce for demo (use 10,000 for real dataset)
    print(f"\nGenerating dataset of {n_samples} samples...")

    all_masks = []
    all_images = []

    # Generate in batches
    batch_size = 10
    for batch_idx in range(0, n_samples, batch_size):
        batch_end = min(batch_idx + batch_size, n_samples)
        batch_n = batch_end - batch_idx

        print(f"\nBatch {batch_idx//batch_size + 1}/{(n_samples-1)//batch_size + 1}")

        # Generate random masks
        masks = []
        for _ in range(batch_n):
            pattern_type = np.random.choice(['lines', 'contacts', 'random'])
            mask = generate_test_mask(size=512, pattern_type=pattern_type)
            masks.append(mask)

        masks_batch = tf.stack(masks, axis=0)

        # Simulate
        images_batch = simulator.batch_simulate(masks_batch, verbose=False)

        all_masks.append(masks_batch)
        all_images.append(images_batch)

        print(f"  Generated {batch_n} samples")

    # Concatenate all batches
    final_masks = tf.concat(all_masks, axis=0)
    final_images = tf.concat(all_images, axis=0)

    print(f"\n✓ Dataset generation complete!")
    print(f"  Final masks shape: {final_masks.shape}")
    print(f"  Final images shape: {final_images.shape}")

    # Save dataset
    output_dir = './training_data'
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_dir, 'litho_dataset.npz'),
        masks=final_masks.numpy(),
        aerials=final_images.numpy(),
        wavelength=settings.wavelength,
        na=settings.na,
        pixel_size=settings.pixel_size,
    )

    print(f"  Saved to {output_dir}/litho_dataset.npz")

    return final_masks, final_images


if __name__ == '__main__':
    """
    Run all examples
    """
    print("\n" + "█"*70)
    print("Advanced Lithography Simulator - Example Usage")
    print("█"*70 + "\n")

    # Example 1: Basic usage
    simulator, mask, image = example_basic_usage()

    # Example 2: Batch generation
    masks_batch, images_batch = example_batch_generation()

    # Example 3: TCC vs SOCC comparison
    comparison = example_tcc_vs_socc()

    # Example 4: Custom settings
    custom_settings, custom_simulator = example_custom_settings()

    # Example 5: Validation
    validation_results = example_validation()

    # Example 6: Large dataset (commented out for quick demo)
    # final_masks, final_images = example_dataset_generation()

    print("\n" + "█"*70)
    print("All examples completed successfully!")
    print("█"*70 + "\n")
