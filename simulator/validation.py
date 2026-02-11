"""
Validation and Cross-Checking
==============================

Functions for validating simulator accuracy:
- Cross-check TCC vs SOCC
- Compare with existing Abbe method
- Analytical test cases
- Numerical accuracy metrics
"""

import numpy as np
import tensorflow as tf
import sys
import os
from typing import Dict, Optional, Tuple

# Add parent directory to path to import existing simulator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .imaging import simulate_tcc, simulate_socc, compare_tcc_socc
from .tcc import TCCKernel
from .socc import SOCCDecomposition
from .optics import OpticalSettings


def compare_methods(
    mask: tf.Tensor,
    settings: OpticalSettings,
    methods: list = ['tcc', 'socc', 'abbe'],
    cache_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Compare multiple simulation methods on the same mask.

    Args:
        mask: Test mask pattern
        settings: Optical settings
        methods: List of methods to compare
        cache_dir: Cache directory for TCC/SOCC
        verbose: Print comparison results

    Returns:
        Dictionary with results and comparison metrics
    """
    results = {}
    images = {}
    times = {}

    # TCC method
    if 'tcc' in methods:
        if verbose:
            print("\n=== TCC Method ===")
        tcc_kernel = TCCKernel(settings, cache_dir=cache_dir)
        tcc_kernel.compute(verbose=verbose)

        import time
        start = time.time()
        img_tcc = simulate_tcc(mask, tcc_kernel)
        times['tcc'] = time.time() - start
        images['tcc'] = img_tcc

        if verbose:
            print(f"TCC computation time: {times['tcc']:.3f} seconds")

    # SOCC method
    if 'socc' in methods:
        if verbose:
            print("\n=== SOCC Method ===")

        # Use existing TCC if available
        if 'tcc' in methods:
            socc_decomp = SOCCDecomposition(tcc_kernel, n_modes=30, cache_dir=cache_dir)
        else:
            tcc_kernel = TCCKernel(settings, cache_dir=cache_dir)
            tcc_kernel.compute(verbose=verbose)
            socc_decomp = SOCCDecomposition(tcc_kernel, n_modes=30, cache_dir=cache_dir)

        socc_decomp.decompose(verbose=verbose)

        import time
        start = time.time()
        img_socc = simulate_socc(mask, socc_decomp)
        times['socc'] = time.time() - start
        images['socc'] = img_socc

        if verbose:
            print(f"SOCC computation time: {times['socc']:.3f} seconds")

    # Abbe method (existing implementation)
    if 'abbe' in methods:
        if verbose:
            print("\n=== Abbe Method (Reference) ===")

        try:
            # Import existing Abbe simulator
            from litho_sim_tf import simulate as abbe_simulate

            import time
            start = time.time()

            # Convert settings to Abbe parameters
            img_abbe = _simulate_with_abbe(mask, settings, abbe_simulate)
            times['abbe'] = time.time() - start
            images['abbe'] = img_abbe

            if verbose:
                print(f"Abbe computation time: {times['abbe']:.3f} seconds")

        except ImportError:
            if verbose:
                print("Warning: Abbe simulator (litho_sim_tf) not available")
            methods.remove('abbe')

    # Compute comparison metrics
    comparison = {}

    if 'tcc' in methods and 'socc' in methods:
        comparison['tcc_vs_socc'] = _compute_difference_metrics(
            images['tcc'], images['socc'], 'TCC', 'SOCC', verbose=verbose
        )

    if 'tcc' in methods and 'abbe' in methods:
        comparison['tcc_vs_abbe'] = _compute_difference_metrics(
            images['tcc'], images['abbe'], 'TCC', 'Abbe', verbose=verbose
        )

    if 'socc' in methods and 'abbe' in methods:
        comparison['socc_vs_abbe'] = _compute_difference_metrics(
            images['socc'], images['abbe'], 'SOCC', 'Abbe', verbose=verbose
        )

    results['images'] = images
    results['times'] = times
    results['comparison'] = comparison

    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        for method in methods:
            print(f"{method.upper():8s}: {times[method]:6.3f} seconds")

        if 'tcc' in methods and 'socc' in methods:
            speedup = times['tcc'] / (times['socc'] + 1e-10)
            print(f"\nSOCC Speedup: {speedup:.1f}×")

    return results


def _simulate_with_abbe(
    mask: tf.Tensor,
    settings: OpticalSettings,
    abbe_simulate_func
) -> tf.Tensor:
    """
    Run Abbe simulator with settings from OpticalSettings.

    Args:
        mask: Mask pattern
        settings: Optical settings
        abbe_simulate_func: Abbe simulation function

    Returns:
        Aerial image from Abbe method
    """
    # Convert mask to numpy
    mask_np = mask.numpy()

    # Call Abbe simulator
    # Note: This is a simplified interface - actual Abbe may have different API
    try:
        img_abbe = abbe_simulate_func(
            mask_np,
            wavelength=settings.wavelength,
            na=settings.na,
            sigma=settings.sigma_outer,  # Use outer sigma
            defocus=settings.defocus
        )
    except TypeError:
        # Try simpler call
        img_abbe = abbe_simulate_func(mask_np)

    return tf.constant(img_abbe, dtype=tf.float32)


def _compute_difference_metrics(
    img1: tf.Tensor,
    img2: tf.Tensor,
    name1: str,
    name2: str,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute difference metrics between two images.

    Args:
        img1, img2: Images to compare
        name1, name2: Names for display
        verbose: Print results

    Returns:
        Dictionary of metrics
    """
    # Ensure same shape
    if img1.shape != img2.shape:
        # Resize if needed
        size = min(img1.shape[0], img2.shape[0])
        img1 = tf.image.resize(img1[..., None], [size, size])[:, :, 0]
        img2 = tf.image.resize(img2[..., None], [size, size])[:, :, 0]

    # Compute differences
    diff = img1 - img2

    mse = float(tf.reduce_mean(diff ** 2))
    mae = float(tf.reduce_mean(tf.abs(diff)))
    max_error = float(tf.reduce_max(tf.abs(diff)))

    # Normalize by image range
    img_range = float(tf.reduce_max(img1) - tf.reduce_min(img1))
    relative_mse = mse / (img_range ** 2 + 1e-10)
    relative_mae = mae / (img_range + 1e-10)
    relative_max = max_error / (img_range + 1e-10)

    # SSIM (Structural Similarity Index)
    try:
        # Reshape for tf.image.ssim
        img1_4d = img1[None, :, :, None]
        img2_4d = img2[None, :, :, None]
        ssim = float(tf.image.ssim(img1_4d, img2_4d, max_val=float(tf.reduce_max(img1))))
    except:
        ssim = None

    metrics = {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'relative_mse': relative_mse,
        'relative_mae': relative_mae,
        'relative_max_error': relative_max,
        'ssim': ssim,
    }

    if verbose:
        print(f"\n{name1} vs {name2}:")
        print(f"  MSE:               {mse:.2e}")
        print(f"  MAE:               {mae:.2e}")
        print(f"  Max error:         {max_error:.2e}")
        print(f"  Relative MSE:      {relative_mse:.2e}")
        print(f"  Relative MAE:      {relative_mae * 100:.2f}%")
        print(f"  Relative max:      {relative_max * 100:.2f}%")
        if ssim is not None:
            print(f"  SSIM:              {ssim:.4f}")

    return metrics


def validate_accuracy(
    settings: OpticalSettings,
    test_cases: list = ['lines', 'contacts'],
    cache_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Validate simulator accuracy on standard test patterns.

    Args:
        settings: Optical settings
        test_cases: List of test pattern types
        cache_dir: Cache directory
        verbose: Print results

    Returns:
        Validation results dictionary
    """
    from .imaging import generate_test_mask

    results = {}

    for test_case in test_cases:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Test Case: {test_case.upper()}")
            print(f"{'='*60}")

        # Generate test mask
        mask = generate_test_mask(size=512, pattern_type=test_case)

        # Compare methods
        case_results = compare_methods(
            mask, settings,
            methods=['tcc', 'socc'],  # Only TCC and SOCC for now
            cache_dir=cache_dir,
            verbose=verbose
        )

        results[test_case] = case_results

    return results


def validate_tcc_symmetry(
    tcc_kernel: TCCKernel,
    n_samples: int = 100,
    verbose: bool = True
) -> bool:
    """
    Validate TCC Hermitian symmetry.

    TCC(f1, f2) should equal TCC*(f2, f1)

    Args:
        tcc_kernel: TCC kernel to validate
        n_samples: Number of random samples
        verbose: Print results

    Returns:
        True if validation passes
    """
    if verbose:
        print(f"\nValidating TCC Hermitian symmetry ({n_samples} samples)...")

    stats = tcc_kernel.validate_symmetry(n_samples=n_samples)

    if verbose:
        print(f"  Max error:  {stats['max_error']:.2e}")
        print(f"  Mean error: {stats['mean_error']:.2e}")
        print(f"  Status:     {'PASS' if stats['passed'] else 'FAIL'}")

    return stats['passed']


def validate_socc_convergence(
    socc_decomp: SOCCDecomposition,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Validate SOCC mode convergence.

    Checks energy distribution and convergence rate.

    Args:
        socc_decomp: SOCC decomposition
        verbose: Print results

    Returns:
        Convergence statistics
    """
    if verbose:
        print(f"\nValidating SOCC convergence...")

    stats = socc_decomp.analyze_convergence()

    if verbose:
        print(f"  Total modes:       {stats['total_singular_values']}")
        print(f"  Modes retained:    {stats['modes_retained']}")
        print(f"  Energy captured:   {stats['energy_captured'] * 100:.2f}%")
        print(f"  Modes for 95%:     {stats['modes_for_95_percent']}")
        print(f"  Modes for 99%:     {stats['modes_for_99_percent']}")
        print(f"  Modes for 99.9%:   {stats['modes_for_99.9_percent']}")

    return stats


def run_all_validations(
    settings: Optional[OpticalSettings] = None,
    cache_dir: str = './cache',
    verbose: bool = True
) -> Dict[str, any]:
    """
    Run comprehensive validation suite.

    Args:
        settings: Optical settings (None = use defaults)
        cache_dir: Cache directory
        verbose: Print detailed results

    Returns:
        Complete validation results
    """
    if settings is None:
        settings = OpticalSettings()  # Use defaults

    if verbose:
        print("="*60)
        print("COMPREHENSIVE VALIDATION SUITE")
        print("="*60)
        print(settings.summary())

    results = {}

    # 1. Validate TCC symmetry
    if verbose:
        print("\n1. TCC Symmetry Validation")
        print("-" * 60)

    tcc_kernel = TCCKernel(settings, cache_dir=cache_dir)
    tcc_kernel.compute(verbose=verbose)

    results['tcc_symmetry'] = validate_tcc_symmetry(tcc_kernel, verbose=verbose)

    # 2. Validate SOCC convergence
    if verbose:
        print("\n2. SOCC Convergence Validation")
        print("-" * 60)

    socc_decomp = SOCCDecomposition(tcc_kernel, n_modes=50, cache_dir=cache_dir)
    socc_decomp.decompose(verbose=verbose)

    results['socc_convergence'] = validate_socc_convergence(socc_decomp, verbose=verbose)

    # 3. Validate accuracy on test patterns
    if verbose:
        print("\n3. Accuracy Validation on Test Patterns")
        print("-" * 60)

    results['test_patterns'] = validate_accuracy(
        settings,
        test_cases=['lines', 'contacts'],
        cache_dir=cache_dir,
        verbose=verbose
    )

    # 4. Summary
    if verbose:
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"TCC Symmetry:  {'PASS' if results['tcc_symmetry'] else 'FAIL'}")
        conv_stats = results['socc_convergence']
        print(f"SOCC Energy:   {conv_stats['energy_captured'] * 100:.1f}% with {conv_stats['modes_retained']} modes")
        print("="*60)

    return results
