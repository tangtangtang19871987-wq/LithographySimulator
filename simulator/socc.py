"""
SOCC (Sum of Coherent Components) Decomposition
================================================

Decomposes TCC into coherent modes via Singular Value Decomposition.

TCC(f₁, f₂) ≈ Σᵢ₌₁ᴷ λᵢ φᵢ(f₁) φᵢ*(f₂)

Where:
- λᵢ = Singular values (eigenvalues)
- φᵢ(f) = Coherent modes (eigenfunctions)
- K = Number of modes (typically 20-50 for 99% accuracy)

Aerial image via SOCC:
I(x) = Σᵢ₌₁ᴷ λᵢ |IFFT[M(f) φᵢ(f)]|²

This is much faster than full TCC: K coherent propagations instead
of full 4D convolution.

References:
- Flagello et al. (1996). "Theory of high-NA imaging"
- Liu & Zakhor (2002). "Binary and phase shifting masks"
"""

import numpy as np
import tensorflow as tf
import os
import time
from typing import Optional, List, Tuple, Dict, Union
from .tcc import TCCKernel


class SOCCDecomposition:
    """
    Sum of Coherent Components decomposition of TCC.

    Performs SVD of TCC matrix to extract dominant coherent modes.
    Dramatically speeds up image computation while maintaining accuracy.
    """

    def __init__(
        self,
        tcc_kernel: TCCKernel,
        n_modes: int = 30,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize SOCC decomposition.

        Args:
            tcc_kernel: Precomputed TCC kernel
            n_modes: Number of modes to retain (higher = more accurate)
            cache_dir: Directory for caching decomposition
        """
        self.tcc_kernel = tcc_kernel
        self.n_modes_requested = n_modes
        self.cache_dir = cache_dir

        # Decomposition results
        self._modes = None  # List of coherent mode kernels φᵢ(f)
        self._eigenvalues = None  # Singular values λᵢ
        self._n_modes_actual = 0

        # Statistics
        self.decomposition_time = None
        self.cumulative_energy = None

    @property
    def modes(self) -> List[tf.Tensor]:
        """Get coherent mode kernels."""
        if self._modes is None:
            raise ValueError("SOCC not decomposed. Call decompose() first.")
        return self._modes

    @property
    def eigenvalues(self) -> tf.Tensor:
        """Get singular values (eigenvalues)."""
        if self._eigenvalues is None:
            raise ValueError("SOCC not decomposed. Call decompose() first.")
        return self._eigenvalues

    @property
    def n_modes(self) -> int:
        """Number of modes actually retained."""
        return self._n_modes_actual

    def decompose(
        self,
        method: str = 'svd',
        force_recompute: bool = False,
        verbose: bool = True
    ) -> Tuple[List[tf.Tensor], tf.Tensor]:
        """
        Perform SOCC decomposition.

        Args:
            method: Decomposition method ('svd', 'randomized', 'iterative')
            force_recompute: Recompute even if cached
            verbose: Print progress information

        Returns:
            (modes, eigenvalues) tuple
        """
        # Check cache
        if not force_recompute and self.cache_dir is not None:
            cached = self._load_from_cache()
            if cached is not None:
                self._modes, self._eigenvalues = cached
                self._n_modes_actual = len(self._modes)
                if verbose:
                    print(f"✓ Loaded SOCC decomposition from cache ({self.n_modes} modes)")
                return self._modes, self._eigenvalues

        # Ensure TCC is computed
        if self.tcc_kernel.tcc is None:
            if verbose:
                print("Computing TCC kernel first...")
            self.tcc_kernel.compute(verbose=verbose)

        if verbose:
            print(f"Performing SOCC decomposition ({method} method)...")
            print(f"  Target modes: {self.n_modes_requested}")

        start_time = time.time()

        # Perform decomposition based on method
        if method == 'svd':
            modes, eigenvalues = self._decompose_full_svd(verbose=verbose)
        elif method == 'randomized':
            modes, eigenvalues = self._decompose_randomized_svd(verbose=verbose)
        elif method == 'iterative':
            modes, eigenvalues = self._decompose_iterative(verbose=verbose)
        else:
            raise ValueError(f"Unknown decomposition method: {method}")

        self._modes = modes
        self._eigenvalues = eigenvalues
        self._n_modes_actual = len(modes)

        self.decomposition_time = time.time() - start_time

        # Compute energy statistics
        self._compute_energy_statistics()

        if verbose:
            print(f"✓ SOCC decomposition completed in {self.decomposition_time:.1f} seconds")
            print(f"  Modes retained: {self.n_modes}")
            print(f"  Energy captured: {self.cumulative_energy[-1] * 100:.2f}%")

        # Cache result
        if self.cache_dir is not None:
            self._save_to_cache()
            if verbose:
                print(f"✓ SOCC cached")

        return self._modes, self._eigenvalues

    def _decompose_full_svd(self, verbose: bool = True) -> Tuple[List[tf.Tensor], tf.Tensor]:
        """
        Full SVD decomposition.

        Algorithm:
        1. Reshape TCC(f1, f2) -> Matrix A (N² × N²)
        2. Perform SVD: A = U Σ V^H
        3. Extract top k modes: φᵢ = sqrt(σᵢ) * uᵢ
        4. Eigenvalues: λᵢ = σᵢ

        Time complexity: O(N⁶) - very expensive!
        Memory: O(N⁴)
        """
        tcc = self.tcc_kernel.tcc
        n_freq = self.tcc_kernel.n_freq

        if verbose:
            print(f"  Reshaping TCC: ({n_freq}, {n_freq}, {n_freq}, {n_freq}) -> ({n_freq**2}, {n_freq**2})")

        # Reshape to matrix
        # TCC: (n, n, n, n) -> (n*n, n*n)
        tcc_matrix = tf.reshape(tcc, [n_freq * n_freq, n_freq * n_freq])

        if verbose:
            print(f"  Matrix shape: {tcc_matrix.shape}")
            print(f"  Computing SVD (this may take a while)...")

        # SVD decomposition
        # Note: For very large matrices, this can be extremely slow
        s, u, v = tf.linalg.svd(tcc_matrix, full_matrices=False)

        # Keep top k modes
        k = min(self.n_modes_requested, len(s))

        if verbose:
            print(f"  Extracting {k} dominant modes...")

        eigenvalues = s[:k]

        modes = []
        for i in range(k):
            # Mode: sqrt(σᵢ) * reshape(uᵢ, (n, n))
            mode_vector = u[:, i] * tf.sqrt(s[i])
            mode_2d = tf.reshape(mode_vector, [n_freq, n_freq])
            modes.append(mode_2d)

        return modes, eigenvalues

    def _decompose_randomized_svd(self, verbose: bool = True) -> Tuple[List[tf.Tensor], tf.Tensor]:
        """
        Randomized SVD for faster computation.

        Uses randomized algorithms (Halko et al. 2011) to compute
        approximate SVD in O(k²n) time instead of O(n³).

        For now, falls back to full SVD (future optimization).
        """
        if verbose:
            print("  Note: Randomized SVD not yet implemented, using full SVD")

        return self._decompose_full_svd(verbose=verbose)

    def _decompose_iterative(self, verbose: bool = True) -> Tuple[List[tf.Tensor], tf.Tensor]:
        """
        Iterative power method for top modes.

        Computes dominant eigenmodes one at a time via power iteration.
        Good for extracting small number of top modes.

        For now, falls back to full SVD (future optimization).
        """
        if verbose:
            print("  Note: Iterative decomposition not yet implemented, using full SVD")

        return self._decompose_full_svd(verbose=verbose)

    def compute_aerial_image(
        self,
        mask: tf.Tensor,
        return_mode_images: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]]:
        """
        Compute aerial image using SOCC modes.

        I(x) = Σᵢ₌₁ᴷ λᵢ |IFFT[M(f) φᵢ(f)]|²

        This is the fast imaging method!

        Args:
            mask: Binary mask pattern (2D tensor)
            return_mode_images: If True, return individual mode contributions

        Returns:
            Aerial image intensity (or tuple with mode images if requested)
        """
        if self._modes is None:
            raise ValueError("SOCC not decomposed. Call decompose() first.")

        # Compute mask spectrum
        mask_complex = tf.cast(mask, tf.complex64)
        mask_spectrum = tf.signal.fft2d(mask_complex)

        # Initialize image
        image = tf.zeros_like(mask, dtype=tf.float32)

        # Store mode images if requested
        mode_images = [] if return_mode_images else None

        # Sum coherent mode contributions
        for i, (mode, eigenvalue) in enumerate(zip(self._modes, self._eigenvalues)):
            # Coherent propagation for this mode
            # φᵢ(f) * M(f)
            coherent_spectrum = mode * mask_spectrum

            # IFFT to spatial domain
            coherent_field = tf.signal.ifft2d(coherent_spectrum)

            # Intensity contribution
            mode_intensity = tf.abs(coherent_field) ** 2

            # Weight by eigenvalue and add
            weighted_intensity = eigenvalue * mode_intensity
            image += tf.cast(weighted_intensity, tf.float32)

            if return_mode_images:
                mode_images.append(mode_intensity)

        if return_mode_images:
            return image, mode_images
        else:
            return image

    def _compute_energy_statistics(self) -> None:
        """Compute energy distribution across modes."""
        if self._eigenvalues is None:
            return

        # Total energy
        total_energy = tf.reduce_sum(self._eigenvalues)

        # Cumulative energy
        cumulative = tf.cumsum(self._eigenvalues) / total_energy

        self.cumulative_energy = cumulative.numpy()

    def analyze_convergence(self) -> Dict[str, any]:
        """
        Analyze SOCC convergence with number of modes.

        Returns:
            Dictionary with convergence statistics
        """
        if self._eigenvalues is None:
            raise ValueError("SOCC not decomposed")

        # Find number of modes for various accuracy levels
        k_95 = int(np.argmax(self.cumulative_energy >= 0.95)) + 1
        k_99 = int(np.argmax(self.cumulative_energy >= 0.99)) + 1
        k_999 = int(np.argmax(self.cumulative_energy >= 0.999)) + 1

        return {
            'total_singular_values': len(self._eigenvalues),
            'modes_retained': self.n_modes,
            'energy_captured': self.cumulative_energy[-1],
            'modes_for_95_percent': k_95,
            'modes_for_99_percent': k_99,
            'modes_for_99.9_percent': k_999,
            'cumulative_energy': self.cumulative_energy,
            'eigenvalues': self._eigenvalues.numpy(),
        }

    def plot_convergence(self) -> None:
        """
        Plot mode convergence (requires matplotlib).

        Shows:
        - Eigenvalue spectrum
        - Cumulative energy vs. number of modes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return

        if self._eigenvalues is None:
            raise ValueError("SOCC not decomposed")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Eigenvalue spectrum
        ax = axes[0]
        eigenvalues_np = self._eigenvalues.numpy()
        ax.semilogy(eigenvalues_np, 'o-')
        ax.set_xlabel('Mode index')
        ax.set_ylabel('Eigenvalue (log scale)')
        ax.set_title('Eigenvalue Spectrum')
        ax.grid(True, alpha=0.3)

        # Cumulative energy
        ax = axes[1]
        ax.plot(self.cumulative_energy * 100, 'o-')
        ax.axhline(95, color='r', linestyle='--', alpha=0.5, label='95%')
        ax.axhline(99, color='g', linestyle='--', alpha=0.5, label='99%')
        ax.axhline(99.9, color='b', linestyle='--', alpha=0.5, label='99.9%')
        ax.set_xlabel('Number of modes')
        ax.set_ylabel('Cumulative energy (%)')
        ax.set_title('SOCC Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([90, 100])

        plt.tight_layout()
        plt.show()

    def _get_cache_filename(self) -> str:
        """Generate cache filename."""
        tcc_cache = self.tcc_kernel._get_cache_filename()
        # SOCC cache based on TCC cache name
        base_name = tcc_cache.replace('tcc_', 'socc_').replace('.npz', '')
        filename = f"{base_name}_k{self.n_modes_requested}.npz"
        return filename

    def _save_to_cache(self) -> None:
        """Save SOCC decomposition to cache."""
        if self.cache_dir is None or self._modes is None:
            return

        os.makedirs(self.cache_dir, exist_ok=True)

        cache_path = os.path.join(self.cache_dir, self._get_cache_filename())

        # Convert modes to numpy
        modes_list = []
        for mode in self._modes:
            mode_np = mode.numpy()
            modes_list.append({
                'real': np.real(mode_np),
                'imag': np.imag(mode_np),
            })

        eigenvalues_np = self._eigenvalues.numpy()

        # Save
        save_dict = {
            'eigenvalues': eigenvalues_np,
            'n_modes': self.n_modes,
            'decomposition_time': self.decomposition_time,
        }

        # Add modes
        for i, mode_dict in enumerate(modes_list):
            save_dict[f'mode_{i}_real'] = mode_dict['real']
            save_dict[f'mode_{i}_imag'] = mode_dict['imag']

        np.savez_compressed(cache_path, **save_dict)

    def _load_from_cache(self) -> Optional[Tuple[List[tf.Tensor], tf.Tensor]]:
        """Load SOCC decomposition from cache."""
        if self.cache_dir is None:
            return None

        cache_path = os.path.join(self.cache_dir, self._get_cache_filename())

        if not os.path.exists(cache_path):
            return None

        try:
            data = np.load(cache_path)

            # Load eigenvalues
            eigenvalues = tf.constant(data['eigenvalues'], dtype=tf.float32)

            # Load modes
            n_modes = int(data['n_modes'])
            modes = []

            for i in range(n_modes):
                mode_real = data[f'mode_{i}_real']
                mode_imag = data[f'mode_{i}_imag']
                mode_np = mode_real + 1j * mode_imag
                mode = tf.constant(mode_np, dtype=tf.complex64)
                modes.append(mode)

            # Load metadata
            if 'decomposition_time' in data:
                self.decomposition_time = float(data['decomposition_time'])

            return modes, eigenvalues

        except Exception as e:
            print(f"Warning: Failed to load SOCC cache: {e}")
            return None

    def summary(self) -> str:
        """Return summary of SOCC decomposition."""
        if self._modes is None:
            status = "Not decomposed"
            energy = 0.0
        else:
            status = f"Decomposed ({self.n_modes} modes)"
            energy = self.cumulative_energy[-1] * 100 if self.cumulative_energy is not None else 0.0

        decomp_time = self.decomposition_time if self.decomposition_time is not None else 0.0

        return f"""
SOCC Decomposition Summary:
==========================
Status:            {status}
Modes retained:    {self.n_modes} / {self.n_modes_requested} requested
Energy captured:   {energy:.2f}%
Decomposition time: {decomp_time:.1f} seconds
Cache directory:   {self.cache_dir if self.cache_dir else 'None'}

Based on TCC:
-------------
{self.tcc_kernel.summary()}
"""
