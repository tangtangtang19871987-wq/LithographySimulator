"""
TCC (Transmission Cross Coefficient) Computation
=================================================

Implements Hopkins formulation for partially coherent imaging.

TCC(f₁, f₂) = ∫∫ P(f₁ - s) P*(f₂ - s) S(s) ds

Where:
- P(f) = Pupil function
- S(s) = Source distribution
- f₁, f₂ = Spatial frequency coordinates

The TCC is precomputed once for a given optical configuration and
can be reused for any mask pattern.

References:
- Hopkins, H.H. (1953). "On the diffraction theory of optical images"
- Yeung, M.S. (1988). "Fast computation of aerial images"
"""

import numpy as np
import tensorflow as tf
import os
import hashlib
import json
from typing import Optional, Tuple, Dict
from .optics import OpticalSettings, PupilFunction, SourceDistribution


class TCCKernel:
    """
    Transmission Cross Coefficient kernel computation and storage.

    The TCC is a 4D complex tensor: TCC[f1x, f1y, f2x, f2y]

    For efficiency:
    - Exploits Hermitian symmetry: TCC(f1, f2) = TCC*(f2, f1)
    - Supports sparse storage for large grids
    - GPU-accelerated computation
    - Caching system for reuse
    """

    def __init__(
        self,
        settings: OpticalSettings,
        cache_dir: Optional[str] = None,
        use_sparse: bool = False
    ):
        """
        Initialize TCC kernel.

        Args:
            settings: Optical system configuration
            cache_dir: Directory for caching precomputed TCC (None = no caching)
            use_sparse: Use sparse tensor representation (for large grids)
        """
        self.settings = settings
        self.cache_dir = cache_dir
        self.use_sparse = use_sparse

        self.pupil = PupilFunction(settings)
        self.source = SourceDistribution(settings)

        # TCC tensor (computed on demand)
        self._tcc = None
        self._frequency_grid = None
        self._f_coords = None

        # Computational statistics
        self.computation_time = None
        self.memory_usage = None

    @property
    def n_freq(self) -> int:
        """Number of frequency samples."""
        return self.settings.frequency_samples

    @property
    def tcc(self) -> tf.Tensor:
        """Get TCC tensor (compute if not cached)."""
        if self._tcc is None:
            raise ValueError("TCC not computed. Call compute() first.")
        return self._tcc

    @property
    def frequency_coords(self) -> np.ndarray:
        """Get frequency coordinate grid."""
        if self._f_coords is None:
            self._f_coords = self._generate_frequency_grid()
        return self._f_coords

    def compute(self, force_recompute: bool = False, verbose: bool = True) -> tf.Tensor:
        """
        Compute TCC tensor.

        Args:
            force_recompute: Recompute even if cached
            verbose: Print progress information

        Returns:
            TCC tensor of shape (n_freq, n_freq, n_freq, n_freq)
        """
        # Check cache first
        if not force_recompute and self.cache_dir is not None:
            cached_tcc = self._load_from_cache()
            if cached_tcc is not None:
                self._tcc = cached_tcc
                if verbose:
                    print("✓ Loaded TCC from cache")
                return self._tcc

        # Compute TCC
        if verbose:
            print(f"Computing TCC kernel ({self.n_freq}×{self.n_freq} frequency grid)...")
            print(f"  Source: {self.settings.source_type}, σ={self.settings.sigma_inner:.2f}-{self.settings.sigma_outer:.2f}")
            print(f"  This may take several minutes...")

        import time
        start_time = time.time()

        if self.use_sparse:
            self._tcc = self._compute_tcc_sparse(verbose=verbose)
        else:
            self._tcc = self._compute_tcc_full(verbose=verbose)

        self.computation_time = time.time() - start_time

        if verbose:
            print(f"✓ TCC computation completed in {self.computation_time:.1f} seconds")
            mem_gb = self._estimate_memory_usage()
            print(f"  Memory usage: {mem_gb:.2f} GB")

        # Cache result
        if self.cache_dir is not None:
            self._save_to_cache()
            if verbose:
                print(f"✓ TCC cached to {self.cache_dir}")

        return self._tcc

    def _compute_tcc_full(self, verbose: bool = True) -> tf.Tensor:
        """
        Compute full TCC tensor.

        Algorithm:
        1. Generate frequency grids
        2. Generate source points
        3. For each (f1, f2) pair:
           - Integrate: ∫∫ P(f1-s) P*(f2-s) S(s) ds
        4. Return 4D TCC tensor

        Time complexity: O(N^4 × N_source)
        Space complexity: O(N^4)
        """
        n = self.n_freq
        f_cutoff = self.settings.frequency_cutoff

        # Frequency grid
        f_coords = np.linspace(-f_cutoff * 1.2, f_cutoff * 1.2, n, dtype=np.float32)
        self._f_coords = f_coords

        # Source sampling points
        sx_points, sy_points, weights = self.source.generate_source_points()
        n_source = len(sx_points)

        if verbose:
            print(f"  Frequency range: ±{f_cutoff:.4f} nm⁻¹")
            print(f"  Source points: {n_source}")

        # Initialize TCC tensor
        tcc_tensor = np.zeros((n, n, n, n), dtype=np.complex64)

        # Compute TCC via source integration
        # Due to memory constraints, compute in batches
        batch_size = min(n, 32)  # Process batch_size frequency pairs at a time

        for i1 in range(0, n, batch_size):
            if verbose and i1 % 64 == 0:
                progress = 100.0 * i1 / n
                print(f"  Progress: {progress:.1f}%", end='\r')

            i1_end = min(i1 + batch_size, n)

            for j1 in range(0, n, batch_size):
                j1_end = min(j1 + batch_size, n)

                # Extract frequency batch
                f1x = f_coords[i1:i1_end, None, None, None]
                f1y = f_coords[None, j1:j1_end, None, None]

                for i2 in range(0, n, batch_size):
                    i2_end = min(i2 + batch_size, n)

                    for j2 in range(0, n, batch_size):
                        j2_end = min(j2 + batch_size, n)

                        f2x = f_coords[None, None, i2:i2_end, None]
                        f2y = f_coords[None, None, None, j2:j2_end]

                        # Compute TCC for this batch
                        tcc_batch = self._integrate_tcc(
                            f1x, f1y, f2x, f2y,
                            sx_points, sy_points, weights
                        )

                        # Store
                        tcc_tensor[i1:i1_end, j1:j1_end, i2:i2_end, j2:j2_end] = tcc_batch

        if verbose:
            print(f"  Progress: 100.0%")

        return tf.constant(tcc_tensor, dtype=tf.complex64)

    def _integrate_tcc(
        self,
        f1x: np.ndarray,
        f1y: np.ndarray,
        f2x: np.ndarray,
        f2y: np.ndarray,
        sx_points: np.ndarray,
        sy_points: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Integrate TCC over source:

        TCC(f1, f2) = Σₛ P(f1 - s) P*(f2 - s) S(s) w_s

        Args:
            f1x, f1y: First frequency coordinates (4D grids)
            f2x, f2y: Second frequency coordinates (4D grids)
            sx_points, sy_points: Source points
            weights: Source integration weights

        Returns:
            TCC values for this batch
        """
        n_source = len(sx_points)
        shape = np.broadcast_shapes(f1x.shape, f1y.shape, f2x.shape, f2y.shape)
        tcc_sum = np.zeros(shape, dtype=np.complex64)

        # Integrate over source points
        for k in range(n_source):
            sx = sx_points[k]
            sy = sy_points[k]
            w = weights[k]

            # P(f1 - s)
            p1 = self.pupil.evaluate(f1x - sx, f1y - sy).numpy()

            # P*(f2 - s)
            p2_conj = tf.math.conj(self.pupil.evaluate(f2x - sx, f2y - sy)).numpy()

            # Accumulate
            tcc_sum += w * p1 * p2_conj

        return tcc_sum

    def _compute_tcc_sparse(self, verbose: bool = True) -> tf.Tensor:
        """
        Compute TCC with sparse representation.

        Only stores TCC(f1, f2) where both f1 and f2 are inside pupil support.
        This dramatically reduces memory for large grids.

        Returns:
            Sparse TCC tensor
        """
        # For now, use dense representation
        # True sparse implementation would use tf.sparse.SparseTensor
        # This is a placeholder for future optimization

        if verbose:
            print("  Note: Sparse TCC not yet implemented, using dense")

        return self._compute_tcc_full(verbose=verbose)

    def _generate_frequency_grid(self) -> np.ndarray:
        """Generate frequency coordinate array."""
        f_cutoff = self.settings.frequency_cutoff
        f_coords = np.linspace(-f_cutoff * 1.2, f_cutoff * 1.2, self.n_freq, dtype=np.float32)
        return f_coords

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in GB."""
        if self._tcc is None:
            return 0.0

        # Complex64: 8 bytes per element
        n_elements = self.n_freq ** 4
        bytes_used = n_elements * 8
        gb_used = bytes_used / (1024 ** 3)

        self.memory_usage = gb_used
        return gb_used

    def _get_cache_filename(self) -> str:
        """Generate cache filename from optical settings hash."""
        # Create unique hash from optical parameters
        settings_dict = {
            'wavelength': self.settings.wavelength,
            'na': self.settings.na,
            'sigma_inner': self.settings.sigma_inner,
            'sigma_outer': self.settings.sigma_outer,
            'source_type': self.settings.source_type,
            'n_freq': self.n_freq,
            'defocus': self.settings.defocus,
        }

        settings_str = json.dumps(settings_dict, sort_keys=True)
        settings_hash = hashlib.md5(settings_str.encode()).hexdigest()[:12]

        filename = f"tcc_{settings_hash}_n{self.n_freq}.npz"
        return filename

    def _save_to_cache(self) -> None:
        """Save TCC to cache file."""
        if self.cache_dir is None or self._tcc is None:
            return

        os.makedirs(self.cache_dir, exist_ok=True)

        cache_path = os.path.join(self.cache_dir, self._get_cache_filename())

        # Convert to numpy for saving
        tcc_np = self._tcc.numpy()

        # Save with metadata
        np.savez_compressed(
            cache_path,
            tcc_real=np.real(tcc_np),
            tcc_imag=np.imag(tcc_np),
            f_coords=self._f_coords,
            wavelength=self.settings.wavelength,
            na=self.settings.na,
            sigma_inner=self.settings.sigma_inner,
            sigma_outer=self.settings.sigma_outer,
            computation_time=self.computation_time,
        )

    def _load_from_cache(self) -> Optional[tf.Tensor]:
        """Load TCC from cache if available."""
        if self.cache_dir is None:
            return None

        cache_path = os.path.join(self.cache_dir, self._get_cache_filename())

        if not os.path.exists(cache_path):
            return None

        try:
            data = np.load(cache_path)

            # Reconstruct complex tensor
            tcc_real = data['tcc_real']
            tcc_imag = data['tcc_imag']
            tcc_np = tcc_real + 1j * tcc_imag

            # Load metadata
            self._f_coords = data['f_coords']
            if 'computation_time' in data:
                self.computation_time = float(data['computation_time'])

            return tf.constant(tcc_np, dtype=tf.complex64)

        except Exception as e:
            print(f"Warning: Failed to load TCC cache: {e}")
            return None

    def get_tcc_value(self, f1x: float, f1y: float, f2x: float, f2y: float) -> complex:
        """
        Get TCC value at specific frequency coordinates.

        Args:
            f1x, f1y: First frequency
            f2x, f2y: Second frequency

        Returns:
            TCC(f1, f2) as complex number
        """
        if self._tcc is None:
            raise ValueError("TCC not computed")

        # Find nearest grid indices
        i1 = np.argmin(np.abs(self._f_coords - f1x))
        j1 = np.argmin(np.abs(self._f_coords - f1y))
        i2 = np.argmin(np.abs(self._f_coords - f2x))
        j2 = np.argmin(np.abs(self._f_coords - f2y))

        return complex(self._tcc[i1, j1, i2, j2].numpy())

    def validate_symmetry(self, n_samples: int = 100, tolerance: float = 1e-5) -> Dict[str, float]:
        """
        Validate Hermitian symmetry: TCC(f1, f2) = TCC*(f2, f1).

        Args:
            n_samples: Number of random samples to check
            tolerance: Maximum allowed error

        Returns:
            Dictionary with validation statistics
        """
        if self._tcc is None:
            raise ValueError("TCC not computed")

        n = self.n_freq
        errors = []

        for _ in range(n_samples):
            # Random indices
            i1, j1 = np.random.randint(0, n, size=2)
            i2, j2 = np.random.randint(0, n, size=2)

            # Get values
            val_12 = self._tcc[i1, j1, i2, j2]
            val_21 = self._tcc[i2, j2, i1, j1]

            # Check conjugate symmetry
            error = tf.abs(val_12 - tf.math.conj(val_21))
            errors.append(float(error))

        max_error = max(errors)
        mean_error = np.mean(errors)

        return {
            'max_error': max_error,
            'mean_error': mean_error,
            'passed': max_error < tolerance,
            'n_samples': n_samples,
        }

    def summary(self) -> str:
        """Return summary of TCC kernel."""
        if self._tcc is None:
            status = "Not computed"
        else:
            status = f"Computed ({self.n_freq}×{self.n_freq}×{self.n_freq}×{self.n_freq})"

        mem_gb = self.memory_usage if self.memory_usage is not None else 0.0
        comp_time = self.computation_time if self.computation_time is not None else 0.0

        return f"""
TCC Kernel Summary:
==================
Status:            {status}
Frequency samples: {self.n_freq}
Memory usage:      {mem_gb:.2f} GB
Computation time:  {comp_time:.1f} seconds
Sparse storage:    {self.use_sparse}
Cache directory:   {self.cache_dir if self.cache_dir else 'None'}

Optical Settings:
-----------------
{self.settings.summary()}
"""
