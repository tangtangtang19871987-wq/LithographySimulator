"""
Optical System Module
=====================

Defines optical components for lithography simulation:
- Optical settings and parameters
- Pupil function (lens transfer function)
- Source distribution (illumination)

All parameters are configurable with physically meaningful defaults.
"""

import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union


@dataclass
class OpticalSettings:
    """
    Configuration for lithography optical system.

    Default parameters represent typical ArF immersion lithography:
    - Wavelength: 193nm (ArF excimer laser)
    - NA: 1.35 (immersion lithography)
    - Partial coherence: σ_inner=0.7, σ_outer=0.9 (annular illumination)
    - Pixel size: 8nm (suggested for high resolution)

    All parameters are configurable for experimentation.
    """

    # Wavelength and numerical aperture
    wavelength: float = 193.0  # nm, ArF laser
    na: float = 1.35  # Numerical aperture (immersion)
    immersion_refractive_index: float = 1.44  # Water for immersion

    # Source configuration (partial coherence)
    sigma_inner: float = 0.7  # Inner coherence (annular source)
    sigma_outer: float = 0.9  # Outer coherence (annular source)
    source_type: str = 'annular'  # 'annular', 'circular', 'dipole', 'quadrupole', 'custom'

    # Computational parameters
    pixel_size: float = 8.0  # nm, spatial domain sampling
    frequency_samples: int = 256  # Frequency domain grid size

    # Advanced optical features
    defocus: float = 0.0  # nm, defocus distance
    aberrations: Optional[np.ndarray] = None  # Zernike coefficients
    apodization: Optional[str] = None  # 'gaussian', 'linear', None
    central_obscuration: float = 0.0  # Fraction of pupil radius

    # Polarization settings
    enable_polarization: bool = False  # Enable vector imaging
    polarization_type: str = 'TE'  # 'TE', 'TM', 'unpolarized'

    # Source sampling (for integration)
    source_samples: int = 64  # Number of source points for integration

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.wavelength <= 0:
            raise ValueError(f"Wavelength must be positive, got {self.wavelength}")
        if not 0 < self.na <= self.immersion_refractive_index:
            raise ValueError(f"NA must be in (0, n_immersion], got {self.na}")
        if not 0 <= self.sigma_inner <= self.sigma_outer <= 1.0:
            raise ValueError(f"Must have 0 ≤ σ_inner ≤ σ_outer ≤ 1.0")
        if self.pixel_size <= 0:
            raise ValueError(f"Pixel size must be positive, got {self.pixel_size}")
        if self.frequency_samples < 32:
            raise ValueError(f"Frequency samples must be ≥ 32, got {self.frequency_samples}")

    @property
    def rayleigh_resolution(self) -> float:
        """Rayleigh diffraction limit: 0.61λ/NA (nm)."""
        return 0.61 * self.wavelength / self.na

    @property
    def frequency_cutoff(self) -> float:
        """Frequency cutoff: NA/λ (1/nm)."""
        return self.na / self.wavelength

    @property
    def max_frequency(self) -> float:
        """Maximum spatial frequency based on pixel size (Nyquist)."""
        return 1.0 / (2.0 * self.pixel_size)

    def summary(self) -> str:
        """Return human-readable summary of optical settings."""
        return f"""
Optical Settings Summary:
========================
Wavelength:        {self.wavelength:.1f} nm
Numerical Aperture: {self.na:.2f}
Immersion Index:   {self.immersion_refractive_index:.2f}
Source:            {self.source_type} (σ={self.sigma_inner:.2f}-{self.sigma_outer:.2f})
Pixel Size:        {self.pixel_size:.1f} nm
Frequency Grid:    {self.frequency_samples} × {self.frequency_samples}
Polarization:      {'Enabled' if self.enable_polarization else 'Disabled'}

Derived Properties:
------------------
Rayleigh Limit:    {self.rayleigh_resolution:.2f} nm
Frequency Cutoff:  {self.frequency_cutoff:.6f} nm⁻¹
Nyquist Freq:      {self.max_frequency:.6f} nm⁻¹
"""


class PupilFunction:
    """
    Pupil function representing the lens transfer function.

    The pupil function P(fx, fy) describes how the lens modulates
    spatial frequencies. It includes:
    - Geometric aperture (circular, with optional obscuration)
    - Aberrations (Zernike polynomials)
    - Apodization (amplitude variations)
    - Defocus

    For vector (polarization) imaging, returns 2×2 Jones matrix.
    """

    def __init__(self, settings: OpticalSettings):
        """
        Initialize pupil function.

        Args:
            settings: Optical system configuration
        """
        self.settings = settings
        self._aberration_coeff = settings.aberrations

    def evaluate(
        self,
        fx: Union[float, np.ndarray, tf.Tensor],
        fy: Union[float, np.ndarray, tf.Tensor],
        return_jones: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]]:
        """
        Evaluate pupil function at spatial frequency (fx, fy).

        Args:
            fx: X spatial frequency (1/nm)
            fy: Y spatial frequency (1/nm)
            return_jones: If True and polarization enabled, return 2×2 Jones matrix

        Returns:
            Complex pupil value P(fx, fy), or Jones matrix components if return_jones=True
        """
        # Convert to TensorFlow tensors
        fx = tf.cast(fx, tf.float32)
        fy = tf.cast(fy, tf.float32)

        # Radial frequency
        f_radial = tf.sqrt(fx**2 + fy**2)

        # Geometric aperture (circular)
        f_cutoff = self.settings.frequency_cutoff
        aperture = tf.cast(f_radial <= f_cutoff, tf.float32)

        # Central obscuration
        if self.settings.central_obscuration > 0:
            f_obscuration = f_cutoff * self.settings.central_obscuration
            obscuration_mask = tf.cast(f_radial >= f_obscuration, tf.float32)
            aperture *= obscuration_mask

        # Apodization (amplitude variation)
        if self.settings.apodization == 'gaussian':
            apodization = tf.exp(-0.5 * (f_radial / f_cutoff)**2)
        elif self.settings.apodization == 'linear':
            apodization = 1.0 - (f_radial / f_cutoff)
        else:
            apodization = tf.ones_like(f_radial)

        # Phase from defocus
        # Defocus phase: exp(-i k z √(n² - (λf)²))
        if abs(self.settings.defocus) > 1e-6:
            # Wave vector
            k = 2.0 * np.pi / self.settings.wavelength
            n = self.settings.immersion_refractive_index

            # Under square root term
            lambda_f_sq = (self.settings.wavelength * f_radial)**2
            n_sq = n**2

            # Only compute phase where inside aperture
            sqrt_term = tf.sqrt(
                tf.maximum(n_sq - lambda_f_sq, 0.0)
            )

            defocus_phase = -k * self.settings.defocus * sqrt_term
        else:
            defocus_phase = tf.zeros_like(f_radial)

        # Aberrations (Zernike polynomials)
        if self._aberration_coeff is not None:
            aberration_phase = self._compute_aberration_phase(fx, fy, f_radial, f_cutoff)
        else:
            aberration_phase = tf.zeros_like(f_radial)

        # Total phase
        total_phase = defocus_phase + aberration_phase

        # Complex pupil
        amplitude = aperture * apodization
        pupil = tf.cast(amplitude, tf.complex64) * tf.exp(tf.complex(0.0, total_phase))

        # Return scalar or Jones matrix
        if not self.settings.enable_polarization or not return_jones:
            return pupil
        else:
            # Compute Jones matrix for vector imaging
            return self._compute_jones_pupil(fx, fy, f_radial, pupil)

    def _compute_aberration_phase(
        self,
        fx: tf.Tensor,
        fy: tf.Tensor,
        f_radial: tf.Tensor,
        f_cutoff: float
    ) -> tf.Tensor:
        """
        Compute phase from Zernike aberrations.

        Args:
            fx, fy: Spatial frequencies
            f_radial: Radial frequency
            f_cutoff: Cutoff frequency

        Returns:
            Aberration phase in radians
        """
        # Normalized coordinates
        rho = f_radial / f_cutoff  # [0, 1] inside pupil
        theta = tf.atan2(fy, fx)

        # Implement common Zernike terms
        # This is a simplified implementation - full Zernike requires more terms
        # Coefficients: [defocus, astigmatism_x, astigmatism_y, coma_x, coma_y, ...]

        phase = tf.zeros_like(rho)

        if len(self._aberration_coeff) >= 1:
            # Z_2^0: Defocus
            phase += self._aberration_coeff[0] * (2.0 * rho**2 - 1.0)

        if len(self._aberration_coeff) >= 3:
            # Z_2^-2: Astigmatism at 45°
            phase += self._aberration_coeff[1] * rho**2 * tf.sin(2.0 * theta)
            # Z_2^2: Astigmatism at 0°
            phase += self._aberration_coeff[2] * rho**2 * tf.cos(2.0 * theta)

        if len(self._aberration_coeff) >= 5:
            # Z_3^-1: Coma Y
            phase += self._aberration_coeff[3] * (3.0 * rho**3 - 2.0 * rho) * tf.sin(theta)
            # Z_3^1: Coma X
            phase += self._aberration_coeff[4] * (3.0 * rho**3 - 2.0 * rho) * tf.cos(theta)

        return phase

    def _compute_jones_pupil(
        self,
        fx: tf.Tensor,
        fy: tf.Tensor,
        f_radial: tf.Tensor,
        pupil_scalar: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute Jones matrix pupil for vector imaging.

        Returns 2×2 Jones matrix as (Pxx, Pxy, Pyx, Pyy).

        For high-NA lithography, the polarization state changes
        due to refraction at high angles.
        """
        # Angle of incidence
        f_cutoff = self.settings.frequency_cutoff
        f_norm = f_radial / f_cutoff  # Normalized frequency [0, 1]

        # Obliquity factor for high-NA
        # cos(θ) = √(1 - (NA·λ·f)²/n²)
        n = self.settings.immersion_refractive_index
        sin_theta_sq = f_norm**2
        cos_theta = tf.sqrt(tf.maximum(1.0 - sin_theta_sq, 0.0))

        # Azimuthal angle
        phi = tf.atan2(fy, fx)

        # Apodization factors for TE and TM
        # For details, see: Flagello et al., J. Opt. Soc. Am. A 13, 53 (1996)
        apod_te = tf.sqrt(cos_theta)  # TE component
        apod_tm = tf.sqrt(cos_theta)  # TM component (simplified)

        # Jones matrix components
        cos_phi = tf.cos(phi)
        sin_phi = tf.sin(phi)

        # TE component (perpendicular to plane of incidence)
        te_x = -sin_phi
        te_y = cos_phi

        # TM component (parallel to plane of incidence)
        tm_x = cos_phi * cos_theta
        tm_y = sin_phi * cos_theta

        # Jones matrix
        pxx = pupil_scalar * (apod_te * te_x + apod_tm * tm_x)
        pxy = pupil_scalar * (apod_te * te_y + apod_tm * tm_y)
        pyx = pupil_scalar * (apod_te * te_x - apod_tm * tm_x)
        pyy = pupil_scalar * (apod_te * te_y - apod_tm * tm_y)

        return pxx, pxy, pyx, pyy


class SourceDistribution:
    """
    Source intensity distribution S(sx, sy).

    Represents the illumination source in the pupil plane.
    Common types:
    - Circular: conventional coherence
    - Annular: off-axis illumination
    - Dipole: two opposite poles
    - Quadrupole: four poles
    - Custom: arbitrary pixelated source

    Source is normalized: ∫∫ S(s) ds = 1
    """

    def __init__(self, settings: OpticalSettings, custom_source: Optional[np.ndarray] = None):
        """
        Initialize source distribution.

        Args:
            settings: Optical configuration
            custom_source: Optional custom source map (normalized)
        """
        self.settings = settings
        self._custom_source = custom_source

        if custom_source is not None:
            # Normalize custom source
            total = np.sum(custom_source)
            if total > 0:
                self._custom_source = custom_source / total

    def evaluate(
        self,
        sx: Union[float, np.ndarray, tf.Tensor],
        sy: Union[float, np.ndarray, tf.Tensor]
    ) -> tf.Tensor:
        """
        Evaluate source intensity at (sx, sy).

        Source coordinates are normalized by NA/λ.

        Args:
            sx: X source coordinate (normalized)
            sy: Y source coordinate (normalized)

        Returns:
            Source intensity S(sx, sy) ≥ 0
        """
        sx = tf.cast(sx, tf.float32)
        sy = tf.cast(sy, tf.float32)

        s_radial = tf.sqrt(sx**2 + sy**2)

        source_type = self.settings.source_type.lower()

        if source_type == 'circular':
            # Circular source: uniform within σ_outer
            intensity = tf.cast(
                s_radial <= self.settings.sigma_outer,
                tf.float32
            )
            # Normalize
            area = np.pi * self.settings.sigma_outer**2
            intensity /= area

        elif source_type == 'annular':
            # Annular source: uniform between σ_inner and σ_outer
            inner_mask = tf.cast(s_radial >= self.settings.sigma_inner, tf.float32)
            outer_mask = tf.cast(s_radial <= self.settings.sigma_outer, tf.float32)
            intensity = inner_mask * outer_mask
            # Normalize
            area = np.pi * (self.settings.sigma_outer**2 - self.settings.sigma_inner**2)
            intensity /= area

        elif source_type == 'dipole':
            # Dipole: two opposite poles
            s_azimuth = tf.atan2(sy, sx)
            s_mid = (self.settings.sigma_inner + self.settings.sigma_outer) / 2.0
            s_width = (self.settings.sigma_outer - self.settings.sigma_inner) / 2.0

            # Radial mask
            radial_mask = tf.cast(
                tf.abs(s_radial - s_mid) <= s_width,
                tf.float32
            )

            # Angular mask (two poles at 0° and 180°)
            angular_width = np.pi / 6  # 30° opening
            angle_mask = tf.cast(
                (tf.abs(s_azimuth) < angular_width) |
                (tf.abs(s_azimuth - np.pi) < angular_width) |
                (tf.abs(s_azimuth + np.pi) < angular_width),
                tf.float32
            )

            intensity = radial_mask * angle_mask
            # Normalize (approximate)
            intensity /= tf.reduce_sum(intensity) + 1e-10

        elif source_type == 'quadrupole':
            # Quadrupole: four poles at 45°, 135°, 225°, 315°
            s_azimuth = tf.atan2(sy, sx)
            s_mid = (self.settings.sigma_inner + self.settings.sigma_outer) / 2.0
            s_width = (self.settings.sigma_outer - self.settings.sigma_inner) / 2.0

            # Radial mask
            radial_mask = tf.cast(
                tf.abs(s_radial - s_mid) <= s_width,
                tf.float32
            )

            # Angular mask (four poles)
            angular_width = np.pi / 8  # 22.5° opening
            pole_angles = [np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4]
            angle_mask = tf.zeros_like(s_azimuth)
            for pole_angle in pole_angles:
                angle_mask += tf.cast(
                    tf.abs(s_azimuth - pole_angle) < angular_width,
                    tf.float32
                )
            angle_mask = tf.minimum(angle_mask, 1.0)

            intensity = radial_mask * angle_mask
            # Normalize
            intensity /= tf.reduce_sum(intensity) + 1e-10

        elif source_type == 'custom':
            # Custom pixelated source
            if self._custom_source is None:
                raise ValueError("Custom source type requires custom_source array")

            # Interpolate from custom source map
            # This is a simplified implementation
            # In practice, would use tf.image.resize or interpolation
            intensity = tf.constant(self._custom_source, dtype=tf.float32)

        else:
            raise ValueError(f"Unknown source type: {source_type}")

        return intensity

    def generate_source_points(self, n_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate discrete source points for integration.

        Returns source sampling points (sx, sy) and weights for numerical integration.

        Args:
            n_points: Number of source points (uses settings.source_samples if None)

        Returns:
            (sx_points, sy_points, weights) where weights sum to 1.0
        """
        if n_points is None:
            n_points = self.settings.source_samples

        # Generate grid in source space
        source_type = self.settings.source_type.lower()

        if source_type in ['circular', 'annular']:
            # Use polar grid for circular/annular sources
            if source_type == 'circular':
                r_min, r_max = 0.0, self.settings.sigma_outer
            else:
                r_min, r_max = self.settings.sigma_inner, self.settings.sigma_outer

            # Number of radial and angular samples
            n_radial = int(np.sqrt(n_points))
            n_angular = int(n_points / n_radial)

            # Radial samples (area-weighted for uniform density)
            r_edges = np.linspace(r_min**2, r_max**2, n_radial + 1)
            r_centers = np.sqrt((r_edges[:-1] + r_edges[1:]) / 2.0)

            # Angular samples
            theta = np.linspace(0, 2*np.pi, n_angular + 1)[:-1]

            # Create grid
            r_grid, theta_grid = np.meshgrid(r_centers, theta, indexing='ij')
            sx = r_grid * np.cos(theta_grid)
            sy = r_grid * np.sin(theta_grid)

            # Weights (uniform in this case)
            weights = np.ones_like(sx) / (n_radial * n_angular)

            sx_points = sx.flatten()
            sy_points = sy.flatten()
            weights = weights.flatten()

        else:
            # For other source types, use Cartesian grid
            s_max = self.settings.sigma_outer * 1.2
            s_grid = np.linspace(-s_max, s_max, int(np.sqrt(n_points)))
            sx_grid, sy_grid = np.meshgrid(s_grid, s_grid)

            sx_flat = sx_grid.flatten()
            sy_flat = sy_grid.flatten()

            # Evaluate source at each point
            source_values = self.evaluate(sx_flat, sy_flat).numpy()

            # Keep only non-zero points
            mask = source_values > 1e-6
            sx_points = sx_flat[mask]
            sy_points = sy_flat[mask]
            weights = source_values[mask]

            # Normalize weights
            weights = weights / np.sum(weights)

        return sx_points, sy_points, weights

    def visualize(self, resolution: int = 256) -> np.ndarray:
        """
        Generate source intensity map for visualization.

        Args:
            resolution: Image resolution

        Returns:
            2D source intensity array
        """
        s_max = self.settings.sigma_outer * 1.2
        s_coords = np.linspace(-s_max, s_max, resolution)
        sx_grid, sy_grid = np.meshgrid(s_coords, s_coords)

        source_map = self.evaluate(sx_grid, sy_grid).numpy()

        return source_map
