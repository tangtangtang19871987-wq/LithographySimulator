# Physics-Based Lithography Simulator Implementation Plan
## Advanced Methods: TCC and SOCC

**Date:** 2026-02-10
**Branch:** `claude/integration-all-features-OkWhC`
**Purpose:** Benchmark-quality training data generation

---

## Executive Summary

This plan outlines the implementation of an advanced lithography simulator using **TCC (Transmission Cross Coefficient)** or **SOCC (Sum of Coherent Components)** methods. These approaches are significantly more sophisticated than the current Abbe imaging implementation and provide:

- **Higher physical accuracy** for partially coherent imaging
- **Better computational efficiency** for repeated simulations
- **Industry-standard** modeling approach
- **Suitable for ML benchmarking** with realistic complexity

---

## 1. Current Implementation Analysis

### 1.1 Existing Simulator (`litho_sim_tf.py`)

**Method:** Abbe Partial Coherence Imaging
- Decomposes source into point sources
- Computes coherent image for each source point
- Incoherently sums intensities

**Limitations:**
1. **Computationally expensive:** O(N_source × N_pixels²)
2. **Not optimized for repeated simulations** with same optical settings
3. **Limited physical accuracy** for complex source shapes
4. **No polarization effects**
5. **Simple source decomposition**

**Strengths:**
- Working and tested
- Correct for basic partially coherent imaging
- Good educational tool

### 1.2 Why TCC/SOCC?

**TCC (Transmission Cross Coefficient):**
- Represents the optical system's **impulse response**
- Precomputed once for given optical settings
- **Physically rigorous** formulation
- Used in production lithography software (Calibre, Sentaurus)

**SOCC (Sum of Coherent Components):**
- Decomposes partially coherent system into **coherent modes**
- Singular Value Decomposition (SVD) of TCC
- **Efficient representation** with dominant modes
- Trade-off between accuracy and speed

---

## 2. Mathematical Foundations

### 2.1 TCC Formulation

**Transmission Cross Coefficient:**
```
TCC(f₁, f₂) = ∫∫ P(f - s) P*(f' - s) S(s) ds
```

Where:
- `P(f)` = Pupil function at frequency `f`
- `S(s)` = Source intensity distribution
- `*` = Complex conjugate
- `f₁, f₂` = Spatial frequency coordinates

**Aerial Image via TCC:**
```
I(x) = ∫∫∫∫ M(f₁) M*(f₂) TCC(f₁, f₂) exp[i2π(f₁-f₂)·x] df₁ df₂
```

Where:
- `M(f)` = Mask spectrum (Fourier transform of mask)
- `I(x)` = Aerial image intensity

**Key Insight:** TCC depends **only on optical system**, not mask!

### 2.2 SOCC Formulation

**SVD Decomposition:**
```
TCC(f₁, f₂) ≈ Σᵢ₌₁ᴷ λᵢ φᵢ(f₁) φᵢ*(f₂)
```

Where:
- `λᵢ` = Singular values (eigenvalues)
- `φᵢ` = Coherent modes (eigenfunctions)
- `K` = Number of modes (typically 10-50 for 95%+ accuracy)

**Aerial Image via SOCC:**
```
I(x) = Σᵢ₌₁ᴷ λᵢ |∫ M(f) φᵢ(f) exp(i2πf·x) df|²
```

**Advantages:**
- Only K coherent propagations (K << N_source points)
- Each term is a standard coherent image
- Parallel computation friendly

---

## 3. Implementation Architecture

### 3.1 Proposed Module Structure

```
advanced_litho_simulator/
├── __init__.py
├── tcc.py                    # TCC computation
├── socc.py                   # SOCC decomposition
├── optics.py                 # Optical system (pupil, source)
├── mask_utils.py             # Mask processing
├── imaging.py                # Image formation
├── polarization.py           # Polarization effects (optional)
└── validation.py             # Cross-check with Abbe
```

### 3.2 Class Hierarchy

```python
# Core classes
OpticalSystem:
    - pupil_function(f)       # Complex pupil
    - source_distribution(s)   # Source intensity
    - wavelength, NA, sigma   # Optical parameters

TCCKernel:
    - compute_tcc()           # Build TCC tensor
    - save/load               # Caching for reuse
    - visualize()             # Diagnostic plots

SOCCDecomposition:
    - decompose_tcc()         # SVD of TCC
    - modes, eigenvalues      # Coherent components
    - truncate_modes(k)       # Keep k dominant modes

MaskProcessor:
    - compute_spectrum()      # FFT of mask
    - apply_mask_bias()       # Optical proximity
    - generate_test_patterns() # Standard patterns

ImageSimulator:
    - aerial_image_tcc()      # Using TCC
    - aerial_image_socc()     # Using SOCC
    - aerial_image_abbe()     # Using Abbe (reference)
```

---

## 4. Detailed Implementation Plan

### Phase 1: Foundation (Week 1)

#### 4.1 Optical System Module (`optics.py`)

**Tasks:**
1. **Enhanced Pupil Function**
   ```python
   class PupilFunction:
       def __init__(self, na, wavelength, aberrations=None,
                    apodization=None, obscuration=0):
           # Extended features:
           # - Apodization (Jones pupil)
           # - Central obscuration
           # - Vector (polarization) pupil

   def evaluate(self, fx, fy):
       """Evaluate pupil at spatial frequency (fx, fy)"""
       # Returns complex pupil value
   ```

2. **Source Distribution**
   ```python
   class SourceDistribution:
       def __init__(self, shape='annular', sigma_inner=0.4,
                    sigma_outer=0.8):
           # Support:
           # - Annular, quasar, dipole, quadrupole
           # - Custom (pixelated)
           # - Source mask optimization (SMO) formats

   def normalize(self):
       """Ensure ∫∫ S(s) ds = 1"""
   ```

3. **Optical Settings Container**
   ```python
   @dataclass
   class OpticalSettings:
       wavelength: float = 193.0  # nm
       na: float = 0.93
       sigma_inner: float = 0.7
       sigma_outer: float = 0.9
       aberrations: List[float] = None  # Zernike coefficients
       pixel_size: float = 5.0  # nm (finer than current 25nm)
       frequency_samples: int = 512  # Frequency domain sampling
   ```

#### 4.2 Validation Framework (`validation.py`)

**Tasks:**
1. **Reference Implementations**
   ```python
   def abbe_imaging_reference(mask, optics):
       """High-precision Abbe for validation"""

   def hopkins_analytical(mask, optics):
       """Analytical solution for simple cases"""
   ```

2. **Accuracy Metrics**
   ```python
   def compare_images(img1, img2):
       """
       Returns:
       - MSE, MAE, SSIM
       - Line-edge profiles
       - CD (Critical Dimension) errors
       """
   ```

### Phase 2: TCC Implementation (Week 1-2)

#### 4.3 TCC Computation (`tcc.py`)

**Approach:** Frequency-domain integration

```python
class TCCKernel:
    """
    Transmission Cross Coefficient computation.

    TCC is a 4D tensor: TCC[f1x, f1y, f2x, f2y]
    For N frequency samples: N^4 elements (expensive!)
    """

    def __init__(self, optics: OpticalSettings, cache_dir=None):
        self.optics = optics
        self.cache_dir = cache_dir
        self._tcc = None

    def compute_tcc_full(self, n_freq=128):
        """
        Compute full TCC tensor.

        Algorithm:
        1. Generate frequency grids (f1x, f1y), (f2x, f2y)
        2. For each (f1, f2):
            a. Integrate over source: ∫∫ P(f1-s)P*(f2-s)S(s) ds
            b. Use quadrature or pixel summation
        3. Store as (n_freq, n_freq, n_freq, n_freq) tensor

        Memory: ~16 GB for n_freq=128, float32
        Time: ~10-30 minutes (optimized)
        """
        # Implementation details...

    def compute_tcc_sparse(self, n_freq=256, threshold=1e-6):
        """
        Compute TCC with sparsity exploitation.

        Observation: TCC is sparse due to pupil support
        - Only store non-zero elements
        - Use COO or CSR sparse format
        """
        # Sparse implementation...

    def save_cache(self, filename):
        """Save precomputed TCC for reuse"""
        # Save with optical settings hash

    def load_cache(self, filename):
        """Load precomputed TCC"""
        # Verify optical settings match
```

**Optimization Strategies:**
1. **Symmetry Exploitation:**
   - TCC has Hermitian symmetry: `TCC(f1, f2) = TCC*(f2, f1)`
   - Only compute half the tensor

2. **Sparse Storage:**
   - Pupil support limits frequency range
   - Store only (f1, f2) where both in pupil

3. **GPU Acceleration:**
   - Parallel integration over source points
   - Batch frequency pair computation

4. **Caching:**
   - Precompute TCC for common optical settings
   - Hash optical parameters for cache lookup

#### 4.4 Image Formation with TCC (`imaging.py`)

```python
def aerial_image_from_tcc(mask, tcc_kernel):
    """
    Compute aerial image using precomputed TCC.

    Steps:
    1. Compute mask spectrum: M(f) = FFT(mask)
    2. Evaluate: I(x) = ∫∫∫∫ M(f1)M*(f2)TCC(f1,f2)exp[i2π(f1-f2)·x] df1 df2
    3. Efficient implementation via FFT

    Time: ~1-2 seconds for 512×512 mask
    """
    # Mask FFT
    mask_spectrum = tf.signal.fft2d(tf.cast(mask, tf.complex64))

    # TCC convolution (efficient algorithm)
    image = _tcc_convolution(mask_spectrum, tcc_kernel.tcc)

    return tf.abs(image)

def _tcc_convolution(M, tcc):
    """
    Efficient TCC-based image computation.

    Uses:
    1. Hopkins decomposition for efficient computation
    2. FFT-based convolution where possible
    3. Sparse matrix operations
    """
    # Implementation...
```

### Phase 3: SOCC Implementation (Week 2)

#### 4.5 SOCC Decomposition (`socc.py`)

```python
class SOCCDecomposition:
    """
    Sum of Coherent Components decomposition.

    Decomposes TCC into dominant coherent modes via SVD.
    Much faster than full TCC for repeated simulations.
    """

    def __init__(self, tcc_kernel: TCCKernel):
        self.tcc_kernel = tcc_kernel
        self.modes = None
        self.eigenvalues = None

    def decompose(self, n_modes=50, method='svd'):
        """
        Perform SVD of TCC matrix.

        Steps:
        1. Reshape TCC: (N^2, N^2) matrix
        2. SVD: TCC = U Σ V^H
        3. Extract k dominant modes
        4. Reshape modes back to 2D

        Methods:
        - 'svd': Full SVD (slow but accurate)
        - 'randomized': Randomized SVD (fast approximation)
        - 'iterative': Power iteration for top modes
        """
        if method == 'svd':
            self._decompose_full_svd(n_modes)
        elif method == 'randomized':
            self._decompose_randomized_svd(n_modes)
        elif method == 'iterative':
            self._decompose_iterative(n_modes)

    def _decompose_full_svd(self, n_modes):
        """
        Full SVD decomposition.

        Algorithm:
        1. Reshape TCC(f1, f2) -> Matrix A
        2. A = U Σ V^H  (SVD)
        3. Keep top k singular values/vectors
        4. modes[i] = sqrt(σ_i) * u_i
        5. eigenvalues[i] = σ_i
        """
        n_freq = self.tcc_kernel.n_freq
        tcc_matrix = tf.reshape(
            self.tcc_kernel.tcc,
            [n_freq * n_freq, n_freq * n_freq]
        )

        # TensorFlow SVD
        s, u, v = tf.linalg.svd(tcc_matrix, full_matrices=False)

        # Keep top modes
        self.eigenvalues = s[:n_modes]
        self.modes = [
            tf.reshape(u[:, i] * tf.sqrt(s[i]), [n_freq, n_freq])
            for i in range(n_modes)
        ]

    def compute_aerial_image(self, mask):
        """
        Compute image using SOCC.

        I(x) = Σᵢ λᵢ |IFFTshift( M(f) φᵢ(f) )|²

        Speed: ~0.1 seconds for k=20 modes, 512×512 mask
        """
        mask_spectrum = tf.signal.fft2d(tf.cast(mask, tf.complex64))

        image = tf.zeros_like(mask, dtype=tf.float32)

        for i, (mode, eigenvalue) in enumerate(
            zip(self.modes, self.eigenvalues)
        ):
            # Coherent image for this mode
            coherent_field = mask_spectrum * mode
            spatial_field = tf.signal.ifft2d(coherent_field)

            # Add intensity contribution
            image += eigenvalue * tf.abs(spatial_field) ** 2

        return image

    def analyze_convergence(self):
        """
        Analyze mode convergence.

        Returns:
        - Cumulative energy: Σᵢ₌₁ᵏ λᵢ / Σ λᵢ
        - Optimal k for 95%, 99%, 99.9% accuracy
        """
        total_energy = tf.reduce_sum(self.eigenvalues)
        cumulative = tf.cumsum(self.eigenvalues) / total_energy

        return {
            'cumulative_energy': cumulative.numpy(),
            'k_95': int(tf.argmax(cumulative >= 0.95)),
            'k_99': int(tf.argmax(cumulative >= 0.99)),
            'k_999': int(tf.argmax(cumulative >= 0.999)),
        }
```

#### 4.6 Fast SOCC Algorithm

**Optimizations:**
1. **Randomized SVD** (Halko et al. 2011)
   - O(k²n) instead of O(n³)
   - Suitable for large TCC matrices

2. **Mode Precomputation**
   - Compute modes once for optical settings
   - Reuse for all masks

3. **GPU Batching**
   - Process multiple modes in parallel
   - Batch multiple mask evaluations

### Phase 4: Advanced Features (Week 3)

#### 4.7 Polarization Effects (`polarization.py`)

```python
class VectorImaging:
    """
    Polarization-aware imaging for high-NA lithography.

    For NA > 0.6, vector (TE/TM) effects become significant.
    """

    def compute_vector_tcc(self, optics):
        """
        Compute 2×2 TCC tensor for TE/TM polarization.

        TCC becomes: TCC[pol1, pol2, f1, f2]
        where pol1, pol2 ∈ {TE, TM}
        """
        # Implementation...
```

#### 4.8 Resist Effects (Optional)

```python
class ResistModel:
    """
    Simple resist model for realistic data.

    Aerial image → Latent image → Resist profile
    """

    def apply_acid_diffusion(self, aerial_image, diffusion_length=20):
        """Gaussian blur to simulate acid diffusion"""

    def threshold_development(self, latent_image, threshold=0.5):
        """Binary threshold for development"""
```

---

## 5. Benchmarking and Validation

### 5.1 Test Cases

**Analytical Cases:**
1. **Coherent Imaging (σ=0):**
   - Should match classical Fourier optics
   - Validate TCC → coherent limit

2. **Incoherent Imaging (σ=∞):**
   - Point Spread Function (PSF) based
   - Validate TCC → incoherent limit

3. **Contact Holes:**
   - Standard test pattern
   - Compare with Abbe method

4. **Line-Space Arrays:**
   - CD measurements
   - Proximity effects

**Numerical Validation:**
```python
def validate_tcc_vs_abbe():
    """
    Compare TCC method with Abbe method.

    Expected:
    - MSE < 1e-4 for same optical settings
    - Speed: TCC 10-100× faster after precomputation
    """
    # Test cases...
```

### 5.2 Performance Benchmarks

**Target Metrics:**
```
TCC Precomputation:
- Time: < 30 minutes for 256×256 frequency samples
- Memory: < 32 GB
- Storage: < 1 GB compressed

SOCC Decomposition:
- Time: < 5 minutes for 50 modes
- Memory: < 16 GB

Image Generation (SOCC):
- Time: < 0.5 seconds per 512×512 image
- Accuracy: > 99% vs full TCC
- Batch: > 100 images/minute

Dataset Generation:
- Target: 10,000 diverse masks
- Time: < 2 hours with SOCC
- Storage: ~1-2 GB
```

---

## 6. Integration Plan

### 6.1 Backward Compatibility

```python
# Keep existing Abbe simulator
from litho_sim_tf import simulate as simulate_abbe

# New advanced simulator
from advanced_litho_simulator import simulate_tcc, simulate_socc

# Unified interface
def simulate(mask, method='socc', **kwargs):
    """
    Unified simulation interface.

    Args:
        method: 'abbe', 'tcc', or 'socc'
    """
    if method == 'abbe':
        return simulate_abbe(mask, **kwargs)
    elif method == 'tcc':
        return simulate_tcc(mask, **kwargs)
    elif method == 'socc':
        return simulate_socc(mask, **kwargs)
```

### 6.2 Data Pipeline Integration

```python
# Update data_pipeline.py
from advanced_litho_simulator import SOCCSimulator

class AdvancedSimulationContext:
    """Enhanced simulation using SOCC."""

    def __init__(self, optical_settings, n_modes=30):
        # Precompute TCC and SOCC modes
        self.socc = SOCCSimulator(optical_settings)
        self.socc.precompute_modes(n_modes)

    def simulate(self, geometry):
        """Fast simulation using precomputed modes."""
        return self.socc.compute_image(geometry)
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# test_advanced_simulator.py

class TestTCC:
    def test_tcc_symmetry():
        """TCC(f1, f2) = TCC*(f2, f1)"""

    def test_tcc_coherent_limit():
        """TCC → δ(f1 - f2) as σ → 0"""

    def test_tcc_caching():
        """Save/load preserves accuracy"""

class TestSOCC:
    def test_socc_decomposition():
        """Modes are orthonormal"""

    def test_socc_convergence():
        """Energy convergence with k modes"""

    def test_socc_accuracy():
        """Image error < 1% for k=50"""

class TestImaging:
    def test_aerial_image_range():
        """Image intensity in [0, 1]"""

    def test_numerical_accuracy():
        """Compare with reference implementation"""
```

### 7.2 Integration Tests

```python
def test_full_pipeline():
    """End-to-end: mask → TCC → SOCC → image"""

def test_batch_generation():
    """Generate 100 diverse masks efficiently"""

def test_vs_commercial():
    """Compare with commercial simulator (if available)"""
```

---

## 8. Documentation Plan

### 8.1 Technical Documentation

**Files:**
1. `ADVANCED_SIMULATOR_GUIDE.md`
   - Theory: TCC and SOCC
   - Usage examples
   - Performance characteristics

2. `TCC_SOCC_MATHEMATICS.md`
   - Detailed mathematical derivations
   - References to literature
   - Validation methods

3. `API_REFERENCE.md`
   - Class/function documentation
   - Parameter descriptions
   - Code examples

### 8.2 Usage Examples

```python
# Quick start
from advanced_litho_simulator import SOCCSimulator, OpticalSettings

# Setup optics
optics = OpticalSettings(
    wavelength=193.0,
    na=0.93,
    sigma_inner=0.7,
    sigma_outer=0.9
)

# Precompute modes (one-time, ~5 minutes)
simulator = SOCCSimulator(optics)
simulator.precompute_modes(n_modes=30)
simulator.save_cache('optical_cache.npz')

# Fast simulation (0.1 seconds)
mask = generate_test_mask()
aerial_image = simulator.simulate(mask)
```

---

## 9. Timeline and Milestones

### Week 1: Foundation
- [ ] Optical system module
- [ ] Validation framework
- [ ] Basic TCC computation (small grid)

### Week 2: TCC/SOCC Core
- [ ] Full TCC implementation
- [ ] SOCC decomposition
- [ ] Image formation algorithms

### Week 3: Optimization & Testing
- [ ] Sparse TCC
- [ ] GPU acceleration
- [ ] Comprehensive tests

### Week 4: Integration & Documentation
- [ ] Data pipeline integration
- [ ] Documentation
- [ ] Benchmark results

---

## 10. Success Criteria

### 10.1 Accuracy
- ✅ MSE < 1e-4 vs Abbe method
- ✅ CD error < 1 nm vs reference
- ✅ SOCC convergence > 99% with k=50

### 10.2 Performance
- ✅ Image generation < 0.5 seconds (SOCC)
- ✅ Batch generation > 100 images/minute
- ✅ Dataset (10,000 images) < 2 hours

### 10.3 Physical Realism
- ✅ Correct Hopkins formulation
- ✅ Polarization effects (optional)
- ✅ Validated against literature

### 10.4 Usability
- ✅ Simple API
- ✅ Backward compatible
- ✅ Well documented
- ✅ Cached precomputation

---

## 11. Risks and Mitigations

### Risk 1: Memory Requirements
**Issue:** Full TCC is 4D tensor (N^4 elements)
**Mitigation:**
- Use sparse representation
- Implement SOCC early (reduces to k modes)
- Support reduced frequency sampling

### Risk 2: Computation Time
**Issue:** TCC precomputation can take hours
**Mitigation:**
- GPU acceleration
- Parallel processing
- Provide precomputed caches for standard optical settings

### Risk 3: Numerical Stability
**Issue:** SVD and FFT can have precision issues
**Mitigation:**
- Use float64 for TCC computation
- Validate against analytical cases
- Extensive unit testing

### Risk 4: Complexity
**Issue:** Implementation is mathematically sophisticated
**Mitigation:**
- Phased approach (start with TCC, then SOCC)
- Extensive validation against Abbe
- Clear documentation

---

## 12. Alternatives Considered

### Alternative 1: Hopkins Formulation Only
**Pros:** Simpler than full TCC
**Cons:** Still requires source decomposition

### Alternative 2: Machine Learning Surrogate
**Pros:** Very fast inference
**Cons:** Not physically rigorous (we're training ML models!)

### Alternative 3: Keep Current Abbe Method
**Pros:** Already working
**Cons:** Not suitable for serious benchmarking

**Decision:** Implement TCC/SOCC for physical rigor and efficiency

---

## 13. References

### Academic Papers
1. Hopkins, H.H. (1953). "On the diffraction theory of optical images"
2. Yeung, M.S. (1988). "Fast computation of aerial images"
3. Flagello et al. (1996). "Theory of high-NA imaging"
4. Liu & Zakhor (2002). "Binary and phase shifting masks"

### Industry Standards
1. SEMI P35: Terminology for Microlithography
2. ITRS Roadmap: Lithography section

### Software References
1. Calibre (Mentor Graphics): TCC-based OPC
2. Sentaurus Lithography (Synopsys): Hopkins imaging
3. Dr. LiTHO (Fraunhofer IISB): Academic simulator

---

## 14. Recommendation

### Recommended Approach: **SOCC Method**

**Rationale:**
1. ✅ **Physically rigorous** - Based on Hopkins formulation
2. ✅ **Computationally efficient** - 10-100× faster than Abbe
3. ✅ **Industry standard** - Used in production tools
4. ✅ **Flexible** - Easy to add features (polarization, etc.)
5. ✅ **Cacheable** - Precompute once, reuse for many masks
6. ✅ **Suitable for ML** - Realistic complexity without being intractable

**Implementation Priority:**
1. **Phase 1:** Basic TCC (2 weeks)
2. **Phase 2:** SOCC decomposition (1 week)
3. **Phase 3:** Optimization & validation (1 week)
4. **Phase 4:** Integration (1 week)

**Total Estimated Time:** 4-5 weeks for full implementation

---

## 15. Next Steps (Awaiting Approval)

**After approval, proceed with:**
1. Create `advanced_litho_simulator/` module structure
2. Implement optical system classes
3. Start with basic TCC computation (small grid)
4. Validate against existing Abbe method
5. Progressively add SOCC and optimizations

**Deliverables:**
- Working SOCC-based simulator
- Comprehensive test suite
- Documentation and examples
- Benchmark dataset generator
- Performance comparison report

---

**Status:** 📋 PLAN READY FOR REVIEW
**Waiting for:** User approval to proceed with implementation
**Estimated Effort:** 4-5 weeks full implementation
**Impact:** Production-quality lithography simulator for ML benchmarking

---

**Prepared by:** Claude Code Analysis
**Date:** 2026-02-10
**Branch:** `claude/integration-all-features-OkWhC`
