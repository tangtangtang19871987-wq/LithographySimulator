"""
Advanced Lithography Simulator
================================

Physics-based lithography simulation using TCC (Transmission Cross Coefficient)
and SOCC (Sum of Coherent Components) methods.

Features:
- Industry-standard Hopkins formulation
- Both TCC and SOCC methods for cross-validation
- Configurable optical parameters
- Optional polarization effects
- High-performance GPU-accelerated computation

Author: Claude Code
Date: 2026-02-11
Branch: claude/integration-all-features-OkWhC
"""

from .optics import (
    OpticalSettings,
    PupilFunction,
    SourceDistribution,
)

from .tcc import TCCKernel
from .socc import SOCCDecomposition
from .imaging import simulate_tcc, simulate_socc, ImageSimulator
from .validation import compare_methods, validate_accuracy

__version__ = '1.0.0'

__all__ = [
    # Core classes
    'OpticalSettings',
    'PupilFunction',
    'SourceDistribution',
    'TCCKernel',
    'SOCCDecomposition',
    'ImageSimulator',

    # High-level functions
    'simulate_tcc',
    'simulate_socc',
    'compare_methods',
    'validate_accuracy',
]
