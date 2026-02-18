"""
half_unet_fno — backward-compatible re-export shim
====================================================

The model has moved to ``models/half_unet_fno.py``.
This module re-exports everything so existing imports keep working.

Original port: https://github.com/Robh96/UNet-FNO (MIT License)
"""

# Re-export everything from the canonical location
from models.half_unet_fno import (  # noqa: F401
    _compl_mul2d,
    SpectralConv2d,
    FNOBlock,
    HalfUNetFNO,
    ACTIVATION_MAP,
    build_half_unet_fno,
    build_and_compile,
)

__all__ = [
    "_compl_mul2d",
    "SpectralConv2d",
    "FNOBlock",
    "HalfUNetFNO",
    "ACTIVATION_MAP",
    "build_half_unet_fno",
    "build_and_compile",
]
