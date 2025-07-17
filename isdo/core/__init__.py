"""
ISDO 核心數學模塊
================

包含希爾伯特空間、變分控制和譜投影的核心實現。
"""

from .spectral_basis import SpectralBasis
from .hilbert_space import HilbertSpace
from .variational_controller import VariationalController
from .spectral_projection import SpectralProjection
from .variational_ode_system import VariationalODESystem

__all__ = [
    "SpectralBasis",
    "HilbertSpace",
    "VariationalController",
    "SpectralProjection",
    "VariationalODESystem",
]