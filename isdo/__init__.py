"""
Infinite Spectral Diffusion Odyssey (ISDO)
===========================================

將擴散採樣轉化為變分最優控制問題的無窮維譜方法實現。

主要模塊:
- core: 核心數學操作 (希爾伯特空間、變分控制)
- math: 數學基礎 (譜基底、李群操作)
- samplers: 採樣器實現 (ISDO主採樣器)

作者: AI Assistant
版本: 0.1.0
許可: 遵循 stable-diffusion-webui-forge 許可協議
"""

__version__ = "0.1.0"
__author__ = "AI Assistant"

# 導入主要類別
from .samplers.isdo_sampler import ISDOSampler
from .core.spectral_basis import SpectralBasis
from .core.hilbert_space import HilbertSpace
from .core.variational_controller import VariationalController

__all__ = [
    "ISDOSampler",
    "SpectralBasis",
    "HilbertSpace",
    "VariationalController",
]