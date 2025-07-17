"""
Hilbert Space Operations for ISDO
=================================

實現 Sobolev 空間的各種操作，包括範數計算、嵌入檢查和 Gelfand 三元組。
這是 ISDO 算法的數學基礎，確保函數空間的完備性和收斂性。

數學背景:
- Sobolev 空間: H^s(Ω) = {f ∈ L²(Ω) : ||f||_{H^s} < ∞}
- Sobolev 嵌入定理: H^s ↪ C^0 當 s > d/2
- Gelfand 三元組: H ⊂ H' ⊂ H'' (嵌入緊緻)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Dict, List
import math
from .spectral_basis import SpectralBasis, BasisType


class SobolevNorm:
    """Sobolev 範數計算器"""

    def __init__(self, order: float = 1.0, domain_size: Tuple[float, ...] = (1.0, 1.0)):
        """
        初始化 Sobolev 範數計算器

        Args:
            order: Sobolev 空間的階數 s
            domain_size: 定義域大小 (Lx, Ly)
        """
        self.order = order
        self.domain_size = domain_size

    def __call__(self, f: torch.Tensor, derivatives: Optional[Dict] = None) -> torch.Tensor:
        """
        計算 Sobolev H^s 範數

        ||f||_{H^s}² = Σ_{|α| ≤ s} ||D^α f||_{L²}²

        Args:
            f: 輸入函數 (B, C, H, W) 或 (H, W)
            derivatives: 預計算的導數 (可選)

        Returns:
            sobolev_norm: H^s 範數
        """
        if derivatives is None:
            derivatives = self._compute_derivatives(f)

        norm_squared = torch.tensor(0.0, device=f.device, dtype=f.dtype)

        # L² 範數 (階數 0)
        l2_norm_sq = torch.sum(f.abs() ** 2, dim=(-2, -1))  # 空間維度積分
        norm_squared = norm_squared + l2_norm_sq

        if self.order >= 1:
            # 一階導數項
            for deriv_name in ['dx', 'dy']:
                if deriv_name in derivatives:
                    deriv_norm_sq = torch.sum(derivatives[deriv_name].abs() ** 2, dim=(-2, -1))
                    norm_squared = norm_squared + deriv_norm_sq

        if self.order >= 2:
            # 二階導數項
            for deriv_name in ['dxx', 'dyy', 'dxy']:
                if deriv_name in derivatives:
                    deriv_norm_sq = torch.sum(derivatives[deriv_name].abs() ** 2, dim=(-2, -1))
                    norm_squared = norm_squared + deriv_norm_sq

        # 處理分數階
        if self.order != int(self.order):
            fractional_part = self.order - int(self.order)
            # 使用傅立葉方法計算分數階 Sobolev 範數
            frac_norm_sq = self._fractional_sobolev_norm(f, fractional_part)
            norm_squared = norm_squared + frac_norm_sq

        return torch.sqrt(norm_squared)

    def _compute_derivatives(self, f: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        計算函數的各階導數

        使用中心差分方案計算數值導數
        """
        derivatives = {}

        # 一階導數
        # ∂f/∂x
        dx = torch.zeros_like(f)
        dx[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / 2.0
        dx[..., 0, :] = f[..., 1, :] - f[..., 0, :]
        dx[..., -1, :] = f[..., -1, :] - f[..., -2, :]
        derivatives['dx'] = dx / self.domain_size[0]

        # ∂f/∂y
        dy = torch.zeros_like(f)
        dy[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) / 2.0
        dy[..., :, 0] = f[..., :, 1] - f[..., :, 0]
        dy[..., :, -1] = f[..., :, -1] - f[..., :, -2]
        derivatives['dy'] = dy / self.domain_size[1]

        # 二階導數
        # ∂²f/∂x²
        dxx = torch.zeros_like(f)
        dxx[..., 1:-1, :] = f[..., 2:, :] - 2*f[..., 1:-1, :] + f[..., :-2, :]
        derivatives['dxx'] = dxx / (self.domain_size[0] ** 2)

        # ∂²f/∂y²
        dyy = torch.zeros_like(f)
        dyy[..., :, 1:-1] = f[..., :, 2:] - 2*f[..., :, 1:-1] + f[..., :, :-2]
        derivatives['dyy'] = dyy / (self.domain_size[1] ** 2)

        # ∂²f/∂x∂y (混合導數)
        dxy = torch.zeros_like(f)
        dxy[..., 1:-1, 1:-1] = (f[..., 2:, 2:] - f[..., 2:, :-2] -
                                 f[..., :-2, 2:] + f[..., :-2, :-2]) / 4.0
        derivatives['dxy'] = dxy / (self.domain_size[0] * self.domain_size[1])

        return derivatives

    def _fractional_sobolev_norm(self, f: torch.Tensor, fractional_order: float) -> torch.Tensor:
        """
        使用譜方法計算分數階 Sobolev 範數

        ||f||_{H^s}² = ∫ |f̂(ξ)|² (1 + |ξ|²)^s dξ
        """
        # 傅立葉變換
        f_hat = torch.fft.fft2(f, dim=(-2, -1))

        # 頻率網格
        H, W = f.shape[-2:]
        kx = torch.fft.fftfreq(H, d=1.0/H, device=f.device)
        ky = torch.fft.fftfreq(W, d=1.0/W, device=f.device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')

        # 頻率模長平方
        k_squared = KX**2 + KY**2

        # 分數階權重 (1 + |k|²)^s
        weight = (1 + k_squared) ** fractional_order

        # 計算範數
        weighted_spectrum = f_hat.abs() ** 2 * weight
        norm_squared = torch.sum(weighted_spectrum, dim=(-2, -1))

        return norm_squared


class HilbertSpace:
    """
    希爾伯特空間操作類別

    實現 Sobolev 空間的各種數學操作，包括嵌入檢查、
    Gelfand 三元組和函數空間的幾何結構。
    """

    def __init__(
        self,
        spatial_dims: Tuple[int, ...],
        sobolev_order: float = 1.0,
        domain_size: Tuple[float, ...] = (1.0, 1.0),
        device: Optional[torch.device] = None
    ):
        """
        初始化希爾伯特空間

        Args:
            spatial_dims: 空間維度 (H, W)
            sobolev_order: Sobolev 空間階數 s
            domain_size: 定義域大小
            device: 計算設備
        """
        self.spatial_dims = spatial_dims
        self.sobolev_order = sobolev_order
        self.domain_size = domain_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if len(spatial_dims) == 2:
            self.H, self.W = spatial_dims
            self.spatial_dim = 2  # 空間維數 d
        else:
            raise ValueError(f"目前只支援 2D 空間: {spatial_dims}")

        # 初始化 Sobolev 範數計算器
        self.sobolev_norm = SobolevNorm(sobolev_order, domain_size)

        # 檢查嵌入條件
        self.embedding_threshold = self.spatial_dim / 2.0  # d/2
        self.is_embedded_in_continuous = sobolev_order > self.embedding_threshold

    def compute_sobolev_norm(self, f: torch.Tensor) -> torch.Tensor:
        """
        計算函數的 Sobolev H^s 範數

        Args:
            f: 輸入函數

        Returns:
            norm: Sobolev 範數
        """
        return self.sobolev_norm(f)

    def check_sobolev_embedding(self, s: float) -> Dict[str, bool]:
        """
        檢查 Sobolev 嵌入條件

        根據 Sobolev 嵌入定理檢查各種嵌入關係

        Args:
            s: Sobolev 空間階數

        Returns:
            embedding_info: 嵌入信息字典
        """
        d = self.spatial_dim

        embedding_info = {
            'continuous_embedding': s > d/2,  # H^s ↪ C^0
            'compact_embedding': s > d/2,     # 緊嵌入
            'bounded_embedding': s >= 0,      # H^s ↪ L²
            'critical_sobolev': abs(s - d/2) < 1e-6,  # 臨界情況
        }

        # 更詳細的嵌入信息
        if s > d/2 + 1:
            embedding_info['holder_embedding'] = True
            embedding_info['holder_exponent'] = min(1.0, s - d/2 - 1)
        else:
            embedding_info['holder_embedding'] = False
            embedding_info['holder_exponent'] = 0.0

        # Lebesgue 空間嵌入
        if s >= 0:
            # H^s ↪ L^p 當 1/p = 1/2 - s/d
            if s < d/2:
                p_critical = 2*d / (d - 2*s)
                embedding_info['lebesgue_critical_p'] = p_critical
            else:
                embedding_info['lebesgue_critical_p'] = float('inf')

        return embedding_info

    def gelfand_triple_projection(self, f: torch.Tensor, target_space: str = 'dual') -> torch.Tensor:
        """
        Gelfand 三元組中的投影操作

        實現 H ⊂ H' ⊂ H'' 中的投影

        Args:
            f: 輸入函數
            target_space: 目標空間 ('dual', 'double_dual')

        Returns:
            projected_f: 投影後的函數
        """
        if target_space == 'dual':
            # 投影到對偶空間 H'
            # 使用 Riesz 表示定理: f ↦ ⟨f, ·⟩
            return self._riesz_representation(f)
        elif target_space == 'double_dual':
            # 投影到雙對偶空間 H''
            return self._double_dual_embedding(f)
        else:
            raise ValueError(f"未知的目標空間: {target_space}")

    def _riesz_representation(self, f: torch.Tensor) -> torch.Tensor:
        """
        Riesz 表示定理的實現

        將函數映射到其 Riesz 表示元
        """
        # 簡化實現：使用 (-Δ + I)^{-s/2} 作為 Riesz 映射
        # 在傅立葉域中: F[(-Δ + I)^{-s/2} f] = (|ξ|² + 1)^{-s/2} F[f]

        f_hat = torch.fft.fft2(f, dim=(-2, -1))

        # 頻率網格
        kx = torch.fft.fftfreq(self.H, d=1.0/self.H, device=self.device)
        ky = torch.fft.fftfreq(self.W, d=1.0/self.W, device=self.device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        k_squared = KX**2 + KY**2

        # Riesz 運算子
        riesz_multiplier = (k_squared + 1) ** (-self.sobolev_order/2)

        # 應用運算子
        riesz_f_hat = f_hat * riesz_multiplier

        # 逆傅立葉變換
        riesz_f = torch.fft.ifft2(riesz_f_hat, dim=(-2, -1)).real

        return riesz_f

    def _double_dual_embedding(self, f: torch.Tensor) -> torch.Tensor:
        """
        雙對偶空間的自然嵌入
        """
        # 對於有限維情況，雙對偶嵌入是恆等映射
        return f.clone()

    def inner_product(self, f: torch.Tensor, g: torch.Tensor,
                     space_type: str = 'sobolev') -> torch.Tensor:
        """
        計算不同空間中的內積

        Args:
            f, g: 輸入函數
            space_type: 空間類型 ('l2', 'sobolev', 'dual')

        Returns:
            inner_product: 內積值
        """
        if space_type == 'l2':
            # L² 內積: ⟨f, g⟩ = ∫ f(x) g*(x) dx
            return torch.sum(f.conj() * g, dim=(-2, -1))

        elif space_type == 'sobolev':
            # Sobolev 內積: ⟨f, g⟩_{H^s} = ∑_{|α|≤s} ⟨D^α f, D^α g⟩_{L²}
            return self._sobolev_inner_product(f, g)

        elif space_type == 'dual':
            # 對偶配對: ⟨f, g⟩ = f(g)
            riesz_f = self._riesz_representation(f)
            return torch.sum(riesz_f.conj() * g, dim=(-2, -1))

        else:
            raise ValueError(f"未知的空間類型: {space_type}")

    def _sobolev_inner_product(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        計算 Sobolev 內積
        """
        # 計算導數
        f_derivs = self.sobolev_norm._compute_derivatives(f)
        g_derivs = self.sobolev_norm._compute_derivatives(g)

        inner_prod = torch.tensor(0.0, device=f.device, dtype=f.dtype)

        # L² 項
        inner_prod = inner_prod + torch.sum(f.conj() * g, dim=(-2, -1))

        # 導數項
        if self.sobolev_order >= 1:
            for deriv_name in ['dx', 'dy']:
                if deriv_name in f_derivs and deriv_name in g_derivs:
                    deriv_inner = torch.sum(f_derivs[deriv_name].conj() *
                                          g_derivs[deriv_name], dim=(-2, -1))
                    inner_prod = inner_prod + deriv_inner

        if self.sobolev_order >= 2:
            for deriv_name in ['dxx', 'dyy', 'dxy']:
                if deriv_name in f_derivs and deriv_name in g_derivs:
                    deriv_inner = torch.sum(f_derivs[deriv_name].conj() *
                                          g_derivs[deriv_name], dim=(-2, -1))
                    inner_prod = inner_prod + deriv_inner

        return inner_prod

    def fractional_laplacian(self, f: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        計算分數階拉普拉斯算子 (-Δ)^α f

        在傅立葉域中: F[(-Δ)^α f] = |ξ|^{2α} F[f]

        Args:
            f: 輸入函數
            alpha: 分數階參數

        Returns:
            result: (-Δ)^α f
        """
        f_hat = torch.fft.fft2(f, dim=(-2, -1))

        # 頻率網格
        kx = torch.fft.fftfreq(self.H, d=1.0/self.H, device=self.device)
        ky = torch.fft.fftfreq(self.W, d=1.0/self.W, device=self.device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        k_squared = KX**2 + KY**2

        # 避免 k=0 處的奇異性
        k_squared = torch.clamp(k_squared, min=1e-12)

        # 分數階拉普拉斯算子
        laplacian_multiplier = k_squared ** alpha

        # 應用算子
        result_hat = f_hat * laplacian_multiplier

        # 逆變換
        result = torch.fft.ifft2(result_hat, dim=(-2, -1))

        # 如果輸入是實數，返回實部
        if f.dtype.is_floating_point:
            result = result.real

        return result

    def estimate_sobolev_regularity(self, f: torch.Tensor,
                                  max_order: float = 5.0) -> float:
        """
        估計函數的 Sobolev 正則性

        通過頻譜分析估計函數所屬的最高 Sobolev 空間

        Args:
            f: 輸入函數
            max_order: 最大檢查階數

        Returns:
            estimated_order: 估計的 Sobolev 階數
        """
        f_hat = torch.fft.fft2(f, dim=(-2, -1))

        # 頻率網格
        kx = torch.fft.fftfreq(self.H, d=1.0/self.H, device=self.device)
        ky = torch.fft.fftfreq(self.W, d=1.0/self.W, device=self.device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        k_squared = KX**2 + KY**2

        # 測試不同的 Sobolev 階數
        orders = torch.linspace(0, max_order, 50, device=self.device)
        norms = torch.zeros_like(orders)

        for i, s in enumerate(orders):
            # 計算 ||f||_{H^s}²
            weight = (1 + k_squared) ** s
            weighted_spectrum = f_hat.abs() ** 2 * weight
            norms[i] = torch.sum(weighted_spectrum)

        # 找到範數爆炸的臨界點
        log_norms = torch.log(norms + 1e-12)

        # 尋找範數增長率變化最大的點
        growth_rate = torch.diff(log_norms) / torch.diff(orders)

        # 找到增長率急劇增加的點
        critical_idx = torch.argmax(torch.diff(growth_rate))
        estimated_order = orders[critical_idx].item()

        return estimated_order

    def verify_embedding_constants(self, f: torch.Tensor) -> Dict[str, float]:
        """
        驗證 Sobolev 嵌入的最佳常數

        Args:
            f: 測試函數

        Returns:
            constants: 嵌入常數字典
        """
        constants = {}

        if self.is_embedded_in_continuous:
            # 連續嵌入常數: ||f||_{C^0} ≤ C ||f||_{H^s}
            continuous_norm = torch.max(f.abs())
            sobolev_norm = self.compute_sobolev_norm(f)
            constants['continuous_embedding'] = (continuous_norm / (sobolev_norm + 1e-12)).item()

        # L² 嵌入常數
        l2_norm = torch.sqrt(torch.sum(f.abs() ** 2, dim=(-2, -1)))
        sobolev_norm = self.compute_sobolev_norm(f)
        constants['l2_embedding'] = (l2_norm / (sobolev_norm + 1e-12)).item()

        return constants

    def project_to_sobolev_ball(self, f: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
        """
        將函數投影到 Sobolev 球

        確保 ||f||_{H^s} ≤ radius

        Args:
            f: 輸入函數
            radius: 球半徑

        Returns:
            projected_f: 投影後的函數
        """
        current_norm = self.compute_sobolev_norm(f)

        if current_norm <= radius:
            return f
        else:
            # 重新歸一化
            scaling_factor = radius / (current_norm + 1e-12)
            return f * scaling_factor