"""
Spectral Basis Implementation for ISDO
======================================

實現希爾伯特空間中的各種譜基底，包括傅立葉基底、小波基底等。
提供基底生成、正交化、內積計算和投影操作。

數學背景:
- 傅立葉基底: φ_k(x) = exp(i k·x) / √|Ω|
- 小波基底: 基於 Daubechies 或 Haar 小波
- Sobolev 空間嵌入: H^s ↪ C^0 當 s > d/2
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, List
from enum import Enum
import math


class BasisType(Enum):
    """基底類型枚舉"""
    FOURIER = "fourier"
    WAVELET = "wavelet"
    CHEBYSHEV = "chebyshev"
    LEGENDRE = "legendre"


class SpectralBasis:
    """
    譜基底生成與操作類別

    此類實現無窮維希爾伯特空間中的各種正交基底，
    支援 ISDO 算法的譜投影需求。
    """

    def __init__(
        self,
        spatial_dims: Tuple[int, ...],
        spectral_order: int = 256,
        basis_type: BasisType = BasisType.FOURIER,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        初始化譜基底

        Args:
            spatial_dims: 空間維度 (H, W) 或 (B, C, H, W)
            spectral_order: 譜截斷階數 M
            basis_type: 基底類型
            device: 計算設備
            dtype: 數據類型
        """
        self.spatial_dims = spatial_dims
        self.spectral_order = spectral_order
        self.basis_type = basis_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        # 計算基底維度
        if len(spatial_dims) == 2:  # (H, W)
            self.H, self.W = spatial_dims
            self.total_spatial_dim = self.H * self.W
        elif len(spatial_dims) == 4:  # (B, C, H, W)
            self.B, self.C, self.H, self.W = spatial_dims
            self.total_spatial_dim = self.H * self.W
        else:
            raise ValueError(f"不支援的空間維度: {spatial_dims}")

        # 生成基底
        self.basis_functions = self._generate_basis()
        self.normalization_factors = self._compute_normalization()

    def _generate_basis(self) -> torch.Tensor:
        """
        生成指定類型的基底函數

        Returns:
            basis_functions: (spectral_order, H, W) 的基底函數張量
        """
        if self.basis_type == BasisType.FOURIER:
            return self._generate_fourier_basis()
        elif self.basis_type == BasisType.WAVELET:
            return self._generate_wavelet_basis()
        elif self.basis_type == BasisType.CHEBYSHEV:
            return self._generate_chebyshev_basis()
        else:
            raise NotImplementedError(f"基底類型 {self.basis_type} 尚未實現")

    def _generate_fourier_basis(self) -> torch.Tensor:
        """
        生成 2D 傅立葉基底

        數學公式: φ_{k1,k2}(x,y) = exp(2πi(k1*x/H + k2*y/W)) / √(H*W)

        Returns:
            fourier_basis: (spectral_order, H, W) 複數張量
        """
        # 計算頻率網格
        max_freq = int(math.sqrt(self.spectral_order))
        k1_range = torch.arange(-max_freq//2, max_freq//2 + 1, device=self.device)
        k2_range = torch.arange(-max_freq//2, max_freq//2 + 1, device=self.device)

        k1_grid, k2_grid = torch.meshgrid(k1_range, k2_range, indexing='ij')
        k1_flat = k1_grid.reshape(-1)[:self.spectral_order]
        k2_flat = k2_grid.reshape(-1)[:self.spectral_order]

        # 空間座標
        x = torch.linspace(0, 1, self.H, device=self.device)
        y = torch.linspace(0, 1, self.W, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # 生成基底函數
        basis_list = []
        for i in range(self.spectral_order):
            k1, k2 = k1_flat[i].item(), k2_flat[i].item()

            # φ_k(x,y) = exp(2πi(k1*x + k2*y))
            phase = 2 * math.pi * (k1 * X + k2 * Y)
            basis_func = torch.exp(1j * phase)

            basis_list.append(basis_func)

        basis_functions = torch.stack(basis_list, dim=0)
        return basis_functions.to(self.dtype)

    def _generate_wavelet_basis(self) -> torch.Tensor:
        """
        生成小波基底 (簡化版 Haar 小波)

        Returns:
            wavelet_basis: (spectral_order, H, W) 張量
        """
        # 簡化實現：使用 Haar 小波的 2D 版本
        basis_list = []
        scale_levels = int(math.log2(min(self.H, self.W)))

        # 尺度函數 (scaling function)
        phi_00 = torch.ones(self.H, self.W, device=self.device, dtype=self.dtype)
        phi_00 /= math.sqrt(self.H * self.W)  # 歸一化
        basis_list.append(phi_00)

        # 小波函數在不同尺度和位置
        for j in range(scale_levels):
            scale = 2 ** j
            step_h = self.H // scale
            step_w = self.W // scale

            for h_pos in range(0, self.H, step_h):
                for w_pos in range(0, self.W, step_w):
                    if len(basis_list) >= self.spectral_order:
                        break

                    # 生成 Haar 小波函數
                    psi = torch.zeros(self.H, self.W, device=self.device, dtype=self.dtype)

                    # 水平小波
                    h_end = min(h_pos + step_h, self.H)
                    w_mid = min(w_pos + step_w//2, self.W)
                    w_end = min(w_pos + step_w, self.W)

                    if w_mid < w_end:
                        psi[h_pos:h_end, w_pos:w_mid] = 1.0
                        psi[h_pos:h_end, w_mid:w_end] = -1.0
                        psi *= math.sqrt(scale) / math.sqrt(self.H * self.W)

                        basis_list.append(psi.clone())

                if len(basis_list) >= self.spectral_order:
                    break
            if len(basis_list) >= self.spectral_order:
                break

        # 填滿到指定數量
        while len(basis_list) < self.spectral_order:
            noise_basis = torch.randn(self.H, self.W, device=self.device, dtype=self.dtype)
            noise_basis /= torch.norm(noise_basis)
            basis_list.append(noise_basis)

        return torch.stack(basis_list[:self.spectral_order], dim=0)

    def _generate_chebyshev_basis(self) -> torch.Tensor:
        """
        生成 Chebyshev 多項式基底

        Returns:
            chebyshev_basis: (spectral_order, H, W) 張量
        """
        # 映射到 [-1, 1] 區間
        x = torch.linspace(-1, 1, self.H, device=self.device)
        y = torch.linspace(-1, 1, self.W, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        basis_list = []
        max_degree = int(math.sqrt(self.spectral_order))

        for n in range(max_degree + 1):
            for m in range(max_degree + 1):
                if len(basis_list) >= self.spectral_order:
                    break

                # T_n(x) * T_m(y) - Chebyshev 多項式的張量積
                T_n_x = self._chebyshev_polynomial(n, X)
                T_m_y = self._chebyshev_polynomial(m, Y)

                basis_func = T_n_x * T_m_y
                # 歸一化
                norm = torch.sqrt(torch.sum(basis_func.abs() ** 2))
                if norm > 1e-10:
                    basis_func /= norm

                basis_list.append(basis_func.to(self.dtype))

        # 填滿剩餘
        while len(basis_list) < self.spectral_order:
            random_basis = torch.randn(self.H, self.W, device=self.device, dtype=self.dtype)
            random_basis /= torch.norm(random_basis)
            basis_list.append(random_basis)

        return torch.stack(basis_list[:self.spectral_order], dim=0)

    def _chebyshev_polynomial(self, n: int, x: torch.Tensor) -> torch.Tensor:
        """
        計算第 n 階 Chebyshev 多項式

        使用遞推關係: T_0(x)=1, T_1(x)=x, T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
        """
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return x
        else:
            T_prev2 = torch.ones_like(x)  # T_0
            T_prev1 = x                   # T_1

            for k in range(2, n + 1):
                T_curr = 2 * x * T_prev1 - T_prev2
                T_prev2 = T_prev1
                T_prev1 = T_curr

            return T_prev1

    def _compute_normalization(self) -> torch.Tensor:
        """
        計算基底函數的歸一化因子

        確保 <φ_i, φ_j> = δ_{ij} (Kronecker delta)
        """
        norms = torch.zeros(self.spectral_order, device=self.device, dtype=torch.float32)

        for i in range(self.spectral_order):
            # 計算 ||φ_i||_2
            basis_i = self.basis_functions[i]
            norm_squared = torch.sum(basis_i.conj() * basis_i).real
            norms[i] = torch.sqrt(norm_squared)

        return norms

    def inner_product(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        計算兩個函數的 L^2 內積

        Args:
            f, g: (H, W) 或 (B, C, H, W) 張量

        Returns:
            inner_prod: 內積值
        """
        if f.dim() == 4 and g.dim() == 4:  # (B, C, H, W)
            # 對空間維度積分
            prod = torch.sum(f.conj() * g, dim=(-2, -1))  # (B, C)
            return prod
        elif f.dim() == 2 and g.dim() == 2:  # (H, W)
            prod = torch.sum(f.conj() * g)
            return prod
        else:
            raise ValueError(f"維度不匹配: f.shape={f.shape}, g.shape={g.shape}")

    def project_to_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        將函數投影到譜基底

        計算投影係數: c_k = <x, φ_k>

        Args:
            x: (H, W) 或 (B, C, H, W) 輸入函數

        Returns:
            coefficients: (spectral_order,) 或 (B, C, spectral_order) 投影係數
        """
        if x.dim() == 2:  # (H, W)
            coeffs = torch.zeros(self.spectral_order, device=self.device, dtype=self.dtype)

            for k in range(self.spectral_order):
                # c_k = <x, φ_k> / ||φ_k||^2
                basis_k = self.basis_functions[k]
                inner_prod = self.inner_product(x, basis_k)
                norm_k = self.normalization_factors[k]
                coeffs[k] = inner_prod / (norm_k ** 2 + 1e-10)

            return coeffs

        elif x.dim() == 4:  # (B, C, H, W)
            B, C = x.shape[:2]
            coeffs = torch.zeros(B, C, self.spectral_order, device=self.device, dtype=self.dtype)

            for k in range(self.spectral_order):
                basis_k = self.basis_functions[k]  # (H, W)
                # 廣播計算
                inner_prod = torch.sum(x.conj() * basis_k[None, None, :, :], dim=(-2, -1))  # (B, C)
                norm_k = self.normalization_factors[k]
                coeffs[:, :, k] = inner_prod / (norm_k ** 2 + 1e-10)

            return coeffs
        else:
            raise ValueError(f"不支援的張量維度: {x.shape}")

    def reconstruct_from_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        從譜係數重建函數

        x = Σ_k c_k φ_k

        Args:
            coefficients: (spectral_order,) 或 (B, C, spectral_order) 譜係數

        Returns:
            x_reconstructed: 重建的函數
        """
        if coefficients.dim() == 1:  # (spectral_order,)
            x_recon = torch.zeros(self.H, self.W, device=self.device, dtype=self.dtype)

            for k in range(min(self.spectral_order, len(coefficients))):
                norm_k = self.normalization_factors[k]
                x_recon += coefficients[k] * norm_k**2 * self.basis_functions[k]

            return x_recon

        elif coefficients.dim() == 3:  # (B, C, spectral_order)
            B, C, _ = coefficients.shape
            x_recon = torch.zeros(B, C, self.H, self.W, device=self.device, dtype=self.dtype)

            for k in range(min(self.spectral_order, coefficients.shape[-1])):
                # 廣播相乘
                norm_k = self.normalization_factors[k]
                coeff_k = coefficients[:, :, k]  # (B, C)
                basis_k = self.basis_functions[k]  # (H, W)
                x_recon += coeff_k[:, :, None, None] * norm_k**2 * basis_k[None, None, :, :]

            return x_recon
        else:
            raise ValueError(f"不支援的係數維度: {coefficients.shape}")

    def gram_schmidt_orthogonalization(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Gram-Schmidt 正交化程序

        Args:
            vectors: (num_vectors, H, W) 待正交化的向量組

        Returns:
            orthogonal_vectors: 正交化後的向量組
        """
        num_vectors = vectors.shape[0]
        ortho_vectors = torch.zeros_like(vectors)

        for i in range(num_vectors):
            # 開始正交化第 i 個向量
            v_i = vectors[i].clone()

            # 減去與前面所有正交向量的投影
            for j in range(i):
                u_j = ortho_vectors[j]
                proj_coeff = self.inner_product(v_i, u_j) / (self.inner_product(u_j, u_j) + 1e-10)
                v_i = v_i - proj_coeff * u_j

            # 歸一化
            norm_v_i = torch.sqrt(self.inner_product(v_i, v_i).real + 1e-10)
            ortho_vectors[i] = v_i / norm_v_i

        return ortho_vectors

    def truncate_spectrum(self, coefficients: torch.Tensor, energy_threshold: float = 0.99) -> Tuple[torch.Tensor, int]:
        """
        自適應譜截斷

        保留累積能量達到閾值的譜模式

        Args:
            coefficients: 譜係數
            energy_threshold: 能量保留閾值

        Returns:
            truncated_coeffs: 截斷後的係數
            effective_order: 有效譜階數
        """
        # 計算能量 (係數的模長平方)
        if coefficients.dim() == 1:
            energies = (coefficients.conj() * coefficients).real
        else:
            energies = torch.sum((coefficients.conj() * coefficients).real, dim=tuple(range(coefficients.dim() - 1)))

        # 按能量排序
        sorted_energies, sorted_indices = torch.sort(energies, descending=True)
        cumulative_energy = torch.cumsum(sorted_energies, dim=0)
        total_energy = cumulative_energy[-1]

        # 找到達到閾值的位置
        threshold_mask = cumulative_energy / total_energy >= energy_threshold
        if threshold_mask.any():
            effective_order = torch.argmax(threshold_mask.float()) + 1
        else:
            effective_order = len(coefficients)

        # 創建截斷版本
        truncated_coeffs = torch.zeros_like(coefficients)
        important_indices = sorted_indices[:effective_order]

        if coefficients.dim() == 1:
            truncated_coeffs[important_indices] = coefficients[important_indices]
        else:
            truncated_coeffs[..., important_indices] = coefficients[..., important_indices]

        return truncated_coeffs, effective_order.item()

    def get_frequency_content(self, coefficients: torch.Tensor) -> dict:
        """
        分析譜係數的頻率內容

        Args:
            coefficients: 譜係數

        Returns:
            frequency_analysis: 包含低頻、中頻、高頻能量分佈的字典
        """
        if coefficients.dim() == 1:
            energies = (coefficients.conj() * coefficients).real
        else:
            energies = torch.mean((coefficients.conj() * coefficients).real, dim=tuple(range(coefficients.dim() - 1)))

        total_energy = torch.sum(energies)

        # 簡單的頻率分段
        low_freq_end = self.spectral_order // 4
        high_freq_start = 3 * self.spectral_order // 4

        low_freq_energy = torch.sum(energies[:low_freq_end])
        mid_freq_energy = torch.sum(energies[low_freq_end:high_freq_start])
        high_freq_energy = torch.sum(energies[high_freq_start:])

        return {
            'total_energy': total_energy.item(),
            'low_freq_ratio': (low_freq_energy / total_energy).item(),
            'mid_freq_ratio': (mid_freq_energy / total_energy).item(),
            'high_freq_ratio': (high_freq_energy / total_energy).item(),
            'effective_bandwidth': torch.sum(energies > 0.01 * torch.max(energies)).item()
        }