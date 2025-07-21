"""
Spectral Projection System for ISDO
==================================

實現圖像在空間域和譜域之間的高效轉換，是連接 SpectralBasis 和數值計算的核心組件。
提供 FFT 優化、批次處理和自適應截斷等高級功能。

數學背景:
- 投影算子: P_M: L²(Ω) → span{φ₁, ..., φ_M}
- 譜重建: x = Σ_{k=1}^M c_k φ_k
- Galerkin 投影: ⟨x - P_M x, φ_k⟩ = 0, ∀k ≤ M
"""

import torch
import torch.nn.functional as F
import torch.fft
import numpy as np
from typing import Tuple, Optional, Union, Dict, List
import math
from .spectral_basis import SpectralBasis, BasisType


class SpectralProjector:
    """
    譜投影器的基礎類

    定義譜投影的基本接口和通用功能
    """

    def __init__(
        self,
        spectral_basis: SpectralBasis,
        device: Optional[torch.device] = None
    ):
        """
        初始化譜投影器

        Args:
            spectral_basis: 譜基底對象
            device: 計算設備
        """
        self.basis = spectral_basis
        self.device = device or spectral_basis.device
        self.spectral_order = spectral_basis.spectral_order

    def forward_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向投影: 空間域 → 譜域

        計算 c_k = ⟨x, φ_k⟩

        Args:
            x: 空間域函數

        Returns:
            coefficients: 譜係數
        """
        return self.basis.project_to_basis(x)

    def inverse_projection(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        逆投影: 譜域 → 空間域

        重建 x = Σ c_k φ_k

        Args:
            coefficients: 譜係數

        Returns:
            x_reconstructed: 重建的空間域函數
        """
        return self.basis.reconstruct_from_coefficients(coefficients)

    def project_and_truncate(
        self,
        x: torch.Tensor,
        energy_threshold: float = 0.99
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        投影並自適應截斷

        Args:
            x: 輸入函數
            energy_threshold: 能量保留閾值

        Returns:
            truncated_coeffs: 截斷後的係數
            reconstructed: 重建的函數
            effective_order: 有效譜階數
        """
        # 完整投影
        full_coeffs = self.forward_projection(x)

        # 自適應截斷
        truncated_coeffs, effective_order = self.basis.truncate_spectrum(
            full_coeffs, energy_threshold
        )

        # 重建
        reconstructed = self.inverse_projection(truncated_coeffs)

        return truncated_coeffs, reconstructed, effective_order


class FFTSpectralProjection(SpectralProjector):
    """
    基於 FFT 的快速譜投影

    專門針對傅立葉基底優化，利用 FFT 加速計算
    """

    def __init__(
        self,
        spatial_dims: Tuple[int, ...],
        spectral_order: int = 256,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        初始化 FFT 譜投影

        Args:
            spatial_dims: 空間維度
            spectral_order: 譜截斷階數
            device: 計算設備
            dtype: 數據類型
        """
        # 創建傅立葉基底
        fourier_basis = SpectralBasis(
            spatial_dims=spatial_dims,
            spectral_order=spectral_order,
            basis_type=BasisType.FOURIER,
            device=device,
            dtype=dtype
        )

        super().__init__(fourier_basis, device)

        self.spatial_dims = spatial_dims
        self.dtype = dtype

        if len(spatial_dims) == 2:
            self.H, self.W = spatial_dims
        else:
            raise ValueError(f"目前只支援 2D: {spatial_dims}")

        # 預計算頻率映射
        self._setup_frequency_mapping()

    def _setup_frequency_mapping(self):
        """
        設置頻率索引映射，用於 FFT 和譜係數之間的轉換
        """
        # 計算最大頻率
        max_freq = int(math.sqrt(self.spectral_order))

        # 建立頻率索引
        k1_range = torch.arange(-max_freq//2, max_freq//2 + 1, device=self.device)
        k2_range = torch.arange(-max_freq//2, max_freq//2 + 1, device=self.device)

        k1_grid, k2_grid = torch.meshgrid(k1_range, k2_range, indexing='ij')

        # 展平並截斷到 spectral_order
        self.k1_indices = k1_grid.reshape(-1)[:self.spectral_order]
        self.k2_indices = k2_grid.reshape(-1)[:self.spectral_order]

        # 建立 FFT 頻率映射
        fft_k1 = torch.fft.fftfreq(self.H, device=self.device) * self.H
        fft_k2 = torch.fft.fftfreq(self.W, device=self.device) * self.W

        # 創建映射表
        self.fft_to_spectral_map = torch.zeros(self.spectral_order, 2, dtype=torch.long, device=self.device)

        for i in range(self.spectral_order):
            k1, k2 = self.k1_indices[i].item(), self.k2_indices[i].item()

            # 找到對應的 FFT 索引
            fft_idx1 = torch.argmin(torch.abs(fft_k1 - k1))
            fft_idx2 = torch.argmin(torch.abs(fft_k2 - k2))

            self.fft_to_spectral_map[i] = torch.tensor([fft_idx1, fft_idx2])

    def forward_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 FFT 的快速前向投影

        Args:
            x: 空間域函數 (B, C, H, W) 或 (H, W)

        Returns:
            coefficients: 譜係數
        """
        original_shape = x.shape

        # 確保輸入是複數
        if x.dtype.is_floating_point:
            x_complex = x.to(self.dtype)
        else:
            x_complex = x

        # 2D FFT
        x_fft = torch.fft.fft2(x_complex, dim=(-2, -1))

        # 歸一化
        x_fft = x_fft / math.sqrt(self.H * self.W)

        # 提取譜係數
        if x.dim() == 4:  # (B, C, H, W)
            B, C = x.shape[:2]
            coefficients = torch.zeros(B, C, self.spectral_order, device=self.device, dtype=self.dtype)

            for i in range(self.spectral_order):
                idx1, idx2 = self.fft_to_spectral_map[i]
                coefficients[:, :, i] = x_fft[:, :, idx1, idx2]

        elif x.dim() == 2:  # (H, W)
            coefficients = torch.zeros(self.spectral_order, device=self.device, dtype=self.dtype)

            for i in range(self.spectral_order):
                idx1, idx2 = self.fft_to_spectral_map[i]
                coefficients[i] = x_fft[idx1, idx2]
        else:
            raise ValueError(f"不支援的張量維度: {x.shape}")

        return coefficients

    def inverse_projection(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        使用 IFFT 的快速逆投影

        Args:
            coefficients: 譜係數

        Returns:
            x_reconstructed: 重建的空間域函數
        """
        if coefficients.dim() == 3:  # (B, C, spectral_order)
            B, C, _ = coefficients.shape
            x_fft = torch.zeros(B, C, self.H, self.W, device=self.device, dtype=self.dtype)

            # 填充 FFT 數組
            for i in range(min(self.spectral_order, coefficients.shape[-1])):
                idx1, idx2 = self.fft_to_spectral_map[i]
                x_fft[:, :, idx1, idx2] = coefficients[:, :, i]

        elif coefficients.dim() == 1:  # (spectral_order,)
            x_fft = torch.zeros(self.H, self.W, device=self.device, dtype=self.dtype)

            for i in range(min(self.spectral_order, len(coefficients))):
                idx1, idx2 = self.fft_to_spectral_map[i]
                x_fft[idx1, idx2] = coefficients[i]
        else:
            raise ValueError(f"不支援的係數維度: {coefficients.shape}")

        # 逆 FFT
        x_reconstructed = torch.fft.ifft2(x_fft, dim=(-2, -1))

        # 歸一化
        x_reconstructed = x_reconstructed * math.sqrt(self.H * self.W)

        # 如果原始是實數，返回實部
        if coefficients.dtype.is_floating_point:
            x_reconstructed = x_reconstructed.real

        return x_reconstructed

    def compute_spectral_derivatives(self, coefficients: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        在譜域中計算導數

        利用 ∇φ_k 在頻域的簡單形式

        Args:
            coefficients: 譜係數

        Returns:
            derivatives: 包含各階導數的字典
        """
        derivatives = {}

        # 計算頻率乘子
        k1_multipliers = 2j * math.pi * self.k1_indices / self.H
        k2_multipliers = 2j * math.pi * self.k2_indices / self.W

        if coefficients.dim() == 3:  # (B, C, spectral_order)
            # ∂/∂x 導數
            dx_coeffs = coefficients * k1_multipliers[None, None, :]
            derivatives['dx'] = self.inverse_projection(dx_coeffs)

            # ∂/∂y 導數
            dy_coeffs = coefficients * k2_multipliers[None, None, :]
            derivatives['dy'] = self.inverse_projection(dy_coeffs)

            # 二階導數
            dxx_coeffs = coefficients * (k1_multipliers[None, None, :] ** 2)
            derivatives['dxx'] = self.inverse_projection(dxx_coeffs)

            dyy_coeffs = coefficients * (k2_multipliers[None, None, :] ** 2)
            derivatives['dyy'] = self.inverse_projection(dyy_coeffs)

            # 混合導數
            dxy_coeffs = coefficients * (k1_multipliers[None, None, :] * k2_multipliers[None, None, :])
            derivatives['dxy'] = self.inverse_projection(dxy_coeffs)

        elif coefficients.dim() == 1:  # (spectral_order,)
            dx_coeffs = coefficients * k1_multipliers
            derivatives['dx'] = self.inverse_projection(dx_coeffs)

            dy_coeffs = coefficients * k2_multipliers
            derivatives['dy'] = self.inverse_projection(dy_coeffs)

            dxx_coeffs = coefficients * (k1_multipliers ** 2)
            derivatives['dxx'] = self.inverse_projection(dxx_coeffs)

            dyy_coeffs = coefficients * (k2_multipliers ** 2)
            derivatives['dyy'] = self.inverse_projection(dyy_coeffs)

            dxy_coeffs = coefficients * (k1_multipliers * k2_multipliers)
            derivatives['dxy'] = self.inverse_projection(dxy_coeffs)

        return derivatives

    def apply_spectral_filter(
        self,
        coefficients: torch.Tensor,
        filter_func: callable
    ) -> torch.Tensor:
        """
        在譜域應用濾波器

        Args:
            coefficients: 譜係數
            filter_func: 濾波函數 f(k1, k2) → weight

        Returns:
            filtered_coefficients: 濾波後的係數
        """
        # 計算每個模式的濾波權重
        weights = torch.zeros(self.spectral_order, device=self.device, dtype=coefficients.dtype)

        for i in range(self.spectral_order):
            k1 = self.k1_indices[i].item()
            k2 = self.k2_indices[i].item()
            weights[i] = filter_func(k1, k2)

        # 應用濾波
        if coefficients.dim() == 3:  # (B, C, spectral_order)
            filtered = coefficients * weights[None, None, :]
        elif coefficients.dim() == 1:  # (spectral_order,)
            filtered = coefficients * weights
        else:
            raise ValueError(f"不支援的係數維度: {coefficients.shape}")

        return filtered


class AdaptiveSpectralProjection(FFTSpectralProjection):
    """
    自適應譜投影系統

    根據函數特性動態調整譜階數和投影策略
    """

    def __init__(
        self,
        spatial_dims: Tuple[int, ...],
        initial_spectral_order: int = 128,
        max_spectral_order: int = 512,
        adaptation_threshold: float = 0.01,
        device: Optional[torch.device] = None
    ):
        """
        初始化自適應譜投影

        Args:
            spatial_dims: 空間維度
            initial_spectral_order: 初始譜階數
            max_spectral_order: 最大譜階數
            adaptation_threshold: 自適應閾值
            device: 計算設備
        """
        super().__init__(spatial_dims, initial_spectral_order, device)

        self.initial_order = initial_spectral_order
        self.max_order = max_spectral_order
        self.adaptation_threshold = adaptation_threshold

        # 創建多個不同階數的投影器
        self.projectors = {}
        for order in [64, 128, 256, 512]:
            if order <= max_spectral_order:
                self.projectors[order] = FFTSpectralProjection(
                    spatial_dims, order, device
                )

    def adaptive_projection(
        self,
        x: torch.Tensor,
        target_error: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        自適應譜投影

        根據重建誤差動態選擇最適譜階數

        Args:
            x: 輸入函數
            target_error: 目標誤差

        Returns:
            coefficients: 最優譜係數
            reconstructed: 重建函數
            optimal_order: 最優譜階數
        """
        # 從低階開始測試
        orders_to_test = sorted([order for order in self.projectors.keys()])

        best_order = orders_to_test[0]
        best_coeffs = None
        best_reconstructed = None

        for order in orders_to_test:
            projector = self.projectors[order]

            # 投影和重建
            coeffs = projector.forward_projection(x)
            reconstructed = projector.inverse_projection(coeffs)

            # 計算重建誤差
            error = self._compute_reconstruction_error(x, reconstructed)

            if error <= target_error:
                best_order = order
                best_coeffs = coeffs
                best_reconstructed = reconstructed
                break
            else:
                # 保存當前最佳結果
                best_order = order
                best_coeffs = coeffs
                best_reconstructed = reconstructed

        return best_coeffs, best_reconstructed, best_order

    def _compute_reconstruction_error(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> float:
        """
        計算重建誤差
        """
        # 相對 L² 誤差
        diff = original - reconstructed
        error_norm = torch.norm(diff.view(diff.shape[0], -1) if diff.dim() > 2 else diff.view(-1))
        original_norm = torch.norm(original.view(original.shape[0], -1) if original.dim() > 2 else original.view(-1))

        relative_error = error_norm / (original_norm + 1e-12)
        return relative_error.item()

    def analyze_spectral_content(self, x: torch.Tensor) -> Dict[str, float]:
        """
        分析函數的譜內容

        Args:
            x: 輸入函數

        Returns:
            spectral_analysis: 譜分析結果
        """
        # 使用最高階投影器分析
        max_order_projector = self.projectors[max(self.projectors.keys())]
        coeffs = max_order_projector.forward_projection(x)

        # 計算能量分佈
        energies = (coeffs.conj() * coeffs).real
        if energies.dim() > 1:
            energies = torch.mean(energies, dim=tuple(range(energies.dim() - 1)))

        total_energy = torch.sum(energies)

        # 分析頻率分佈
        num_modes = len(energies)
        low_freq_end = num_modes // 4
        high_freq_start = 3 * num_modes // 4

        low_freq_energy = torch.sum(energies[:low_freq_end])
        mid_freq_energy = torch.sum(energies[low_freq_end:high_freq_start])
        high_freq_energy = torch.sum(energies[high_freq_start:])

        # 計算有效帶寬
        energy_threshold = 0.01 * torch.max(energies)
        effective_bandwidth = torch.sum(energies > energy_threshold).item()

        # 計算譜衰減率
        log_energies = torch.log(energies + 1e-12)
        k_indices = torch.arange(len(energies), dtype=torch.float32, device=energies.device)

        # 線性回歸估計衰減率
        k_mean = torch.mean(k_indices)
        log_e_mean = torch.mean(log_energies)

        numerator = torch.sum((k_indices - k_mean) * (log_energies - log_e_mean))
        denominator = torch.sum((k_indices - k_mean) ** 2)

        decay_rate = numerator / (denominator + 1e-12)

        return {
            'total_energy': total_energy.item(),
            'low_freq_ratio': (low_freq_energy / total_energy).item(),
            'mid_freq_ratio': (mid_freq_energy / total_energy).item(),
            'high_freq_ratio': (high_freq_energy / total_energy).item(),
            'effective_bandwidth': effective_bandwidth,
            'spectral_decay_rate': decay_rate.item(),
            'recommended_order': min(max(64, effective_bandwidth * 2), self.max_order)
        }


class SpectralProjection:
    """
    統一的譜投影接口

    整合各種譜投影方法，提供統一的高級接口
    """

    def __init__(
        self,
        spatial_dims: Tuple[int, ...],
        spectral_order: int = 256,
        projection_type: str = 'fft',
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        初始化譜投影系統

        Args:
            spatial_dims: 空間維度
            spectral_order: 譜階數
            projection_type: 投影類型 ('fft', 'adaptive')
            device: 計算設備
            **kwargs: 額外參數
        """
        self.spatial_dims = spatial_dims
        self.spectral_order = spectral_order
        self.projection_type = projection_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 創建投影器
        if projection_type == 'fft':
            self.projector = FFTSpectralProjection(
                spatial_dims, spectral_order, device, **kwargs
            )
        elif projection_type == 'adaptive':
            self.projector = AdaptiveSpectralProjection(
                spatial_dims, spectral_order, device=device, **kwargs
            )
        else:
            raise ValueError(f"未知的投影類型: {projection_type}")

    def __call__(
        self,
        x: torch.Tensor,
        mode: str = 'forward'
    ) -> torch.Tensor:
        """
        執行譜投影

        Args:
            x: 輸入張量
            mode: 投影模式 ('forward', 'inverse')

        Returns:
            result: 投影結果
        """
        if mode == 'forward':
            return self.projector.forward_projection(x)
        elif mode == 'inverse':
            return self.projector.inverse_projection(x)
        else:
            raise ValueError(f"未知的投影模式: {mode}")

    def project_and_reconstruct(
        self,
        x: torch.Tensor,
        energy_threshold: float = 0.99
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        完整的投影-重建流程

        Args:
            x: 輸入函數
            energy_threshold: 能量保留閾值

        Returns:
            coefficients: 譜係數
            reconstructed: 重建函數
            info: 投影信息
        """
        if hasattr(self.projector, 'adaptive_projection'):
            # 自適應投影
            target_error = 1 - energy_threshold
            coeffs, reconstructed, optimal_order = self.projector.adaptive_projection(
                x, target_error
            )

            info = {
                'projection_type': 'adaptive',
                'optimal_order': optimal_order,
                'reconstruction_error': self.projector._compute_reconstruction_error(x, reconstructed)
            }

        else:
            # 標準投影
            coeffs, reconstructed, effective_order = self.projector.project_and_truncate(
                x, energy_threshold
            )

            info = {
                'projection_type': 'standard',
                'effective_order': effective_order,
                'energy_threshold': energy_threshold
            }

        return coeffs, reconstructed, info

    def compute_derivatives(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        計算函數的各階導數

        Args:
            x: 輸入函數

        Returns:
            derivatives: 導數字典
        """
        # 先投影到譜域
        coeffs = self.projector.forward_projection(x)

        # 在譜域計算導數
        if hasattr(self.projector, 'compute_spectral_derivatives'):
            return self.projector.compute_spectral_derivatives(coeffs)
        else:
            # 回退到數值微分
            return self._numerical_derivatives(x)

    def project_with_ode_optimization(
        self,
        x: torch.Tensor,
        target_sigma: float = 0.1,
        model_wrapper = None,
        optimization_steps: int = 10,
        extra_args: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        使用 VariationalODESystem 進行最佳化投影

        這是一個高級功能，通過求解變分 ODE 來找到最佳的譜係數

        Args:
            x: 輸入函數
            target_sigma: 目標噪聲水平（用於優化過程）
            model_wrapper: 模型包裝器（如果需要）
            optimization_steps: 優化步數
            extra_args: 額外參數

        Returns:
            optimized_coeffs: 優化後的譜係數
            reconstructed: 重建函數
            optimization_info: 優化信息
        """
        from .variational_ode_system import VariationalODESystem

        extra_args = extra_args or {}

        # 初始標準投影
        initial_coeffs = self.projector.forward_projection(x)

        if model_wrapper is None:
            # 沒有模型，返回標準投影
            reconstructed = self.projector.inverse_projection(initial_coeffs)
            return initial_coeffs, reconstructed, {'method': 'standard', 'optimization_used': False}

        try:
            # 創建 VariationalODESystem 用於優化
            ode_system = VariationalODESystem(
                spatial_dims=x.shape[-2:],
                spectral_order=min(self.spectral_order, initial_coeffs.shape[-1]),
                sobolev_order=1.5,
                regularization_lambda=0.001,  # 小的正則化用於投影優化
                sobolev_penalty=0.0001,
                device=x.device
            )

            # 通過變分 ODE 優化譜係數
            # 從高噪聲到低噪聲，引導係數到更好的配置
            optimized_coeffs, trajectory, solve_info = ode_system.solve_variational_ode(
                initial_coefficients=initial_coeffs,
                sigma_start=1.0,  # 高噪聲起點
                sigma_end=target_sigma,
                model_wrapper=model_wrapper,
                extra_args=extra_args,
                max_steps=optimization_steps,
                adaptive_stepping=True
            )

            # 重建優化後的函數
            reconstructed = ode_system.spectral_basis.reconstruct_from_coefficients(optimized_coeffs)

            optimization_info = {
                'method': 'variational_ode',
                'optimization_used': True,
                'converged': solve_info.get('converged', False),
                'optimization_steps': solve_info['final_step'],
                'initial_coeffs_norm': torch.norm(initial_coeffs).item(),
                'optimized_coeffs_norm': torch.norm(optimized_coeffs).item(),
                'reconstruction_improvement': self._compute_reconstruction_improvement(
                    x, initial_coeffs, optimized_coeffs, ode_system
                )
            }

            return optimized_coeffs, reconstructed, optimization_info

        except Exception as e:
            # 優化失敗，回退到標準投影
            reconstructed = self.projector.inverse_projection(initial_coeffs)
            return initial_coeffs, reconstructed, {
                'method': 'standard_fallback',
                'optimization_used': False,
                'error': str(e)
            }

    def _compute_reconstruction_improvement(
        self,
        original: torch.Tensor,
        initial_coeffs: torch.Tensor,
        optimized_coeffs: torch.Tensor,
        ode_system
    ) -> float:
        """
        計算重建改進程度
        """
        # 重建兩個版本
        initial_reconstruction = ode_system.spectral_basis.reconstruct_from_coefficients(initial_coeffs)
        optimized_reconstruction = ode_system.spectral_basis.reconstruct_from_coefficients(optimized_coeffs)

        # 計算重建誤差
        initial_error = torch.norm(original - initial_reconstruction)
        optimized_error = torch.norm(original - optimized_reconstruction)

        # 返回改進比例（負數表示變差）
        if initial_error > 1e-12:
            improvement = (initial_error - optimized_error) / initial_error
            return improvement.item()
        else:
            return 0.0

    def analyze_spectral_dynamics(
        self,
        x: torch.Tensor,
        model_wrapper,
        sigma_schedule: torch.Tensor,
        extra_args: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        分析譜動力學演化

        使用 VariationalODESystem 分析函數在不同噪聲水平下的譜演化

        Args:
            x: 輸入函數
            model_wrapper: 模型包裝器
            sigma_schedule: 噪聲水平序列
            extra_args: 額外參數

        Returns:
            dynamics_analysis: 動力學分析結果
        """
        from .variational_ode_system import VariationalODESystem

        extra_args = extra_args or {}

        try:
            # 創建 VariationalODESystem
            ode_system = VariationalODESystem(
                spatial_dims=x.shape[-2:],
                spectral_order=self.spectral_order,
                sobolev_order=1.5,
                device=x.device
            )

            # 初始投影
            initial_coeffs = ode_system.spectral_basis.project_to_basis(x)

            # 分析不同 sigma 水平下的動力學
            dynamics_results = {}

            for i, sigma in enumerate(sigma_schedule):
                # 計算在此 sigma 下的譜動力學
                rhs = ode_system.spectral_dynamics.compute_spectral_rhs(
                    initial_coeffs, sigma.item(), model_wrapper, extra_args
                )

                dynamics_results[f'sigma_{sigma.item():.3f}'] = {
                    'spectral_rhs': rhs,
                    'rhs_norm': torch.norm(rhs),
                    'dominant_modes': torch.argsort(torch.abs(rhs), dim=-1, descending=True)[..., :10],
                    'energy_distribution': torch.abs(rhs) ** 2
                }

            # 計算整體動力學特性
            all_rhs = torch.stack([dynamics_results[k]['spectral_rhs'] for k in dynamics_results.keys()])

            dynamics_analysis = {
                'sigma_values': sigma_schedule,
                'dynamics_per_sigma': dynamics_results,
                'temporal_evolution': {
                    'rhs_trajectory': all_rhs,
                    'energy_evolution': torch.stack([dynamics_results[k]['energy_distribution'] for k in dynamics_results.keys()]),
                    'dominant_mode_stability': self._analyze_mode_stability(dynamics_results),
                    'total_energy_trend': torch.stack([dynamics_results[k]['rhs_norm'] for k in dynamics_results.keys()])
                }
            }

            return dynamics_analysis

        except Exception as e:
            return {
                'error': str(e),
                'sigma_values': sigma_schedule,
                'analysis_failed': True
            }

    def _analyze_mode_stability(self, dynamics_results: Dict) -> torch.Tensor:
        """
        分析模式穩定性
        """
        dominant_modes_list = [dynamics_results[k]['dominant_modes'] for k in dynamics_results.keys()]

        if len(dominant_modes_list) > 1:
            # 計算主導模式的變化
            mode_changes = []
            for i in range(1, len(dominant_modes_list)):
                # 計算前後主導模式的重疊度
                prev_modes = set(dominant_modes_list[i-1].flatten().tolist())
                curr_modes = set(dominant_modes_list[i].flatten().tolist())
                overlap = len(prev_modes.intersection(curr_modes)) / len(prev_modes.union(curr_modes))
                mode_changes.append(overlap)

            return torch.tensor(mode_changes)
        else:
            return torch.tensor([1.0])  # 單一時間點，完全穩定

    def _numerical_derivatives(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        數值微分後備方案
        """
        derivatives = {}

        if x.dim() == 4:  # (B, C, H, W)
            # 使用 Sobel 算子
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                                 device=x.device, dtype=x.dtype)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                                 device=x.device, dtype=x.dtype)

            derivatives['dx'] = F.conv2d(x, sobel_x.expand(x.shape[1], 1, -1, -1),
                                       padding=1, groups=x.shape[1]) / 8.0
            derivatives['dy'] = F.conv2d(x, sobel_y.expand(x.shape[1], 1, -1, -1),
                                       padding=1, groups=x.shape[1]) / 8.0

        return derivatives

    def apply_filter(
        self,
        x: torch.Tensor,
        filter_type: str = 'lowpass',
        cutoff: float = 0.5
    ) -> torch.Tensor:
        """
        應用譜域濾波

        Args:
            x: 輸入函數
            filter_type: 濾波類型
            cutoff: 截止頻率

        Returns:
            filtered: 濾波後的函數
        """
        # 投影到譜域
        coeffs = self.projector.forward_projection(x)

        # 定義濾波函數
        if filter_type == 'lowpass':
            def filter_func(k1, k2):
                k_norm = math.sqrt(k1**2 + k2**2)
                max_k = math.sqrt(self.spectral_order)
                return 1.0 if k_norm <= cutoff * max_k else 0.0

        elif filter_type == 'highpass':
            def filter_func(k1, k2):
                k_norm = math.sqrt(k1**2 + k2**2)
                max_k = math.sqrt(self.spectral_order)
                return 0.0 if k_norm <= cutoff * max_k else 1.0

        elif filter_type == 'bandpass':
            def filter_func(k1, k2):
                k_norm = math.sqrt(k1**2 + k2**2)
                max_k = math.sqrt(self.spectral_order)
                normalized_k = k_norm / max_k
                return 1.0 if cutoff <= normalized_k <= 2*cutoff else 0.0
        else:
            raise ValueError(f"未知的濾波類型: {filter_type}")

        # 應用濾波
        if hasattr(self.projector, 'apply_spectral_filter'):
            filtered_coeffs = self.projector.apply_spectral_filter(coeffs, filter_func)
        else:
            # 簡化實現
            filtered_coeffs = coeffs  # 無濾波

        # 重建
        return self.projector.inverse_projection(filtered_coeffs)