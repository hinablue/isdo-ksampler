"""
Variational ODE System Implementation for ISDO
==============================================

實現變分 ODE 系統，將無窮維變分最優控制問題離散化為有限維 ODE 系統。
這是 ISDO 算法的核心計算引擎，負責在譜空間中求解動力學方程。

數學背景:
- 無窮維 ODE: dx/dσ = F(x, σ) 在希爾伯特空間中
- 譜截斷: x(σ) = Σ[k=1 to M] c_k(σ) φ_k
- 變分動力學: dc_k/dσ = G_k(c, σ)
- 正則化: 包含 Sobolev 範數約束
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Dict, List, Callable, Any
import math
from .hilbert_space import HilbertSpace
from .variational_controller import VariationalController
from .spectral_basis import SpectralBasis, BasisType
from ..samplers.unified_model_wrapper import UnifiedModelWrapper


class SpectralDynamics:
    """
    譜空間動力學系統

    實現譜係數的動力學方程：dc_k/dσ = G_k(c, σ)
    """

    def __init__(
        self,
        spectral_basis: SpectralBasis,
        hilbert_space: HilbertSpace,
        regularization_lambda: float = 1e-4,
        sobolev_penalty: float = 0.001
    ):
        """
        初始化譜動力學

        Args:
            spectral_basis: 譜基底
            hilbert_space: 希爾伯特空間
            regularization_lambda: 正則化參數
            sobolev_penalty: Sobolev 範數懲罰
        """
        self.spectral_basis = spectral_basis
        self.hilbert_space = hilbert_space
        self.lambda_reg = regularization_lambda
        self.sobolev_penalty = sobolev_penalty
        self.device = spectral_basis.device

    def compute_spectral_rhs(
        self,
        coefficients: torch.Tensor,
        sigma: float,
        model_wrapper: UnifiedModelWrapper,
        extra_args: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        計算譜係數的右端函數 dc/dσ = G(c, σ)

        Args:
            coefficients: 當前譜係數 (B, C, M)
            sigma: 當前噪聲水平
            model_wrapper: 統一模型包裝器
            extra_args: 額外參數

        Returns:
            rhs: 右端函數值 dc/dσ
        """
        extra_args = extra_args or {}

        # 從譜係數重建圖像
        x_current = self.spectral_basis.reconstruct_from_coefficients(coefficients)

        # 獲取去噪模型輸出
        f_denoiser = model_wrapper(x_current, sigma, **extra_args)

        # 計算理想漂移項在譜空間的投影
        drift_term = f_denoiser / (sigma + 1e-8)
        drift_coeffs = self.spectral_basis.project_to_basis(drift_term)

        # 變分修正項
        variational_correction = self._compute_variational_correction(
            coefficients, x_current, f_denoiser, sigma
        )

        # Sobolev 正則化項
        sobolev_correction = self._compute_sobolev_correction(coefficients)

        # 總的右端函數
        rhs = drift_coeffs + variational_correction + sobolev_correction

        return rhs

    def _compute_variational_correction(
        self,
        coefficients: torch.Tensor,
        x_current: torch.Tensor,
        f_denoiser: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """
        計算變分修正項

        這項來自於變分最優控制的 Euler-Lagrange 方程
        """
        # 計算 f 的梯度（數值近似）
        grad_f = self._approximate_gradient(f_denoiser)

        # 將梯度投影到譜空間
        grad_f_coeffs = self.spectral_basis.project_to_basis(grad_f)

        # 變分修正：-λ * 梯度項
        variational_correction = -self.lambda_reg * grad_f_coeffs

        return variational_correction

    def _compute_sobolev_correction(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        計算 Sobolev 正則化修正項

        實現譜空間中的 Sobolev 範數約束
        """
        if coefficients.dim() == 3:  # (B, C, M)
            B, C, M = coefficients.shape
        else:  # (M,)
            M = coefficients.shape[0]
            B, C = 1, 1

        # 計算每個模式的頻率權重
        # 高頻模式獲得更大的懲罰
        frequency_weights = torch.arange(1, M + 1, device=self.device, dtype=torch.float32)
        frequency_weights = frequency_weights ** self.hilbert_space.sobolev_order

        # Sobolev 懲罰：高頻模式衰減更快
        sobolev_correction = torch.zeros_like(coefficients)

        if coefficients.dim() == 3:
            for k in range(M):
                sobolev_correction[:, :, k] = -self.sobolev_penalty * frequency_weights[k] * coefficients[:, :, k]
        else:
            for k in range(M):
                sobolev_correction[k] = -self.sobolev_penalty * frequency_weights[k] * coefficients[k]

        return sobolev_correction

    def _approximate_gradient(self, f: torch.Tensor) -> torch.Tensor:
        """
        數值近似梯度
        """
        if f.dim() == 4:  # (B, C, H, W)
            # 使用 Sobel 算子
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                                 device=f.device, dtype=f.dtype)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                                 device=f.device, dtype=f.dtype)

            grad_x = F.conv2d(f, sobel_x.expand(f.shape[1], 1, -1, -1),
                            padding=1, groups=f.shape[1]) / 8.0
            grad_y = F.conv2d(f, sobel_y.expand(f.shape[1], 1, -1, -1),
                            padding=1, groups=f.shape[1]) / 8.0

            # 梯度模長
            grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-12)

        else:  # (H, W)
            # 簡單差分
            grad_x = torch.zeros_like(f)
            grad_y = torch.zeros_like(f)

            grad_x[1:-1, :] = (f[2:, :] - f[:-2, :]) / 2.0
            grad_y[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2.0

            grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-12)

        return grad_magnitude


class AdaptiveStepSizeController:
    """
    自適應步長控制器

    根據局部誤差和 Sobolev 範數動態調整積分步長
    """

    def __init__(
        self,
        initial_step_size: float = 0.01,
        min_step_size: float = 1e-6,
        max_step_size: float = 0.1,
        error_tolerance: float = 1e-4,
        safety_factor: float = 0.8
    ):
        """
        初始化自適應步長控制器

        Args:
            initial_step_size: 初始步長
            min_step_size: 最小步長
            max_step_size: 最大步長
            error_tolerance: 誤差容忍度
            safety_factor: 安全因子
        """
        self.h = initial_step_size
        self.h_min = min_step_size
        self.h_max = max_step_size
        self.tol = error_tolerance
        self.safety = safety_factor

    def adapt_step_size(
        self,
        error_estimate: torch.Tensor,
        current_coeffs: torch.Tensor,
        sobolev_norm: torch.Tensor
    ) -> float:
        """
        基於誤差估計調整步長

        Args:
            error_estimate: 局部截斷誤差估計
            current_coeffs: 當前譜係數
            sobolev_norm: 當前 Sobolev 範數

        Returns:
            new_step_size: 新步長
        """
        # 計算相對誤差
        relative_error = torch.norm(error_estimate) / (torch.norm(current_coeffs) + 1e-12)

        # 基於誤差的步長調整
        if relative_error > 0:
            step_factor = (self.tol / relative_error.item()) ** 0.2
        else:
            step_factor = 2.0  # 如果誤差很小，增大步長

        # 基於 Sobolev 範數的額外約束
        # 如果範數增長過快，減小步長
        if sobolev_norm > 100.0:  # 閾值可調
            step_factor *= 0.5
        elif sobolev_norm < 1.0:
            step_factor *= 1.2

        # 應用安全因子
        new_step_size = self.safety * step_factor * self.h

        # 限制在合理範圍內
        new_step_size = torch.clamp(
            torch.tensor(new_step_size),
            self.h_min,
            self.h_max
        ).item()

        self.h = new_step_size
        return new_step_size


class ConvergenceDetector:
    """
    收斂檢測器

    監控譜係數的收斂性，決定何時停止積分
    """

    def __init__(
        self,
        convergence_threshold: float = 1e-5,
        window_size: int = 5,
        min_iterations: int = 10
    ):
        """
        初始化收斂檢測器

        Args:
            convergence_threshold: 收斂閾值
            window_size: 滑動窗口大小
            min_iterations: 最小迭代數
        """
        self.threshold = convergence_threshold
        self.window_size = window_size
        self.min_iterations = min_iterations
        self.history = []

    def check_convergence(
        self,
        current_coeffs: torch.Tensor,
        iteration: int
    ) -> Tuple[bool, Dict[str, float]]:
        """
        檢查是否已收斂

        Args:
            current_coeffs: 當前譜係數
            iteration: 當前迭代數

        Returns:
            converged: 是否收斂
            convergence_info: 收斂信息
        """
        # 記錄歷史
        current_norm = torch.norm(current_coeffs).item()
        self.history.append(current_norm)

        # 保持窗口大小
        if len(self.history) > self.window_size:
            self.history.pop(0)

        convergence_info = {
            'current_norm': current_norm,
            'iteration': iteration,
            'window_size': len(self.history)
        }

        # 需要足夠的歷史數據
        if iteration < self.min_iterations or len(self.history) < self.window_size:
            return False, convergence_info

        # 計算變化率
        recent_changes = []
        for i in range(1, len(self.history)):
            change = abs(self.history[i] - self.history[i-1]) / (self.history[i-1] + 1e-12)
            recent_changes.append(change)

        avg_change_rate = np.mean(recent_changes)
        max_change_rate = np.max(recent_changes)

        convergence_info.update({
            'avg_change_rate': avg_change_rate,
            'max_change_rate': max_change_rate,
            'threshold': self.threshold
        })

        # 收斂判定
        converged = (avg_change_rate < self.threshold and
                    max_change_rate < self.threshold * 2)

        return converged, convergence_info


class VariationalODESystem:
    """
    變分 ODE 系統

    ISDO 算法的核心計算引擎，負責求解譜空間中的變分動力學方程。
    實現從無窮維變分問題到有限維 ODE 系統的完整轉換。
    """

    def __init__(
        self,
        spatial_dims: Tuple[int, ...],
        spectral_order: int = 256,
        sobolev_order: float = 1.5,
        regularization_lambda: float = 1e-4,
        sobolev_penalty: float = 0.001,
        basis_type: BasisType = BasisType.FOURIER,
        device: Optional[torch.device] = None
    ):
        """
        初始化變分 ODE 系統

        Args:
            spatial_dims: 空間維度 (H, W)
            spectral_order: 譜截斷階數
            sobolev_order: Sobolev 空間階數
            regularization_lambda: 正則化參數
            sobolev_penalty: Sobolev 懲罰係數
            basis_type: 基底類型
            device: 計算設備
        """
        self.spatial_dims = spatial_dims
        self.spectral_order = spectral_order
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化組件
        self.spectral_basis = SpectralBasis(
            spatial_dims=spatial_dims,
            spectral_order=spectral_order,
            basis_type=basis_type,
            device=self.device
        )

        self.hilbert_space = HilbertSpace(
            spatial_dims=spatial_dims,
            sobolev_order=sobolev_order,
            device=self.device
        )

        self.spectral_dynamics = SpectralDynamics(
            spectral_basis=self.spectral_basis,
            hilbert_space=self.hilbert_space,
            regularization_lambda=regularization_lambda,
            sobolev_penalty=sobolev_penalty
        )

        self.step_controller = AdaptiveStepSizeController()
        self.convergence_detector = ConvergenceDetector()

        # 統計信息
        self.stats = {
            'total_steps': 0,
            'adaptive_adjustments': 0,
            'convergence_checks': 0,
            'min_step_size': float('inf'),
            'max_step_size': 0.0
        }

    def solve_variational_ode(
        self,
        initial_coefficients: torch.Tensor,
        sigma_start: float,
        sigma_end: float,
        model_wrapper: UnifiedModelWrapper,
        extra_args: Optional[Dict] = None,
        max_steps: int = 1000,
        adaptive_stepping: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        求解變分 ODE 系統

        核心算法：在譜空間中積分 dc/dσ = G(c, σ)

        Args:
            initial_coefficients: 初始譜係數
            sigma_start: 起始噪聲水平
            sigma_end: 結束噪聲水平
            model_wrapper: 統一模型包裝器
            extra_args: 額外參數
            max_steps: 最大步數
            adaptive_stepping: 是否使用自適應步長

        Returns:
            final_coefficients: 最終譜係數
            solution_trajectory: 解軌跡
            solve_info: 求解信息
        """
        extra_args = extra_args or {}

        # 初始化
        coeffs_current = initial_coefficients.clone()
        sigma_current = sigma_start

        # 記錄軌跡
        trajectory_coeffs = [coeffs_current.clone()]
        trajectory_sigma = [sigma_current]

        solve_info = {
            'converged': False,
            'final_step': 0,
            'final_sigma': sigma_current,
            'convergence_history': [],
            'step_size_history': [],
            'sobolev_norm_history': []
        }

        # 主積分循環
        for step in range(max_steps):
            # 計算當前 Sobolev 範數
            current_x = self.spectral_basis.reconstruct_from_coefficients(coeffs_current)
            sobolev_norm = self.hilbert_space.compute_sobolev_norm(current_x)

            # 記錄統計
            solve_info['sobolev_norm_history'].append(sobolev_norm.item())
            self.stats['total_steps'] += 1

            # 檢查收斂
            converged, conv_info = self.convergence_detector.check_convergence(
                coeffs_current, step
            )
            solve_info['convergence_history'].append(conv_info)
            self.stats['convergence_checks'] += 1

            if converged and step > 10:
                solve_info['converged'] = True
                solve_info['convergence_reason'] = 'coefficients_converged'
                break

            # 檢查是否到達目標
            if abs(sigma_current - sigma_end) < 1e-8:
                solve_info['convergence_reason'] = 'target_sigma_reached'
                break

            # 計算右端函數
            rhs = self.spectral_dynamics.compute_spectral_rhs(
                coeffs_current, sigma_current, model_wrapper, extra_args
            )

            # 自適應步長控制
            if adaptive_stepping:
                # 估計局部誤差（使用嵌入式 RK 方法）
                error_estimate = self._estimate_local_error(
                    coeffs_current, rhs, sigma_current, model_wrapper, extra_args
                )

                # 調整步長
                step_size = self.step_controller.adapt_step_size(
                    error_estimate, coeffs_current, sobolev_norm
                )
                self.stats['adaptive_adjustments'] += 1
            else:
                step_size = (sigma_end - sigma_start) / max_steps

            # 記錄步長統計
            solve_info['step_size_history'].append(step_size)
            self.stats['min_step_size'] = min(self.stats['min_step_size'], step_size)
            self.stats['max_step_size'] = max(self.stats['max_step_size'], step_size)

            # 執行積分步驟（使用 RK4）
            coeffs_next, sigma_next = self._rk4_step(
                coeffs_current, sigma_current, step_size,
                model_wrapper, extra_args
            )

            # 更新狀態
            coeffs_current = coeffs_next
            sigma_current = sigma_next

            # 記錄軌跡
            trajectory_coeffs.append(coeffs_current.clone())
            trajectory_sigma.append(sigma_current)

            # 檢查數值穩定性
            if torch.isnan(coeffs_current).any() or torch.isinf(coeffs_current).any():
                solve_info['convergence_reason'] = 'numerical_instability'
                break

        # 最終處理
        solve_info['final_step'] = step
        solve_info['final_sigma'] = sigma_current

        # 組織軌跡
        solution_trajectory = {
            'coefficients': torch.stack(trajectory_coeffs),
            'sigma_values': torch.tensor(trajectory_sigma, device=self.device),
            'spatial_trajectory': self._reconstruct_spatial_trajectory(trajectory_coeffs)
        }

        return coeffs_current, solution_trajectory, solve_info

    def _estimate_local_error(
        self,
        coeffs: torch.Tensor,
        rhs: torch.Tensor,
        sigma: float,
        model_wrapper: UnifiedModelWrapper,
        extra_args: Dict,
        h_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        估計局部截斷誤差

        使用兩種不同步長的結果比較
        """
        h = self.step_controller.h

        # 全步長
        k1 = rhs
        k2 = self.spectral_dynamics.compute_spectral_rhs(
            coeffs + h * k1 / 2, sigma - h/2, model_wrapper, extra_args
        )
        k3 = self.spectral_dynamics.compute_spectral_rhs(
            coeffs + h * k2 / 2, sigma - h/2, model_wrapper, extra_args
        )
        k4 = self.spectral_dynamics.compute_spectral_rhs(
            coeffs + h * k3, sigma - h, model_wrapper, extra_args
        )

        full_step = coeffs + h * (k1 + 2*k2 + 2*k3 + k4) / 6

        # 兩個半步長
        h_half = h * h_ratio

        # 第一個半步
        k1_1 = rhs
        k2_1 = self.spectral_dynamics.compute_spectral_rhs(
            coeffs + h_half * k1_1 / 2, sigma - h_half/2, model_wrapper, extra_args
        )
        k3_1 = self.spectral_dynamics.compute_spectral_rhs(
            coeffs + h_half * k2_1 / 2, sigma - h_half/2, model_wrapper, extra_args
        )
        k4_1 = self.spectral_dynamics.compute_spectral_rhs(
            coeffs + h_half * k3_1, sigma - h_half, model_wrapper, extra_args
        )

        half_step_1 = coeffs + h_half * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1) / 6

        # 第二個半步
        k1_2 = self.spectral_dynamics.compute_spectral_rhs(
            half_step_1, sigma - h_half, model_wrapper, extra_args
        )
        k2_2 = self.spectral_dynamics.compute_spectral_rhs(
            half_step_1 + h_half * k1_2 / 2, sigma - h_half - h_half/2, model_wrapper, extra_args
        )
        k3_2 = self.spectral_dynamics.compute_spectral_rhs(
            half_step_1 + h_half * k2_2 / 2, sigma - h_half - h_half/2, model_wrapper, extra_args
        )
        k4_2 = self.spectral_dynamics.compute_spectral_rhs(
            half_step_1 + h_half * k3_2, sigma - h, model_wrapper, extra_args
        )

        two_half_steps = half_step_1 + h_half * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2) / 6

        # 誤差估計
        error_estimate = two_half_steps - full_step

        return error_estimate

    def _rk4_step(
        self,
        coeffs: torch.Tensor,
        sigma: float,
        step_size: float,
        model_wrapper: UnifiedModelWrapper,
        extra_args: Dict
    ) -> Tuple[torch.Tensor, float]:
        """
        執行一步 RK4 積分
        """
        h = step_size

        # RK4 階段
        k1 = self.spectral_dynamics.compute_spectral_rhs(
            coeffs, sigma, model_wrapper, extra_args
        )

        k2 = self.spectral_dynamics.compute_spectral_rhs(
            coeffs + h * k1 / 2, sigma - h/2, model_wrapper, extra_args
        )

        k3 = self.spectral_dynamics.compute_spectral_rhs(
            coeffs + h * k2 / 2, sigma - h/2, model_wrapper, extra_args
        )

        k4 = self.spectral_dynamics.compute_spectral_rhs(
            coeffs + h * k3, sigma - h, model_wrapper, extra_args
        )

        # 更新
        coeffs_next = coeffs + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        sigma_next = sigma - h

        return coeffs_next, sigma_next

    def _reconstruct_spatial_trajectory(self, trajectory_coeffs: List[torch.Tensor]) -> torch.Tensor:
        """
        從譜係數軌跡重建空間軌跡
        """
        spatial_trajectory = []

        for coeffs in trajectory_coeffs:
            x_spatial = self.spectral_basis.reconstruct_from_coefficients(coeffs)
            spatial_trajectory.append(x_spatial)

        return torch.stack(spatial_trajectory)

    def analyze_solution_quality(
        self,
        solution_trajectory: Dict,
        solve_info: Dict
    ) -> Dict[str, Any]:
        """
        分析解的質量

        Args:
            solution_trajectory: 解軌跡
            solve_info: 求解信息

        Returns:
            quality_analysis: 質量分析結果
        """
        analysis = {}

        # 收斂性分析
        analysis['convergence'] = {
            'converged': solve_info['converged'],
            'final_step': solve_info['final_step'],
            'convergence_rate': self._estimate_convergence_rate(solve_info['convergence_history'])
        }

        # Sobolev 範數分析
        sobolev_history = solve_info['sobolev_norm_history']
        analysis['sobolev_analysis'] = {
            'initial_norm': sobolev_history[0] if sobolev_history else 0,
            'final_norm': sobolev_history[-1] if sobolev_history else 0,
            'max_norm': max(sobolev_history) if sobolev_history else 0,
            'norm_stability': np.std(sobolev_history) if len(sobolev_history) > 1 else 0
        }

        # 步長分析
        step_history = solve_info['step_size_history']
        analysis['step_size_analysis'] = {
            'avg_step_size': np.mean(step_history) if step_history else 0,
            'min_step_size': min(step_history) if step_history else 0,
            'max_step_size': max(step_history) if step_history else 0,
            'step_adaptations': len([s for i, s in enumerate(step_history[1:])
                                   if abs(s - step_history[i]) > 1e-10])
        }

        # 譜內容分析
        coeffs_trajectory = solution_trajectory['coefficients']
        analysis['spectral_analysis'] = self._analyze_spectral_evolution(coeffs_trajectory)

        # 數值穩定性
        analysis['numerical_stability'] = {
            'max_coefficient': torch.max(torch.abs(coeffs_trajectory)).item(),
            'condition_number': self._estimate_condition_number(coeffs_trajectory),
            'energy_conservation': self._check_energy_conservation(coeffs_trajectory)
        }

        return analysis

    def _estimate_convergence_rate(self, convergence_history: List[Dict]) -> float:
        """
        估計收斂率
        """
        if len(convergence_history) < 10:
            return 0.0

        # 提取範數變化
        norms = [info.get('current_norm', 0) for info in convergence_history[-10:]]

        # 計算衰減率
        if len(norms) >= 2:
            ratios = [norms[i] / norms[i-1] if norms[i-1] > 1e-12 else 1.0
                     for i in range(1, len(norms))]
            avg_ratio = np.mean(ratios)
            convergence_rate = -np.log(max(avg_ratio, 1e-12))
            return convergence_rate

        return 0.0

    def _analyze_spectral_evolution(self, coeffs_trajectory: torch.Tensor) -> Dict:
        """
        分析譜演化
        """
        T = coeffs_trajectory.shape[0]

        # 計算每個時刻的譜能量分佈
        energy_evolution = []
        for t in range(T):
            coeffs_t = coeffs_trajectory[t]
            if coeffs_t.dim() == 3:  # (B, C, M)
                energies = torch.mean(torch.abs(coeffs_t) ** 2, dim=(0, 1))  # (M,)
            else:  # (M,)
                energies = torch.abs(coeffs_t) ** 2

            energy_evolution.append(energies)

        energy_evolution = torch.stack(energy_evolution)  # (T, M)

        # 分析高低頻演化
        M = energy_evolution.shape[1]
        low_freq_end = M // 4
        high_freq_start = 3 * M // 4

        low_freq_energy = torch.sum(energy_evolution[:, :low_freq_end], dim=1)
        high_freq_energy = torch.sum(energy_evolution[:, high_freq_start:], dim=1)
        total_energy = torch.sum(energy_evolution, dim=1)

        return {
            'initial_low_freq_ratio': (low_freq_energy[0] / total_energy[0]).item(),
            'final_low_freq_ratio': (low_freq_energy[-1] / total_energy[-1]).item(),
            'initial_high_freq_ratio': (high_freq_energy[0] / total_energy[0]).item(),
            'final_high_freq_ratio': (high_freq_energy[-1] / total_energy[-1]).item(),
            'energy_conservation_error': (total_energy[-1] / total_energy[0] - 1.0).item(),
            'spectral_diffusion_rate': torch.mean(torch.diff(high_freq_energy)).item()
        }

    def _estimate_condition_number(self, coeffs_trajectory: torch.Tensor) -> float:
        """
        估計條件數
        """
        # 計算係數矩陣的條件數
        if coeffs_trajectory.dim() == 4:  # (T, B, C, M)
            coeffs_matrix = coeffs_trajectory.view(coeffs_trajectory.shape[0], -1)  # (T, B*C*M)
        else:  # (T, M)
            coeffs_matrix = coeffs_trajectory

        try:
            # SVD 分解
            U, S, V = torch.svd(coeffs_matrix)
            condition_number = (S[0] / S[-1]).item() if S[-1] > 1e-12 else float('inf')
            return condition_number
        except:
            return float('inf')

    def _check_energy_conservation(self, coeffs_trajectory: torch.Tensor) -> float:
        """
        檢查能量守恆
        """
        # 計算每個時刻的總能量
        energies = []
        for t in range(coeffs_trajectory.shape[0]):
            coeffs_t = coeffs_trajectory[t]
            energy = torch.sum(torch.abs(coeffs_t) ** 2).item()
            energies.append(energy)

        if len(energies) >= 2:
            energy_change = abs(energies[-1] - energies[0]) / (energies[0] + 1e-12)
            return energy_change

        return 0.0

    def get_system_statistics(self) -> Dict[str, Any]:
        """
        獲取系統統計信息
        """
        return {
            'system_config': {
                'spatial_dims': self.spatial_dims,
                'spectral_order': self.spectral_order,
                'sobolev_order': self.hilbert_space.sobolev_order,
                'basis_type': self.spectral_basis.basis_type.value
            },
            'runtime_stats': self.stats.copy(),
            'component_info': {
                'spectral_basis_normalization': self.spectral_basis.normalization_factors.cpu().numpy().tolist()[:10],  # 前10個
                'hilbert_space_embedding': self.hilbert_space.is_embedded_in_continuous,
                'current_step_size': self.step_controller.h,
                'convergence_window': self.convergence_detector.window_size
            }
        }

    def reset_statistics(self):
        """
        重置統計計數器
        """
        self.stats = {
            'total_steps': 0,
            'adaptive_adjustments': 0,
            'convergence_checks': 0,
            'min_step_size': float('inf'),
            'max_step_size': 0.0
        }

        # 重置組件狀態
        self.step_controller = AdaptiveStepSizeController()
        self.convergence_detector = ConvergenceDetector()