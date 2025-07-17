"""
Variational Controller Implementation for ISDO
=============================================

實現變分最優控制問題的核心算法，將擴散採樣轉化為最優控制問題。
包括動作積分計算、Euler-Lagrange 方程求解和 Hamilton-Jacobi-Bellman 方程。

數學背景:
- 變分最優控制: min ∫ L(x, ẋ, σ) dσ
- Euler-Lagrange 方程: d/dσ(∂L/∂ẋ) - ∂L/∂x = 0
- Hamilton-Jacobi-Bellman: ∂V/∂σ + H(x, ∇V, σ) = 0
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Dict, List, Callable
import math

# 延遲導入避免循環依賴
def _get_variational_ode_system():
    from .variational_ode_system import VariationalODESystem
    return VariationalODESystem

def _get_hilbert_space():
    from .hilbert_space import HilbertSpace
    return HilbertSpace


class ActionIntegral:
    """
    動作積分計算器

    實現變分最優控制中的動作泛函:
    𝒜[x] = ∫[σ_max to 0] ℒ(x, ẋ, σ) dσ
    """

    def __init__(
        self,
        regularization_lambda: float = 0.01,
        curvature_penalty: float = 0.001,
        domain_size: Tuple[float, ...] = (1.0, 1.0)
    ):
        """
        初始化動作積分

        Args:
            regularization_lambda: 正則化參數 λ
            curvature_penalty: 曲率懲罰係數
            domain_size: 定義域大小
        """
        self.lambda_reg = regularization_lambda
        self.curvature_penalty = curvature_penalty
        self.domain_size = domain_size

    def compute_lagrangian(
        self,
        x: torch.Tensor,
        x_dot: torch.Tensor,
        f_denoiser: torch.Tensor,
        sigma: float,
        grad_f: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        計算拉格朗日函數 ℒ(x, ẋ, σ)

        ℒ = ½|ẋ - f(x;σ)|²_H + λ|∇_x f|²_op + μ|∇²x|²

        Args:
            x: 當前狀態
            x_dot: 狀態導數 dx/dσ
            f_denoiser: 去噪函數輸出 f(x;σ)
            sigma: 噪聲水平
            grad_f: f 的梯度 (可選)

        Returns:
            lagrangian: 拉格朗日函數值
        """
        batch_size = x.shape[0] if x.dim() > 2 else 1

        # 主要項: ½|ẋ - f|²_H
        drift_term = f_denoiser / sigma if sigma > 1e-8 else f_denoiser
        velocity_error = x_dot - drift_term

        # L² 範數的主要貢獻
        main_term = 0.5 * torch.sum(velocity_error ** 2, dim=(-2, -1))

        # 正則化項: λ|∇_x f|²
        regularization_term = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if grad_f is not None:
            grad_norm_sq = torch.sum(grad_f ** 2, dim=(-2, -1))
            regularization_term = self.lambda_reg * grad_norm_sq
        else:
            # 數值近似梯度
            grad_f_approx = self._approximate_gradient(f_denoiser)
            grad_norm_sq = torch.sum(grad_f_approx ** 2, dim=(-2, -1))
            regularization_term = self.lambda_reg * grad_norm_sq

        # 曲率懲罰項: μ|∇²x|²
        curvature_term = self._compute_curvature_penalty(x)

        lagrangian = main_term + regularization_term + self.curvature_penalty * curvature_term

        # 確保形狀一致性
        if lagrangian.dim() == 0:
            lagrangian = lagrangian.unsqueeze(0)
        if batch_size > 1 and lagrangian.shape[0] != batch_size:
            lagrangian = lagrangian.expand(batch_size)

        return lagrangian

    def _approximate_gradient(self, f: torch.Tensor) -> torch.Tensor:
        """
        數值近似梯度計算
        """
        # 使用 Sobel 算子近似梯度
        if f.dim() == 4:  # (B, C, H, W)
            # x 方向梯度
            grad_x = F.conv2d(
                f,
                torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                           device=f.device, dtype=f.dtype).expand(f.shape[1], 1, -1, -1),
                padding=1,
                groups=f.shape[1]
            ) / 8.0

            # y 方向梯度
            grad_y = F.conv2d(
                f,
                torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                           device=f.device, dtype=f.dtype).expand(f.shape[1], 1, -1, -1),
                padding=1,
                groups=f.shape[1]
            ) / 8.0

            # 梯度模長
            grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-12)

        elif f.dim() == 2:  # (H, W)
            # 簡單差分
            grad_x = torch.zeros_like(f)
            grad_y = torch.zeros_like(f)

            grad_x[1:-1, :] = (f[2:, :] - f[:-2, :]) / 2.0
            grad_y[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2.0

            grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-12)
        else:
            raise ValueError(f"不支援的張量維度: {f.shape}")

        return grad_magnitude

    def _compute_curvature_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        計算曲率懲罰項 |∇²x|²
        """
        if x.dim() == 4:  # (B, C, H, W)
            # 二階導數 (拉普拉斯算子)
            laplacian = F.conv2d(
                x,
                torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                           device=x.device, dtype=x.dtype).expand(x.shape[1], 1, -1, -1),
                padding=1,
                groups=x.shape[1]
            )

        elif x.dim() == 2:  # (H, W)
            # 手動計算拉普拉斯算子
            laplacian = torch.zeros_like(x)
            laplacian[1:-1, 1:-1] = (
                x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2] - 4 * x[1:-1, 1:-1]
            )
        else:
            raise ValueError(f"不支援的張量維度: {x.shape}")

        curvature_norm_sq = torch.sum(laplacian ** 2, dim=(-2, -1))
        return curvature_norm_sq

    def compute_action(
        self,
        trajectory: torch.Tensor,
        sigma_schedule: torch.Tensor,
        denoiser_function: Callable,
        extra_args: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        計算整個軌跡的動作積分

        𝒜[x] = ∫[σ_max to 0] ℒ(x, ẋ, σ) dσ

        Args:
            trajectory: 軌跡 (T, B, C, H, W)
            sigma_schedule: 噪聲水平序列 (T,)
            denoiser_function: 去噪函數
            extra_args: 額外參數

        Returns:
            action: 動作值
        """
        extra_args = extra_args or {}
        T = len(sigma_schedule)

        # 計算軌跡導數
        x_dot = self._compute_trajectory_derivative(trajectory, sigma_schedule)

        action = torch.tensor(0.0, device=trajectory.device, dtype=trajectory.dtype)

        for t in range(T - 1):
            x_t = trajectory[t]
            x_dot_t = x_dot[t]
            sigma_t = sigma_schedule[t]

            # 獲取去噪函數輸出
            s_in = torch.ones(x_t.shape[0], device=x_t.device) * sigma_t
            f_t = denoiser_function(x_t, s_in, **extra_args)

            # 計算拉格朗日函數
            lagrangian_t = self.compute_lagrangian(x_t, x_dot_t, f_t, sigma_t.item())

            # 積分 (梯形法則)
            dt = (sigma_schedule[t] - sigma_schedule[t + 1]).abs()
            action = action + lagrangian_t * dt

        return action

    def _compute_trajectory_derivative(
        self,
        trajectory: torch.Tensor,
        sigma_schedule: torch.Tensor
    ) -> torch.Tensor:
        """
        計算軌跡的時間導數 dx/dσ
        """
        T = trajectory.shape[0]
        x_dot = torch.zeros_like(trajectory)

        # 中心差分
        for t in range(1, T - 1):
            dt_forward = (sigma_schedule[t] - sigma_schedule[t + 1]).abs()
            dt_backward = (sigma_schedule[t - 1] - sigma_schedule[t]).abs()

            if dt_forward > 1e-12 and dt_backward > 1e-12:
                # 自適應差分權重
                w_f = dt_backward / (dt_forward + dt_backward)
                w_b = dt_forward / (dt_forward + dt_backward)

                x_dot[t] = w_f * (trajectory[t + 1] - trajectory[t]) / dt_forward + \
                          w_b * (trajectory[t] - trajectory[t - 1]) / dt_backward

        # 邊界條件
        if T > 1:
            dt_0 = (sigma_schedule[0] - sigma_schedule[1]).abs()
            if dt_0 > 1e-12:
                x_dot[0] = (trajectory[1] - trajectory[0]) / dt_0

            dt_T = (sigma_schedule[T-2] - sigma_schedule[T-1]).abs()
            if dt_T > 1e-12:
                x_dot[T-1] = (trajectory[T-1] - trajectory[T-2]) / dt_T

        return x_dot


class EulerLagrangeSystem:
    """
    Euler-Lagrange 方程系統求解器

    求解變分問題的必要條件:
    d/dσ(∂ℒ/∂ẋ) - ∂ℒ/∂x = 0
    """

    def __init__(
        self,
        action_integral: ActionIntegral,
        numerical_epsilon: float = 1e-6
    ):
        """
        初始化 Euler-Lagrange 系統

        Args:
            action_integral: 動作積分計算器
            numerical_epsilon: 數值微分精度
        """
        self.action_integral = action_integral
        self.eps = numerical_epsilon

    def compute_euler_lagrange_residual(
        self,
        x: torch.Tensor,
        x_dot: torch.Tensor,
        x_ddot: torch.Tensor,
        f_denoiser: torch.Tensor,
        sigma: float,
        grad_f: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        計算 Euler-Lagrange 方程的殘差

        Args:
            x: 當前狀態
            x_dot: 一階導數
            x_ddot: 二階導數
            f_denoiser: 去噪函數
            sigma: 噪聲水平
            grad_f: f 的梯度

        Returns:
            residual: EL 方程殘差
        """
        # 計算 ∂ℒ/∂ẋ
        partial_L_partial_xdot = self._compute_partial_L_partial_xdot(
            x, x_dot, f_denoiser, sigma
        )

        # 計算 d/dσ(∂ℒ/∂ẋ) (近似為時間導數)
        d_partial_L_partial_xdot = x_ddot - f_denoiser / (sigma + 1e-8)

        # 計算 ∂ℒ/∂x
        partial_L_partial_x = self._compute_partial_L_partial_x(
            x, x_dot, f_denoiser, sigma, grad_f
        )

        # Euler-Lagrange 方程: d/dσ(∂ℒ/∂ẋ) - ∂ℒ/∂x = 0
        residual = d_partial_L_partial_xdot - partial_L_partial_x

        return residual

    def _compute_partial_L_partial_xdot(
        self,
        x: torch.Tensor,
        x_dot: torch.Tensor,
        f_denoiser: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """
        計算 ∂ℒ/∂ẋ

        對於 ℒ = ½|ẋ - f|², 有 ∂ℒ/∂ẋ = ẋ - f
        """
        drift_term = f_denoiser / (sigma + 1e-8)
        return x_dot - drift_term

    def _compute_partial_L_partial_x(
        self,
        x: torch.Tensor,
        x_dot: torch.Tensor,
        f_denoiser: torch.Tensor,
        sigma: float,
        grad_f: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        計算 ∂ℒ/∂x (使用數值微分)
        """
        # 由於 ℒ 對 x 的依賴主要通過 f(x;σ)，需要計算 ∂f/∂x
        # 這需要對去噪模型進行數值微分

        if grad_f is not None:
            # 如果提供了解析梯度
            partial_f_partial_x = grad_f
        else:
            # 數值微分近似
            partial_f_partial_x = self._numerical_gradient_f(x, f_denoiser)

        # ∂ℒ/∂x ≈ -(ẋ - f/σ) · ∂f/∂x / σ + 正則化項
        drift_term = f_denoiser / (sigma + 1e-8)
        velocity_error = x_dot - drift_term

        # 主要項
        main_term = -torch.sum(
            velocity_error.unsqueeze(-1).unsqueeze(-1) * partial_f_partial_x,
            dim=1, keepdim=True
        ) / (sigma + 1e-8)

        # 正則化項的梯度
        reg_term = self.action_integral.lambda_reg * self._compute_regularization_gradient(x)

        # 曲率懲罰項的梯度
        curvature_grad = self.action_integral.curvature_penalty * self._compute_curvature_gradient(x)

        return main_term + reg_term + curvature_grad

    def _numerical_gradient_f(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        數值計算 ∂f/∂x
        """
        # 簡化實現：使用有限差分
        grad_f = torch.zeros_like(x)

        if x.dim() == 4:  # (B, C, H, W)
            # x 方向
            grad_f[:, :, 1:-1, :] = (f[:, :, 2:, :] - f[:, :, :-2, :]) / (2 * self.eps)
            # y 方向 (簡化為只計算 x 方向)
            grad_f[:, :, :, 1:-1] += (f[:, :, :, 2:] - f[:, :, :, :-2]) / (2 * self.eps)

        return grad_f

    def _compute_regularization_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        計算正則化項的梯度
        """
        # 簡化實現
        return torch.zeros_like(x)

    def _compute_curvature_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        計算曲率懲罰項的梯度
        """
        # ∇(|∇²x|²) = 2∇²(∇²x) = 2∇⁴x
        # 簡化為拉普拉斯算子的拉普拉斯

        if x.dim() == 4:  # (B, C, H, W)
            # 四階微分算子 (簡化實現)
            biharmonic = F.conv2d(
                x,
                torch.tensor([[[[0, 0, 1, 0, 0],
                              [0, 2, -8, 2, 0],
                              [1, -8, 20, -8, 1],
                              [0, 2, -8, 2, 0],
                              [0, 0, 1, 0, 0]]]],
                           device=x.device, dtype=x.dtype).expand(x.shape[1], 1, -1, -1),
                padding=2,
                groups=x.shape[1]
            )
        else:
            biharmonic = torch.zeros_like(x)

        return 2 * biharmonic


class VariationalController:
    """
    變分最優控制器

    ISDO 算法的核心控制器，實現從噪聲到數據的最優路徑求解。
    """

    def __init__(
        self,
        spatial_dims: Tuple[int, ...],
        regularization_lambda: float = 0.01,
        curvature_penalty: float = 0.001,
        sobolev_order: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        初始化變分控制器

        Args:
            spatial_dims: 空間維度
            regularization_lambda: 正則化參數
            curvature_penalty: 曲率懲罰
            sobolev_order: Sobolev 空間階數
            device: 計算設備
        """
        self.spatial_dims = spatial_dims
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化組件
        self.action_integral = ActionIntegral(
            regularization_lambda=regularization_lambda,
            curvature_penalty=curvature_penalty,
            domain_size=(1.0, 1.0)
        )

        self.euler_lagrange_system = EulerLagrangeSystem(self.action_integral)

        self.hilbert_space = _get_hilbert_space()(
            spatial_dims=spatial_dims,
            sobolev_order=sobolev_order,
            device=device
        )

    def compute_optimal_control(
        self,
        x_current: torch.Tensor,
        sigma_current: float,
        denoiser_function: Callable,
        target_sigma: float,
        num_steps: int = 10,
        extra_args: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算最優控制軌跡

        求解變分問題: min ∫ ℒ(x, ẋ, σ) dσ

        Args:
            x_current: 當前狀態
            sigma_current: 當前噪聲水平
            denoiser_function: 去噪函數
            target_sigma: 目標噪聲水平
            num_steps: 積分步數
            extra_args: 額外參數

        Returns:
            optimal_trajectory: 最優軌跡
            optimal_controls: 最優控制序列
        """
        extra_args = extra_args or {}

        # 生成 σ 調度
        sigma_schedule = torch.linspace(
            sigma_current, target_sigma, num_steps + 1, device=self.device
        )

        # 初始化軌跡
        trajectory = torch.zeros(
            num_steps + 1, *x_current.shape, device=self.device, dtype=x_current.dtype
        )
        trajectory[0] = x_current

        controls = torch.zeros_like(trajectory[:-1])

        # 使用變分原理求解最優軌跡
        for t in range(num_steps):
            x_t = trajectory[t]
            sigma_t = sigma_schedule[t]
            sigma_next = sigma_schedule[t + 1]

            # 計算最優控制
            u_optimal = self._solve_optimal_control_step(
                x_t, sigma_t, sigma_next, denoiser_function, extra_args
            )

            controls[t] = u_optimal

            # 更新狀態
            dt = sigma_next - sigma_t
            trajectory[t + 1] = x_t + u_optimal * dt

        return trajectory, controls

    def compute_optimal_control_with_ode_system(
        self,
        x_current: torch.Tensor,
        sigma_current: float,
        denoiser_function: Callable,
        target_sigma: float,
        spectral_order: int = 256,
        extra_args: Optional[Dict] = None,
        use_adaptive_stepping: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        使用 VariationalODESystem 計算高精度最優控制軌跡

        這是改進的求解方法，使用完整的變分 ODE 系統

        Args:
            x_current: 當前狀態
            sigma_current: 當前噪聲水平
            denoiser_function: 去噪函數
            target_sigma: 目標噪聲水平
            spectral_order: 譜截斷階數
            extra_args: 額外參數
            use_adaptive_stepping: 是否使用自適應步長

        Returns:
            optimal_trajectory: 最優軌跡
            trajectory_info: 軌跡詳細信息
            solve_info: 求解統計信息
        """
        from ..samplers.unified_model_wrapper import UnifiedModelWrapper

        extra_args = extra_args or {}

        # 創建 VariationalODESystem 實例
        VariationalODESystem = _get_variational_ode_system()
        ode_system = VariationalODESystem(
            spatial_dims=x_current.shape[-2:],
            spectral_order=spectral_order,
            sobolev_order=self.hilbert_space.sobolev_order,
            regularization_lambda=self.action_integral.lambda_reg,
            sobolev_penalty=self.action_integral.curvature_penalty,
            device=self.device
        )

        # 包裝去噪函數
        if not isinstance(denoiser_function, UnifiedModelWrapper):
            model_wrapper = UnifiedModelWrapper(denoiser_function, device=self.device)
        else:
            model_wrapper = denoiser_function

        # 投影到譜空間
        initial_coefficients = ode_system.spectral_basis.project_to_basis(x_current)

        # 使用 VariationalODESystem 求解
        final_coefficients, solution_trajectory, solve_info = ode_system.solve_variational_ode(
            initial_coefficients=initial_coefficients,
            sigma_start=sigma_current,
            sigma_end=target_sigma,
            model_wrapper=model_wrapper,
            extra_args=extra_args,
            max_steps=100,
            adaptive_stepping=use_adaptive_stepping
        )

        # 重建空間軌跡
        spatial_trajectory = solution_trajectory['spatial_trajectory']

        # 計算控制序列（從軌跡推導）
        controls = self._extract_controls_from_trajectory(
            spatial_trajectory, solution_trajectory['sigma_values'], model_wrapper, extra_args
        )

        # 組織返回信息
        trajectory_info = {
            'spectral_coefficients': solution_trajectory['coefficients'],
            'sigma_values': solution_trajectory['sigma_values'],
            'sobolev_norms': solve_info['sobolev_norm_history'],
            'convergence_history': solve_info['convergence_history'],
            'step_sizes': solve_info['step_size_history']
        }

        return spatial_trajectory, trajectory_info, solve_info

    def _extract_controls_from_trajectory(
        self,
        trajectory: torch.Tensor,
        sigma_values: torch.Tensor,
        model_wrapper,
        extra_args: Dict
    ) -> torch.Tensor:
        """
        從最優軌跡中提取控制序列

        Args:
            trajectory: 空間軌跡 (T, B, C, H, W)
            sigma_values: 對應的 σ 值
            model_wrapper: 模型包裝器
            extra_args: 額外參數

        Returns:
            controls: 控制序列 (T-1, B, C, H, W)
        """
        T = trajectory.shape[0]
        controls = torch.zeros_like(trajectory[:-1])

        for t in range(T - 1):
            x_t = trajectory[t]
            x_next = trajectory[t + 1]
            sigma_t = sigma_values[t]
            sigma_next = sigma_values[t + 1]

            # 計算時間步長
            dt = sigma_next - sigma_t

            if abs(dt) > 1e-12:
                # 控制 u = dx/dt
                controls[t] = (x_next - x_t) / dt
            else:
                # 極小步長，使用模型預測
                s_in = torch.ones(x_t.shape[0], device=x_t.device) * sigma_t
                f_denoiser = model_wrapper(x_t, s_in, **extra_args)
                controls[t] = f_denoiser / (sigma_t + 1e-8)

        return controls

    def _solve_optimal_control_step(
        self,
        x: torch.Tensor,
        sigma: float,
        sigma_next: float,
        denoiser_function: Callable,
        extra_args: Dict
    ) -> torch.Tensor:
        """
        求解單步最優控制

        使用 Hamilton-Jacobi-Bellman 方程的簡化形式
        """
        # 獲取去噪函數輸出
        s_in = torch.ones(x.shape[0], device=x.device) * sigma
        f_denoiser = denoiser_function(x, s_in, **extra_args)

        # 計算理想漂移項
        drift_term = f_denoiser / (sigma + 1e-8)

        # 添加正則化修正
        regularization_correction = self._compute_regularization_correction(x, f_denoiser, sigma)

        # 最優控制: u* = f/σ + 修正項
        optimal_control = drift_term + regularization_correction

        return optimal_control

    def _compute_regularization_correction(
        self,
        x: torch.Tensor,
        f_denoiser: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """
        計算正則化修正項
        """
        # 計算 f 的梯度
        grad_f = self.action_integral._approximate_gradient(f_denoiser)

        # 正則化修正: -λ ∇(|∇f|²) / 2
        grad_correction = self._compute_gradient_correction(grad_f)

        # 曲率修正
        curvature_correction = self._compute_curvature_correction(x)

        total_correction = (
            -self.action_integral.lambda_reg * grad_correction / 2 +
            -self.action_integral.curvature_penalty * curvature_correction
        )

        return total_correction

    def _compute_gradient_correction(self, grad_f: torch.Tensor) -> torch.Tensor:
        """
        計算梯度修正項
        """
        # ∇(|∇f|²) ≈ 2∇f · ∇(∇f)
        # 簡化實現
        return torch.zeros_like(grad_f)

    def _compute_curvature_correction(self, x: torch.Tensor) -> torch.Tensor:
        """
        計算曲率修正項
        """
        # 拉普拉斯算子
        if x.dim() == 4:
            laplacian = F.conv2d(
                x,
                torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                           device=x.device, dtype=x.dtype).expand(x.shape[1], 1, -1, -1),
                padding=1,
                groups=x.shape[1]
            )
        else:
            laplacian = torch.zeros_like(x)

        return laplacian

    def evaluate_trajectory_quality(
        self,
        trajectory: torch.Tensor,
        sigma_values: torch.Tensor,
        denoiser_function: Callable,
        extra_args: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        評估軌跡質量

        使用變分動作積分和其他品質指標

        Args:
            trajectory: 軌跡 (T, B, C, H, W)
            sigma_values: σ 調度
            denoiser_function: 去噪函數
            extra_args: 額外參數

        Returns:
            quality_metrics: 品質評估指標
        """
        extra_args = extra_args or {}

        # 計算動作積分
        action_value = self.action_integral.compute_action(
            trajectory, sigma_values, denoiser_function, extra_args
        )

        # 計算 Sobolev 範數變化
        sobolev_norms = []
        for t in range(trajectory.shape[0]):
            norm = self.hilbert_space.compute_sobolev_norm(trajectory[t])
            sobolev_norms.append(norm.item())

        # 計算 Euler-Lagrange 殘差
        residuals = []
        for t in range(1, trajectory.shape[0] - 1):
            x_prev = trajectory[t - 1]
            x_curr = trajectory[t]
            x_next = trajectory[t + 1]
            sigma_t = sigma_values[t]

            # 計算導數
            dt_prev = (sigma_values[t] - sigma_values[t - 1]).item()
            dt_next = (sigma_values[t + 1] - sigma_values[t]).item()

            if abs(dt_prev) > 1e-12 and abs(dt_next) > 1e-12:
                x_dot = (x_next - x_prev) / (dt_next + dt_prev)
                x_ddot = (x_next - 2 * x_curr + x_prev) / (dt_next * dt_prev / 2)

                # 計算去噪輸出
                s_in = torch.ones(x_curr.shape[0], device=x_curr.device) * sigma_t
                f_denoiser = denoiser_function(x_curr, s_in, **extra_args)

                # EL 殘差
                residual = self.euler_lagrange_system.compute_euler_lagrange_residual(
                    x_curr, x_dot, x_ddot, f_denoiser, sigma_t.item()
                )
                residuals.append(torch.norm(residual).item())

        return {
            'action_value': action_value.item(),
            'mean_sobolev_norm': np.mean(sobolev_norms),
            'sobolev_norm_variation': np.std(sobolev_norms),
            'mean_euler_lagrange_residual': np.mean(residuals) if residuals else 0.0,
            'max_euler_lagrange_residual': np.max(residuals) if residuals else 0.0,
            'trajectory_smoothness': self._compute_trajectory_smoothness(trajectory),
            'energy_conservation': self._compute_energy_conservation(trajectory, sigma_values)
        }

    def verify_optimality_conditions(
        self,
        trajectory: torch.Tensor,
        sigma_schedule: torch.Tensor,
        denoiser_function: Callable,
        extra_args: Optional[Dict] = None,
        tolerance: float = 1e-3
    ) -> Dict[str, bool]:
        """
        驗證最優性條件

        檢查軌跡是否滿足 Euler-Lagrange 方程

        Args:
            trajectory: 軌跡
            sigma_schedule: σ 調度
            denoiser_function: 去噪函數
            extra_args: 額外參數
            tolerance: 容忍度

        Returns:
            optimality_check: 最優性檢查結果
        """
        extra_args = extra_args or {}

        # 計算軌跡導數
        x_dot = self.action_integral._compute_trajectory_derivative(trajectory, sigma_schedule)

        # 計算二階導數 (簡化)
        x_ddot = torch.zeros_like(x_dot)
        for t in range(1, len(x_dot) - 1):
            dt1 = sigma_schedule[t] - sigma_schedule[t-1]
            dt2 = sigma_schedule[t+1] - sigma_schedule[t]
            if abs(dt1) > 1e-12 and abs(dt2) > 1e-12:
                x_ddot[t] = (x_dot[t+1] - x_dot[t-1]) / (dt1 + dt2)

        euler_lagrange_satisfied = True
        max_residual = 0.0

        for t in range(1, len(trajectory) - 1):
            x_t = trajectory[t]
            x_dot_t = x_dot[t]
            x_ddot_t = x_ddot[t]
            sigma_t = sigma_schedule[t]

            # 獲取去噪函數輸出
            s_in = torch.ones(x_t.shape[0], device=x_t.device) * sigma_t
            f_t = denoiser_function(x_t, s_in, **extra_args)

            # 計算 Euler-Lagrange 殘差
            residual = self.euler_lagrange_system.compute_euler_lagrange_residual(
                x_t, x_dot_t, x_ddot_t, f_t, sigma_t.item()
            )

            residual_norm = torch.norm(residual).item()
            max_residual = max(max_residual, residual_norm)

            if residual_norm > tolerance:
                euler_lagrange_satisfied = False

        return {
            'euler_lagrange_satisfied': euler_lagrange_satisfied,
            'max_residual_norm': max_residual,
            'tolerance': tolerance,
            'residual_within_tolerance': max_residual <= tolerance
        }