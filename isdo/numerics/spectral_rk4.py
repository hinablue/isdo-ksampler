"""
Spectral Runge-Kutta Solver for ISDO
====================================

在譜空間中實現 Runge-Kutta 求解器，結合變分最優控制。
這是 ISDO 算法的核心數值求解器。
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Callable
import math


class SpectralRK4:
    """
    譜 Runge-Kutta 4階求解器

    在譜係數空間中執行數值積分，解決變分 ODE 系統
    """

    def __init__(
        self,
        spectral_projection,
        variational_controller,
        device: Optional[torch.device] = None
    ):
        """
        初始化譜 RK4 求解器

        Args:
            spectral_projection: 譜投影系統
            variational_controller: 變分控制器
            device: 計算設備
        """
        self.spectral_projection = spectral_projection
        self.variational_controller = variational_controller
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def step(
        self,
        coefficients: torch.Tensor,
        sigma_current: float,
        sigma_next: float,
        model: Callable,
        extra_args: Dict
    ) -> torch.Tensor:
        """
        執行一步譜 RK4 積分

        Args:
            coefficients: 當前譜係數
            sigma_current: 當前噪聲水平
            sigma_next: 下一步噪聲水平
            model: 去噪模型
            extra_args: 額外參數

        Returns:
            updated_coefficients: 更新後的譜係數
        """
        dt = sigma_next - sigma_current

        # RK4 計算
        k1 = self._compute_rhs(coefficients, sigma_current, model, extra_args)
        k2 = self._compute_rhs(coefficients + 0.5 * dt * k1, sigma_current + 0.5 * dt, model, extra_args)
        k3 = self._compute_rhs(coefficients + 0.5 * dt * k2, sigma_current + 0.5 * dt, model, extra_args)
        k4 = self._compute_rhs(coefficients + dt * k3, sigma_next, model, extra_args)

        # 組合結果
        updated_coefficients = coefficients + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return updated_coefficients

    def _compute_rhs(
        self,
        coefficients: torch.Tensor,
        sigma: float,
        model: Callable,
        extra_args: Dict
    ) -> torch.Tensor:
        """
        計算 ODE 右端項

        這是譜空間中的變分動力學
        """
        # 重建到空間域
        x = self.spectral_projection(coefficients, mode='inverse')

        # 獲取模型輸出
        s_in = torch.ones(x.shape[0], device=x.device) * sigma
        denoised = model(x, s_in, **extra_args)

        # 計算漂移項 (簡化的變分動力學)
        drift = (x - denoised) / (sigma + 1e-8)

        # 投影回譜空間
        drift_coeffs = self.spectral_projection(drift, mode='forward')

        return drift_coeffs