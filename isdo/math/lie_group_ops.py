"""
Lie Group Operations for ISDO
=============================

實現李群對稱操作，用於保持圖像的拓撲結構不變性。
通過 SE(3) 群作用修正對稱破壞，確保生成結果的結構完整性。
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math


class LieGroupOps:
    """
    李群操作類別

    實現 SE(3) 群的各種操作，用於對稱性保持和結構修正
    """

    def __init__(
        self,
        spatial_dims: Tuple[int, ...],
        device: Optional[torch.device] = None
    ):
        """
        初始化李群操作

        Args:
            spatial_dims: 空間維度
            device: 計算設備
        """
        self.spatial_dims = spatial_dims
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if len(spatial_dims) == 2:
            self.H, self.W = spatial_dims
        else:
            raise ValueError(f"目前只支援 2D: {spatial_dims}")

    def detect_symmetry_violations(self, x: torch.Tensor) -> Dict[str, float]:
        """
        檢測對稱性破壞

        簡化實現：檢測明顯的結構異常

        Args:
            x: 輸入圖像

        Returns:
            violations: 對稱性破壞指標
        """
        violations = {}

        # 檢測極值
        max_val = torch.max(x.abs()).item()
        min_val = -max_val
        range_violation = max(abs(max_val) - 5.0, abs(min_val) + 5.0, 0.0)

        # 檢測梯度異常
        if x.dim() == 4:  # (B, C, H, W)
            grad_x = torch.diff(x, dim=-1)
            grad_y = torch.diff(x, dim=-2)

            gx = grad_x[..., :-1, :].real  # 保證為實數
            gy = grad_y[..., :, :-1].real
            grad_norm = torch.mean(torch.sqrt(torch.clamp(gx**2 + gy**2, min=0.0)))
            gradient_violation = max(grad_norm.item() - 2.0, 0.0)
        else:
            gradient_violation = 0.0

        # 檢測能量集中
        energy = torch.sum(x**2, dim=(-2, -1))
        energy_std = torch.std(energy).item()
        energy_violation = max(energy_std - 1.0, 0.0)

        violations['range_violation'] = range_violation
        violations['gradient_violation'] = gradient_violation
        violations['energy_violation'] = energy_violation
        violations['total_violation'] = range_violation + gradient_violation + energy_violation

        return violations

    def generate_symmetry_perturbations(
        self,
        x: torch.Tensor,
        violation_strength: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        生成對稱性修正擾動

        Args:
            x: 輸入圖像
            violation_strength: 破壞強度

        Returns:
            perturbations: 修正擾動
        """
        perturbations = {}

        # 生成小的旋轉擾動
        angle = violation_strength * 0.1  # 小角度

        perturbations['rotation_angle'] = angle
        perturbations['translation'] = torch.randn(2, device=self.device) * violation_strength * 0.01
        perturbations['scaling'] = 1.0 + torch.randn(1, device=self.device) * violation_strength * 0.01

        return perturbations

    def apply_group_action(
        self,
        x: torch.Tensor,
        perturbations: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        應用李群作用

        簡化實現：應用小的幾何變換
        """

        if torch.is_complex(x):
            x = x.real
        x = x.to(self.device).to(torch.float32)

        if x.dim() != 4:
            return x  # 簡化：只處理 4D 張量

        # 應用縮放
        if 'scaling' in perturbations:
            scaling = perturbations['scaling'].item()
            x = x * scaling

        # 輕微的空間變換 (簡化為高斯平滑)
        if 'rotation_angle' in perturbations:
            angle = perturbations['rotation_angle']
            if abs(angle) > 1e-6:
                # 使用高斯核近似小旋轉
                kernel_size = 3
                sigma = abs(angle) * 10
                kernel = self._gaussian_kernel(kernel_size, sigma)

                # 對每個通道應用
                B, C, H, W = x.shape
                x_smooth = torch.zeros_like(x)

                for c in range(C):
                    x_smooth[:, c:c+1] = F.conv2d(
                        x[:, c:c+1],
                        kernel.expand(1, 1, -1, -1),
                        padding=kernel_size//2
                    )

                # 混合原始和平滑版本
                mix_ratio = min(abs(angle) * 100, 0.1)
                x = (1 - mix_ratio) * x + mix_ratio * x_smooth

        return x

    def compute_symmetry_correction(self, x: torch.Tensor, violation_threshold: float = 0.01) -> torch.Tensor:
        """
        自動對輸入圖像進行對稱性修正：
        1. 檢測對稱性破壞
        2. 若超過閾值則生成修正擾動
        3. 應用李群作用進行修正

        Args:
            x: 輸入圖像 (B, C, H, W)
            violation_threshold: 觸發修正的違規閾值

        Returns:
            x_corrected: 修正後的圖像
        """
        violations = self.detect_symmetry_violations(x)
        if violations['total_violation'] > violation_threshold:
            perturbations = self.generate_symmetry_perturbations(x, violation_strength=violations['total_violation'])
            x_corrected = self.apply_group_action(x, perturbations)
            return x_corrected
        else:
            return x

    def _gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """生成高斯核"""
        coords = torch.arange(size, dtype=torch.float32, device=self.device)
        coords = coords - (size - 1) / 2

        g1d = torch.exp(-(coords**2) / (2 * sigma**2))
        g2d = g1d[:, None] * g1d[None, :]
        g2d = g2d / g2d.sum()

        return g2d