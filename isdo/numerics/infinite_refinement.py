"""
Infinite Refinement Loop for ISDO
=================================

實現無窮細化循環系統，用於李群對稱性修正和結構保持。
通過迭代擾動和退火調度，確保生成結果的拓撲完整性。

核心功能：
- 細化迭代主循環
- 智能擾動生成
- 自適應退火調度
- 拓撲保真度檢查
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Callable, Union
import math
import numpy as np
from tqdm import trange


class InfiniteRefinement:
    """
    無窮細化循環類別

    實現李群對稱性保持的細化算法，通過迭代過程
    修正對稱破壞並保持拓撲結構不變性。
    """

    def __init__(
        self,
        spatial_dims: Tuple[int, ...],
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-5,
        perturbation_strength: float = 0.01,
        annealing_rate: float = 0.99,
        topology_check_frequency: int = 10,
        device: Optional[torch.device] = None
    ):
        """
        初始化無窮細化系統

        Args:
            spatial_dims: 空間維度
            max_iterations: 最大迭代數
            convergence_threshold: 收斂閾值
            perturbation_strength: 擾動強度
            annealing_rate: 退火速率
            topology_check_frequency: 拓撲檢查頻率
            device: 計算設備
        """
        self.spatial_dims = spatial_dims
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.perturbation_strength = perturbation_strength
        self.annealing_rate = annealing_rate
        self.topology_check_frequency = topology_check_frequency
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if len(spatial_dims) == 2:
            self.H, self.W = spatial_dims
        else:
            raise ValueError(f"目前只支援 2D 空間維度: {spatial_dims}")

        # 細化統計
        self.stats = {
            'total_iterations': 0,
            'convergence_achieved': 0,
            'topology_violations': 0,
            'average_perturbation': 0.0
        }

    def refinement_loop(
        self,
        x: torch.Tensor,
        target_function: Callable[[torch.Tensor], torch.Tensor],
        lie_group_ops: Optional[object] = None,
        spectral_projection: Optional[object] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        執行主要的細化迭代循環

        Args:
            x: 輸入張量
            target_function: 目標函數（如去噪模型）
            lie_group_ops: 李群操作對象
            spectral_projection: 譜投影對象
            progress_callback: 進度回調函數

        Returns:
            Dict containing refined tensor and statistics
        """
        batch_size, channels = x.shape[:2]
        x_refined = x.clone()
        current_perturbation_strength = self.perturbation_strength

        convergence_history = []
        topology_violations = []

        # 計算初始目標值
        initial_target = target_function(x_refined)
        previous_loss = self._compute_refinement_loss(x_refined, initial_target)

        for iteration in range(self.max_iterations):
            # 生成擾動
            perturbations = self.generate_perturbations(
                x_refined,
                strength=current_perturbation_strength,
                iteration=iteration
            )

            # 應用擾動
            x_perturbed = self._apply_perturbations(x_refined, perturbations)

            # 計算擾動後的目標值
            perturbed_target = target_function(x_perturbed)
            current_loss = self._compute_refinement_loss(x_perturbed, perturbed_target)

            # 檢查是否改善
            if current_loss < previous_loss:
                x_refined = x_perturbed
                previous_loss = current_loss
                improvement = True
            else:
                improvement = False

            # 退火調度
            current_perturbation_strength = self.annealing_schedule(
                current_perturbation_strength,
                iteration,
                improvement
            )

            # 記錄收斂歷史
            convergence_history.append(current_loss.item())

            # 拓撲保真檢查
            if iteration % self.topology_check_frequency == 0:
                topology_score = self.check_topology_preservation(x_refined, x)
                topology_violations.append(topology_score)

                # 如果拓撲破壞嚴重，應用修正
                if topology_score > 0.1 and lie_group_ops is not None:
                    x_refined = self._apply_topology_correction(
                        x_refined, lie_group_ops
                    )

            # 檢查收斂
            if len(convergence_history) >= 5:
                recent_change = abs(
                    convergence_history[-1] - convergence_history[-5]
                )
                if recent_change < self.convergence_threshold:
                    self.stats['convergence_achieved'] += 1
                    break

            # 進度回調
            if progress_callback is not None and iteration % 100 == 0:
                progress_callback(iteration, current_loss.item(), topology_violations[-1] if topology_violations else 0.0)

        # 更新統計
        self.stats['total_iterations'] += iteration + 1
        self.stats['topology_violations'] += len([v for v in topology_violations if v > 0.05])
        self.stats['average_perturbation'] = (
            self.stats['average_perturbation'] * 0.9 +
            current_perturbation_strength * 0.1
        )

        return {
            'refined_tensor': x_refined,
            'convergence_history': convergence_history,
            'topology_violations': topology_violations,
            'final_loss': previous_loss.item(),
            'iterations_used': iteration + 1,
            'stats': self.stats.copy()
        }

    def generate_perturbations(
        self,
        x: torch.Tensor,
        strength: float = 0.01,
        iteration: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        生成智能擾動

        根據當前狀態和迭代次數生成適應性擾動

        Args:
            x: 輸入張量
            strength: 擾動強度
            iteration: 當前迭代次數

        Returns:
            Dict containing various types of perturbations
        """
        batch_size, channels = x.shape[:2]
        perturbations = {}

        # 1. 高斯噪聲擾動（基礎）
        perturbations['gaussian'] = torch.randn_like(x) * strength

        # 2. 結構化擾動（頻域）
        if iteration % 3 == 0:
            # 低頻擾動
            low_freq_noise = F.interpolate(
                torch.randn(batch_size, channels, self.H//4, self.W//4, device=self.device),
                size=(self.H, self.W),
                mode='bilinear',
                align_corners=False
            ) * strength * 0.5
            perturbations['low_frequency'] = low_freq_noise

        # 3. 邊緣保持擾動
        if iteration % 5 == 0:
            # 計算梯度
            grad_x = torch.diff(x, dim=-1, prepend=x[..., :1])
            grad_y = torch.diff(x, dim=-2, prepend=x[..., :1, :])

            # 在梯度較小的區域加強擾動
            edge_mask = (grad_x.abs() + grad_y.abs()) < 0.1
            edge_perturbation = torch.randn_like(x) * strength * 0.3
            edge_perturbation = edge_perturbation * edge_mask.float()
            perturbations['edge_preserving'] = edge_perturbation

        # 4. 對稱性導向擾動
        if iteration % 7 == 0:
            # 生成對稱破壞修正擾動
            center_h, center_w = self.H // 2, self.W // 2

            # 徑向對稱擾動
            h_coords = torch.arange(self.H, device=self.device).float() - center_h
            w_coords = torch.arange(self.W, device=self.device).float() - center_w
            h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')
            r = torch.sqrt(h_grid**2 + w_grid**2)

            radial_weight = torch.exp(-r / (min(self.H, self.W) / 4))
            radial_perturbation = torch.randn_like(x) * strength * 0.2
            radial_perturbation = radial_perturbation * radial_weight.unsqueeze(0).unsqueeze(0)
            perturbations['radial_symmetry'] = radial_perturbation

        return perturbations

    def annealing_schedule(
        self,
        current_strength: float,
        iteration: int,
        improvement: bool
    ) -> float:
        """
        自適應退火調度

        根據迭代進展和改善情況調整擾動強度

        Args:
            current_strength: 當前擾動強度
            iteration: 迭代次數
            improvement: 是否有改善

        Returns:
            Updated perturbation strength
        """
        # 基礎指數退火
        base_decay = current_strength * self.annealing_rate

        # 根據改善情況調整
        if improvement:
            # 如果有改善，稍微增強探索
            adjustment = 1.05
        else:
            # 如果沒有改善，加強衰減
            adjustment = 0.95

        # 階段性調整
        if iteration < 100:
            # 初期：保持較高探索
            phase_factor = 1.0
        elif iteration < 500:
            # 中期：逐漸收斂
            phase_factor = 0.8
        else:
            # 後期：精細調整
            phase_factor = 0.5

        new_strength = base_decay * adjustment * phase_factor

        # 確保不會過小或過大
        new_strength = max(new_strength, 1e-6)
        new_strength = min(new_strength, 0.1)

        return new_strength

    def check_topology_preservation(
        self,
        x_current: torch.Tensor,
        x_original: torch.Tensor,
        method: str = 'structural'
    ) -> float:
        """
        檢查拓撲保真度

        評估細化過程中結構完整性的保持程度

        Args:
            x_current: 當前狀態
            x_original: 原始狀態
            method: 檢查方法

        Returns:
            Topology violation score (0 = perfect, 1 = completely broken)
        """
        if method == 'structural':
            return self._structural_topology_check(x_current, x_original)
        elif method == 'spectral':
            return self._spectral_topology_check(x_current, x_original)
        else:
            # 綜合檢查
            structural_score = self._structural_topology_check(x_current, x_original)
            spectral_score = self._spectral_topology_check(x_current, x_original)
            return (structural_score + spectral_score) / 2

    def _structural_topology_check(
        self,
        x_current: torch.Tensor,
        x_original: torch.Tensor
    ) -> float:
        """結構化拓撲檢查"""
        # 計算梯度場的變化
        grad_orig_x = torch.diff(x_original, dim=-1)
        grad_orig_y = torch.diff(x_original, dim=-2)
        grad_curr_x = torch.diff(x_current, dim=-1)
        grad_curr_y = torch.diff(x_current, dim=-2)

        # 梯度方向的變化
        grad_orig_mag = torch.sqrt(grad_orig_x**2 + grad_orig_y**2 + 1e-8)
        grad_curr_mag = torch.sqrt(grad_curr_x**2 + grad_curr_y**2 + 1e-8)

        # 正規化梯度
        grad_orig_norm_x = grad_orig_x / grad_orig_mag
        grad_orig_norm_y = grad_orig_y / grad_orig_mag
        grad_curr_norm_x = grad_curr_x / grad_curr_mag
        grad_curr_norm_y = grad_curr_y / grad_curr_mag

        # 計算梯度方向的相似性
        direction_similarity = (
            grad_orig_norm_x * grad_curr_norm_x +
            grad_orig_norm_y * grad_curr_norm_y
        )

        # 拓撲破壞分數（1 - 平均相似性）
        violation_score = 1.0 - direction_similarity.mean().item()
        return max(0.0, min(1.0, violation_score))

    def _spectral_topology_check(
        self,
        x_current: torch.Tensor,
        x_original: torch.Tensor
    ) -> float:
        """譜域拓撲檢查"""
        # FFT 譜分析
        fft_orig = torch.fft.fft2(x_original)
        fft_curr = torch.fft.fft2(x_current)

        # 比較功率譜
        power_orig = torch.abs(fft_orig)**2
        power_curr = torch.abs(fft_curr)**2

        # 正規化
        power_orig = power_orig / (power_orig.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        power_curr = power_curr / (power_curr.sum(dim=(-2, -1), keepdim=True) + 1e-8)

        # KL 散度作為譜差異度量
        kl_div = F.kl_div(
            power_curr.log(),
            power_orig,
            reduction='mean'
        )

        # 轉換為 0-1 分數
        violation_score = torch.tanh(kl_div).item()
        return max(0.0, min(1.0, violation_score))

    def _compute_refinement_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """計算細化損失"""
        # 主要重建損失
        recon_loss = F.mse_loss(x, target)

        # 結構保持損失（總變分）
        tv_loss = self._total_variation_loss(x)

        # 組合損失
        total_loss = recon_loss + 0.01 * tv_loss
        return total_loss

    def _total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """計算總變分損失"""
        diff_h = torch.abs(x[..., 1:, :] - x[..., :-1, :])
        diff_w = torch.abs(x[..., :, 1:] - x[..., :, :-1])
        return diff_h.mean() + diff_w.mean()

    def _apply_perturbations(
        self,
        x: torch.Tensor,
        perturbations: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """應用擾動"""
        x_perturbed = x.clone()

        # 加權組合所有擾動
        total_perturbation = torch.zeros_like(x)

        if 'gaussian' in perturbations:
            total_perturbation += perturbations['gaussian'] * 0.4

        if 'low_frequency' in perturbations:
            total_perturbation += perturbations['low_frequency'] * 0.3

        if 'edge_preserving' in perturbations:
            total_perturbation += perturbations['edge_preserving'] * 0.2

        if 'radial_symmetry' in perturbations:
            total_perturbation += perturbations['radial_symmetry'] * 0.1

        x_perturbed = x + total_perturbation
        return x_perturbed

    def _apply_topology_correction(
        self,
        x: torch.Tensor,
        lie_group_ops: object
    ) -> torch.Tensor:
        """應用李群拓撲修正"""
        # 檢測對稱性破壞
        violations = lie_group_ops.detect_symmetry_violations(x)

        if violations['total_violation'] > 0.05:
            # 生成修正擾動
            correction_perturbations = lie_group_ops.generate_symmetry_perturbations(
                x, violation_strength=violations['total_violation']
            )

            # 應用李群作用
            x_corrected = lie_group_ops.apply_group_action(x, correction_perturbations)
            return x_corrected

        return x

    def get_statistics(self) -> Dict[str, float]:
        """獲取細化統計信息"""
        return self.stats.copy()

    def reset_statistics(self) -> None:
        """重置統計信息"""
        self.stats = {
            'total_iterations': 0,
            'convergence_achieved': 0,
            'topology_violations': 0,
            'average_perturbation': 0.0
        }