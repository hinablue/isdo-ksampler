"""
ISDO Sampler Implementation
===========================

Infinite Spectral Diffusion Odyssey 主採樣器實現。
整合變分最優控制、譜投影和李群細化，提供與傳統採樣器相同的接口。

核心創新:
- 變分最優控制替代傳統 ODE
- 譜方法處理無窮維問題
- 李群對稱保證結構不變性
- 自適應調度優化效率
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Dict, List, Callable
import math
from tqdm import trange

# 導入核心組件
from ..core.spectral_basis import SpectralBasis, BasisType
from ..core.hilbert_space import HilbertSpace
from ..core.variational_controller import VariationalController
from ..core.spectral_projection import SpectralProjection
from ..numerics.spectral_rk4 import SpectralRK4
from ..numerics.lie_group_ops import LieGroupOps
from ..numerics.infinite_refinement import InfiniteRefinement
from .unified_model_wrapper import UnifiedModelWrapper


def to_d(x, sigma, denoised):
    """
    兼容函數：計算 ODE 的漂移項
    與 k_diffusion/sampling.py 中的 to_d 函數相同
    """
    return (x - denoised) / sigma


class ISDOSampler:
    """
    Infinite Spectral Diffusion Odyssey 主採樣器

    實現完整的 ISDO 採樣算法，包括：
    1. 變分最優控制
    2. 譜投影與重建
    3. 李群對稱細化
    4. 自適應調度
    """

    def __init__(
        self,
        spatial_dims: Tuple[int, ...] = (64, 64),
        spectral_order: int = 256,
        sobolev_order: float = 1.5,
        regularization_lambda: float = 1e-4, # Adjusted from 0.01
        curvature_penalty: float = 0.001,
        refinement_iterations: int = 1000,
        adaptive_scheduling: bool = True,
        lie_group_refinement: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        初始化 ISDO 採樣器

        Args:
            spatial_dims: 空間維度 (H, W)
            spectral_order: 譜截斷階數 M
            sobolev_order: Sobolev 空間階數 s
            regularization_lambda: 變分正則化參數 λ
            curvature_penalty: 曲率懲罰係數 μ
            refinement_iterations: 李群細化迭代數 K
            adaptive_scheduling: 是否使用自適應調度
            lie_group_refinement: 是否使用李群細化
            device: 計算設備
            dtype: 數據類型
        """
        self.spatial_dims = spatial_dims
        self.spectral_order = spectral_order
        self.sobolev_order = sobolev_order
        self.regularization_lambda = regularization_lambda
        self.curvature_penalty = curvature_penalty
        self.refinement_iterations = refinement_iterations
        self.adaptive_scheduling = adaptive_scheduling
        self.lie_group_refinement = lie_group_refinement
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        # 初始化核心組件
        self._initialize_components()

        # 統計信息
        self.stats = {
            'total_denoiser_calls': 0,
            'spectral_projections': 0,
            'lie_group_operations': 0,
            'adaptive_adjustments': 0,
            'infinite_refinements': 0,
            'refinement_iterations_total': 0,
            'refinement_convergence_rate': 0.0
        }

    def _initialize_components(self):
        """初始化所有核心組件"""

        # 希爾伯特空間
        self.hilbert_space = HilbertSpace(
            spatial_dims=self.spatial_dims,
            sobolev_order=self.sobolev_order,
            device=self.device
        )

        # 譜投影系統
        self.spectral_projection = SpectralProjection(
            spatial_dims=self.spatial_dims,
            spectral_order=self.spectral_order,
            projection_type='adaptive' if self.adaptive_scheduling else 'fft',
            device=self.device
        )

        # 變分控制器
        self.variational_controller = VariationalController(
            spatial_dims=self.spatial_dims,
            regularization_lambda=self.regularization_lambda,
            curvature_penalty=self.curvature_penalty,
            sobolev_order=self.sobolev_order,
            device=self.device
        )

        # 變分 ODE 系統 (取代 SpectralRK4)
        from ..core.variational_ode_system import VariationalODESystem
        self.variational_ode_system = VariationalODESystem(
            spatial_dims=self.spatial_dims,
            spectral_order=self.spectral_order,
            sobolev_order=self.sobolev_order,
            regularization_lambda=self.regularization_lambda,
            sobolev_penalty=self.curvature_penalty,
            device=self.device
        )

        # 保留 SpectralRK4 作為回退選項
        self.spectral_rk4 = SpectralRK4(
            spectral_projection=self.spectral_projection,
            variational_controller=self.variational_controller,
            device=self.device
        )

        # 李群操作 (如果啟用)
        if self.lie_group_refinement:
            self.lie_group_ops = LieGroupOps(
                spatial_dims=self.spatial_dims,
                device=self.device
            )

            # 無窮細化系統
            self.infinite_refinement = InfiniteRefinement(
                spatial_dims=self.spatial_dims,
                max_iterations=min(self.refinement_iterations, 100),  # 限制最大迭代數
                convergence_threshold=1e-5,
                perturbation_strength=0.01,
                annealing_rate=0.99,
                topology_check_frequency=10,
                device=self.device
            )

    @torch.no_grad()
    def sample_isdo(
        self,
        model,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        extra_args: Optional[Dict] = None,
        callback: Optional[Callable] = None,
        disable: Optional[bool] = None,
        # ISDO 特有參數
        energy_threshold: float = 0.99,
        quality_threshold: float = 0.01,
        max_refinement_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        ISDO 主採樣函數

        與 sample_euler 兼容的接口，但使用變分最優控制

        Args:
            model: 去噪模型
            x: 初始噪聲 (B, C, H, W)
            sigmas: 噪聲水平序列 (T,)
            extra_args: 額外參數
            callback: 回調函數
            disable: 是否禁用進度條
            energy_threshold: 譜能量保留閾值
            quality_threshold: 軌跡質量閾值
            max_refinement_steps: 最大細化步數

        Returns:
            x_final: 最終採樣結果
        """
        extra_args = extra_args or {}
        max_refinement_steps = max_refinement_steps or self.refinement_iterations

        # 統一模型接口
        unified_model = UnifiedModelWrapper(model)

        # 初始化
        x_current = x.clone()
        s_in = x.new_ones([x.shape[0]])

        # 主採樣循環
        for i in trange(len(sigmas) - 1, disable=disable, desc="ISDO Sampling"):
            sigma_current = sigmas[i]
            sigma_next = sigmas[i + 1]

            # 階段 1: 變分最優控制求解
            x_current = self._variational_control_step(
                x_current, sigma_current, sigma_next,
                unified_model, extra_args, energy_threshold
            )

            # 階段 2: 李群對稱細化 (可選)
            if self.lie_group_refinement and i % 5 == 0:  # 每5步細化一次
                x_current = self._lie_group_refinement_step(
                    x_current, sigma_next, unified_model,
                    extra_args, max_refinement_steps
                )

            # 回調函數
            if callback is not None:
                # 計算去噪結果用於回調
                denoised = unified_model(x_current, sigma_current * s_in, **extra_args)
                callback({
                    'x': x_current,
                    'i': i,
                    'sigma': sigmas[i],
                    'sigma_hat': sigma_current,
                    'denoised': denoised,
                    # ISDO 特有信息
                    'isdo_stats': self._get_step_stats(),
                    'sobolev_norm': self.hilbert_space.compute_sobolev_norm(x_current).item()
                })

        return x_current

    def advanced_refinement(
        self,
        x: torch.Tensor,
        model: UnifiedModelWrapper,
        sigma: float,
        extra_args: Optional[Dict] = None,
        max_iterations: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        use_spectral_projection: bool = True,
        detailed_stats: bool = False
    ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        高級細化接口：直接使用 InfiniteRefinement 系統

        提供對無窮細化循環的完全控制，適用於高質量細化需求

        Args:
            x: 輸入張量
            model: 統一模型包裝器
            sigma: 噪聲水平
            extra_args: 額外參數
            max_iterations: 最大迭代數（覆蓋默認值）
            convergence_threshold: 收斂閾值（覆蓋默認值）
            use_spectral_projection: 是否使用譜投影
            detailed_stats: 是否返回詳細統計信息

        Returns:
            refinement_result: 細化結果和統計信息
        """
        if not hasattr(self, 'infinite_refinement'):
            raise RuntimeError("InfiniteRefinement not initialized. Enable lie_group_refinement=True.")

        extra_args = extra_args or {}

        # 動態調整細化參數
        if max_iterations is not None:
            self.infinite_refinement.max_iterations = max_iterations
        if convergence_threshold is not None:
            self.infinite_refinement.convergence_threshold = convergence_threshold

        # 創建增強的目標函數
        def enhanced_target_function(x_input: torch.Tensor) -> torch.Tensor:
            """
            增強的目標函數：結合多種技術
            """
            # 基本去噪
            denoised = model(x_input, sigma, **extra_args)

            if use_spectral_projection and hasattr(self, 'spectral_projection'):
                # 譜投影增強
                try:
                    spectral_coeffs = self.spectral_projection(x_input, mode='forward')
                    spectral_enhanced = self.spectral_projection(spectral_coeffs, mode='inverse')

                    # 混合原始去噪和譜增強結果
                    alpha = 0.8  # 去噪權重
                    enhanced_target = alpha * denoised + (1 - alpha) * spectral_enhanced
                    return enhanced_target
                except Exception:
                    pass

            return denoised

        # 細化進度追蹤
        progress_data = {'iterations': [], 'losses': [], 'violations': []} if detailed_stats else None

        def detailed_progress_callback(iteration: int, loss: float, topology_violation: float):
            if detailed_stats:
                progress_data['iterations'].append(iteration)
                progress_data['losses'].append(loss)
                progress_data['violations'].append(topology_violation)

            # 每50步報告一次
            if iteration % 50 == 0:
                print(f"Advanced refinement iteration {iteration}: loss={loss:.6f}, violation={topology_violation:.6f}")

        # 執行高級細化
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if start_time:
            start_time.record()

        try:
            lie_group_ops = self.lie_group_ops if hasattr(self, 'lie_group_ops') else None
            spectral_projection = self.spectral_projection if use_spectral_projection else None

            refinement_result = self.infinite_refinement.refinement_loop(
                x=x,
                target_function=enhanced_target_function,
                lie_group_ops=lie_group_ops,
                spectral_projection=spectral_projection,
                progress_callback=detailed_progress_callback if detailed_stats else None
            )

            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                execution_time = start_time.elapsed_time(end_time) / 1000.0  # 轉換為秒
            else:
                execution_time = None

            # 增強結果信息
            enhanced_result = refinement_result.copy()
            enhanced_result['execution_time'] = execution_time
            enhanced_result['progress_data'] = progress_data
            enhanced_result['initial_sobolev_norm'] = self.hilbert_space.compute_sobolev_norm(x).item()
            enhanced_result['final_sobolev_norm'] = self.hilbert_space.compute_sobolev_norm(
                refinement_result['refined_tensor']
            ).item()

            # 計算質量改善指標
            if lie_group_ops:
                initial_violations = lie_group_ops.detect_symmetry_violations(x)
                final_violations = lie_group_ops.detect_symmetry_violations(refinement_result['refined_tensor'])
                enhanced_result['symmetry_improvement'] = {
                    'initial_violation': initial_violations['total_violation'],
                    'final_violation': final_violations['total_violation'],
                    'improvement_ratio': (
                        initial_violations['total_violation'] - final_violations['total_violation']
                    ) / (initial_violations['total_violation'] + 1e-8)
                }

            return enhanced_result

        except Exception as e:
            return {
                'refined_tensor': x,
                'error': str(e),
                'convergence_history': [],
                'iterations_used': 0,
                'execution_time': execution_time
            }

    def _variational_control_step(
        self,
        x: torch.Tensor,
        sigma_current: float,
        sigma_next: float,
        model: UnifiedModelWrapper,
        extra_args: Dict,
        energy_threshold: float
    ) -> torch.Tensor:
        """
        執行單步變分最優控制

        使用 VariationalODESystem 進行高精度求解
        """
        try:
            # 使用 VariationalODESystem 進行精確求解
            x_updated = self._ode_system_step(
                x, sigma_current, sigma_next, model, extra_args, energy_threshold
            )
            return x_updated

        except Exception as e:
            # 回退到傳統方法
            print(f"VariationalODESystem 失敗，回退到 SpectralRK4: {e}")
            return self._fallback_spectral_step(
                x, sigma_current, sigma_next, model, extra_args, energy_threshold
            )

    def _ode_system_step(
        self,
        x: torch.Tensor,
        sigma_current: float,
        sigma_next: float,
        model: UnifiedModelWrapper,
        extra_args: Dict,
        energy_threshold: float
    ) -> torch.Tensor:
        """
        使用 VariationalODESystem 執行單步
        """
        # 投影到譜空間
        initial_coeffs = self.variational_ode_system.spectral_basis.project_to_basis(x)

        # 使用變分 ODE 系統求解
        final_coeffs, trajectory, solve_info = self.variational_ode_system.solve_variational_ode(
            initial_coefficients=initial_coeffs,
            sigma_start=sigma_current,
            sigma_end=sigma_next,
            model_wrapper=model,
            extra_args=extra_args,
            max_steps=20,  # 較少步數用於單步求解
            adaptive_stepping=self.adaptive_scheduling
        )

        # 檢查求解質量
        if solve_info.get('converged', False) or solve_info['final_step'] > 0:
            # 重建到空間域
            x_updated = self.variational_ode_system.spectral_basis.reconstruct_from_coefficients(final_coeffs)

            # 更新統計
            self.stats['spectral_projections'] += 1
            if solve_info.get('converged', False):
                self.stats['adaptive_adjustments'] += 1

            return x_updated
        else:
            # 求解失敗，使用回退方法
            raise RuntimeError("VariationalODESystem 求解失敗")

    def _fallback_spectral_step(
        self,
        x: torch.Tensor,
        sigma_current: float,
        sigma_next: float,
        model: UnifiedModelWrapper,
        extra_args: Dict,
        energy_threshold: float
    ) -> torch.Tensor:
        """
        回退到原有的譜方法
        """
        # 投影到譜空間
        spectral_coeffs = self.spectral_projection(x, mode='forward')
        self.stats['spectral_projections'] += 1

        # 在譜空間執行變分控制
        if self.adaptive_scheduling:
            # 自適應步長控制
            optimal_coeffs = self._adaptive_spectral_step(
                spectral_coeffs, sigma_current, sigma_next,
                model, extra_args, energy_threshold
            )
        else:
            # 標準譜 RK4 步驟
            optimal_coeffs = self.spectral_rk4.step(
                spectral_coeffs, sigma_current, sigma_next,
                model, extra_args
            )

        # 重建到空間域
        x_updated = self.spectral_projection(optimal_coeffs, mode='inverse')

        return x_updated

    def _adaptive_spectral_step(
        self,
        spectral_coeffs: torch.Tensor,
        sigma_current: float,
        sigma_next: float,
        model: UnifiedModelWrapper,
        extra_args: Dict,
        energy_threshold: float
    ) -> torch.Tensor:
        """
        自適應譜步驟

        根據局部誤差估計調整步長和譜階數
        """
        # 估計局部截斷誤差
        dt = sigma_next - sigma_current

        # 嘗試全步長
        full_step_coeffs = self.spectral_rk4.step(
            spectral_coeffs, sigma_current, sigma_next, model, extra_args
        )

        # 嘗試半步長
        sigma_mid = (sigma_current + sigma_next) / 2
        half_step1_coeffs = self.spectral_rk4.step(
            spectral_coeffs, sigma_current, sigma_mid, model, extra_args
        )
        half_step2_coeffs = self.spectral_rk4.step(
            half_step1_coeffs, sigma_mid, sigma_next, model, extra_args
        )

        # 估計局部誤差
        error_estimate = torch.norm(full_step_coeffs - half_step2_coeffs)

        # 自適應策略
        if error_estimate > energy_threshold:
            # 誤差太大，使用半步長結果並調整譜階數
            self.stats['adaptive_adjustments'] += 1

            # 動態調整譜階數
            if hasattr(self.spectral_projection.projector, 'adaptive_projection'):
                current_x = self.spectral_projection(spectral_coeffs, mode='inverse')
                analysis = self.spectral_projection.projector.analyze_spectral_content(current_x)
                recommended_order = min(analysis['recommended_order'], self.spectral_order)

                # 截斷到推薦階數
                if recommended_order < len(half_step2_coeffs):
                    half_step2_coeffs = half_step2_coeffs[..., :recommended_order]

            return half_step2_coeffs
        else:
            # 誤差可接受，使用全步長結果
            return full_step_coeffs

    def _lie_group_refinement_step(
        self,
        x: torch.Tensor,
        sigma: float,
        model: UnifiedModelWrapper,
        extra_args: Dict,
        max_steps: int
    ) -> torch.Tensor:
        """
        執行李群對稱細化 - 使用 InfiniteRefinement 系統

        通過 SE(3) 群作用和無窮細化循環保持拓撲結構不變性
        """
        if not hasattr(self, 'lie_group_ops') or not hasattr(self, 'infinite_refinement'):
            return x  # 如果未啟用李群操作或無窮細化

        # 檢測是否需要細化
        symmetry_violations = self.lie_group_ops.detect_symmetry_violations(x)

        # 如果對稱性破壞很小，跳過細化
        if symmetry_violations['total_violation'] < 0.01:
            return x

        # 創建目標函數：包裝去噪模型
        def target_function(x_input: torch.Tensor) -> torch.Tensor:
            """
            目標函數：結合去噪和變分控制

            這個函數定義了我們希望細化達到的目標狀態
            """
            try:
                # 基本去噪預測
                denoised = model(x_input, sigma, **extra_args)

                # 可選：添加變分控制修正
                if hasattr(self, 'variational_controller'):
                    # 計算變分控制的理想漂移
                    drift_correction = denoised / (sigma + 1e-8)

                    # 計算正則化修正
                    regularization_correction = self.variational_controller._compute_regularization_correction(
                        x_input, denoised, sigma
                    )

                    # 組合修正
                    corrected_target = x_input + (drift_correction + regularization_correction) * 0.1
                    return corrected_target
                else:
                    return denoised

            except Exception as e:
                # 回退到簡單的去噪
                return model(x_input, sigma, **extra_args)

        # 進度回調函數
        def progress_callback(iteration: int, loss: float, topology_violation: float):
            """細化進度回調"""
            if iteration % 20 == 0:  # 每20步報告一次
                print(f"  Refinement iteration {iteration}: loss={loss:.6f}, topology_violation={topology_violation:.6f}")

        try:
            # 執行無窮細化循環
            refinement_result = self.infinite_refinement.refinement_loop(
                x=x,
                target_function=target_function,
                lie_group_ops=self.lie_group_ops,
                spectral_projection=self.spectral_projection,
                progress_callback=progress_callback if max_steps > 50 else None  # 只在長時間細化時顯示進度
            )

            # 提取細化結果
            refined_x = refinement_result['refined_tensor']
            convergence_history = refinement_result['convergence_history']
            final_loss = refinement_result['final_loss']
            iterations_used = refinement_result['iterations_used']

            # 更新統計信息
            self.stats['lie_group_operations'] += 1
            self.stats['infinite_refinements'] += 1
            self.stats['refinement_iterations_total'] += iterations_used

            # 計算收斂率
            if iterations_used > 0:
                convergence_rate = 1.0 if final_loss < 1e-4 else max(0.0, 1.0 - final_loss)
                self.stats['refinement_convergence_rate'] = (
                    self.stats['refinement_convergence_rate'] * 0.9 + convergence_rate * 0.1
                )

            # 驗證細化質量
            final_violations = self.lie_group_ops.detect_symmetry_violations(refined_x)
            improvement_ratio = (
                symmetry_violations['total_violation'] - final_violations['total_violation']
            ) / (symmetry_violations['total_violation'] + 1e-8)

            # 只有在確實改善的情況下才返回細化結果
            if improvement_ratio > 0.1:  # 改善至少 10%
                print(f"  ✅ Refinement successful: {improvement_ratio:.2%} improvement in {iterations_used} iterations")
                return refined_x
            else:
                print(f"  ⚠️ Refinement didn't improve significantly ({improvement_ratio:.2%}), using original")
                return x

        except Exception as e:
            print(f"  ❌ Refinement failed: {e}, falling back to short spectral refinement")
            # 回退到原來的短程譜細化
            return self._short_spectral_refinement(x, sigma, model, extra_args, num_steps=5)

    def _short_spectral_refinement(
        self,
        x: torch.Tensor,
        sigma: float,
        model: UnifiedModelWrapper,
        extra_args: Dict,
        num_steps: int = 5
    ) -> torch.Tensor:
        """
        短程譜細化

        從小擾動快速回到數據流形
        """
        # 小的 σ 調度用於細化
        sigma_min = max(sigma * 0.9, 0.01)
        sigma_schedule = torch.linspace(sigma, sigma_min, num_steps + 1, device=self.device)

        x_current = x

        for i in range(num_steps):
            sigma_curr = sigma_schedule[i]
            sigma_next = sigma_schedule[i + 1]

            # 簡化的變分步驟
            spectral_coeffs = self.spectral_projection(x_current, mode='forward')
            updated_coeffs = self.spectral_rk4.step(
                spectral_coeffs, sigma_curr.item(), sigma_next.item(),
                model, extra_args
            )
            x_current = self.spectral_projection(updated_coeffs, mode='inverse')

        return x_current

    def _get_step_stats(self) -> Dict:
        """獲取當前步驟的統計信息"""
        return {
            'denoiser_calls': self.stats['total_denoiser_calls'],
            'spectral_projections': self.stats['spectral_projections'],
            'lie_group_ops': self.stats['lie_group_operations'],
            'adaptive_adjustments': self.stats['adaptive_adjustments'],
            'infinite_refinements': self.stats['infinite_refinements'],
            'refinement_iterations_total': self.stats['refinement_iterations_total'],
            'refinement_convergence_rate': self.stats['refinement_convergence_rate']
        }

    def evaluate_sampling_quality(
        self,
        x_samples: torch.Tensor,
        reference_samples: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        評估採樣質量

        Args:
            x_samples: ISDO 採樣結果
            reference_samples: 參考樣本 (可選)

        Returns:
            quality_metrics: 質量評估指標
        """
        metrics = {}

        # Sobolev 範數分析
        sobolev_norms = []
        for i in range(x_samples.shape[0]):
            norm = self.hilbert_space.compute_sobolev_norm(x_samples[i:i+1])
            sobolev_norms.append(norm.item())

        metrics['mean_sobolev_norm'] = np.mean(sobolev_norms)
        metrics['std_sobolev_norm'] = np.std(sobolev_norms)

        # 譜內容分析
        if hasattr(self.spectral_projection.projector, 'analyze_spectral_content'):
            spectral_analyses = []
            for i in range(min(x_samples.shape[0], 10)):  # 分析前10個樣本
                analysis = self.spectral_projection.projector.analyze_spectral_content(
                    x_samples[i:i+1]
                )
                spectral_analyses.append(analysis)

            # 平均譜特性
            avg_low_freq = np.mean([a['low_freq_ratio'] for a in spectral_analyses])
            avg_high_freq = np.mean([a['high_freq_ratio'] for a in spectral_analyses])
            avg_bandwidth = np.mean([a['effective_bandwidth'] for a in spectral_analyses])

            metrics['avg_low_freq_ratio'] = avg_low_freq
            metrics['avg_high_freq_ratio'] = avg_high_freq
            metrics['avg_effective_bandwidth'] = avg_bandwidth

        # 結構完整性檢查
        if self.lie_group_refinement and hasattr(self, 'lie_group_ops'):
            structure_violations = []
            for i in range(min(x_samples.shape[0], 10)):
                violations = self.lie_group_ops.detect_symmetry_violations(
                    x_samples[i:i+1]
                )
                structure_violations.append(violations['total_violation'])

            metrics['mean_structure_violation'] = np.mean(structure_violations)
            metrics['max_structure_violation'] = np.max(structure_violations)

        # 與參考樣本的比較 (如果提供)
        if reference_samples is not None:
            # 計算分佈距離 (簡化版)
            samples_flat = x_samples.view(x_samples.shape[0], -1)
            reference_flat = reference_samples.view(reference_samples.shape[0], -1)

            # 平均值差異
            mean_diff = torch.norm(
                torch.mean(samples_flat, dim=0) - torch.mean(reference_flat, dim=0)
            )
            metrics['mean_difference'] = mean_diff.item()

            # 方差差異
            var_diff = torch.norm(
                torch.var(samples_flat, dim=0) - torch.var(reference_flat, dim=0)
            )
            metrics['variance_difference'] = var_diff.item()

        return metrics

    def get_sampling_statistics(self) -> Dict:
        """獲取完整的採樣統計信息"""
        return {
            'configuration': {
                'spatial_dims': self.spatial_dims,
                'spectral_order': self.spectral_order,
                'sobolev_order': self.sobolev_order,
                'regularization_lambda': self.regularization_lambda,
                'curvature_penalty': self.curvature_penalty,
                'adaptive_scheduling': self.adaptive_scheduling,
                'lie_group_refinement': self.lie_group_refinement
            },
            'runtime_stats': self.stats.copy(),
            'efficiency_metrics': {
                'projections_per_step': self.stats['spectral_projections'] / max(1, self.stats['total_denoiser_calls']),
                'refinement_ratio': self.stats['lie_group_operations'] / max(1, self.stats['total_denoiser_calls']),
                'adaptation_frequency': self.stats['adaptive_adjustments'] / max(1, self.stats['total_denoiser_calls'])
            }
        }

    def reset_statistics(self):
        """重置統計計數器"""
        self.stats = {
            'total_denoiser_calls': 0,
            'spectral_projections': 0,
            'lie_group_operations': 0,
            'adaptive_adjustments': 0,
            'infinite_refinements': 0,
            'refinement_iterations_total': 0,
            'refinement_convergence_rate': 0.0
        }


# 便利函數：與 k_diffusion 兼容的接口
def sample_isdo(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    # ISDO 參數
    spectral_order: int = 256,
    sobolev_order: float = 1.5,
    regularization_lambda: float = 1e-4, # Adjusted from 0.01
    refinement_iterations: int = 1000,
    adaptive_scheduling: bool = True,
    lie_group_refinement: bool = True
) -> torch.Tensor:
    """
    ISDO 採樣的便利函數

    與 k_diffusion.sampling.sample_euler 兼容的接口

    Args:
        model: 去噪模型
        x: 初始噪聲
        sigmas: 噪聲水平序列
        extra_args: 額外參數
        callback: 回調函數
        disable: 禁用進度條
        spectral_order: 譜階數
        sobolev_order: Sobolev 階數
        regularization_lambda: 正則化參數
        refinement_iterations: 細化迭代數
        adaptive_scheduling: 自適應調度
        lie_group_refinement: 李群細化

    Returns:
        samples: 採樣結果
    """
    # 推斷空間維度
    spatial_dims = x.shape[-2:]

    # 創建 ISDO 採樣器
    sampler = ISDOSampler(
        spatial_dims=spatial_dims,
        spectral_order=spectral_order,
        sobolev_order=sobolev_order,
        regularization_lambda=regularization_lambda,
        refinement_iterations=refinement_iterations,
        adaptive_scheduling=adaptive_scheduling,
        lie_group_refinement=lie_group_refinement,
        device=x.device,
        dtype=x.dtype
    )

    # 執行採樣
    return sampler.sample_isdo(
        model=model,
        x=x,
        sigmas=sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable
    )