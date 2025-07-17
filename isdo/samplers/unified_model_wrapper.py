"""
Unified Model Wrapper for ISDO
==============================

統一不同去噪模型的接口，支援 ε-pred、v-pred、flow matching 等預測目標。
提供一致的 D(x; σ) 黑盒接口，簡化 ISDO 算法的實現。

支援的模型類型:
- EDM 風格模型 (ε-prediction)
- v-parameterization
- Flow matching
- Score-based 模型
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Dict, List, Callable, Any
import inspect


class ModelType:
    """模型類型識別"""
    EPSILON = "epsilon"    # ε-prediction (標準 DDPM)
    V_PARAM = "v_param"    # v-parameterization
    FLOW = "flow"          # Flow matching
    SCORE = "score"        # Score-based
    AUTO = "auto"          # 自動檢測


class UnifiedModelWrapper:
    """
    統一模型包裝器

    將不同類型的去噪模型統一為 D(x; σ) 接口，
    其中 D(x; σ) ≈ x₀ (乾淨數據的估計)
    """

    def __init__(
        self,
        model: Any,
        model_type: str = ModelType.AUTO,
        parameterization: Optional[str] = None,
        sigma_data: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        初始化統一模型包裝器

        Args:
            model: 原始去噪模型
            model_type: 模型類型
            parameterization: 參數化方式
            sigma_data: 數據縮放參數
            device: 計算設備
        """
        self.model = model
        self.model_type = model_type
        self.parameterization = parameterization
        self.sigma_data = sigma_data
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 移動模型到指定設備
        if hasattr(model, 'to'):
            self.model = model.to(self.device)

        # 自動檢測模型類型
        if model_type == ModelType.AUTO:
            self.model_type = self._auto_detect_model_type()

        # 緩存計算結果以提高效率
        self._cache = {}
        self._cache_size = 100

    def _auto_detect_model_type(self) -> str:
        """
        自動檢測模型類型

        根據模型的簽名、屬性或輸出特徵推斷模型類型
        """
        # 檢查模型屬性
        if hasattr(self.model, 'parameterization'):
            param = getattr(self.model, 'parameterization')
            if 'v' in param.lower():
                return ModelType.V_PARAM
            elif 'eps' in param.lower() or 'epsilon' in param.lower():
                return ModelType.EPSILON
            elif 'flow' in param.lower():
                return ModelType.FLOW

        # 檢查方法簽名
        if hasattr(self.model, 'forward') or callable(self.model):
            sig = inspect.signature(self.model.forward if hasattr(self.model, 'forward') else self.model)
            params = list(sig.parameters.keys())

            # 常見的參數名稱模式
            if 'timestep' in params or 't' in params:
                return ModelType.EPSILON  # 可能是擴散模型
            elif 'sigma' in params:
                return ModelType.EPSILON  # EDM 風格
            elif 'flow_time' in params:
                return ModelType.FLOW

        # 默認假設為 ε-prediction
        return ModelType.EPSILON

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        統一調用接口，包含對 Classifier-Free Guidance 的處理

        Args:
            x: 噪聲圖像 (B, C, H, W)
            sigma: 噪聲水平 (B,) 或標量
            **kwargs: 額外參數，可能包含 'cond', 'uncond', 'cond_scale'

        Returns:
            x0_pred: 預測的乾淨圖像 D(x; σ)
        """
        # 檢查是否需要 CFG
        cond = kwargs.get('cond', None)
        uncond = kwargs.get('uncond', None)
        cond_scale = kwargs.get('cond_scale', 1.0)

        # 如果沒有提供引導參數，則直接調用
        if cond is None or uncond is None or cond_scale == 1.0:
            return self._get_x0_pred(x, sigma, **kwargs)

        # --- CFG 邏輯 ---
        # 1. 分別計算條件性和非條件性預測
        
        # 準備條件性預測的參數
        cond_kwargs = kwargs.copy()
        cond_kwargs['cond'] = cond
        
        # 準備非條件性預測的參數
        uncond_kwargs = kwargs.copy()
        uncond_kwargs['cond'] = uncond # 模型通常期望 'cond' 參數
        
        # 計算 x0 預測
        x0_cond_pred = self._get_x0_pred(x, sigma, **cond_kwargs)
        x0_uncond_pred = self._get_x0_pred(x, sigma, **uncond_kwargs)

        # 2. 應用引導
        # 公式: x0 = uncond + guidance_scale * (cond - uncond)
        x0_pred = x0_uncond_pred + cond_scale * (x0_cond_pred - x0_uncond_pred)

        return x0_pred

    def _get_x0_pred(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        內部函數，計算單個 x0 預測（無 CFG）
        """
        # 確保 sigma 的形狀正確
        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0])
        elif sigma.dim() == 1 and sigma.shape[0] != x.shape[0]:
            sigma = sigma.expand(x.shape[0])

        # 檢查緩存
        cache_key = self._compute_cache_key(x, sigma, kwargs)
        if cache_key in self._cache:
            return self._cache[cache_key].clone()

        # 根據模型類型調用相應的轉換函數
        if self.model_type == ModelType.EPSILON:
            x0_pred = self._epsilon_to_x0(x, sigma, **kwargs)
        elif self.model_type == ModelType.V_PARAM:
            x0_pred = self._v_param_to_x0(x, sigma, **kwargs)
        elif self.model_type == ModelType.FLOW:
            x0_pred = self._flow_to_x0(x, sigma, **kwargs)
        elif self.model_type == ModelType.SCORE:
            x0_pred = self._score_to_x0(x, sigma, **kwargs)
        else:
            raise ValueError(f"不支援的模型類型: {self.model_type}")

        # 更新緩存
        self._update_cache(cache_key, x0_pred)

        return x0_pred

    def _epsilon_to_x0(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        ε-prediction 模型轉換為 x₀

        x = x₀ + σ·ε
        => x₀ = x - σ·ε_θ(x, σ)
        """
        # 調用模型獲取噪聲預測
        eps_pred = self._call_model(x, sigma, **kwargs)

        # 轉換為 x₀
        sigma_expanded = sigma.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        x0_pred = x - sigma_expanded * eps_pred

        return x0_pred

    def _v_param_to_x0(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        v-parameterization 模型轉換為 x₀

        v = α_t·ε - σ_t·x₀
        => x₀ = (α_t·x - σ_t·v) / α_t²
        """
        # 調用模型獲取 v 預測
        v_pred = self._call_model(x, sigma, **kwargs)

        # 計算 SNR 參數
        sigma_expanded = sigma.view(-1, 1, 1, 1)
        sigma_data_sq = self.sigma_data ** 2

        # EDM 參數化
        alpha_t = sigma_data_sq / (sigma_expanded ** 2 + sigma_data_sq) ** 0.5
        sigma_t = sigma_expanded / (sigma_expanded ** 2 + sigma_data_sq) ** 0.5

        # 轉換為 x₀
        x0_pred = (alpha_t * x - sigma_t * v_pred) / alpha_t

        return x0_pred

    def _flow_to_x0(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Flow matching 模型轉換為 x₀

        在 flow matching 中，模型直接預測速度場 v_t
        需要積分來獲取 x₀
        """
        # 調用模型獲取速度場
        v_pred = self._call_model(x, sigma, **kwargs)

        # 簡化假設：在當前時間步，v ≈ (x₀ - x) / t
        # 這是一個近似，實際 flow matching 需要更複雜的積分
        sigma_expanded = sigma.view(-1, 1, 1, 1)

        # 避免除零
        time_factor = torch.clamp(sigma_expanded, min=1e-6)
        x0_pred = x + v_pred * time_factor

        return x0_pred

    def _score_to_x0(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Score-based 模型轉換為 x₀

        score = ∇_x log p(x_t)
        通過 Tweedie 公式: x₀ = x + σ²·score
        """
        # 調用模型獲取 score
        score_pred = self._call_model(x, sigma, **kwargs)

        # Tweedie 公式
        sigma_expanded = sigma.view(-1, 1, 1, 1)
        x0_pred = x + sigma_expanded ** 2 * score_pred

        return x0_pred

    def _call_model(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        調用底層模型的統一接口

        處理不同模型的參數命名差異
        """
        # 準備模型輸入
        model_kwargs = kwargs.copy()

        # 根據模型簽名調整參數名稱
        if hasattr(self.model, 'forward'):
            sig = inspect.signature(self.model.forward)
        elif callable(self.model):
            sig = inspect.signature(self.model)
        else:
            raise ValueError("模型不可調用")

        param_names = list(sig.parameters.keys())

        # 處理時間/噪聲參數的不同命名
        if 'timestep' in param_names:
            model_kwargs['timestep'] = sigma
        elif 't' in param_names:
            model_kwargs['t'] = sigma
        elif 'sigma' in param_names:
            model_kwargs['sigma'] = sigma
        elif 'noise_level' in param_names:
            model_kwargs['noise_level'] = sigma

        # 調用模型
        try:
            if hasattr(self.model, 'forward'):
                output = self.model.forward(x, **model_kwargs)
            else:
                output = self.model(x, **model_kwargs)
        except Exception as e:
            # 回退到最簡單的調用方式
            try:
                output = self.model(x, sigma)
            except Exception as e2:
                raise RuntimeError(f"模型調用失敗: {e}, 回退調用也失敗: {e2}")

        return output

    def _compute_cache_key(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        kwargs: Dict
    ) -> str:
        """
        計算緩存鍵值
        """
        # 簡化的哈希計算（實際使用中可能需要更複雜的方案）
        x_hash = hash(tuple(x.flatten()[:100].tolist()))  # 只取前100個元素
        sigma_hash = hash(tuple(sigma.flatten().tolist()))
        kwargs_hash = hash(str(sorted(kwargs.items())))

        return f"{x_hash}_{sigma_hash}_{kwargs_hash}"

    def _update_cache(self, key: str, value: torch.Tensor):
        """
        更新緩存，實現 LRU 策略
        """
        if len(self._cache) >= self._cache_size:
            # 移除最舊的項目
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = value.clone()

    def get_model_info(self) -> Dict[str, Any]:
        """
        獲取模型信息
        """
        info = {
            'model_type': self.model_type,
            'parameterization': self.parameterization,
            'sigma_data': self.sigma_data,
            'device': str(self.device),
            'cache_size': len(self._cache),
            'cache_limit': self._cache_size
        }

        # 添加模型特定信息
        if hasattr(self.model, '__class__'):
            info['model_class'] = self.model.__class__.__name__

        if hasattr(self.model, 'config'):
            info['model_config'] = str(self.model.config)

        return info

    def clear_cache(self):
        """清空緩存"""
        self._cache.clear()

    def set_cache_size(self, size: int):
        """設置緩存大小"""
        self._cache_size = size
        if len(self._cache) > size:
            # 保留最新的項目
            items = list(self._cache.items())
            self._cache = dict(items[-size:])

    def compute_denoising_strength(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        計算去噪強度

        用於評估模型在特定噪聲水平下的去噪效果
        """
        x0_pred = self(x, sigma, **kwargs)

        # 計算去噪強度 = ||x - x₀|| / ||x||
        noise_removed = torch.norm(x - x0_pred, dim=(1, 2, 3))
        original_norm = torch.norm(x, dim=(1, 2, 3))

        denoising_strength = noise_removed / (original_norm + 1e-8)

        return denoising_strength

    def estimate_sigma_data(
        self,
        x_samples: torch.Tensor,
        num_samples: int = 100
    ) -> float:
        """
        從樣本估計 σ_data 參數

        Args:
            x_samples: 乾淨樣本
            num_samples: 用於估計的樣本數

        Returns:
            estimated_sigma_data: 估計的 σ_data
        """
        if x_samples.shape[0] > num_samples:
            indices = torch.randperm(x_samples.shape[0])[:num_samples]
            x_samples = x_samples[indices]

        # 計算樣本的標準差
        samples_flat = x_samples.view(x_samples.shape[0], -1)
        sigma_data_est = torch.std(samples_flat).item()

        return sigma_data_est

    def validate_model_output(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs
    ) -> Dict[str, bool]:
        """
        驗證模型輸出的正確性

        Returns:
            validation_results: 驗證結果
        """
        try:
            output = self(x, sigma, **kwargs)

            results = {
                'output_shape_correct': output.shape == x.shape,
                'output_finite': torch.isfinite(output).all().item(),
                'output_not_nan': not torch.isnan(output).any().item(),
                'output_reasonable_range': (output.abs() < 100).all().item(),
                'output_type_correct': output.dtype == x.dtype
            }

            results['all_checks_passed'] = all(results.values())

        except Exception as e:
            results = {
                'output_shape_correct': False,
                'output_finite': False,
                'output_not_nan': False,
                'output_reasonable_range': False,
                'output_type_correct': False,
                'all_checks_passed': False,
                'error': str(e)
            }

        return results