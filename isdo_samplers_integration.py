"""
ISDO Samplers Integration for Forge WebUI
==========================================

將 Infinite Spectral Diffusion Odyssey (ISDO) 採樣器整合到 Stable Diffusion WebUI Forge 中。
參考 k_diffusion/sampling.py 的標準實現模式，提供完全兼容的採樣器接口。

整合的採樣器：
- ISDO Standard: 標準變分最優控制
- ISDO Adaptive: 自適應譜調度
- ISDO HQ: 高質量模式 (更多細化)
- ISDO Fast: 快速模式 (減少細化)
"""

import torch
from typing import Optional, Dict, Any, Callable
from tqdm import trange
from modules import sd_samplers_common, sd_samplers_kdiffusion

import traceback

def to_d(x, sigma, denoised):
    """
    兼容函數：計算 ODE 的漂移項
    與 k_diffusion/sampling.py 中的 to_d 函數相同
    """
    try:
        from k_diffusion import utils
        return (x - denoised) / utils.append_dims(sigma, x.ndim)
    except ImportError:
        # 回退實現：手動實現 append_dims
        dims_to_append = x.ndim - sigma.ndim
        if dims_to_append > 0:
            sigma = sigma[(...,) + (None,) * dims_to_append]
        return (x - denoised) / sigma


@torch.no_grad()
def sample_isdo_standard(model, x, sigmas, extra_args=None, callback=None, disable=None,
                        spectral_order=256, sobolev_order=1.5, regularization_lambda=1e-4):
    """
    ISDO Standard 採樣器 - 標準變分最優控制

    參考 sample_euler 的實現模式，但使用 ISDO 變分最優控制算法

    Args:
        model: 去噪模型
        x: 初始噪聲張量 (B, C, H, W)
        sigmas: 噪聲水平序列 (T,)
        extra_args: 額外參數字典
        callback: 回調函數
        disable: 是否禁用進度條
        spectral_order: 譜截斷階數
        sobolev_order: Sobolev 空間階數
        regularization_lambda: 變分正則化參數

    Returns:
        x: 最終採樣結果
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # 嘗試初始化 ISDO 組件
    try:
        from .isdo.samplers.isdo_sampler import ISDOSampler

        # 獲取空間維度
        spatial_dims = x.shape[-2:]

        # 創建 ISDO 採樣器
        isdo_sampler = ISDOSampler(
            spatial_dims=spatial_dims,
            spectral_order=spectral_order,
            sobolev_order=sobolev_order,
            regularization_lambda=regularization_lambda,
            curvature_penalty=0.001,
            refinement_iterations=1000,
            adaptive_scheduling=False,  # 標準模式不使用自適應
            lie_group_refinement=False,  # 標準模式不使用李群細化
            device=x.device,
            dtype=x.dtype
        )

        use_isdo = True

    except Exception as e:
        print(f"ISDO 初始化失敗，回退到 Euler 方法: {e}")
        print(traceback.format_exc())
        use_isdo = False

    # 主採樣循環 - 遵循 sample_euler 的結構
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_current = sigmas[i]
        sigma_next = sigmas[i + 1]

        if use_isdo:
            # 使用 ISDO 變分最優控制步驟
            try:
                # ISDO 內部會調用模型
                x = _isdo_variational_step(
                    x, sigma_current, sigma_next, None, # denoised is now None
                    isdo_sampler, model, extra_args
                )
            except Exception as e:
                print(f"ISDO 步驟失敗，回退到 Euler: {e}")
                print(traceback.format_exc())
                # 回退時需要計算 denoised
                denoised = model(x, sigma_current * s_in, **extra_args)
                d = to_d(x, sigma_current, denoised)
                dt = sigma_next - sigma_current
                x = x + d * dt
        else:
            # 標準 Euler 方法
            denoised = model(x, sigma_current * s_in, **extra_args)
            d = to_d(x, sigma_current, denoised)
            dt = sigma_next - sigma_current
            x = x + d * dt

        # 執行回調
        if callback is not None:
            # 為了回調，我們需要計算一次 denoised
            denoised_for_callback = model(x, sigma_current * s_in, **extra_args)
            callback({
                'x': x,
                'i': i,
                'sigma': sigmas[i],
                'sigma_hat': sigma_current,
                'denoised': denoised_for_callback
            })

    return x


@torch.no_grad()
def sample_isdo_adaptive(model, x, sigmas, extra_args=None, callback=None, disable=None,
                        spectral_order=512, sobolev_order=1.8, regularization_lambda=1e-4):
    """
    ISDO Adaptive 採樣器 - 自適應譜調度
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # 初始化 ISDO 組件
    try:
        from .isdo.samplers.isdo_sampler import ISDOSampler

        spatial_dims = x.shape[-2:]

        isdo_sampler = ISDOSampler(
            spatial_dims=spatial_dims,
            spectral_order=spectral_order,
            sobolev_order=sobolev_order,
            regularization_lambda=regularization_lambda,
            curvature_penalty=0.002,
            refinement_iterations=1500,
            adaptive_scheduling=True,  # 啟用自適應調度
            lie_group_refinement=False,
            device=x.device,
            dtype=x.dtype
        )

        use_isdo = True

    except Exception as e:
        print(f"ISDO Adaptive 初始化失敗，回退到 Euler: {e}")
        print(traceback.format_exc())
        use_isdo = False

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_current = sigmas[i]
        sigma_next = sigmas[i + 1]

        if use_isdo:
            try:
                # 自適應 ISDO 步驟
                x = _isdo_adaptive_step(
                    x, sigma_current, sigma_next, None, # denoised is now None
                    isdo_sampler, model, extra_args, i
                )
            except Exception as e:
                print(f"ISDO Adaptive 步驟失敗，回退到 Euler: {e}")
                print(traceback.format_exc())
                denoised = model(x, sigma_current * s_in, **extra_args)
                d = to_d(x, sigma_current, denoised)
                dt = sigma_next - sigma_current
                x = x + d * dt
        else:
            denoised = model(x, sigma_current * s_in, **extra_args)
            d = to_d(x, sigma_current, denoised)
            dt = sigma_next - sigma_current
            x = x + d * dt

        if callback is not None:
            denoised_for_callback = model(x, sigma_current * s_in, **extra_args)
            callback({
                'x': x,
                'i': i,
                'sigma': sigmas[i],
                'sigma_hat': sigma_current,
                'denoised': denoised_for_callback
            })

    return x


@torch.no_grad()
def sample_isdo_hq(model, x, sigmas, extra_args=None, callback=None, disable=None,
                   spectral_order=1024, sobolev_order=2.0, regularization_lambda=1e-4):
    """
    ISDO HQ 採樣器 - 高質量模式 (更多細化)
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # 初始化 ISDO 組件
    try:
        from .isdo.samplers.isdo_sampler import ISDOSampler

        spatial_dims = x.shape[-2:]

        isdo_sampler = ISDOSampler(
            spatial_dims=spatial_dims,
            spectral_order=spectral_order,
            sobolev_order=sobolev_order,
            regularization_lambda=regularization_lambda,
            curvature_penalty=0.003,
            refinement_iterations=3000,
            adaptive_scheduling=True,
            lie_group_refinement=True,  # 啟用李群細化
            device=x.device,
            dtype=x.dtype
        )

        use_isdo = True

    except Exception as e:
        print(f"ISDO HQ 初始化失敗，回退到 Heun: {e}")
        print(traceback.format_exc())
        use_isdo = False

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_current = sigmas[i]
        sigma_next = sigmas[i + 1]

        if use_isdo:
            try:
                # 高質量 ISDO 步驟 (包含李群細化)
                x = _isdo_hq_step(
                    x, sigma_current, sigma_next, None, # denoised is now None
                    isdo_sampler, model, extra_args, i
                )
            except Exception as e:
                print(f"ISDO HQ 步驟失敗，回退到 Heun: {e}")
                print(traceback.format_exc())
                # 回退到 Heun 方法 (更好的二階方法)
                denoised = model(x, sigma_current * s_in, **extra_args)
                d = to_d(x, sigma_current, denoised)
                dt = sigma_next - sigma_current
                if sigma_next == 0:
                    x = x + d * dt
                else:
                    x_2 = x + d * dt
                    denoised_2 = model(x_2, sigma_next * s_in, **extra_args)
                    d_2 = to_d(x_2, sigma_next, denoised_2)
                    d_prime = (d + d_2) / 2
                    x = x + d_prime * dt
        else:
            # Heun 方法作為高質量回退
            denoised = model(x, sigma_current * s_in, **extra_args)
            d = to_d(x, sigma_current, denoised)
            dt = sigma_next - sigma_current
            if sigma_next == 0:
                x = x + d * dt
            else:
                x_2 = x + d * dt
                denoised_2 = model(x_2, sigma_next * s_in, **extra_args)
                d_2 = to_d(x_2, sigma_next, denoised_2)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt

        if callback is not None:
            denoised_for_callback = model(x, sigma_current * s_in, **extra_args)
            callback({
                'x': x,
                'i': i,
                'sigma': sigmas[i],
                'sigma_hat': sigma_current,
                'denoised': denoised_for_callback
            })

    return x


@torch.no_grad()
def sample_isdo_fast(model, x, sigmas, extra_args=None, callback=None, disable=None,
                     spectral_order=128, sobolev_order=1.2, regularization_lambda=1e-4):
    """
    ISDO Fast 採樣器 - 快速模式 (減少細化)
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # 初始化 ISDO 組件
    try:
        from .isdo.samplers.isdo_sampler import ISDOSampler

        spatial_dims = x.shape[-2:]

        isdo_sampler = ISDOSampler(
            spatial_dims=spatial_dims,
            spectral_order=spectral_order,
            sobolev_order=sobolev_order,
            regularization_lambda=regularization_lambda,
            curvature_penalty=0.0005,
            refinement_iterations=500,
            adaptive_scheduling=False,  # 快速模式不使用自適應
            lie_group_refinement=False,  # 快速模式不使用李群細化
            device=x.device,
            dtype=x.dtype
        )

        use_isdo = True

    except Exception as e:
        print(f"ISDO Fast 初始化失敗，回退到 Euler: {e}")
        print(traceback.format_exc())
        use_isdo = False

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_current = sigmas[i]
        sigma_next = sigmas[i + 1]

        if use_isdo:
            try:
                # 快速 ISDO 步驟
                x = _isdo_fast_step(
                    x, sigma_current, sigma_next, None, # denoised is now None
                    isdo_sampler, model, extra_args
                )
            except Exception as e:
                print(f"ISDO Fast 步驟失敗，回退到 Euler: {e}")
                print(traceback.format_exc())
                denoised = model(x, sigma_current * s_in, **extra_args)
                d = to_d(x, sigma_current, denoised)
                dt = sigma_next - sigma_current
                x = x + d * dt
        else:
            denoised = model(x, sigma_current * s_in, **extra_args)
            d = to_d(x, sigma_current, denoised)
            dt = sigma_next - sigma_current
            x = x + d * dt

        if callback is not None:
            denoised_for_callback = model(x, sigma_current * s_in, **extra_args)
            callback({
                'x': x,
                'i': i,
                'sigma': sigmas[i],
                'sigma_hat': sigma_current,
                'denoised': denoised_for_callback
            })

    return x


def _isdo_variational_step(x, sigma_current, sigma_next,
                          isdo_sampler, model, extra_args):
    """
    執行單步 ISDO 變分最優控制
    """
    try:
        # 投影到譜空間
        spectral_coeffs = isdo_sampler.spectral_projection(x, mode='forward')

        # 在譜空間執行變分控制步驟
        updated_coeffs = isdo_sampler.spectral_rk4.step(
            spectral_coeffs, float(sigma_current), float(sigma_next),
            model, extra_args
        )

        # 重建到空間域
        x_updated = isdo_sampler.spectral_projection(updated_coeffs, mode='inverse')

        return x_updated

    except Exception as e:
        print(f"變分控制步驟失敗: {e}")
        print(traceback.format_exc())
        # 回退到標準 Euler
        d = to_d(x, sigma_current, denoised)
        dt = sigma_next - sigma_current
        return x + d * dt


def _isdo_adaptive_step(x, sigma_current, sigma_next,
                       isdo_sampler, model, extra_args, step_index):
    """
    執行自適應 ISDO 步驟
    """
    try:
        # 投影到譜空間
        spectral_coeffs = isdo_sampler.spectral_projection(x, mode='forward')

        # 自適應步長控制
        dt = sigma_next - sigma_current

        # 嘗試全步長
        full_step_coeffs = isdo_sampler.spectral_rk4.step(
            spectral_coeffs, float(sigma_current), float(sigma_next),
            model, extra_args
        )

        # 嘗試半步長 (用於誤差估計)
        sigma_mid = (sigma_current + sigma_next) / 2
        half_step1_coeffs = isdo_sampler.spectral_rk4.step(
            spectral_coeffs, float(sigma_current), float(sigma_mid),
            model, extra_args
        )
        half_step2_coeffs = isdo_sampler.spectral_rk4.step(
            half_step1_coeffs, float(sigma_mid), float(sigma_next),
            model, extra_args
        )

        # 估計局部誤差
        error_estimate = torch.norm(full_step_coeffs - half_step2_coeffs)

        # 根據誤差選擇結果
        if error_estimate > 0.1:  # 誤差閾值
            final_coeffs = half_step2_coeffs  # 使用更精確的半步長結果
        else:
            final_coeffs = full_step_coeffs

        # 重建到空間域
        x_updated = isdo_sampler.spectral_projection(final_coeffs, mode='inverse')

        return x_updated

    except Exception as e:
        print(f"自適應步驟失敗: {e}")
        print(traceback.format_exc())
        d = to_d(x, sigma_current, denoised)
        dt = sigma_next - sigma_current
        return x + d * dt


def _isdo_hq_step(x, sigma_current, sigma_next,
                 isdo_sampler, model, extra_args, step_index):
    """
    執行高質量 ISDO 步驟 (包含李群細化)
    """
    try:
        # 首先執行標準變分步驟
        x_variational = _isdo_variational_step(
            x, sigma_current, sigma_next, denoised,
            isdo_sampler, model, extra_args
        )

        # 每5步執行一次李群細化
        if step_index % 5 == 0 and hasattr(isdo_sampler, 'lie_group_ops'):
            try:
                # 檢測對稱性破壞
                violations = isdo_sampler.lie_group_ops.detect_symmetry_violations(x_variational)

                if violations['total_violation'] > 0.01:  # 需要細化
                    # 執行短程細化 (避免過長的細化過程)
                    refined_x = _short_lie_group_refinement(
                        x_variational, float(sigma_next), model, extra_args, isdo_sampler
                    )
                    return refined_x

            except Exception as e:
                print(f"李群細化失敗: {e}")
                print(traceback.format_exc())

        return x_variational

    except Exception as e:
        print(f"高質量步驟失敗: {e}")
        print(traceback.format_exc())
        # 回退到 Heun 方法
        d = to_d(x, sigma_current, denoised)
        dt = sigma_next - sigma_current
        if sigma_next == 0:
            return x + d * dt
        else:
            s_in = x.new_ones([x.shape[0]])
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigma_next * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_next, denoised_2)
            d_prime = (d + d_2) / 2
            return x + d_prime * dt


def _isdo_fast_step(x, sigma_current, sigma_next,
                   isdo_sampler, model, extra_args):
    """
    執行快速 ISDO 步驟 (簡化版變分控制)
    """
    try:
        # 簡化的譜投影 (使用較低的階數)
        spatial_dims = x.shape[-2:]

        # 直接在空間域執行簡化的變分修正
        denoised = model(x, sigma_current * x.new_ones([x.shape[0]]), **extra_args)
        d_euler = to_d(x, sigma_current, denoised)

        # 計算 Sobolev 範數修正 (簡化版)
        sobolev_correction = isdo_sampler.hilbert_space.compute_sobolev_norm(d_euler)
        # 自動根據 d_euler 的 shape 調整 correction_factor
        if sobolev_correction.ndim == 1 and d_euler.ndim == 4:
            # [B] -> [B, 1, 1, 1]
            correction_factor = 1.0 / (1.0 + isdo_sampler.regularization_lambda * sobolev_correction.view(-1, 1, 1, 1))
        elif sobolev_correction.ndim == 2 and d_euler.ndim == 4:
            # [B, C] -> [B, C, 1, 1]
            correction_factor = 1.0 / (1.0 + isdo_sampler.regularization_lambda * sobolev_correction.view(sobolev_correction.shape[0], sobolev_correction.shape[1], 1, 1))
        else:
            # 通用展開
            expand_shape = list(sobolev_correction.shape) + [1] * (d_euler.ndim - sobolev_correction.ndim)
            correction_factor = 1.0 / (1.0 + isdo_sampler.regularization_lambda * sobolev_correction.view(*expand_shape))
        d_corrected = d_euler * correction_factor

        dt = sigma_next - sigma_current
        return x + d_corrected * dt

    except Exception as e:
        print(f"快速步驟失敗: {e}")
        print(traceback.format_exc())
        # 直接回退到 Euler
        d = to_d(x, sigma_current, denoised)
        dt = sigma_next - sigma_current
        return x + d * dt


def _short_lie_group_refinement(x, sigma, model, extra_args, isdo_sampler, num_steps=3):
    """
    短程李群細化 (用於 HQ 模式)
    """
    try:
        x_current = x.clone()

        for _ in range(num_steps):
            # 小的對稱性修正
            if hasattr(isdo_sampler, 'lie_group_ops'):
                correction = isdo_sampler.lie_group_ops.compute_symmetry_correction(x_current)
                x_current = x_current + 0.01 * correction  # 小修正

            # 重新投影到流形
            if hasattr(isdo_sampler, 'spectral_projection'):
                spectral_coeffs = isdo_sampler.spectral_projection(x_current, mode='forward')
                x_current = isdo_sampler.spectral_projection(spectral_coeffs, mode='inverse')

        return x_current

    except Exception as e:
        print(f"短程細化失敗: {e}")
        print(traceback.format_exc())
        return x


# 創建採樣器構造函數
def build_isdo_constructor(sampler_func):
    """
    建構 ISDO 採樣器的構造函數

    Args:
        sampler_func: ISDO 採樣函數

    Returns:
        constructor: 採樣器構造函數
    """
    def constructor(model):
        return sd_samplers_kdiffusion.KDiffusionSampler(sampler_func, model)

    return constructor


# ISDO 採樣器數據定義
samplers_data_isdo = [
    sd_samplers_common.SamplerData(
        'ISDO',
        build_isdo_constructor(sample_isdo_standard),
        ['isdo', 'isdo_standard'],
        {
            'description': '標準 ISDO 變分最優控制採樣',
            'scheduler': 'automatic',
            'second_order': False,
        }
    ),
    sd_samplers_common.SamplerData(
        'ISDO Adaptive',
        build_isdo_constructor(sample_isdo_adaptive),
        ['isdo_adaptive', 'isdo_adapt'],
        {
            'description': '自適應 ISDO，動態調整譜階數和步長',
            'scheduler': 'automatic',
            'second_order': False,
        }
    ),
    sd_samplers_common.SamplerData(
        'ISDO HQ',
        build_isdo_constructor(sample_isdo_hq),
        ['isdo_hq', 'isdo_high_quality'],
        {
            'description': '高質量 ISDO，包含李群對稱細化',
            'scheduler': 'automatic',
            'second_order': True,  # 標記為二階方法 (因為有額外細化)
        }
    ),
    sd_samplers_common.SamplerData(
        'ISDO Fast',
        build_isdo_constructor(sample_isdo_fast),
        ['isdo_fast', 'isdo_speed'],
        {
            'description': '快速 ISDO，優化速度的簡化版本',
            'scheduler': 'automatic',
            'second_order': False,
        }
    ),
]


def get_isdo_samplers():
    """獲取所有 ISDO 採樣器"""
    return samplers_data_isdo


def register_isdo_samplers():
    """註冊 ISDO 採樣器到系統"""
    try:
        from modules import sd_samplers

        for sampler_data in samplers_data_isdo:
            sd_samplers.add_sampler(sampler_data)

        print(f"成功註冊 {len(samplers_data_isdo)} 個 ISDO 採樣器")
        for sampler_data in samplers_data_isdo:
            print(f"  - {sampler_data.name}: {sampler_data.options.get('description', 'ISDO 採樣器')}")

    except Exception as e:
        print(f"註冊 ISDO 採樣器時發生錯誤: {e}")
        print(traceback.format_exc())


# 如果直接運行此模組，註冊採樣器
if __name__ == "__main__":
    register_isdo_samplers()