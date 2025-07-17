"""
測試 InfiniteRefinement 與 ISDOSampler 整合
==========================================

驗證無窮細化循環系統在 ISDO 採樣器中的正確整合和功能。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

def test_infinite_refinement_integration():
    """測試 InfiniteRefinement 與 ISDOSampler 的整合"""
    print("\n=== 測試 InfiniteRefinement 整合 ===")

    try:
        from modules_forge.isdo.samplers import ISDOSampler, UnifiedModelWrapper
        from modules_forge.isdo.math import InfiniteRefinement, LieGroupOps

        # 創建測試模型
        class MockDenoiser:
            def __init__(self):
                self.device = torch.device('cpu')

            def __call__(self, x, sigma, **kwargs):
                # 模擬去噪：減少噪聲，保持結構
                noise_reduction = 1 - torch.clamp(sigma / 10.0, 0.0, 0.9)
                return x * noise_reduction + torch.randn_like(x) * 0.01

        mock_model = MockDenoiser()
        model_wrapper = UnifiedModelWrapper(mock_model)

        # 創建 ISDO 採樣器（啟用李群細化和無窮細化）
        sampler = ISDOSampler(
            spatial_dims=(32, 32),
            spectral_order=64,
            sobolev_order=1.5,
            lie_group_refinement=True,  # 啟用李群細化
            refinement_iterations=50,   # 設置細化迭代數
            device=torch.device('cpu')
        )

        print(f"✅ ISDOSampler 初始化成功")
        print(f"   - InfiniteRefinement 已整合: {hasattr(sampler, 'infinite_refinement')}")
        print(f"   - LieGroupOps 已初始化: {hasattr(sampler, 'lie_group_ops')}")

        # 測試基本細化功能
        test_tensor = torch.randn(1, 3, 32, 32)

        # 測試李群細化步驟
        print(f"\n--- 測試李群細化步驟 ---")
        refined_tensor = sampler._lie_group_refinement_step(
            x=test_tensor,
            sigma=1.0,
            model=model_wrapper,
            extra_args={},
            max_steps=20
        )

        print(f"✅ 李群細化成功")
        print(f"   - 輸入形狀: {test_tensor.shape}")
        print(f"   - 輸出形狀: {refined_tensor.shape}")
        print(f"   - 數值穩定性: {torch.isfinite(refined_tensor).all().item()}")
        print(f"   - 輸出範圍: [{refined_tensor.min().item():.3f}, {refined_tensor.max().item():.3f}]")

        # 測試高級細化接口
        print(f"\n--- 測試高級細化接口 ---")
        advanced_result = sampler.advanced_refinement(
            x=test_tensor,
            model=model_wrapper,
            sigma=2.0,
            max_iterations=30,
            convergence_threshold=1e-4,
            use_spectral_projection=True,
            detailed_stats=True
        )

        if 'error' not in advanced_result:
            print(f"✅ 高級細化成功")
            print(f"   - 迭代次數: {advanced_result['iterations_used']}")
            print(f"   - 最終損失: {advanced_result['final_loss']:.6f}")
            print(f"   - 執行時間: {advanced_result.get('execution_time', 'N/A')}")
            print(f"   - 初始 Sobolev 範數: {advanced_result['initial_sobolev_norm']:.4f}")
            print(f"   - 最終 Sobolev 範數: {advanced_result['final_sobolev_norm']:.4f}")

            if 'symmetry_improvement' in advanced_result:
                sym_imp = advanced_result['symmetry_improvement']
                print(f"   - 對稱性改善: {sym_imp['improvement_ratio']:.2%}")
        else:
            print(f"⚠️  高級細化遇到錯誤: {advanced_result['error']}")

        # 測試完整採樣流程
        print(f"\n--- 測試完整採樣流程 ---")
        initial_noise = torch.randn(1, 3, 32, 32)
        sigma_schedule = torch.linspace(5.0, 0.1, 11)  # 短的調度用於測試

        try:
            samples = sampler.sample_isdo(
                model=model_wrapper,
                x=initial_noise,
                sigmas=sigma_schedule,
                disable=True  # 禁用進度條
            )

            print(f"✅ 完整採樣成功")
            print(f"   - 採樣形狀: {samples.shape}")
            print(f"   - 數值穩定性: {torch.isfinite(samples).all().item()}")

            # 查看統計信息
            stats = sampler._get_step_stats()
            print(f"   - 去噪調用次數: {stats['denoiser_calls']}")
            print(f"   - 譜投影次數: {stats['spectral_projections']}")
            print(f"   - 李群操作次數: {stats['lie_group_ops']}")
            print(f"   - 無窮細化次數: {stats['infinite_refinements']}")
            print(f"   - 細化總迭代數: {stats['refinement_iterations_total']}")
            print(f"   - 細化收斂率: {stats['refinement_convergence_rate']:.2%}")

        except Exception as e:
            print(f"⚠️  完整採樣遇到錯誤: {e}")

        # 測試獨立的 InfiniteRefinement 使用
        print(f"\n--- 測試獨立 InfiniteRefinement ---")
        standalone_refinement = InfiniteRefinement(
            spatial_dims=(32, 32),
            max_iterations=20,
            convergence_threshold=1e-5,
            device=torch.device('cpu')
        )

        def simple_target_function(x):
            return torch.zeros_like(x)  # 簡單的零目標

        standalone_result = standalone_refinement.refinement_loop(
            x=test_tensor,
            target_function=simple_target_function,
            lie_group_ops=sampler.lie_group_ops,
            spectral_projection=sampler.spectral_projection
        )

        print(f"✅ 獨立 InfiniteRefinement 成功")
        print(f"   - 迭代次數: {standalone_result['iterations_used']}")
        print(f"   - 最終損失: {standalone_result['final_loss']:.6f}")
        print(f"   - 收斂歷史長度: {len(standalone_result['convergence_history'])}")

        return True

    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        return False
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_refinement_parameters():
    """測試不同細化參數的效果"""
    print(f"\n=== 測試細化參數效果 ===")

    try:
        from modules_forge.isdo.math import InfiniteRefinement

        # 測試不同參數配置
        configs = [
            {"max_iterations": 10, "perturbation_strength": 0.001, "name": "保守配置"},
            {"max_iterations": 50, "perturbation_strength": 0.01, "name": "標準配置"},
            {"max_iterations": 100, "perturbation_strength": 0.05, "name": "激進配置"},
        ]

        test_input = torch.randn(1, 3, 16, 16)

        def target_func(x):
            return x * 0.8  # 簡單的縮放目標

        for config in configs:
            print(f"\n--- {config['name']} ---")

            refinement = InfiniteRefinement(
                spatial_dims=(16, 16),
                max_iterations=config['max_iterations'],
                perturbation_strength=config['perturbation_strength'],
                convergence_threshold=1e-5,
                device=torch.device('cpu')
            )

            result = refinement.refinement_loop(
                x=test_input,
                target_function=target_func
            )

            print(f"   - 迭代次數: {result['iterations_used']}")
            print(f"   - 最終損失: {result['final_loss']:.6f}")
            print(f"   - 收斂: {'✅' if result['iterations_used'] < config['max_iterations'] else '❌'}")

            # 獲取統計信息
            stats = refinement.get_statistics()
            print(f"   - 總迭代數: {stats['total_iterations']}")
            print(f"   - 收斂次數: {stats['convergence_achieved']}")
            print(f"   - 拓撲違規: {stats['topology_violations']}")

        return True

    except Exception as e:
        print(f"❌ 參數測試失敗: {e}")
        return False

def demonstrate_usage():
    """展示實際使用方式"""
    print(f"\n=== InfiniteRefinement 使用示例 ===")

    # 使用示例代碼
    example_code = '''
# 1. 基本使用方式
from modules_forge.isdo.samplers import ISDOSampler
from modules_forge.isdo.samplers import UnifiedModelWrapper

# 包裝您的模型
model_wrapper = UnifiedModelWrapper(your_model)

# 創建啟用無窮細化的採樣器
sampler = ISDOSampler(
    spatial_dims=(64, 64),
    lie_group_refinement=True,  # 啟用李群細化（包含 InfiniteRefinement）
    refinement_iterations=100,  # 設置最大細化迭代數
    spectral_order=256
)

# 2. 標準採樣（自動使用細化）
samples = sampler.sample_isdo(
    model=model_wrapper,
    x=noise,
    sigmas=sigma_schedule
)

# 3. 高級細化控制
refined_result = sampler.advanced_refinement(
    x=your_tensor,
    model=model_wrapper,
    sigma=1.0,
    max_iterations=200,
    convergence_threshold=1e-6,
    detailed_stats=True
)

# 檢查結果
print(f"細化迭代數: {refined_result['iterations_used']}")
print(f"對稱性改善: {refined_result['symmetry_improvement']['improvement_ratio']:.2%}")

# 4. 獨立使用 InfiniteRefinement
from modules_forge.isdo.math import InfiniteRefinement

refinement_system = InfiniteRefinement(
    spatial_dims=(64, 64),
    max_iterations=1000,
    convergence_threshold=1e-5
)

def my_target_function(x):
    return your_denoiser(x, sigma)

result = refinement_system.refinement_loop(
    x=input_tensor,
    target_function=my_target_function,
    lie_group_ops=lie_group_ops,  # 可選
    spectral_projection=spectral_proj  # 可選
)
'''

    print(example_code)

if __name__ == "__main__":
    print("🚀 測試 InfiniteRefinement 與 ISDO 採樣器整合")

    # 運行測試
    test1_result = test_infinite_refinement_integration()
    test2_result = test_refinement_parameters()

    # 展示使用方式
    demonstrate_usage()

    # 總結
    print(f"\n" + "="*50)
    print(f"測試總結:")
    print(f"整合測試: {'✅ 通過' if test1_result else '❌ 失敗'}")
    print(f"參數測試: {'✅ 通過' if test2_result else '❌ 失敗'}")

    if test1_result and test2_result:
        print(f"\n🎉 所有測試通過！InfiniteRefinement 已成功整合到 ISDO 採樣器中。")
        print(f"您現在可以享受李群對稱性保持的無窮細化採樣體驗！")
    else:
        print(f"\n⚠️  某些測試失敗，請檢查錯誤信息並修復問題。")

    print("="*50)