#!/usr/bin/env python3
"""
ISDO 核心模組整合測試
===================

測試 VariationalODESystem 移動到 core 目錄後的整合是否成功。
"""

import sys
import os
import torch

# 添加項目根目錄到路徑
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_core_module_imports():
    """測試核心模組導入"""
    print("=== 測試核心模組導入 ===")

    try:
        # 測試從 core 導入 VariationalODESystem
        from ..isdo.core import VariationalODESystem
        print("✅ 成功從 core 導入 VariationalODESystem")

        # 測試直接導入
        from ..isdo.core.variational_ode_system import (
            VariationalODESystem,
            SpectralDynamics,
            AdaptiveStepSizeController,
            ConvergenceDetector
        )
        print("✅ 成功導入所有 VariationalODESystem 組件")

        # 測試其他核心模組
        from ..isdo.core import (
            SpectralBasis,
            HilbertSpace,
            VariationalController,
            SpectralProjection
        )
        print("✅ 成功導入所有核心模組")

        return True

    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False

def test_variational_ode_system_initialization():
    """測試 VariationalODESystem 初始化"""
    print("\n=== 測試 VariationalODESystem 初始化 ===")

    try:
        from ..isdo.core import VariationalODESystem

        # 基本初始化
        ode_system = VariationalODESystem(
            spatial_dims=(64, 64),
            spectral_order=128,
            sobolev_order=1.5,
            regularization_lambda=1e-4,
            sobolev_penalty=0.001
        )

        print("✅ VariationalODESystem 初始化成功")
        print(f"   - 空間維度: {ode_system.spatial_dims}")
        print(f"   - 譜階數: {ode_system.spectral_order}")
        print(f"   - 設備: {ode_system.device}")

        # 檢查組件
        assert hasattr(ode_system, 'spectral_basis'), "缺少 spectral_basis 組件"
        assert hasattr(ode_system, 'hilbert_space'), "缺少 hilbert_space 組件"
        assert hasattr(ode_system, 'spectral_dynamics'), "缺少 spectral_dynamics 組件"
        assert hasattr(ode_system, 'step_controller'), "缺少 step_controller 組件"
        assert hasattr(ode_system, 'convergence_detector'), "缺少 convergence_detector 組件"

        print("✅ 所有核心組件初始化成功")

        return ode_system

    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return None

def test_samplers_module_integration():
    """測試採樣器模組整合"""
    print("\n=== 測試採樣器模組整合 ===")

    try:
        # 測試採樣器模組不再直接導入 VariationalODESystem
        from ..isdo.samplers import ISDOSampler, UnifiedModelWrapper, sample_isdo
        print("✅ 成功導入採樣器模組")

        # 測試 ISDO 採樣器可以間接使用 VariationalODESystem（通過核心模組）
        sampler = ISDOSampler(
            spatial_dims=(32, 32),
            spectral_order=64
        )
        print("✅ ISDOSampler 初始化成功")

        return True

    except Exception as e:
        print(f"❌ 採樣器模組整合失敗: {e}")
        return False

def test_variational_ode_functionality():
    """測試 VariationalODESystem 基本功能"""
    print("\n=== 測試 VariationalODESystem 基本功能 ===")

    try:
        from ..isdo.core import VariationalODESystem
        from ..isdo.samplers import UnifiedModelWrapper

        # 創建測試模型
        class MockModel:
            def __init__(self):
                self.device = torch.device('cpu')

            def __call__(self, x, sigma, **kwargs):
                # 簡單的去噪模型模擬
                return x * 0.9

        mock_model = MockModel()
        model_wrapper = UnifiedModelWrapper(mock_model)

        # 初始化 ODE 系統
        ode_system = VariationalODESystem(
            spatial_dims=(16, 16),
            spectral_order=32,
            device=torch.device('cpu')
        )

        # 創建測試數據
        batch_size, channels = 1, 3
        initial_coeffs = torch.randn(batch_size, channels, ode_system.spectral_order)

        # 測試 ODE 求解
        final_coeffs, trajectory, solve_info = ode_system.solve_variational_ode(
            initial_coefficients=initial_coeffs,
            sigma_start=1.0,
            sigma_end=0.1,
            model_wrapper=model_wrapper,
            max_steps=10,
            adaptive_stepping=False
        )

        print("✅ VariationalODESystem 求解成功")
        print(f"   - 初始形狀: {initial_coeffs.shape}")
        print(f"   - 最終形狀: {final_coeffs.shape}")
        print(f"   - 求解步數: {solve_info['final_step']}")
        print(f"   - 收斂狀態: {solve_info.get('converged', 'Unknown')}")

        return True

    except Exception as e:
        print(f"❌ 功能測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forge_integration():
    """測試與 Forge WebUI 的整合"""
    print("\n=== 測試 Forge WebUI 整合 ===")

    try:
        # 測試整合模組
        from ..isdo_samplers_integration import ISDOSamplerWrapper, samplers_data_isdo
        print("✅ 成功導入 ISDO 整合模組")

        # 檢查採樣器數據
        print(f"✅ 可用 ISDO 採樣器數量: {len(samplers_data_isdo)}")
        for sampler_data in samplers_data_isdo:
            print(f"   - {sampler_data.name}")

        return True

    except Exception as e:
        print(f"❌ Forge 整合測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("ISDO 核心模組整合測試")
    print("=" * 50)

    # 運行所有測試
    tests = [
        test_core_module_imports,
        test_variational_ode_system_initialization,
        test_samplers_module_integration,
        test_variational_ode_functionality,
        test_forge_integration
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 測試 {test.__name__} 發生異常: {e}")
            results.append(False)

    # 總結
    print("\n" + "=" * 50)
    print("測試總結:")
    passed = sum(1 for r in results if r)
    total = len(results)

    if passed == total:
        print(f"🎉 所有測試通過 ({passed}/{total})")
        print("\nVariationalODESystem 已成功移動到 core 目錄並整合完成！")
        print("\n建議下一步:")
        print("1. 重啟 Forge WebUI")
        print("2. 在採樣器下拉選單中查找 ISDO 採樣器")
        print("3. 使用 ISDO 採樣器進行圖像生成")
    else:
        print(f"❌ 部分測試失敗 ({passed}/{total})")
        print("\n需要檢查的問題:")
        print("1. 檢查模組導入路徑")
        print("2. 確認所有依賴項已安裝")
        print("3. 檢查 Python 路徑設置")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)