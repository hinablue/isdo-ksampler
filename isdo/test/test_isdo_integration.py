#!/usr/bin/env python3
"""
ISDO 採樣器整合測試腳本
=====================

測試 ISDO 採樣器是否成功整合到 Forge WebUI 中。
"""

import sys
import os

# 添加項目根目錄到路徑
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_isdo_imports():
    """測試 ISDO 模組導入"""
    print("測試 ISDO 模組導入...")

    try:
        # 測試核心模組
        print("  導入核心模組...")
        from ..isdo.core.spectral_basis import SpectralBasis
        from ..isdo.core.hilbert_space import HilbertSpace
        from ..isdo.core.variational_controller import VariationalController
        from ..isdo.core.spectral_projection import SpectralProjection
        print("    ✓ 核心模組導入成功")

        # 測試數學模組
        print("  導入數學模組...")
        from isdo.numerics.spectral_rk4 import SpectralRK4
from isdo.numerics.lie_group_ops import LieGroupOps
        print("    ✓ 數學模組導入成功")

        # 測試採樣器模組
        print("  導入採樣器模組...")
        from ..isdo.samplers.isdo_sampler import ISDOSampler, sample_isdo
        from ..isdo.samplers.unified_model_wrapper import UnifiedModelWrapper
        print("    ✓ 採樣器模組導入成功")

        # 測試整合模組
        print("  導入整合模組...")
        from ..isdo_samplers_integration import (
            ISDOSamplerWrapper,
            samplers_data_isdo,
            register_isdo_samplers
        )
        print("    ✓ 整合模組導入成功")

        return True

    except ImportError as e:
        print(f"    ✗ 導入錯誤: {e}")
        return False
    except Exception as e:
        print(f"    ✗ 其他錯誤: {e}")
        return False


def test_sampler_registration():
    """測試採樣器註冊"""
    print("測試採樣器註冊...")

    try:
        # 導入註冊函數
        from ..isdo_samplers_integration import (
            samplers_data_isdo,
            ISDO_SAMPLER_CONFIGS
        )

        # 檢查採樣器數據
        print(f"  發現 {len(samplers_data_isdo)} 個 ISDO 採樣器:")
        for sampler_data in samplers_data_isdo:
            config = sampler_data.options.get('isdo_config', {})
            print(f"    - {sampler_data.name}: {config.get('description', 'ISDO 採樣器')}")
            print(f"      別名: {sampler_data.aliases}")

        # 檢查配置
        print(f"  配置項目:")
        for name, config in ISDO_SAMPLER_CONFIGS.items():
            print(f"    - {name}:")
            print(f"      譜階數: {config['spectral_order']}")
            print(f"      Sobolev 階數: {config['sobolev_order']}")
            print(f"      自適應調度: {config['adaptive_scheduling']}")
            print(f"      李群細化: {config['lie_group_refinement']}")

        return True

    except Exception as e:
        print(f"    ✗ 註冊測試錯誤: {e}")
        return False


def test_basic_functionality():
    """測試基本功能"""
    print("測試基本功能...")

    try:
        import torch
        from ..isdo.samplers.isdo_sampler import ISDOSampler

        # 創建測試採樣器
        print("  創建 ISDO 採樣器...")
        sampler = ISDOSampler(
            spatial_dims=(32, 32),
            spectral_order=64,
            sobolev_order=1.5,
            refinement_iterations=100,
            device=torch.device('cpu')  # 使用 CPU 測試
        )
        print("    ✓ 採樣器創建成功")

        # 測試統計功能
        print("  測試統計功能...")
        stats = sampler.get_sampling_statistics()
        print(f"    配置: {stats['configuration']['spatial_dims']}")
        print(f"    譜階數: {stats['configuration']['spectral_order']}")
        print("    ✓ 統計功能正常")

        return True

    except Exception as e:
        print(f"    ✗ 功能測試錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_webui_integration():
    """測試 WebUI 整合"""
    print("測試 WebUI 整合...")

    try:
        # 測試是否可以找到 modules
        print("  檢查 WebUI 模組...")
        from modules import sd_samplers_common
        print("    ✓ WebUI 模組可用")

        # 測試採樣器數據結構
        print("  檢查採樣器數據結構...")
        from ..isdo_samplers_integration import samplers_data_isdo

        for sampler_data in samplers_data_isdo:
            # 檢查 SamplerData 結構
            assert hasattr(sampler_data, 'name'), "SamplerData 缺少 name 屬性"
            assert hasattr(sampler_data, 'constructor'), "SamplerData 缺少 constructor 屬性"
            assert hasattr(sampler_data, 'aliases'), "SamplerData 缺少 aliases 屬性"
            assert hasattr(sampler_data, 'options'), "SamplerData 缺少 options 屬性"

            print(f"    ✓ {sampler_data.name} 數據結構正確")

        return True

    except Exception as e:
        print(f"    ✗ WebUI 整合測試錯誤: {e}")
        return False


def main():
    """主測試函數"""
    print("=" * 50)
    print("ISDO 採樣器整合測試")
    print("=" * 50)

    tests = [
        ("模組導入", test_isdo_imports),
        ("採樣器註冊", test_sampler_registration),
        ("基本功能", test_basic_functionality),
        ("WebUI 整合", test_webui_integration),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}測試:")
        success = test_func()
        results.append((test_name, success))
        print(f"{test_name}測試: {'通過' if success else '失敗'}")

    print("\n" + "=" * 50)
    print("測試總結:")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ 通過" if success else "✗ 失敗"
        print(f"  {test_name}: {status}")

    print(f"\n總計: {passed}/{total} 測試通過")

    if passed == total:
        print("\n🎉 所有測試通過！ISDO 採樣器整合成功。")
        print("\n下一步:")
        print("1. 重啟 Forge WebUI")
        print("2. 在採樣器下拉選單中尋找 ISDO 選項")
        print("3. 選擇 ISDO、ISDO Adaptive、ISDO HQ 或 ISDO Fast")
        print("4. 享受變分最優控制採樣的高質量結果！")
    else:
        print(f"\n⚠️  {total - passed} 個測試失敗。請檢查上述錯誤信息。")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)