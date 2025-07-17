"""
ISDO 系統整合測試
=================

簡單的整合測試，驗證 ISDO 各個組件能夠正常協作。
這個測試使用模擬的去噪模型來避免依賴實際的擴散模型。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import time

# 導入 ISDO 組件
from .samplers.isdo_sampler import ISDOSampler, sample_isdo
from .samplers.unified_model_wrapper import UnifiedModelWrapper, ModelType
from .core.spectral_basis import SpectralBasis, BasisType
from .core.hilbert_space import HilbertSpace
from .core.variational_controller import VariationalController
from .core.spectral_projection import SpectralProjection


class MockDenoiserModel(nn.Module):
    """
    模擬去噪模型

    簡單的 UNet 風格模型，用於測試 ISDO 功能
    """

    def __init__(self, channels=4, image_size=64):
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        # 簡單的編碼器-解碼器結構
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, padding=1)
        )

    def forward(self, x, sigma):
        """
        前向傳播

        Args:
            x: 噪聲圖像 (B, C, H, W)
            sigma: 噪聲水平 (B,) 或標量

        Returns:
            eps_pred: 預測的噪聲 (epsilon-prediction)
        """
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma, device=x.device)

        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0])

        # 編碼圖像
        h = self.encoder(x)

        # 時間嵌入
        t_emb = self.time_embed(sigma.unsqueeze(-1))
        t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)

        # 添加時間信息
        h = h + t_emb

        # 解碼
        eps_pred = self.decoder(h)

        # 添加一些隨機性來模擬真實模型
        eps_pred = eps_pred + torch.randn_like(eps_pred) * 0.01

        return eps_pred


def test_spectral_basis():
    """測試譜基底功能"""
    print("測試 SpectralBasis...")

    # 創建傅立葉基底
    basis = SpectralBasis(
        spatial_dims=(32, 32),
        spectral_order=64,
        basis_type=BasisType.FOURIER,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # 測試投影和重建
    x = torch.randn(1, 3, 32, 32, device=basis.device)
    coeffs = basis.project_to_basis(x)
    x_reconstructed = basis.reconstruct_from_coefficients(coeffs)

    # 計算重建誤差
    reconstruction_error = torch.norm(x - x_reconstructed).item()
    print(f"  重建誤差: {reconstruction_error:.6f}")

    assert reconstruction_error < 1.0, f"重建誤差過大: {reconstruction_error}"
    print("  ✓ SpectralBasis 測試通過")


def test_hilbert_space():
    """測試希爾伯特空間功能"""
    print("測試 HilbertSpace...")

    hilbert = HilbertSpace(
        spatial_dims=(32, 32),
        sobolev_order=1.5,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # 測試 Sobolev 範數計算
    x = torch.randn(2, 3, 32, 32, device=hilbert.device)
    sobolev_norm = hilbert.compute_sobolev_norm(x)

    print(f"  Sobolev 範數: {sobolev_norm.mean().item():.3f}")

    # 測試嵌入條件
    embedding_info = hilbert.check_sobolev_embedding(1.5)
    print(f"  連續嵌入: {embedding_info['continuous_embedding']}")

    assert sobolev_norm.mean().item() > 0, "Sobolev 範數應該為正"
    print("  ✓ HilbertSpace 測試通過")


def test_spectral_projection():
    """測試譜投影功能"""
    print("測試 SpectralProjection...")

    projection = SpectralProjection(
        spatial_dims=(32, 32),
        spectral_order=128,
        projection_type='fft',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # 測試前向和逆投影
    x = torch.randn(1, 3, 32, 32, device=projection.device)

    # 前向投影
    coeffs = projection(x, mode='forward')
    print(f"  譜係數形狀: {coeffs.shape}")

    # 逆投影
    x_reconstructed = projection(coeffs, mode='inverse')

    # 計算誤差
    projection_error = torch.norm(x - x_reconstructed).item()
    print(f"  投影重建誤差: {projection_error:.6f}")

    assert projection_error < 1.0, f"投影誤差過大: {projection_error}"
    print("  ✓ SpectralProjection 測試通過")


def test_unified_model_wrapper():
    """測試統一模型包裝器"""
    print("測試 UnifiedModelWrapper...")

    # 創建模擬模型
    mock_model = MockDenoiserModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mock_model = mock_model.to(device)

    # 包裝模型
    wrapped_model = UnifiedModelWrapper(
        model=mock_model,
        model_type=ModelType.EPSILON,
        device=device
    )

    # 測試模型調用
    x = torch.randn(2, 4, 32, 32, device=device)
    sigma = torch.tensor([5.0, 3.0], device=device)

    x0_pred = wrapped_model(x, sigma)

    print(f"  輸入形狀: {x.shape}")
    print(f"  輸出形狀: {x0_pred.shape}")
    print(f"  模型類型: {wrapped_model.model_type}")

    # 驗證輸出
    validation = wrapped_model.validate_model_output(x, sigma)
    print(f"  模型輸出驗證: {validation['all_checks_passed']}")

    assert validation['all_checks_passed'], "模型輸出驗證失敗"
    print("  ✓ UnifiedModelWrapper 測試通過")


def test_isdo_sampler():
    """測試 ISDO 採樣器"""
    print("測試 ISDOSampler...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建模擬模型
    mock_model = MockDenoiserModel(channels=4, image_size=32)
    mock_model = mock_model.to(device)

    # 創建 ISDO 採樣器
    sampler = ISDOSampler(
        spatial_dims=(32, 32),
        spectral_order=64,          # 較小的譜階數用於測試
        sobolev_order=1.5,
        regularization_lambda=0.01,
        curvature_penalty=0.001,
        refinement_iterations=100,   # 較少的迭代用於測試
        adaptive_scheduling=True,
        lie_group_refinement=True,
        device=device
    )

    # 準備採樣參數
    x_init = torch.randn(1, 4, 32, 32, device=device)
    sigmas = torch.linspace(10.0, 0.1, 11, device=device)  # 只用10步測試

    print(f"  初始噪聲形狀: {x_init.shape}")
    print(f"  Sigma 調度: {len(sigmas)} 步")

    # 執行採樣
    start_time = time.time()

    try:
        result = sampler.sample_isdo(
            model=mock_model,
            x=x_init,
            sigmas=sigmas,
            energy_threshold=0.95,    # 較低的閾值用於測試
            quality_threshold=0.05,   # 較鬆的質量要求
            max_refinement_steps=10   # 限制細化步數
        )

        sampling_time = time.time() - start_time

        print(f"  採樣完成，用時: {sampling_time:.2f} 秒")
        print(f"  結果形狀: {result.shape}")

        # 獲取統計信息
        stats = sampler.get_sampling_statistics()
        print(f"  譜投影次數: {stats['runtime_stats']['spectral_projections']}")
        print(f"  李群操作次數: {stats['runtime_stats']['lie_group_operations']}")
        print(f"  自適應調整次數: {stats['runtime_stats']['adaptive_adjustments']}")

        # 評估採樣質量
        quality = sampler.evaluate_sampling_quality(result.unsqueeze(0))
        print(f"  平均 Sobolev 範數: {quality['mean_sobolev_norm']:.3f}")

        assert result.shape == x_init.shape, "輸出形狀不匹配"
        assert not torch.isnan(result).any(), "結果包含 NaN"
        assert torch.isfinite(result).all(), "結果包含無限值"

        print("  ✓ ISDOSampler 測試通過")

    except Exception as e:
        print(f"  ✗ ISDO 採樣失敗: {e}")
        raise


def test_convenience_function():
    """測試便利函數"""
    print("測試 sample_isdo 便利函數...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建模擬模型
    mock_model = MockDenoiserModel(channels=4, image_size=32)
    mock_model = mock_model.to(device)

    # 準備參數
    x_init = torch.randn(1, 4, 32, 32, device=device)
    sigmas = torch.linspace(5.0, 0.1, 6, device=device)  # 只用5步測試

    # 使用便利函數
    result = sample_isdo(
        model=mock_model,
        x=x_init,
        sigmas=sigmas,
        spectral_order=64,
        sobolev_order=1.0,
        regularization_lambda=0.02,
        adaptive_scheduling=False,  # 禁用以加速測試
        lie_group_refinement=False  # 禁用以加速測試
    )

    print(f"  便利函數結果形狀: {result.shape}")

    assert result.shape == x_init.shape, "便利函數輸出形狀不匹配"
    print("  ✓ sample_isdo 便利函數測試通過")


def run_integration_tests():
    """運行所有整合測試"""
    print("=" * 50)
    print("ISDO 系統整合測試")
    print("=" * 50)

    try:
        # 基礎組件測試
        test_spectral_basis()
        print()

        test_hilbert_space()
        print()

        test_spectral_projection()
        print()

        test_unified_model_wrapper()
        print()

        # 整合測試
        test_isdo_sampler()
        print()

        test_convenience_function()
        print()

        print("=" * 50)
        print("🎉 所有測試通過！ISDO 系統運行正常。")
        print("=" * 50)

        return True

    except Exception as e:
        print("=" * 50)
        print(f"❌ 測試失敗: {e}")
        print("=" * 50)
        return False


if __name__ == "__main__":
    # 運行測試
    success = run_integration_tests()

    if success:
        print("\n✅ ISDO 系統整合測試完成，所有功能正常！")
        print("\n🚀 現在可以開始使用 ISDO 進行採樣了。")
        print("\n📖 請參閱 docs/isdo_integration_guide.md 了解詳細用法。")
    else:
        print("\n❌ 整合測試失敗，請檢查配置和依賴。")
        exit(1)