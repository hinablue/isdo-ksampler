# Infinite Spectral Diffusion Odyssey (ISDO)

**將擴散採樣轉化為變分最優控制問題的無窮維譜方法實現**

---

## 來源 & 特別感謝
[@win10ogod UNI.md](https://github.com/win10ogod/ComfyUI-my/blob/master/comfy/doc/Uni.md)

## 專案簡介

ISDO（Infinite Spectral Diffusion Odyssey）是一套以**無窮維希爾伯特空間**與**譜方法**為核心，將擴散模型採樣問題重構為**變分最優控制**的創新框架。其理論基礎結合 Sobolev 空間、傅立葉/小波譜展開、李群對稱細化，實現高精度、高穩定性的生成採樣，突破傳統 ODE 求解器的限制。

- **變分最優控制**：尋找從噪聲到數據的最短路徑，理論最優。
- **無窮維譜展開**：高低頻模式分離，細節與結構兼顧。
- **李群對稱細化**：保持拓撲結構，解決構圖畸形。
- **自適應譜截斷**：效率與品質兼得。

---

## 目錄結構（簡要）

```
├── isdo/                  # 主程式庫
│   ├── core/              # 譜基底、希爾伯特空間、變分控制
│   ├── math/              # 李群、譜 RK4 等數學輔助
│   ├── samplers/          # ISDO 採樣器與模型包裝
│   ├── test/              # 單元與整合測試
│   ├── docs/              # 詳細技術文檔
│   └── chat/              # 開發紀錄
├── isdo_samplers_integration.py # WebUI/Forge 整合入口
├── alter_samplers.py      # 採樣器註冊
└── README.md              # 本說明文件
```

---

## 安裝需求
- Python 3.8+
- PyTorch 1.12+
- NumPy 1.21+
- 建議於 stable-diffusion-webui-forge 環境下運行

---

## 快速開始

```python
import torch
from isdo.samplers.isdo_sampler import sample_isdo

# 基本 ISDO 採樣
result = sample_isdo(
    model=your_diffusion_model,           # 任意擴散模型
    x=torch.randn(1, 4, 64, 64),         # 初始噪聲
    sigmas=torch.logspace(1, -2, 50),    # 噪聲調度
    spectral_order=256,                  # 譜階數
    sobolev_order=1.5,                   # Sobolev 階數
    regularization_lambda=0.01,          # 變分正則化
    adaptive_scheduling=True,            # 自適應調度
    lie_group_refinement=True            # 李群細化
)
```

### 進階自定義

```python
from isdo.samplers.isdo_sampler import ISDOSampler

sampler = ISDOSampler(
    spatial_dims=(64, 64),
    spectral_order=512,
    sobolev_order=2.0,
    regularization_lambda=0.005,
    curvature_penalty=0.001,
    refinement_iterations=2000,
    adaptive_scheduling=True,
    lie_group_refinement=True
)

result = sampler.sample_isdo(
    model=model,
    x=x_init,
    sigmas=sigmas,
    energy_threshold=0.995,
    quality_threshold=0.005,
    max_refinement_steps=100
)
```

---

## 測試方式

於專案根目錄執行：

```bash
python -m isdo.test_integration
```

或執行單元測試：

```bash
python -m isdo.test.test_isdo_integration
python -m isdo.test.test_isdo_core_integration
python -m isdo.test.test_ode_system_integration
```

---

## 主要文檔與資源
- [isdo/README.md](isdo/README.md)：完整理論與用法說明
- [isdo/docs/](isdo/docs/)：譜基底、希爾伯特空間、變分控制等技術指南
- [isdo/docs/isdo_integration_guide.md](isdo/docs/isdo_integration_guide.md)：WebUI/Forge 整合教學
- [isdo/chat/](isdo/chat/)：開發紀錄與設計討論

---

## 授權

本專案採用 MIT License 授權。

Copyright (c) 2025 Hina Chen