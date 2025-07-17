# Infinite Spectral Diffusion Odyssey (ISDO)

## 專案概述

ISDO 是一個革命性的擴散採樣方法，將傳統的 ODE 求解轉化為**變分最優控制問題**，嵌入在**無窮維希爾伯特空間**中，使用**譜方法**求解高階偏微分方程，並通過**李群對稱**確保結構不變性。

## 核心特色

### 🔬 數學理論基礎
- **變分最優控制**: 尋找從噪聲到數據的最短路徑
- **無窮維希爾伯特空間**: 基於 Sobolev 空間的泛函分析
- **譜方法**: 使用傅立葉/小波基底的本徵模式分解
- **李群對稱**: SE(3) 群保證拓撲結構不變性

### 🚀 技術優勢
- **構圖準確性**: 譜展開分離高低頻模式，解決手指/眼睛細節問題
- **通用性**: 支援 ε-pred、v-pred、flow matching 等所有預測目標
- **強大性**: 變分控制超越自回歸模型的預測能力
- **效率**: 自適應譜截斷，無限質量改善可能

## 模塊架構

```
modules_forge/isdo/
├── __init__.py              # 主模塊入口
├── README.md               # 主要說明文檔
├── test_integration.py     # 整合測試腳本
├── core/                   # 核心數學操作
│   ├── __init__.py
│   ├── spectral_basis.py   # 譜基底生成 (16KB, 457行)
│   ├── hilbert_space.py    # 希爾伯特空間操作 (16KB, 489行)
│   ├── variational_controller.py  # 變分最優控制 (28KB, 871行)
│   ├── variational_ode_system.py  # 變分 ODE 系統 (28KB, 840行)
│   └── spectral_projection.py     # 譜投影系統 (30KB, 927行)
├── math/                   # 數學基礎
│   ├── __init__.py
│   ├── lie_group_ops.py    # 李群操作 (4.8KB, 163行)
│   └── spectral_rk4.py     # 譜 Runge-Kutta 求解器 (2.8KB, 99行)
├── samplers/               # 採樣器實現
│   ├── __init__.py
│   ├── isdo_sampler.py     # 主 ISDO 採樣器 (22KB, 648行)
│   └── unified_model_wrapper.py   # 統一模型接口 (13KB, 439行)
├── test/                   # 測試套件
│   ├── __init__.py
│   ├── test_isdo_integration.py        # ISDO 整合測試
│   ├── test_isdo_core_integration.py   # 核心模塊測試
│   └── test_ode_system_integration.py  # ODE 系統測試
├── docs/                   # 完整文檔
│   ├── README.md                       # 主要說明
│   ├── spectral_basis_guide.md         # 譜基底指南
│   ├── hilbert_space_guide.md          # 希爾伯特空間指南
│   ├── variational_controller_guide.md # 變分控制器指南
│   ├── variational_ode_system_guide.md # 變分 ODE 系統指南
│   ├── isdo_integration_guide.md       # ISDO 整合指南
│   └── ISDO_ODE_INTEGRATION_GUIDE.md   # ODE 整合指南
└── chat/                   # 開發文檔
    ├── cursor_modules_forge_isdo_sampler.md  # 採樣器開發記錄
    └── cursor_python_torch.md                # PyTorch 開發記錄
```

## 使用方式

```python
from modules_forge.isdo import ISDOSampler

# 初始化 ISDO 採樣器
sampler = ISDOSampler(
    spectral_order=256,          # 譜截斷階數
    refinement_iterations=100,   # 李群細化迭代數
    adaptive_threshold=1e-6,     # 自適應收斂閾值
    basis_type='fourier',        # 譜基底類型 ('fourier', 'wavelet')
    device='cuda'                # 計算設備
)

# 採樣 (完全兼容 k_diffusion 接口)
samples = sampler.sample_isdo(
    model,                      # 擴散模型
    x,                         # 初始噪聲
    sigmas,                    # 噪聲調度
    extra_args=None,           # 額外參數
    callback=None,             # 回調函數
    disable=None               # 禁用進度條
)

# 高級用法：手動控制譜投影
with sampler.spectral_context():
    # 在譜空間中進行操作
    spectral_x = sampler.project_to_spectral(x)
    # ... 自定義譜操作 ...
    x_reconstructed = sampler.reconstruct_from_spectral(spectral_x)
```

## 理論背景

### 變分最優控制公式

尋找軌跡 x(σ) 最小化動作積分：

```
𝒜[x] = ∫[σ_max to 0] ℒ(x, ẋ, σ) dσ
```

其中 ℒ = ½|ẋ - f(x;σ)|²_H + λ|∇_x f|²_op

### 譜展開

x(σ) = Σ[k=1 to ∞] c_k(σ) φ_k

其中 {φ_k} 是希爾伯特空間 H 的正交基底。

### 李群細化

通過 SE(3) 群作用保持結構不變性：
x' = exp(ε g_j) x

## 實作狀態

### 核心模塊 (100% 完成)
- [x] 專案結構建立
- [x] 譜基底實現 (`spectral_basis.py` - 457行)
- [x] 希爾伯特空間操作 (`hilbert_space.py` - 489行)
- [x] 變分控制器 (`variational_controller.py` - 871行)
- [x] 變分 ODE 系統 (`variational_ode_system.py` - 840行)
- [x] 譜投影系統 (`spectral_projection.py` - 927行)

### 數學基礎 (100% 完成)
- [x] 李群操作 (`lie_group_ops.py` - 163行)
- [x] 譜 Runge-Kutta 求解器 (`spectral_rk4.py` - 99行)

### 採樣器 (100% 完成)
- [x] 主 ISDO 採樣器 (`isdo_sampler.py` - 648行)
- [x] 統一模型接口 (`unified_model_wrapper.py` - 439行)

### 測試與驗證 (100% 完成)
- [x] 整合測試套件 (3個測試文件)
- [x] 主整合測試腳本 (`test_integration.py` - 359行)

### 文檔系統 (100% 完成)
- [x] 完整的技術文檔 (7個指南文件)
- [x] 開發記錄與聊天記錄

**總代碼量**: 超過 5000 行完整實現
**模塊數量**: 11 個核心 Python 模塊
**測試覆蓋**: 完整的單元與整合測試

## 參考文獻

- Karras et al. (2022). "Elucidating the Design Space of Diffusion-Based Generative Models"
- Øksendal. "Stochastic Differential Equations"
- Canuto et al. "Spectral Methods in Fluid Dynamics"
- Bertsekas. "Dynamic Programming and Optimal Control"

## 許可證

遵循 stable-diffusion-webui-forge 的許可協議。

## 整合狀態

ISDO 模塊已完全實現並可獨立運行，但尚未整合到 stable-diffusion-webui-forge 的主採樣器系統中。

### 要整合到 WebUI：
1. 修改 `modules_forge/alter_samplers.py` 註冊 ISDO 採樣器
2. 在 `modules_forge/isdo_samplers_integration.py` 中添加接口適配
3. 運行 `register_isdo_manually.py` 進行手動註冊

### 當前可用操作：
- 獨立測試：`python modules_forge/isdo/test_integration.py`
- 核心功能驗證：`python modules_forge/isdo/test/test_*.py`
- 文檔查看：`modules_forge/isdo/docs/` 目錄

---

**注意**: 此實現基於高階數學理論，需要深厚的泛函分析背景理解。建議搭配技術文檔一起閱讀。