# ISDO VariationalODESystem 整合使用指南

## 概述

本文檔說明如何在 ISDO 系統中使用 `VariationalODESystem` 的新整合功能。通過將 `VariationalODESystem` 整合到 `core/` 和 `samplers/` 模組中，ISDO 現在提供了更精確、更穩定的變分採樣能力。

## 主要改進

### 1. VariationalController 高精度求解

`VariationalController` 現在提供兩種求解方法：

#### 標準方法（原有）
```python
from modules_forge.isdo.core import VariationalController

controller = VariationalController(
    spatial_dims=(64, 64),
    regularization_lambda=0.01,
    curvature_penalty=0.001
)

# 標準變分控制
trajectory, controls = controller.compute_optimal_control(
    x_current=x_initial,
    sigma_current=1.0,
    denoiser_function=model,
    target_sigma=0.1,
    num_steps=10
)
```

#### 高精度 ODE 方法（新增）
```python
# 使用 VariationalODESystem 的高精度求解
trajectory, info, solve_info = controller.compute_optimal_control_with_ode_system(
    x_current=x_initial,
    sigma_current=1.0,
    denoiser_function=model,
    target_sigma=0.1,
    spectral_order=256,           # 譜截斷階數
    use_adaptive_stepping=True    # 自適應步長
)

print(f"收斂狀態: {solve_info['converged']}")
print(f"求解步數: {solve_info['final_step']}")
print(f"Sobolev 範數演化: {info['sobolev_norms']}")
```

#### 軌跡質量評估
```python
# 評估軌跡質量
quality_metrics = controller.evaluate_trajectory_quality(
    trajectory, info['sigma_values'], model
)

print(f"動作積分值: {quality_metrics['action_value']}")
print(f"Euler-Lagrange 殘差: {quality_metrics['mean_euler_lagrange_residual']}")
print(f"軌跡平滑度: {quality_metrics['trajectory_smoothness']}")
```

### 2. ISDOSampler 自動 ODE 集成

`ISDOSampler` 現在自動使用 `VariationalODESystem`，並在失敗時回退到 `SpectralRK4`：

```python
from modules_forge.isdo.samplers import ISDOSampler

# 創建採樣器（自動整合 VariationalODESystem）
sampler = ISDOSampler(
    spatial_dims=(64, 64),
    spectral_order=256,
    sobolev_order=1.5,
    adaptive_scheduling=True,    # 啟用自適應調度
    lie_group_refinement=True    # 啟用李群細化
)

# 採樣會自動使用 VariationalODESystem
samples = sampler.sample_isdo(
    model=your_model,
    x=noise,
    sigmas=sigma_schedule
)

# 查看使用統計
stats = sampler.get_sampling_statistics()
print(f"ODE 求解成功率: {stats['efficiency_metrics']['adaptation_frequency']}")
```

#### 強制使用回退方法
```python
# 如果需要測試或比較，可以強制使用回退方法
x_updated = sampler._fallback_spectral_step(
    x=x_current,
    sigma_current=1.0,
    sigma_next=0.5,
    model=model_wrapper,
    extra_args={},
    energy_threshold=0.99
)
```

### 3. SpectralProjection ODE 優化

`SpectralProjection` 新增了基於 `VariationalODESystem` 的優化功能：

#### ODE 優化投影
```python
from modules_forge.isdo.core import SpectralProjection

projection = SpectralProjection(
    spatial_dims=(64, 64),
    spectral_order=256
)

# 標準投影
coeffs_std, reconstructed_std, info_std = projection.project_and_reconstruct(x)

# ODE 優化投影
coeffs_opt, reconstructed_opt, info_opt = projection.project_with_ode_optimization(
    x=x,
    target_sigma=0.1,
    model_wrapper=model_wrapper,
    optimization_steps=10
)

if info_opt['optimization_used']:
    print(f"優化收斂: {info_opt['converged']}")
    print(f"重建改進: {info_opt['reconstruction_improvement']:.4f}")
```

#### 譜動力學分析
```python
# 分析函數在不同噪聲水平下的譜演化
sigma_schedule = torch.linspace(1.0, 0.1, 10)
dynamics_analysis = projection.analyze_spectral_dynamics(
    x=x,
    model_wrapper=model_wrapper,
    sigma_schedule=sigma_schedule
)

# 查看動力學特性
print(f"主導模式穩定性: {dynamics_analysis['temporal_evolution']['dominant_mode_stability']}")
print(f"能量演化趨勢: {dynamics_analysis['temporal_evolution']['total_energy_trend']}")
```

## 實際使用範例

### 高質量圖像生成

```python
import torch
from modules_forge.isdo.samplers import ISDOSampler, UnifiedModelWrapper
from modules_forge.isdo.core import VariationalController

# 包裝您的擴散模型
model_wrapper = UnifiedModelWrapper(
    your_diffusion_model,
    model_type="epsilon",  # 或 "v_param", "flow", "score"
    sigma_data=1.0
)

# 創建高精度 ISDO 採樣器
sampler = ISDOSampler(
    spatial_dims=(512, 512),     # 高解析度
    spectral_order=512,          # 高譜階數
    sobolev_order=1.5,          # Sobolev 正則化
    regularization_lambda=0.005, # 較強正則化
    adaptive_scheduling=True,    # 自適應調度
    lie_group_refinement=True,   # 李群細化
    device=torch.device('cuda')
)

# 生成噪聲調度
sigma_max, sigma_min = 10.0, 0.01
num_steps = 50
sigmas = torch.linspace(sigma_max, sigma_min, num_steps + 1)

# 初始噪聲
batch_size = 4
noise = torch.randn(batch_size, 3, 512, 512, device='cuda')

# 高質量採樣
samples = sampler.sample_isdo(
    model=model_wrapper,
    x=noise,
    sigmas=sigmas,
    energy_threshold=0.995,      # 高能量保留
    quality_threshold=0.001      # 高質量閾值
)

# 評估採樣質量
quality_metrics = sampler.evaluate_sampling_quality(samples)
print(f"平均 Sobolev 範數: {quality_metrics['mean_sobolev_norm']:.4f}")
print(f"結構完整性: {quality_metrics.get('mean_structure_violation', 'N/A')}")
```

### 變分軌跡分析

```python
from modules_forge.isdo.core import VariationalController

# 創建變分控制器
controller = VariationalController(
    spatial_dims=(256, 256),
    regularization_lambda=0.01,
    curvature_penalty=0.001,
    sobolev_order=2.0  # 更高階 Sobolev 空間
)

# 計算最優軌跡
x_start = torch.randn(1, 3, 256, 256)
trajectory, info, solve_info = controller.compute_optimal_control_with_ode_system(
    x_current=x_start,
    sigma_current=5.0,
    denoiser_function=model_wrapper,
    target_sigma=0.05,
    spectral_order=256,
    use_adaptive_stepping=True
)

# 分析軌跡品質
quality = controller.evaluate_trajectory_quality(
    trajectory, info['sigma_values'], model_wrapper
)

print("=== 軌跡分析結果 ===")
print(f"動作積分: {quality['action_value']:.6f}")
print(f"平均 EL 殘差: {quality['mean_euler_lagrange_residual']:.8f}")
print(f"能量守恆: {quality['energy_conservation']:.6f}")
print(f"軌跡平滑度: {quality['trajectory_smoothness']:.6f}")

# 可視化收斂歷史
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(info['sobolev_norms'])
plt.title('Sobolev 範數演化')
plt.xlabel('步數')
plt.ylabel('範數')

plt.subplot(2, 2, 2)
plt.plot(info['step_sizes'])
plt.title('自適應步長')
plt.xlabel('步數')
plt.ylabel('步長')

plt.subplot(2, 2, 3)
convergence_values = [c['convergence_rate'] for c in info['convergence_history']]
plt.plot(convergence_values)
plt.title('收斂率')
plt.xlabel('步數')
plt.ylabel('收斂率')

plt.subplot(2, 2, 4)
sigma_values = info['sigma_values'].cpu().numpy()
plt.plot(sigma_values)
plt.title('噪聲水平調度')
plt.xlabel('步數')
plt.ylabel('σ')

plt.tight_layout()
plt.savefig('variational_trajectory_analysis.png', dpi=300)
plt.show()
```

## 性能優化建議

### 1. 記憶體優化

```python
# 對於大圖像，使用較小的譜階數
sampler = ISDOSampler(
    spatial_dims=(1024, 1024),
    spectral_order=256,  # 不要設得太高
    device=torch.device('cuda')
)

# 在GPU記憶體不足時，使用CPU
if torch.cuda.get_device_properties(0).total_memory < 8e9:  # < 8GB
    device = torch.device('cpu')
```

### 2. 計算效率

```python
# 對於快速預覽，關閉某些功能
sampler = ISDOSampler(
    adaptive_scheduling=False,   # 關閉自適應調度
    lie_group_refinement=False,  # 關閉李群細化
    spectral_order=128          # 較低譜階數
)

# 對於批量處理，重用採樣器
for batch in dataloader:
    samples = sampler.sample_isdo(model, batch, sigmas)
    sampler.reset_statistics()  # 重置統計
```

### 3. 品質 vs 速度權衡

| 設置 | 速度 | 品質 | 適用場景 |
|------|------|------|----------|
| 快速模式 | 最快 | 中等 | 預覽、測試 |
| 標準模式 | 中等 | 高 | 一般生成 |
| 高質量模式 | 慢 | 最高 | 專業用途 |

```python
# 快速模式
fast_sampler = ISDOSampler(
    spectral_order=64,
    adaptive_scheduling=False,
    lie_group_refinement=False
)

# 標準模式
standard_sampler = ISDOSampler(
    spectral_order=256,
    adaptive_scheduling=True,
    lie_group_refinement=False
)

# 高質量模式
hq_sampler = ISDOSampler(
    spectral_order=512,
    adaptive_scheduling=True,
    lie_group_refinement=True,
    sobolev_order=2.0
)
```

## 故障排除

### 常見問題

1. **VariationalODESystem 求解失敗**
   - 檢查譜階數是否過高
   - 降低正則化參數
   - 檢查模型輸出是否有效

2. **記憶體不足**
   - 降低批次大小
   - 減少譜階數
   - 使用CPU

3. **收斂緩慢**
   - 增加最大步數
   - 調整自適應參數
   - 檢查初始條件

### 調試模式

```python
# 啟用詳細日誌
import logging
logging.basicConfig(level=logging.DEBUG)

# 強制回退到 SpectralRK4
sampler._fallback_only = True  # 非正式標誌

# 檢查中間結果
def debug_callback(callback_dict):
    print(f"步驟 {callback_dict['i']}: σ={callback_dict['sigma']:.4f}")
    print(f"Sobolev 範數: {callback_dict['sobolev_norm']:.4f}")

samples = sampler.sample_isdo(model, noise, sigmas, callback=debug_callback)
```

## 結論

通過整合 `VariationalODESystem`，ISDO 現在提供了：

1. **更高的精度**: 基於變分原理的精確 ODE 求解
2. **更好的穩定性**: 自動回退機制確保可靠性
3. **更豐富的分析**: 詳細的軌跡質量評估
4. **更靈活的配置**: 多層級的品質/速度權衡

這些改進使 ISDO 能夠生成更高質量的圖像，同時保持與現有工作流程的兼容性。