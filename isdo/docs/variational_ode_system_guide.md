# VariationalODESystem 模塊說明文檔

## 概述

`VariationalODESystem` 是 ISDO 系統的核心計算引擎，負責將無窮維變分最優控制問題轉化為有限維 ODE 系統並求解。這是整個 ISDO 算法的數學心臟，實現了從抽象的變分理論到具體數值計算的完整橋梁。

## 數學理論基礎

### 無窮維變分問題的離散化

ISDO 的核心思想是將無窮維問題：

```
min 𝒜[x] = ∫[σ_max to 0] ℒ(x, ẋ, σ) dσ
```

通過譜截斷轉化為有限維問題：

```
x(σ) = Σ[k=1 to M] c_k(σ) φ_k
```

這導致了譜係數的動力學方程：

```
dc_k/dσ = G_k(c_1, ..., c_M, σ)
```

### 譜空間動力學

在譜空間中，動力學方程變為：

```
dc/dσ = P[f(Σc_k φ_k; σ)/σ] + 變分修正項 + Sobolev 正則化項
```

其中 P[·] 是到譜空間的投影算子。

## 核心組件

### 1. SpectralDynamics - 譜動力學系統

負責計算譜係數的右端函數：

```python
from modules_forge.isdo.samplers.variational_ode_system import VariationalODESystem
from modules_forge.isdo.samplers.unified_model_wrapper import UnifiedModelWrapper

# 初始化變分 ODE 系統
ode_system = VariationalODESystem(
    spatial_dims=(64, 64),
    spectral_order=256,
    sobolev_order=1.5,
    regularization_lambda=0.01,
    sobolev_penalty=0.001
)

# 統一模型包裝器
model_wrapper = UnifiedModelWrapper(your_diffusion_model)

# 求解變分 ODE
initial_coeffs = torch.randn(1, 3, 256)  # (B, C, M)
final_coeffs, trajectory, solve_info = ode_system.solve_variational_ode(
    initial_coefficients=initial_coeffs,
    sigma_start=10.0,
    sigma_end=0.01,
    model_wrapper=model_wrapper,
    max_steps=1000,
    adaptive_stepping=True
)
```

### 2. AdaptiveStepSizeController - 自適應步長控制

根據局部誤差動態調整積分步長：

```python
# 自適應步長的工作原理
step_controller = ode_system.step_controller

# 基於誤差估計調整步長
error_estimate = torch.randn_like(initial_coeffs) * 0.01
sobolev_norm = torch.tensor(1.5)
new_step_size = step_controller.adapt_step_size(
    error_estimate, initial_coeffs, sobolev_norm
)

print(f"新步長: {new_step_size:.6f}")
```

### 3. ConvergenceDetector - 收斂檢測

監控求解過程的收斂性：

```python
# 檢查收斂狀態
convergence_detector = ode_system.convergence_detector
converged, conv_info = convergence_detector.check_convergence(
    current_coeffs=final_coeffs,
    iteration=100
)

print(f"是否收斂: {converged}")
print(f"收斂信息: {conv_info}")
```

## 主要功能

### 1. 變分 ODE 求解

核心功能是求解譜空間中的動力學方程：

```python
# 完整的求解流程
def solve_isdo_step(model, x_current, sigma_current, sigma_next):
    # 投影到譜空間
    coeffs_initial = ode_system.spectral_basis.project_to_basis(x_current)

    # 求解變分 ODE
    coeffs_final, trajectory, info = ode_system.solve_variational_ode(
        initial_coefficients=coeffs_initial,
        sigma_start=sigma_current,
        sigma_end=sigma_next,
        model_wrapper=UnifiedModelWrapper(model)
    )

    # 重建到空間域
    x_next = ode_system.spectral_basis.reconstruct_from_coefficients(coeffs_final)

    return x_next, trajectory, info

# 使用示例
x_current = torch.randn(1, 3, 64, 64)
x_next, trajectory, solve_info = solve_isdo_step(
    model=your_model,
    x_current=x_current,
    sigma_current=1.0,
    sigma_next=0.1
)

print(f"求解步數: {solve_info['final_step']}")
print(f"是否收斂: {solve_info['converged']}")
```

### 2. 解質量分析

評估求解結果的質量：

```python
# 分析解的質量
quality_analysis = ode_system.analyze_solution_quality(trajectory, solve_info)

print("收斂性分析:")
print(f"  收斂率: {quality_analysis['convergence']['convergence_rate']:.6f}")

print("Sobolev 範數分析:")
print(f"  初始範數: {quality_analysis['sobolev_analysis']['initial_norm']:.3f}")
print(f"  最終範數: {quality_analysis['sobolev_analysis']['final_norm']:.3f}")
print(f"  範數穩定性: {quality_analysis['sobolev_analysis']['norm_stability']:.6f}")

print("譜分析:")
print(f"  低頻能量比例變化: {quality_analysis['spectral_analysis']['initial_low_freq_ratio']:.3f} → {quality_analysis['spectral_analysis']['final_low_freq_ratio']:.3f}")
print(f"  能量守恆誤差: {quality_analysis['spectral_analysis']['energy_conservation_error']:.6f}")
```

### 3. 數值穩定性控制

確保數值計算的穩定性：

```python
# 監控數值穩定性
stability_info = quality_analysis['numerical_stability']

print("數值穩定性:")
print(f"  最大係數: {stability_info['max_coefficient']:.2e}")
print(f"  條件數: {stability_info['condition_number']:.2e}")
print(f"  能量守恆誤差: {stability_info['energy_conservation']:.6f}")

# 如果檢測到不穩定，可以調整參數
if stability_info['condition_number'] > 1e12:
    print("警告: 條件數過大，建議減少譜階數或增加正則化")
```

## 高級功能

### 1. 自定義變分修正

可以自定義變分修正項以適應特定需求：

```python
class CustomVariationalCorrection:
    def __init__(self, custom_lambda=0.02):
        self.custom_lambda = custom_lambda

    def compute_correction(self, coeffs, x_current, f_denoiser, sigma):
        # 自定義修正邏輯
        # 例如：基於圖像內容的自適應正則化
        image_complexity = torch.std(x_current)
        adaptive_reg = self.custom_lambda * (1 + image_complexity)

        # 計算梯度並應用自適應正則化
        grad_f = self._approximate_gradient(f_denoiser)
        correction = -adaptive_reg * grad_f

        return correction

# 集成自定義修正
# 這需要修改 SpectralDynamics 類別的實現
```

### 2. 多尺度求解策略

實現多尺度的變分 ODE 求解：

```python
def multiscale_solve(ode_system, initial_coeffs, sigma_start, sigma_end):
    """
    多尺度求解策略
    """
    # 粗尺度求解
    coarse_system = VariationalODESystem(
        spatial_dims=ode_system.spatial_dims,
        spectral_order=ode_system.spectral_order // 4,  # 降低解析度
        sobolev_order=ode_system.hilbert_space.sobolev_order
    )

    # 粗尺度初步求解
    coarse_coeffs = initial_coeffs[:, :, :coarse_system.spectral_order]
    coarse_result, _, _ = coarse_system.solve_variational_ode(
        coarse_coeffs, sigma_start, sigma_end, model_wrapper
    )

    # 細尺度精細求解
    fine_coeffs = torch.zeros_like(initial_coeffs)
    fine_coeffs[:, :, :coarse_system.spectral_order] = coarse_result

    final_result, trajectory, info = ode_system.solve_variational_ode(
        fine_coeffs, sigma_start, sigma_end, model_wrapper
    )

    return final_result, trajectory, info
```

### 3. 並行化求解

對於大批次處理，可以實現並行求解：

```python
def parallel_batch_solve(ode_system, batch_coeffs, sigma_pairs, model_wrapper):
    """
    並行批次求解
    """
    import concurrent.futures

    def solve_single(coeffs, sigma_start, sigma_end):
        return ode_system.solve_variational_ode(
            coeffs.unsqueeze(0), sigma_start, sigma_end, model_wrapper
        )

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        for i in range(batch_coeffs.shape[0]):
            coeffs_i = batch_coeffs[i]
            sigma_start, sigma_end = sigma_pairs[i]

            future = executor.submit(solve_single, coeffs_i, sigma_start, sigma_end)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

    return results
```

## 性能優化

### 1. 記憶體管理

對於大規模問題，優化記憶體使用：

```python
def memory_efficient_solve(ode_system, initial_coeffs, sigma_start, sigma_end):
    """
    記憶體高效的求解方式
    """
    # 分塊處理大的譜係數
    if initial_coeffs.shape[-1] > 512:  # 如果譜階數太大
        chunk_size = 256
        num_chunks = (initial_coeffs.shape[-1] + chunk_size - 1) // chunk_size

        results = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, initial_coeffs.shape[-1])

            chunk_coeffs = initial_coeffs[:, :, start_idx:end_idx]

            # 創建對應大小的系統
            chunk_system = VariationalODESystem(
                spatial_dims=ode_system.spatial_dims,
                spectral_order=end_idx - start_idx,
                sobolev_order=ode_system.hilbert_space.sobolev_order
            )

            chunk_result, _, _ = chunk_system.solve_variational_ode(
                chunk_coeffs, sigma_start, sigma_end, model_wrapper
            )

            results.append(chunk_result)

            # 清理記憶體
            del chunk_system, chunk_result
            torch.cuda.empty_cache()  # 如果使用 GPU

        # 合併結果
        final_coeffs = torch.cat(results, dim=-1)
        return final_coeffs

    else:
        # 正常求解
        final_coeffs, _, _ = ode_system.solve_variational_ode(
            initial_coeffs, sigma_start, sigma_end, model_wrapper
        )
        return final_coeffs
```

### 2. GPU 加速

充分利用 GPU 並行計算能力：

```python
def gpu_optimized_solve(ode_system, initial_coeffs, sigma_start, sigma_end):
    """
    GPU 優化的求解
    """
    # 確保所有數據在 GPU 上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    initial_coeffs = initial_coeffs.to(device)
    ode_system.spectral_basis.device = device
    ode_system.hilbert_space.device = device

    # 使用混合精度
    with torch.cuda.amp.autocast():
        final_coeffs, trajectory, info = ode_system.solve_variational_ode(
            initial_coeffs, sigma_start, sigma_end, model_wrapper
        )

    return final_coeffs, trajectory, info
```

## 調試與診斷

### 1. 詳細日誌

啟用詳細的求解日誌：

```python
def debug_solve(ode_system, initial_coeffs, sigma_start, sigma_end):
    """
    帶詳細日誌的求解
    """
    class DebugModelWrapper:
        def __init__(self, base_wrapper):
            self.base_wrapper = base_wrapper
            self.call_count = 0

        def __call__(self, x, sigma, **kwargs):
            self.call_count += 1
            result = self.base_wrapper(x, sigma, **kwargs)

            if self.call_count % 10 == 0:
                print(f"模型調用 {self.call_count}: σ={sigma:.4f}, "
                      f"輸入範圍=[{x.min():.3f}, {x.max():.3f}], "
                      f"輸出範圍=[{result.min():.3f}, {result.max():.3f}]")

            return result

    debug_wrapper = DebugModelWrapper(model_wrapper)

    # 求解並記錄
    final_coeffs, trajectory, info = ode_system.solve_variational_ode(
        initial_coeffs, sigma_start, sigma_end, debug_wrapper
    )

    print(f"總模型調用次數: {debug_wrapper.call_count}")
    print(f"求解信息: {info}")

    return final_coeffs, trajectory, info
```

### 2. 可視化工具

創建可視化工具監控求解過程：

```python
def visualize_solution_trajectory(trajectory, solve_info):
    """
    可視化求解軌跡
    """
    import matplotlib.pyplot as plt

    # 1. Sobolev 範數演化
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    sobolev_history = solve_info['sobolev_norm_history']
    plt.plot(sobolev_history)
    plt.title('Sobolev 範數演化')
    plt.xlabel('步數')
    plt.ylabel('範數')
    plt.yscale('log')

    # 2. 步長變化
    plt.subplot(1, 3, 2)
    step_history = solve_info['step_size_history']
    plt.plot(step_history)
    plt.title('自適應步長')
    plt.xlabel('步數')
    plt.ylabel('步長')
    plt.yscale('log')

    # 3. 譜能量分佈
    plt.subplot(1, 3, 3)
    coeffs_traj = trajectory['coefficients']
    if coeffs_traj.dim() == 4:  # (T, B, C, M)
        energy_evolution = torch.mean(torch.abs(coeffs_traj) ** 2, dim=(1, 2))  # (T, M)
    else:  # (T, M)
        energy_evolution = torch.abs(coeffs_traj) ** 2

    # 顯示前50個模式的能量演化
    for k in range(min(5, energy_evolution.shape[1])):
        plt.plot(energy_evolution[:, k].cpu(), label=f'模式 {k+1}')

    plt.title('譜能量演化')
    plt.xlabel('步數')
    plt.ylabel('能量')
    plt.yscale('log')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 使用可視化
final_coeffs, trajectory, solve_info = ode_system.solve_variational_ode(
    initial_coeffs, sigma_start, sigma_end, model_wrapper
)
visualize_solution_trajectory(trajectory, solve_info)
```

## 疑難排解

### 常見問題及解決方案

1. **數值不穩定**
   ```python
   # 檢查條件數
   quality = ode_system.analyze_solution_quality(trajectory, solve_info)
   if quality['numerical_stability']['condition_number'] > 1e12:
       print("解決方案:")
       print("- 減少 spectral_order")
       print("- 增加 sobolev_penalty")
       print("- 使用更高精度 (torch.float64)")
   ```

2. **收斂緩慢**
   ```python
   # 分析收斂率
   conv_rate = quality['convergence']['convergence_rate']
   if conv_rate < 0.1:
       print("建議:")
       print("- 調整 regularization_lambda")
       print("- 使用自適應步長")
       print("- 檢查初始條件")
   ```

3. **記憶體不足**
   ```python
   # 檢查記憶體使用
   current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
   if current_memory > 8e9:  # 8GB
       print("記憶體優化建議:")
       print("- 減少 spectral_order")
       print("- 使用分塊處理")
       print("- 啟用 gradient checkpointing")
   ```

## 理論驗證

### 驗證變分原理

```python
def verify_variational_principle(ode_system, trajectory, solve_info):
    """
    驗證變分原理是否得到滿足
    """
    # 計算動作值
    coeffs_traj = trajectory['coefficients']
    sigma_traj = trajectory['sigma_values']

    # 計算 Euler-Lagrange 殘差
    el_residuals = []

    for t in range(1, len(coeffs_traj) - 1):
        # 計算導數
        dt_prev = sigma_traj[t] - sigma_traj[t-1]
        dt_next = sigma_traj[t+1] - sigma_traj[t]

        if abs(dt_prev) > 1e-12 and abs(dt_next) > 1e-12:
            c_dot = (coeffs_traj[t+1] - coeffs_traj[t-1]) / (dt_next - dt_prev)
            c_ddot = (coeffs_traj[t+1] - 2*coeffs_traj[t] + coeffs_traj[t-1]) / (dt_next * dt_prev)

            # 計算 Euler-Lagrange 殘差 (簡化版)
            residual = torch.norm(c_ddot + 0.1 * c_dot)  # 簡化的 EL 方程
            el_residuals.append(residual.item())

    avg_residual = np.mean(el_residuals) if el_residuals else float('inf')

    print(f"Euler-Lagrange 平均殘差: {avg_residual:.2e}")

    if avg_residual < 1e-3:
        print("✓ 變分原理驗證通過")
    else:
        print("✗ 變分原理可能未得到良好滿足")

    return avg_residual < 1e-3
```

## 參考文獻

- Evans, L.C. "Partial Differential Equations"
- Øksendal, B. "Stochastic Differential Equations"
- Quarteroni, A. "Numerical Approximation of Partial Differential Equations"
- Canuto, C. "Spectral Methods in Fluid Dynamics"

---

**注意**: `VariationalODESystem` 是 ISDO 系統最複雜的組件，需要深厚的數值分析和變分計算背景。建議先熟悉基礎的 ODE 求解理論，然後逐步理解變分方法的數值實現。