# VariationalController 模塊說明文檔

## 概述

`VariationalController` 是 ISDO 系統的核心控制器，實現了變分最優控制理論，將傳統的擴散採樣問題轉化為在希爾伯特空間中求解最優路徑的變分問題。這是 ISDO 算法超越傳統 ODE 求解器的關鍵所在。

## 數學理論基礎

### 變分最優控制問題

ISDO 將擴散採樣重新建模為最優控制問題：

**目標**: 尋找從噪聲 x(σ_max) 到數據 x(0) 的最優軌跡，最小化動作積分：

```
𝒜[x] = ∫[σ_max to 0] ℒ(x, ẋ, σ) dσ
```

其中拉格朗日函數為：
```
ℒ(x, ẋ, σ) = ½|ẋ - f(x;σ)/σ|²_H + λ|∇_x f|²_op + μ|∇²x|²
```

### Euler-Lagrange 方程

最優軌跡必須滿足 Euler-Lagrange 方程：

```
d/dσ(∂ℒ/∂ẋ) - ∂ℒ/∂x = 0
```

這給出了比傳統 ODE 更精確的動力學方程。

### Hamilton-Jacobi-Bellman 方程

對於值函數 V(x, σ)，HJB 方程為：

```
∂V/∂σ + H(x, ∇V, σ) = 0
```

其中 Hamiltonian 為：
```
H(x, p, σ) = min_u [⟨p, u⟩ + ℒ(x, u, σ)]
```

## 核心功能

### 1. 動作積分計算

```python
from modules_forge.isdo.core.variational_controller import VariationalController

# 初始化變分控制器
controller = VariationalController(
    spatial_dims=(64, 64),
    regularization_lambda=0.01,  # λ: 正則化強度
    curvature_penalty=0.001,     # μ: 曲率懲罰
    sobolev_order=1.5,          # Sobolev 空間階數
    device=torch.device('cuda')
)

# 計算最優控制軌跡
x_init = torch.randn(1, 3, 64, 64)  # 初始噪聲
sigma_start = 10.0
sigma_end = 0.1

optimal_trajectory, optimal_controls = controller.compute_optimal_control(
    x_current=x_init,
    sigma_current=sigma_start,
    denoiser_function=model,  # 去噪模型
    target_sigma=sigma_end,
    num_steps=50,
    extra_args={'conditioning': conditioning_info}
)

print(f"軌跡形狀: {optimal_trajectory.shape}")  # (51, 1, 3, 64, 64)
print(f"控制形狀: {optimal_controls.shape}")   # (50, 1, 3, 64, 64)
```

### 2. 拉格朗日函數計算

單步拉格朗日函數計算：

```python
# 計算單點的拉格朗日函數值
x = torch.randn(1, 3, 64, 64)
x_dot = torch.randn(1, 3, 64, 64)  # dx/dσ
f_denoiser = model(x, sigma * torch.ones(1))  # 去噪函數輸出
sigma = 5.0

lagrangian_value = controller.action_integral.compute_lagrangian(
    x=x,
    x_dot=x_dot,
    f_denoiser=f_denoiser,
    sigma=sigma
)

print(f"拉格朗日函數值: {lagrangian_value.item():.6f}")
```

### 3. 完整軌跡動作計算

```python
# 評估整個軌跡的動作積分
sigma_schedule = torch.linspace(10.0, 0.1, 51)

def denoiser_wrapper(x, s_in, **kwargs):
    return model(x, s_in)

action_value = controller.action_integral.compute_action(
    trajectory=optimal_trajectory,
    sigma_schedule=sigma_schedule,
    denoiser_function=denoiser_wrapper,
    extra_args={}
)

print(f"總動作值: {action_value.item():.3f}")
```

## 高級功能

### 1. 軌跡質量評估

全面評估軌跡的各項質量指標：

```python
quality_metrics = controller.evaluate_trajectory_quality(
    trajectory=optimal_trajectory,
    sigma_schedule=sigma_schedule,
    denoiser_function=denoiser_wrapper
)

print("軌跡質量評估:")
for metric, value in quality_metrics.items():
    print(f"  {metric}: {value:.6f}")
```

輸出示例：
```
軌跡質量評估:
  action_value: 15.234567
  initial_sobolev_norm: 12.345678
  final_sobolev_norm: 3.456789
  trajectory_smoothness: 2.345678
  energy_change: -8.888999
  norm_ratio: 0.280124
```

### 2. 最優性條件驗證

驗證軌跡是否滿足 Euler-Lagrange 方程：

```python
optimality_check = controller.verify_optimality_conditions(
    trajectory=optimal_trajectory,
    sigma_schedule=sigma_schedule,
    denoiser_function=denoiser_wrapper,
    tolerance=1e-3
)

print("最優性檢查:")
print(f"  Euler-Lagrange 滿足: {optimality_check['euler_lagrange_satisfied']}")
print(f"  最大殘差範數: {optimality_check['max_residual_norm']:.2e}")
print(f"  在容忍範圍內: {optimality_check['residual_within_tolerance']}")
```

### 3. 自適應控制策略

實現自適應的變分控制：

```python
def adaptive_variational_sampling(
    model,
    x_init,
    sigma_max=10.0,
    sigma_min=0.01,
    quality_threshold=0.01
):
    """
    自適應變分採樣，根據軌跡質量動態調整步數
    """
    current_x = x_init
    current_sigma = sigma_max
    trajectory_points = [current_x]

    while current_sigma > sigma_min:
        # 估算下一步的目標 sigma
        sigma_step = min(current_sigma * 0.8, current_sigma - sigma_min)
        target_sigma = current_sigma - sigma_step

        # 計算最優控制
        traj, controls = controller.compute_optimal_control(
            x_current=current_x,
            sigma_current=current_sigma,
            denoiser_function=lambda x, s: model(x, s),
            target_sigma=target_sigma,
            num_steps=10
        )

        # 評估質量
        sigma_schedule = torch.linspace(current_sigma, target_sigma, 11)
        quality = controller.evaluate_trajectory_quality(
            trajectory=traj,
            sigma_schedule=sigma_schedule,
            denoiser_function=lambda x, s: model(x, s)
        )

        # 根據質量調整
        if quality['trajectory_smoothness'] > quality_threshold:
            # 質量不佳，減小步長
            sigma_step *= 0.5
            target_sigma = current_sigma - sigma_step
            continue

        # 接受這一步
        current_x = traj[-1]
        current_sigma = target_sigma
        trajectory_points.append(current_x)

        print(f"σ: {current_sigma:.3f}, 動作值: {quality['action_value']:.3f}")

    return torch.stack(trajectory_points, dim=0)

# 使用自適應採樣
adaptive_result = adaptive_variational_sampling(model, x_init)
```

## 實際應用案例

### 1. 超高解析度圖像生成

變分控制特別適合高解析度圖像生成：

```python
def high_resolution_generation(model, resolution=1024):
    """
    高解析度圖像的變分採樣
    """
    # 創建高解析度控制器
    hr_controller = VariationalController(
        spatial_dims=(resolution, resolution),
        regularization_lambda=0.005,  # 降低正則化避免過度平滑
        curvature_penalty=0.0001,     # 輕微曲率約束
        sobolev_order=1.2,           # 較低階數適合高頻細節
    )

    # 初始化噪聲
    x_init = torch.randn(1, 3, resolution, resolution)

    # 使用更精細的 σ 調度
    sigma_schedule = torch.logspace(1, -2, 100)  # 100 步精細調度

    # 分段優化
    trajectory_segments = []
    current_x = x_init

    for i in range(0, len(sigma_schedule)-10, 10):
        segment_sigmas = sigma_schedule[i:i+11]

        segment_traj, _ = hr_controller.compute_optimal_control(
            x_current=current_x,
            sigma_current=segment_sigmas[0].item(),
            denoiser_function=lambda x, s: model(x, s),
            target_sigma=segment_sigmas[-1].item(),
            num_steps=10
        )

        trajectory_segments.append(segment_traj)
        current_x = segment_traj[-1]

        # 監控質量
        quality = hr_controller.evaluate_trajectory_quality(
            trajectory=segment_traj,
            sigma_schedule=segment_sigmas,
            denoiser_function=lambda x, s: model(x, s)
        )
        print(f"段 {i//10}: 動作值 {quality['action_value']:.3f}")

    return torch.cat(trajectory_segments, dim=0)

# 生成高解析度圖像
hr_result = high_resolution_generation(model, resolution=512)
final_image = hr_result[-1]  # 最終結果
```

### 2. 條件生成的精確控制

利用變分控制實現精確的條件生成：

```python
def conditional_variational_generation(
    model,
    conditioning,
    control_strength=1.0
):
    """
    條件生成的變分控制
    """
    # 修改拉格朗日函數以包含條件約束
    class ConditionalActionIntegral(controller.action_integral.__class__):
        def __init__(self, parent, conditioning, strength):
            super().__init__(
                parent.lambda_reg,
                parent.curvature_penalty,
                parent.domain_size
            )
            self.conditioning = conditioning
            self.strength = strength

        def compute_lagrangian(self, x, x_dot, f_denoiser, sigma, grad_f=None):
            # 基礎拉格朗日項
            base_lagrangian = super().compute_lagrangian(
                x, x_dot, f_denoiser, sigma, grad_f
            )

            # 條件約束項
            if self.conditioning is not None:
                condition_error = torch.sum((x - self.conditioning)**2, dim=(-2, -1))
                condition_penalty = self.strength * condition_error
                base_lagrangian = base_lagrangian + condition_penalty

            return base_lagrangian

    # 創建條件控制器
    conditional_controller = VariationalController(
        spatial_dims=(64, 64),
        regularization_lambda=0.01,
        curvature_penalty=0.001,
        sobolev_order=1.5
    )

    # 替換動作積分計算器
    conditional_controller.action_integral = ConditionalActionIntegral(
        conditional_controller.action_integral,
        conditioning,
        control_strength
    )

    # 執行條件生成
    x_init = torch.randn(1, 3, 64, 64)

    trajectory, controls = conditional_controller.compute_optimal_control(
        x_current=x_init,
        sigma_current=10.0,
        denoiser_function=lambda x, s: model(x, s, conditioning=conditioning),
        target_sigma=0.1,
        num_steps=50
    )

    return trajectory[-1]

# 使用條件生成
conditioning_info = torch.randn(1, 3, 64, 64)  # 條件信息
conditional_result = conditional_variational_generation(
    model, conditioning_info, control_strength=0.5
)
```

### 3. 多模態採樣策略

利用變分控制探索多模態分佈：

```python
def multimodal_variational_sampling(model, num_modes=4):
    """
    多模態變分採樣
    """
    results = []

    for mode in range(num_modes):
        # 每個模態使用不同的初始化和參數
        x_init = torch.randn(1, 3, 64, 64) * (1.0 + 0.2 * mode)

        # 動態調整正則化強度
        mode_controller = VariationalController(
            spatial_dims=(64, 64),
            regularization_lambda=0.005 * (1 + mode * 0.5),  # 遞增正則化
            curvature_penalty=0.001 / (1 + mode * 0.2),      # 遞減曲率約束
            sobolev_order=1.5
        )

        trajectory, _ = mode_controller.compute_optimal_control(
            x_current=x_init,
            sigma_current=10.0,
            denoiser_function=lambda x, s: model(x, s),
            target_sigma=0.1,
            num_steps=50
        )

        # 評估模態質量
        quality = mode_controller.evaluate_trajectory_quality(
            trajectory=trajectory,
            sigma_schedule=torch.linspace(10.0, 0.1, 51),
            denoiser_function=lambda x, s: model(x, s)
        )

        results.append({
            'sample': trajectory[-1],
            'quality': quality,
            'mode_id': mode
        })

        print(f"模態 {mode}: 動作值 {quality['action_value']:.3f}")

    # 選擇最佳模態或返回所有結果
    best_mode = min(results, key=lambda x: x['quality']['action_value'])

    return {
        'best_sample': best_mode['sample'],
        'all_samples': [r['sample'] for r in results],
        'qualities': [r['quality'] for r in results]
    }

# 多模態採樣
multimodal_results = multimodal_variational_sampling(model, num_modes=3)
```

## 性能優化

### 1. 並行化軌跡計算

```python
def parallel_trajectory_computation(controller, x_batch, model):
    """
    批次並行計算多個軌跡
    """
    batch_size = x_batch.shape[0]

    # 並行計算所有軌跡
    trajectories = []

    for i in range(batch_size):
        traj, _ = controller.compute_optimal_control(
            x_current=x_batch[i:i+1],
            sigma_current=10.0,
            denoiser_function=lambda x, s: model(x, s),
            target_sigma=0.1,
            num_steps=50
        )
        trajectories.append(traj)

    # 合併結果
    batched_trajectories = torch.stack(trajectories, dim=1)  # (T, B, C, H, W)

    return batched_trajectories

# 批次處理
x_batch = torch.randn(4, 3, 64, 64)  # 4 個樣本
batch_results = parallel_trajectory_computation(controller, x_batch, model)
```

### 2. 記憶體優化的長軌跡

```python
def memory_efficient_long_trajectory(controller, x_init, model, total_steps=1000):
    """
    記憶體優化的長軌跡計算
    """
    checkpoint_interval = 50
    current_x = x_init
    current_sigma = 10.0
    sigma_min = 0.01

    trajectory_checkpoints = [current_x]

    num_checkpoints = total_steps // checkpoint_interval
    sigma_schedule = torch.logspace(1, -2, num_checkpoints + 1)

    for i in range(num_checkpoints):
        target_sigma = sigma_schedule[i + 1].item()

        # 計算段軌跡
        segment_traj, _ = controller.compute_optimal_control(
            x_current=current_x,
            sigma_current=current_sigma,
            denoiser_function=lambda x, s: model(x, s),
            target_sigma=target_sigma,
            num_steps=checkpoint_interval
        )

        # 只保存檢查點，釋放中間結果
        current_x = segment_traj[-1].clone()
        current_sigma = target_sigma
        trajectory_checkpoints.append(current_x)

        # 強制垃圾回收
        del segment_traj
        torch.cuda.empty_cache()

        print(f"檢查點 {i+1}/{num_checkpoints}, σ: {current_sigma:.4f}")

    return torch.stack(trajectory_checkpoints, dim=0)
```

## 調試與診斷

### 1. 收斂性分析

```python
def analyze_convergence(controller, trajectory, sigma_schedule, model):
    """
    分析軌跡的收斂性質
    """
    # 計算每步的動作變化
    action_history = []

    for t in range(len(trajectory) - 1):
        segment = trajectory[t:t+2]
        segment_sigmas = sigma_schedule[t:t+2]

        action_val = controller.action_integral.compute_action(
            segment, segment_sigmas, lambda x, s: model(x, s)
        )
        action_history.append(action_val.item())

    # 分析收斂率
    action_diffs = np.diff(action_history)
    convergence_rate = np.mean(np.abs(action_diffs))

    # 檢查單調性
    is_monotonic = np.all(action_diffs <= 0)  # 動作應該遞減

    print(f"平均收斂率: {convergence_rate:.2e}")
    print(f"動作單調遞減: {is_monotonic}")

    return {
        'action_history': action_history,
        'convergence_rate': convergence_rate,
        'monotonic': is_monotonic
    }

# 分析收斂性
convergence_info = analyze_convergence(
    controller, optimal_trajectory, sigma_schedule, model
)
```

### 2. 數值穩定性檢查

```python
def check_numerical_stability(controller, trajectory):
    """
    檢查數值計算的穩定性
    """
    # 檢查軌跡中的異常值
    trajectory_norms = torch.norm(trajectory.view(len(trajectory), -1), dim=1)

    # 檢測爆炸或消失
    max_norm = torch.max(trajectory_norms)
    min_norm = torch.min(trajectory_norms)
    condition_number = max_norm / (min_norm + 1e-12)

    # 檢查梯度
    x_dot = controller.action_integral._compute_trajectory_derivative(
        trajectory, torch.linspace(10.0, 0.1, len(trajectory))
    )
    gradient_norms = torch.norm(x_dot.view(len(x_dot), -1), dim=1)
    max_gradient = torch.max(gradient_norms)

    stability_ok = (
        condition_number < 1e6 and
        max_norm < 1e3 and
        max_gradient < 1e3
    )

    return {
        'stable': stability_ok,
        'condition_number': condition_number.item(),
        'max_norm': max_norm.item(),
        'max_gradient': max_gradient.item()
    }
```

## 疑難排解

### 常見問題及解決方案

1. **動作積分爆炸**
   ```python
   # 降低正則化參數
   controller = VariationalController(
       spatial_dims=(64, 64),
       regularization_lambda=0.001,  # 從 0.01 降到 0.001
       curvature_penalty=0.0001,
       sobolev_order=1.0  # 降低 Sobolev 階數
   )
   ```

2. **軌跡不平滑**
   ```python
   # 增加曲率懲罰
   controller.action_integral.curvature_penalty = 0.01  # 增加到 0.01

   # 或使用更多積分步數
   trajectory, _ = controller.compute_optimal_control(
       x_current=x_init,
       sigma_current=10.0,
       denoiser_function=model,
       target_sigma=0.1,
       num_steps=100  # 增加步數
   )
   ```

3. **記憶體不足**
   ```python
   # 使用漸進式計算
   def progressive_computation(controller, x_init, model):
       current_x = x_init
       sigma_points = torch.logspace(1, -2, 21)  # 20 段

       for i in range(len(sigma_points) - 1):
           traj, _ = controller.compute_optimal_control(
               x_current=current_x,
               sigma_current=sigma_points[i].item(),
               denoiser_function=model,
               target_sigma=sigma_points[i+1].item(),
               num_steps=5  # 短段計算
           )
           current_x = traj[-1]
           torch.cuda.empty_cache()

       return current_x
   ```

## 參考文獻

- Bertsekas, D.P. "Dynamic Programming and Optimal Control"
- Pontryagin, L.S. "Mathematical Theory of Optimal Processes"
- Fleming, W.H. "Controlled Markov Processes and Viscosity Solutions"
- Evans, L.C. "An Introduction to Mathematical Optimal Control Theory"

---

**注意**: VariationalController 涉及複雜的變分計算，建議先理解經典變分法和最優控制理論。實際使用時，參數調優對結果影響很大。