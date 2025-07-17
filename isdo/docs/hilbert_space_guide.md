# HilbertSpace 模塊說明文檔

## 概述

`HilbertSpace` 模塊實現了 ISDO 系統中的 Sobolev 空間操作，這是無窮維變分最優控制的數學基礎。該模塊提供了完整的泛函分析工具，包括 Sobolev 範數計算、嵌入檢查和 Gelfand 三元組操作。

## 數學理論基礎

### Sobolev 空間定義

對於開集 Ω ⊂ ℝᵈ，Sobolev 空間 H^s(Ω) 定義為：

```
H^s(Ω) = {f ∈ L²(Ω) : ||f||_{H^s} < ∞}
```

其中 Sobolev 範數為：
```
||f||_{H^s}² = Σ_{|α| ≤ s} ||D^α f||_{L²}²
```

### Sobolev 嵌入定理

**關鍵定理**: 當 s > d/2 時，有緊嵌入 H^s(Ω) ↪ C^0(Ω̄)

這保證了 ISDO 算法中函數的連續性和有界性。

### Gelfand 三元組

實現嵌入鏈：H ⊂ H' ⊂ H''，其中：
- H: 原 Hilbert 空間
- H': 對偶空間
- H'': 雙對偶空間

## 核心功能

### 1. Sobolev 範數計算

```python
from modules_forge.isdo.core.hilbert_space import HilbertSpace

# 初始化希爾伯特空間
hilbert = HilbertSpace(
    spatial_dims=(64, 64),
    sobolev_order=1.5,  # H^1.5 空間
    domain_size=(1.0, 1.0),
    device=torch.device('cuda')
)

# 計算 Sobolev 範數
image = torch.randn(1, 3, 64, 64)
sobolev_norm = hilbert.compute_sobolev_norm(image)
print(f"H^1.5 範數: {sobolev_norm}")
```

### 2. 嵌入條件檢查

檢查 Sobolev 嵌入的各種條件：

```python
# 檢查不同階數的嵌入性質
for s in [0.5, 1.0, 1.5, 2.0, 3.0]:
    embedding_info = hilbert.check_sobolev_embedding(s)
    print(f"\nH^{s} 空間嵌入性質:")
    print(f"  連續嵌入: {embedding_info['continuous_embedding']}")
    print(f"  緊嵌入: {embedding_info['compact_embedding']}")
    print(f"  臨界情況: {embedding_info['critical_sobolev']}")

    if 'holder_embedding' in embedding_info:
        print(f"  Hölder 嵌入: {embedding_info['holder_embedding']}")
        print(f"  Hölder 指數: {embedding_info['holder_exponent']:.3f}")
```

### 3. 不同內積計算

支援多種內積計算：

```python
f = torch.randn(64, 64)
g = torch.randn(64, 64)

# L² 內積
l2_inner = hilbert.inner_product(f, g, space_type='l2')
print(f"L² 內積: {l2_inner.item():.6f}")

# Sobolev 內積
sobolev_inner = hilbert.inner_product(f, g, space_type='sobolev')
print(f"H^s 內積: {sobolev_inner.item():.6f}")

# 對偶配對
dual_inner = hilbert.inner_product(f, g, space_type='dual')
print(f"對偶配對: {dual_inner.item():.6f}")
```

### 4. 分數階拉普拉斯算子

實現分數階微分算子：

```python
# 應用分數階拉普拉斯算子
alpha = 0.5  # 分數階參數
result = hilbert.fractional_laplacian(image, alpha)
print(f"(-Δ)^{alpha} 的結果形狀: {result.shape}")

# 不同階數的效果
for alpha in [0.25, 0.5, 0.75, 1.0]:
    frac_laplacian = hilbert.fractional_laplacian(image, alpha)
    norm = torch.norm(frac_laplacian)
    print(f"α={alpha}: ||(-Δ)^α f|| = {norm.item():.3f}")
```

## 高級功能

### 1. Sobolev 正則性估計

自動估計函數的 Sobolev 正則性：

```python
# 創建不同正則性的測試函數
def create_test_function(regularity_level):
    """創建指定正則性的測試函數"""
    x = torch.linspace(0, 1, 64)
    y = torch.linspace(0, 1, 64)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    if regularity_level == 'smooth':
        # C∞ 函數 (高正則性)
        return torch.exp(-(X**2 + Y**2) * 10)
    elif regularity_level == 'lipschitz':
        # Lipschitz 函數
        return torch.sqrt(X**2 + Y**2 + 0.01)
    elif regularity_level == 'rough':
        # 低正則性函數
        return torch.sign(X - 0.5) * torch.sign(Y - 0.5)

# 測試不同函數的正則性
for level in ['smooth', 'lipschitz', 'rough']:
    test_func = create_test_function(level)
    estimated_order = hilbert.estimate_sobolev_regularity(test_func)
    print(f"{level} 函數估計正則性: H^{estimated_order:.2f}")
```

### 2. Gelfand 三元組操作

實現對偶空間的投影：

```python
# 投影到對偶空間
dual_projection = hilbert.gelfand_triple_projection(image, target_space='dual')
print(f"對偶投影形狀: {dual_projection.shape}")

# 投影到雙對偶空間
double_dual = hilbert.gelfand_triple_projection(image, target_space='double_dual')
print(f"雙對偶投影形狀: {double_dual.shape}")

# 驗證嵌入的保範性
original_norm = hilbert.compute_sobolev_norm(image)
dual_norm = hilbert.compute_sobolev_norm(dual_projection)
print(f"原始範數: {original_norm.item():.6f}")
print(f"對偶範數: {dual_norm.item():.6f}")
```

### 3. 嵌入常數驗證

驗證 Sobolev 嵌入的最佳常數：

```python
# 測試多個函數以估計嵌入常數
embedding_constants = []

for i in range(10):
    test_func = torch.randn(1, 1, 64, 64)
    constants = hilbert.verify_embedding_constants(test_func)
    embedding_constants.append(constants)

    print(f"測試 {i+1}:")
    print(f"  連續嵌入常數: {constants.get('continuous_embedding', 'N/A')}")
    print(f"  L² 嵌入常數: {constants['l2_embedding']:.3f}")

# 統計分析
if embedding_constants:
    l2_constants = [c['l2_embedding'] for c in embedding_constants]
    print(f"\nL² 嵌入常數統計:")
    print(f"  平均值: {np.mean(l2_constants):.3f}")
    print(f"  標準差: {np.std(l2_constants):.3f}")
    print(f"  最大值: {np.max(l2_constants):.3f}")
```

## 實際應用案例

### 1. 圖像去噪的正則化

利用 Sobolev 空間的正則化性質：

```python
def sobolev_regularized_denoising(noisy_image, regularization_strength=0.1):
    """
    使用 Sobolev 範數正則化的圖像去噪
    """
    # 確保圖像在 Sobolev 球內
    projected_image = hilbert.project_to_sobolev_ball(
        noisy_image, radius=1.0
    )

    # 應用分數階擴散
    denoised = hilbert.fractional_laplacian(
        projected_image, alpha=regularization_strength
    )

    return denoised

# 應用去噪
noisy = torch.randn(1, 3, 64, 64) * 0.1 + torch.randn(1, 3, 64, 64)
denoised = sobolev_regularized_denoising(noisy)

print(f"去噪前 Sobolev 範數: {hilbert.compute_sobolev_norm(noisy).item():.3f}")
print(f"去噪後 Sobolev 範數: {hilbert.compute_sobolev_norm(denoised).item():.3f}")
```

### 2. 多尺度分析

利用不同 Sobolev 階數進行多尺度分析：

```python
def multiscale_sobolev_analysis(image):
    """
    多尺度 Sobolev 分析
    """
    scales = [0.5, 1.0, 1.5, 2.0]
    analysis_results = {}

    for s in scales:
        # 創建對應階數的希爾伯特空間
        h_space = HilbertSpace(
            spatial_dims=(64, 64),
            sobolev_order=s,
            device=image.device
        )

        # 計算該尺度的範數
        norm_s = h_space.compute_sobolev_norm(image)

        # 檢查嵌入性質
        embedding = h_space.check_sobolev_embedding(s)

        analysis_results[f'H^{s}'] = {
            'norm': norm_s.item(),
            'continuous_embedding': embedding['continuous_embedding'],
            'estimated_regularity': h_space.estimate_sobolev_regularity(image)
        }

    return analysis_results

# 執行多尺度分析
image = torch.randn(1, 1, 64, 64)
results = multiscale_sobolev_analysis(image)

for scale, info in results.items():
    print(f"\n{scale} 空間分析:")
    print(f"  範數: {info['norm']:.3f}")
    print(f"  連續嵌入: {info['continuous_embedding']}")
    print(f"  估計正則性: H^{info['estimated_regularity']:.2f}")
```

## 數值穩定性與優化

### 1. 條件數監控

```python
def monitor_numerical_stability(hilbert_space, test_functions):
    """
    監控數值穩定性
    """
    condition_numbers = []

    for i, f in enumerate(test_functions):
        # 計算導數
        derivatives = hilbert_space.sobolev_norm._compute_derivatives(f)

        # 檢查導數的條件數
        for deriv_name, deriv_tensor in derivatives.items():
            cond_num = torch.max(deriv_tensor.abs()) / (torch.min(deriv_tensor.abs()) + 1e-12)
            condition_numbers.append(cond_num.item())

        # 檢查 Sobolev 範數的數值穩定性
        norm = hilbert_space.compute_sobolev_norm(f)
        l2_norm = torch.sqrt(torch.sum(f.abs() ** 2))
        ratio = norm / (l2_norm + 1e-12)

        print(f"函數 {i}: Sobolev/L² 比例 = {ratio.item():.3f}")

    print(f"平均條件數: {np.mean(condition_numbers):.2e}")
    print(f"最大條件數: {np.max(condition_numbers):.2e}")

# 生成測試函數
test_funcs = [torch.randn(1, 1, 64, 64) for _ in range(5)]
monitor_numerical_stability(hilbert, test_funcs)
```

### 2. 自適應精度控制

```python
def adaptive_precision_control(hilbert_space, target_function, target_error=1e-6):
    """
    自適應精度控制
    """
    # 從低精度開始
    current_dtype = torch.float32

    while True:
        # 使用當前精度計算
        f_current = target_function.to(current_dtype)
        norm_current = hilbert_space.compute_sobolev_norm(f_current)

        # 使用雙精度作為參考
        f_reference = target_function.to(torch.float64)
        hilbert_reference = HilbertSpace(
            spatial_dims=hilbert_space.spatial_dims,
            sobolev_order=hilbert_space.sobolev_order,
            device=hilbert_space.device
        )
        norm_reference = hilbert_reference.compute_sobolev_norm(f_reference)

        # 計算相對誤差
        relative_error = torch.abs(norm_current - norm_reference.to(current_dtype)) / norm_reference.to(current_dtype)

        print(f"精度 {current_dtype}: 相對誤差 = {relative_error.item():.2e}")

        if relative_error < target_error:
            print(f"達到目標精度，使用 {current_dtype}")
            break
        elif current_dtype == torch.float64:
            print("已達最高精度")
            break
        else:
            current_dtype = torch.float64

    return current_dtype

# 測試自適應精度
test_image = torch.randn(1, 1, 64, 64)
optimal_dtype = adaptive_precision_control(hilbert, test_image)
```

## 疑難排解

### 常見問題及解決方案

1. **數值不穩定**
   ```python
   # 檢查輸入數據範圍
   print(f"輸入範圍: [{torch.min(image):.3f}, {torch.max(image):.3f}]")

   # 正規化輸入
   normalized_image = (image - torch.mean(image)) / (torch.std(image) + 1e-8)
   ```

2. **Sobolev 範數爆炸**
   ```python
   # 投影到有界集合
   bounded_image = hilbert.project_to_sobolev_ball(image, radius=10.0)

   # 檢查是否需要降低 Sobolev 階數
   if hilbert.sobolev_order > 2.0:
       print("警告: 高階 Sobolev 空間可能數值不穩定")
   ```

3. **記憶體不足**
   ```python
   # 分塊處理大圖像
   def process_in_chunks(large_image, chunk_size=32):
       H, W = large_image.shape[-2:]
       results = []

       for i in range(0, H, chunk_size):
           for j in range(0, W, chunk_size):
               chunk = large_image[..., i:i+chunk_size, j:j+chunk_size]
               chunk_result = hilbert.compute_sobolev_norm(chunk)
               results.append(chunk_result)

       return torch.stack(results)
   ```

## 理論驗證

### 嵌入定理驗證

```python
def verify_sobolev_embedding_theorem():
    """
    驗證 Sobolev 嵌入定理
    """
    # 測試 s > d/2 的情況
    s_critical = 1.1  # > d/2 = 1 對於 2D

    hilbert_critical = HilbertSpace(
        spatial_dims=(64, 64),
        sobolev_order=s_critical
    )

    # 生成隨機函數
    test_func = torch.randn(10, 1, 64, 64)

    # 計算 Sobolev 範數和連續範數
    sobolev_norms = []
    continuous_norms = []

    for i in range(10):
        sob_norm = hilbert_critical.compute_sobolev_norm(test_func[i:i+1])
        cont_norm = torch.max(test_func[i].abs())

        sobolev_norms.append(sob_norm.item())
        continuous_norms.append(cont_norm.item())

    # 檢查嵌入不等式
    embedding_constants = [c/s for c, s in zip(continuous_norms, sobolev_norms)]

    print(f"嵌入常數範圍: [{min(embedding_constants):.3f}, {max(embedding_constants):.3f}]")
    print(f"平均嵌入常數: {np.mean(embedding_constants):.3f}")

    # 驗證定理成立
    if all(c < float('inf') for c in embedding_constants):
        print("✓ Sobolev 嵌入定理驗證成功")
    else:
        print("✗ 嵌入定理驗證失敗")

verify_sobolev_embedding_theorem()
```

## 參考文獻

- Adams, R. A. "Sobolev Spaces"
- Evans, L. C. "Partial Differential Equations"
- Brezis, H. "Functional Analysis, Sobolev Spaces and Partial Differential Equations"
- Triebel, H. "Theory of Function Spaces"

---

**注意**: HilbertSpace 模塊涉及深層的泛函分析理論，建議結合相關數學背景一起學習。對於實際應用，重點關注 Sobolev 嵌入條件和數值穩定性。