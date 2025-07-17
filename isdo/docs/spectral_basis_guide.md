# SpectralBasis 模塊說明文檔

## 概述

`SpectralBasis` 是 ISDO 系統的核心數學模塊，負責在無窮維希爾伯特空間中生成各種正交基底。此模塊實現了譜方法的數學基礎，是將圖像從空間域轉換到頻域的關鍵組件。

## 數學背景

### 希爾伯特空間理論

在 L²(Ω) 希爾伯特空間中，我們尋找完備的正交基底 {φₖ}，使得任意函數 f ∈ L²(Ω) 可以表示為：

```
f(x) = Σ[k=1 to ∞] cₖ φₖ(x)
```

其中投影係數為：`cₖ = ⟨f, φₖ⟩ / ||φₖ||²`

### 支援的基底類型

#### 1. 傅立葉基底 (FOURIER)

**數學定義**：
```
φₖ₁,ₖ₂(x,y) = exp(2πi(k₁x/H + k₂y/W)) / √(H×W)
```

**特性**：
- 適合分析週期性結構
- 在頻域處理中表現優異
- 支援快速傅立葉變換 (FFT) 加速

**應用場景**：
- 自然圖像的頻率分析
- 紋理和週期性模式

#### 2. 小波基底 (WAVELET)

**數學定義** (Haar 小波)：
```
ψⱼ,ₘ,ₙ(x,y) = 2^(j/2) ψ(2ʲx - m, 2ʲy - n)
```

**特性**：
- 多尺度分析能力
- 時頻局部化
- 適合處理非平穩信號

**應用場景**：
- 邊緣檢測和特徵提取
- 多尺度圖像分析

#### 3. Chebyshev 基底 (CHEBYSHEV)

**數學定義**：
```
Tₙ,ₘ(x,y) = Tₙ(x) × Tₘ(y)
```

其中 Tₙ(x) 是第 n 階 Chebyshev 多項式。

**特性**：
- 在區間 [-1,1] 上正交
- 良好的數值穩定性
- 適合多項式逼近

## 主要功能

### 1. 基底生成

```python
from modules_forge.isdo.core.spectral_basis import SpectralBasis, BasisType

# 創建傅立葉基底
basis = SpectralBasis(
    spatial_dims=(64, 64),
    spectral_order=256,
    basis_type=BasisType.FOURIER,
    device=torch.device('cuda')
)
```

### 2. 函數投影

將圖像投影到譜空間：

```python
# 輸入圖像 (H, W) 或 (B, C, H, W)
image = torch.randn(1, 3, 64, 64)

# 計算譜係數
coefficients = basis.project_to_basis(image)
print(f"譜係數形狀: {coefficients.shape}")  # (1, 3, 256)
```

### 3. 函數重建

從譜係數重建圖像：

```python
# 從譜係數重建
reconstructed = basis.reconstruct_from_coefficients(coefficients)
print(f"重建圖像形狀: {reconstructed.shape}")  # (1, 3, 64, 64)

# 計算重建誤差
error = torch.norm(image - reconstructed)
print(f"重建誤差: {error.item()}")
```

### 4. 自適應譜截斷

根據能量閾值自動截斷譜：

```python
# 保留 99% 能量的譜模式
truncated_coeffs, effective_order = basis.truncate_spectrum(
    coefficients,
    energy_threshold=0.99
)
print(f"有效譜階數: {effective_order}")
```

### 5. 頻率分析

分析信號的頻率內容：

```python
freq_analysis = basis.get_frequency_content(coefficients)
print(f"低頻能量比例: {freq_analysis['low_freq_ratio']:.3f}")
print(f"高頻能量比例: {freq_analysis['high_freq_ratio']:.3f}")
print(f"有效帶寬: {freq_analysis['effective_bandwidth']}")
```

## 核心算法

### 投影算法

1. **內積計算**：`⟨f, φₖ⟩ = ∫ f(x) φₖ*(x) dx`
2. **歸一化**：`cₖ = ⟨f, φₖ⟩ / ||φₖ||²`
3. **截斷控制**：保留重要的 M 個模式

### Gram-Schmidt 正交化

確保基底函數的正交性：

```python
# 正交化一組向量
vectors = torch.randn(10, 64, 64)  # 10 個隨機向量
orthogonal = basis.gram_schmidt_orthogonalization(vectors)

# 驗證正交性
for i in range(5):
    for j in range(i+1, 5):
        inner_prod = basis.inner_product(orthogonal[i], orthogonal[j])
        print(f"⟨φ{i}, φ{j}⟩ = {inner_prod.abs().item():.6f}")
```

## 性能優化

### GPU 加速

- 所有運算支援 CUDA 加速
- 批次化處理提高效率
- 記憶體優化的 FFT 實現

### 數值穩定性

- 自動歸一化防止數值溢出
- 正則化項避免除零錯誤
- 條件數監控保證穩定性

```python
# 檢查基底的條件數
norms = basis.normalization_factors
condition_number = torch.max(norms) / torch.min(norms)
print(f"基底條件數: {condition_number.item():.2e}")
```

## 實際應用示例

### 圖像壓縮

```python
# 使用譜截斷進行圖像壓縮
original_image = load_image("example.jpg")
coeffs = basis.project_to_basis(original_image)

# 保留 95% 能量
compressed_coeffs, _ = basis.truncate_spectrum(coeffs, 0.95)
compressed_image = basis.reconstruct_from_coefficients(compressed_coeffs)

# 計算壓縮比
compression_ratio = torch.count_nonzero(coeffs) / torch.count_nonzero(compressed_coeffs)
print(f"壓縮比: {compression_ratio.item():.1f}:1")
```

### 頻域濾波

```python
# 低通濾波
def lowpass_filter(coefficients, cutoff_freq=0.3):
    freq_analysis = basis.get_frequency_content(coefficients)
    # 保留低頻分量
    filtered_coeffs = coefficients.clone()
    high_freq_start = int(cutoff_freq * len(coefficients))
    filtered_coeffs[high_freq_start:] = 0
    return filtered_coeffs

filtered_coeffs = lowpass_filter(coefficients)
filtered_image = basis.reconstruct_from_coefficients(filtered_coeffs)
```

## 數學證明要點

### 完備性證明

傅立葉基底在 L²([0,1]²) 上是完備的，即：
```
||f - Σ[k=1 to M] cₖφₖ||₂ → 0  as M → ∞
```

### 收斂性分析

對於 Sobolev 空間 H^s 中的函數，譜截斷誤差滿足：
```
||f - P_M f||_{H^s} ≤ C M^{-s} ||f||_{H^s}
```

其中 P_M 是到前 M 個模式的投影算子。

## 疑難排解

### 常見問題

1. **記憶體不足**
   - 減少 `spectral_order`
   - 使用 CPU 模式
   - 批次處理大圖像

2. **數值不穩定**
   - 檢查輸入數據範圍
   - 增加正則化參數
   - 使用更高精度 (torch.float64)

3. **重建誤差過大**
   - 增加譜階數
   - 檢查基底選擇是否合適
   - 驗證歸一化因子

### 調試工具

```python
# 檢查基底質量
def diagnose_basis(basis):
    print(f"譜階數: {basis.spectral_order}")
    print(f"基底類型: {basis.basis_type}")
    print(f"歸一化因子範圍: [{basis.normalization_factors.min():.2e}, {basis.normalization_factors.max():.2e}]")

    # 測試投影-重建精度
    test_image = torch.randn(1, 1, basis.H, basis.W)
    coeffs = basis.project_to_basis(test_image)
    reconstructed = basis.reconstruct_from_coefficients(coeffs)
    error = torch.norm(test_image - reconstructed) / torch.norm(test_image)
    print(f"相對重建誤差: {error.item():.2e}")

diagnose_basis(basis)
```

## 參考資料

- Canuto, C. et al. "Spectral Methods in Fluid Dynamics"
- Daubechies, I. "Ten Lectures on Wavelets"
- Evans, L.C. "Partial Differential Equations"
- Trefethen, L.N. "Spectral Methods in MATLAB"

---

**注意**: 此模塊需要深厚的泛函分析和調和分析背景。建議結合數學理論文檔一起學習。