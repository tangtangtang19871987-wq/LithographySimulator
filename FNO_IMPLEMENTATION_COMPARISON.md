# FNO/FFT 实现对比分析

**参考项目:** https://github.com/Robh96/UNet-FNO (PyTorch)
**本仓库:** LithographySimulator (TensorFlow 2.10)
**日期:** 2026-02-11

---

## 一、总体架构对比

| 维度 | UNet-FNO (Robh96) | 本仓库 (fft_conv / fft_layers) |
|------|-------------------|-------------------------------|
| **框架** | PyTorch | TensorFlow 2.10 |
| **数据格式** | NCHW (B, C, H, W) | NHWC (B, H, W, C) |
| **核心 FFT 用途** | 谱卷积（频域通道混合） | 圆形卷积加速（大核 depthwise） |
| **频域通道混合** | ✅ 是（einsum 在频域做 in→out 混合） | ❌ 否（仅 depthwise，pointwise 在空域） |
| **截断模式数** | ✅ 是（只保留低频 modes1×modes2 个系数） | ❌ 否（保留全谱，无截断） |
| **可学习复数权重** | ✅ 是（complex weight tensors）| ❌ 否（仅实数 depthwise kernel） |
| **1D vs 2D FFT** | 2D rfft2（全空间） | 1D rfft（逐轴：width 或 height） |
| **应用场景** | 湍流场预测（通用 PDE solver） | 光刻仿真（大核圆形卷积加速） |

---

## 二、SpectralConv2d 详细对比

### 2.1 UNet-FNO SpectralConv2d (PyTorch)

```
输入: (B, C_in, H, W)
  ↓
rfft2(input, norm='ortho')  → X_ft: (B, C_in, H, W//2+1)  [complex]
  ↓
截断至前 modes1 行 + 后 modes1 行（低频 + 对称负频率）
仅保留 W//2+1 中的前 modes2//2+1 列
  ↓
einsum("bixy,ioxy->boxy", X_ft_truncated, weights)
  → 权重形状: (C_in, C_out, modes1, modes2//2+1)  [complex]
  → 频域 in→out 通道混合
  ↓
irfft2(out_ft, s=(H, W), norm='ortho')  → 输出: (B, C_out, H, W)
```

**关键点：**
- `weights1`: 对应正频率区域上半部分
- `weights2`: 对应负频率区域（下半部分，`-modes1:`）
- 同时保留正/负频率两组权重，共 `2 × modes1 × modes2//2+1` 个复数参数

### 2.2 本仓库 fft_circular_depthwise_conv1d (TF)

```
输入: (B, H, W, C)
  ↓
转置 → (B, H, C, W) [axis=2 时]
  ↓
rfft(x_t, fft_length=[N])  → X: (B, H, C, N//2+1)  [complex]
  ↓
构建圆形 kernel: h_circ = [causal | zeros | anti_causal]
rfft(h_circ)  → H_freq: (C, N//2+1)  [complex]
  ↓
Y = X * conj(H_freq)  [逐元素，depthwise，无跨通道混合]
  ↓
irfft(Y)  → y_t
  ↓
转置回 → (B, H, W, C)
```

**关键点：**
- 无截断：保留全频谱
- 无跨通道混合：每通道独立（depthwise）
- 圆形卷积：内核以循环方式排列（符合光刻周期边界条件）
- Pointwise Conv2D 在空域完成通道混合

---

## 三、核心差异总结

### 3.1 截断 vs 全谱

| | UNet-FNO | 本仓库 |
|--|--|--|
| 方式 | 低通滤波（保留 modes 个低频系数） | 全谱保留 |
| 意义 | 利用物理信号低频主导特性，正则化效果 | 精确圆形卷积，无频率截断损失 |
| 适用 | PDE 求解（流场通常低频主导） | 大核圆形卷积加速 |

### 3.2 频域通道混合 vs 空域通道混合

| | UNet-FNO | 本仓库 |
|--|--|--|
| 频域 | ✅ 全局感受野通道混合 | ❌ 仅 depthwise |
| 空域 | ✅ 1×1 Conv（FNOBlock中） | ✅ 1×1 Conv（FFTAxisCircularConv中）|
| 参数量 | `C_in × C_out × modes × modes//2+1 × 2` | `K × C`（极小）|

### 3.3 架构用途

| | UNet-FNO | 本仓库 |
|--|--|--|
| 设计目标 | 替代 UNet 中的局部卷积为全局谱卷积 | 加速 UNet 中的大核（≥11）空间卷积 |
| 理论基础 | Fourier Neural Operator（Li et al. 2020） | 圆形卷积 + FFT 计算效率 |
| 信息流 | 全图全频谱混合（真全局感受野） | 局部圆形卷积（周期性局部感受野） |

---

## 四、HalfUNet 架构对比

### 4.1 UNet-FNO 的 HalfUNet

```
Input
  ↓ lifting (Conv2D 1×1)
  ↓
Encoder（levels+1 个 FNOBlock，每层 MaxPool2D 降采样）
  │── enc_out[0] (原始分辨率)
  │── enc_out[1] (1/2)
  │── enc_out[2] (1/4)
  │── ...
  └── enc_out[levels] (bottleneck)
  ↓
Full-Scale Feature Fusion（所有层双线性上采样回原始分辨率，逐元素求和）
  ↓
final_fno_blocks（FNOBlock × num_final_blocks）
  ↓
projection（Conv2D 扩展 → 激活 → Conv2D 输出）
Output
```

**特点：**
- 恒定通道宽度 width（不像标准 UNet 逐层翻倍）
- 解码器极简：无卷积，仅上采样 + 求和（非拼接）
- 所有跳跃连接在最后统一融合（全尺度融合）

### 4.2 本仓库 shift_equivariant_unet.py

```
Input
  ↓ CircularConv2D
  ↓
Encoder（跨层通道翻倍，CircularPad + Conv2D）
  │── skip[0], skip[1], ...
  └── bottleneck
  ↓
Decoder（逐层上采样 + 跳跃连接拼接 + CircularConv2D）
Output
```

**区别：**
- 通道数逐层翻倍
- 跳跃连接：拼接（concat）而非求和
- 卷积：圆形边界条件（shift-equivariant）
- 无谱卷积

---

## 五、复数权重实现方式

### PyTorch（原生复数支持）

```python
self.weights1 = nn.Parameter(
    scale * torch.rand(..., dtype=torch.cfloat)
)
# torch.cfloat = complex64
```

### TensorFlow 2.10（需拆分实虚部）

TF 2.10 中 `tf.Variable` 不直接支持 complex64 参数，需：

```python
# 方式 1：存储实部 + 虚部两个独立变量
self.weights_re = self.add_weight(shape=(...), dtype=tf.float32)
self.weights_im = self.add_weight(shape=(...), dtype=tf.float32)

# 使用时组合
weights = tf.complex(self.weights_re, self.weights_im)

# 方式 2（更新版 TF 支持）：直接 complex64
# 但 TF 2.10 中 add_weight(..., dtype=tf.complex64) 行为有限制，
# 仍推荐方式 1 确保稳定性
```

---

## 六、移植要点

| 项目 | PyTorch API | TensorFlow 2.10 等价 |
|------|------------|----------------------|
| 2D rfft | `torch.fft.rfft2(x, norm='ortho')` | `tf.signal.rfft2d(x)` + 手动除以 `sqrt(H*W)` |
| 2D irfft | `torch.fft.irfft2(x, s=(H,W))` | `tf.signal.irfft2d(x, fft_length=[H,W])` |
| 复数 einsum | `torch.einsum("bixy,ioxy->boxy", x, w)` | `tf.einsum("bhwi,hioc->bhwc", x, w)` (NHWC) |
| 复数权重 | `nn.Parameter(dtype=torch.cfloat)` | `add_weight(dtype=tf.float32)` × 2 (实/虚) |
| MaxPool | `nn.MaxPool2d(2)` | `tf.keras.layers.MaxPool2D(2)` |
| Bilinear upsample | `F.interpolate(..., mode='bilinear')` | `tf.image.resize(..., method='bilinear')` |
| GELU | `nn.GELU()` | `tf.keras.activations.gelu` |
| AdamW | `torch.optim.AdamW` | `tf.keras.optimizers.AdamW` (TF 2.10需自定义或用addons) |
| StepLR | `torch.optim.lr_scheduler.StepLR` | `tf.keras.optimizers.schedules.ExponentialDecay` (近似) |
| 数据格式 | NCHW | NHWC（所有操作需调整维度顺序） |

---

## 七、结论

两个实现代表了 FFT 在神经网络中的**不同应用范式**：

1. **UNet-FNO (Robh96)**：真正的 Fourier Neural Operator，在频域做跨通道混合+截断，是一种全局操作，适合物理场预测（PDE solver）。

2. **本仓库 (fft_conv)**：FFT 加速的大核局部圆形卷积，保持空域卷积语义（局部感受野），仅用 FFT 提升计算效率，适合光刻仿真的周期边界条件。

移植价值：HalfUNetFNO 的谱卷积层为本仓库带来**真正全局感受野**，与现有的圆形卷积互补，对光刻成像（全局光学效应显著）有潜在优势。
