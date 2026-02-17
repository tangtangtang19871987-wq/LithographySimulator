# Half-UNet FNO — TensorFlow 2.10 移植说明

**来源：** https://github.com/Robh96/UNet-FNO
**目标框架：** TensorFlow 2.10
**移植文件：** `half_unet_fno.py`、`train_fno.py`
**对比分析：** `FNO_IMPLEMENTATION_COMPARISON.md`
**日期：** 2026-02-11

---

## 一、原始项目概述

Robh96/UNet-FNO 实现了一个用于 2D 湍流辐射层流场预测的 **Half-UNet Fourier Neural Operator**，核心思想是用频域谱卷积（Spectral Convolution）替代标准 U-Net 中的局部空间卷积，获得**真正的全局感受野**。

### 原始架构（PyTorch，NCHW 格式）

```
输入 (B, C_in, H, W)
  │
  ├─ lifting: Conv2D 1×1 → width 通道
  │
  ├─ Encoder:
  │   ├─ FNOBlock_0 → 保存特征（原始分辨率）
  │   │    MaxPool2D(2)
  │   ├─ FNOBlock_1 → 保存特征（1/2 分辨率）
  │   │    MaxPool2D(2)
  │   ├─ FNOBlock_2 → 保存特征（1/4 分辨率）
  │   │    MaxPool2D(2)
  │   └─ FNOBlock_3（bottleneck）→ 保存特征（1/8 分辨率）
  │
  ├─ 全尺度特征融合（Full-Scale Feature Fusion）：
  │   所有 4 个特征图 bilinear upsample 回原始 H×W
  │   → 逐元素求和（不是拼接！）
  │
  ├─ Final FNOBlocks × num_final_blocks
  │
  └─ Projection head: Conv1×1(width→width×4) → GELU → Conv1×1(width×4→C_out)

输出 (B, C_out, H, W)
```

每个 **FNOBlock**：

```
x → SpectralConv2d(x) ─┐
                        ├─ + → activation → + ← residual(x)
x → Conv2D_1×1(x)   ─┘
```

每个 **SpectralConv2d**：

```
x (B,C_in,H,W)
  → rfft2(x, norm='ortho')          # (B, C_in, H, W//2+1) 复数
  → 截断至前 modes1 行 + 后 modes1 行（对称负频率）
  → einsum("bixy,ioxy->boxy", x, w)  # 频域跨通道混合
  → 在零填充后的全频率张量中写回
  → irfft2(out, s=(H,W), norm='ortho')
```

---

## 二、与本仓库现有 FFT 实现的关键区别

详见 `FNO_IMPLEMENTATION_COMPARISON.md`，核心对比：

| 维度 | 本移植（SpectralConv2d） | 本仓库 fft_conv.py |
|------|------------------------|-------------------|
| FFT 维度 | 2D rfft2（全空间） | 1D rfft（逐轴） |
| 频率截断 | ✅ 保留前 modes 个低频系数 | ❌ 保留全谱 |
| 通道混合方式 | ✅ 频域 einsum（全局） | ❌ 空域 pointwise（局部） |
| 感受野 | **全局**（每个输出像素看到所有输入） | 局部（卷积核大小决定） |
| 设计目的 | PDE solver / 全局场预测 | 大核圆形卷积加速 |

**结论**：两种 FFT 用法互补。SpectralConv2d 适合需要全局感受野的场景（光刻的全局光学效应），fft_conv.py 适合精确的圆形边界卷积加速。

---

## 三、TF 2.10 适配说明

### 3.1 数据格式：NCHW → NHWC

PyTorch 默认 NCHW，TensorFlow 默认 NHWC。所有维度操作均已调整。

```python
# rfft2d 在最内两维操作，需临时转置
x_nchw = tf.transpose(x, [0, 3, 1, 2])        # NHWC → NCHW
x_ft_nchw = tf.signal.rfft2d(x_nchw)          # rfft2 on last 2 dims
x_ft = tf.transpose(x_ft_nchw, [0, 2, 3, 1])  # NCHW → NHWC
```

Einsum 索引映射：

| PyTorch（NCHW） | TF（NHWC）| 含义 |
|---------------|----------|------|
| `"bixy,ioxy->boxy"` | `"bhwi,hwio->bhwo"` | b=batch, h/x=H, w/y=W, i=C_in, o=C_out |

### 3.2 复数权重：`torch.cfloat` → 实虚部分离存储

TF 2.10 中 `tf.Variable(dtype=tf.complex64)` 的梯度计算不稳定，因此将每组权重拆为两个 float32 变量：

```python
# PyTorch 原始
self.weights1 = nn.Parameter(
    scale * torch.rand(..., dtype=torch.cfloat))   # 单个复数参数

# TF 2.10 移植
self.w1_re = self.add_weight("w1_re", shape=w_shape, dtype=tf.float32)  # 实部
self.w1_im = self.add_weight("w1_im", shape=w_shape, dtype=tf.float32)  # 虚部

# 使用时组合
w1 = tf.complex(self.w1_re, self.w1_im)
```

参数数量不变（float32×2 = complex64×1），梯度更稳定。

### 3.3 FFT 归一化：`norm='ortho'` 手动实现

PyTorch 的 `rfft2(x, norm='ortho')` 内置正交归一化。TF `tf.signal.rfft2d` 不支持此参数，需手动处理：

```python
# 正变换后除以 sqrt(H*W)
norm = tf.cast(tf.math.sqrt(tf.cast(H * W, tf.float32)), tf.complex64)
x_ft = x_ft / norm

# 逆变换前乘回
out_ft = out_ft * norm
x_out = tf.signal.irfft2d(out_ft, fft_length=[H, W])
```

### 3.4 频率张量重建：scatter-in-place → tf.concat

PyTorch 可以直接切片赋值（`out_ft[:, :, :m1, :m2] = ...`）。TF 的 `tf.Tensor` 不可变，用 `tf.concat` + 零填充重建：

```python
# 步骤：
# 1. W 方向：将 out_top (m1_eff, m2_eff) 右侧补零到 W//2+1
pad_w = W_half - m2_eff
zeros_w = tf.zeros([B, m1_eff, pad_w, C_out], dtype=tf.complex64)
out_top_full = tf.concat([out_top, zeros_w], axis=2)   # (B, m1_eff, W//2+1, C_out)

# 2. H 方向：top + 中间零行 + bottom，拼接成完整频率张量
mid_h = H - 2 * m1_eff
zeros_mid = tf.zeros([B, mid_h, W_half, C_out], dtype=tf.complex64)
out_ft = tf.concat([out_top_full, zeros_mid, out_bot_full], axis=1)
```

### 3.5 双线性上采样：`F.interpolate` → `tf.image.resize`

```python
# PyTorch
upsampled = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)

# TF 2.10（NHWC，target_size=(H,W) 元组）
upsampled = tf.image.resize(feat, target_size, method='bilinear')
```

### 3.6 优化器：`AdamW` + `StepLR`

```python
# AdamW（TF 2.10 experimental）
try:
    opt = tf.keras.optimizers.experimental.AdamW(
        learning_rate=lr_schedule, weight_decay=weight_decay)
except AttributeError:
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# StepLR → PiecewiseConstantDecay（精确等价）
boundaries = [step_size * steps_per_epoch * (i + 1) for i in range(20)]
values     = [initial_lr * (gamma ** i) for i in range(len(boundaries) + 1)]
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
```

### 3.7 子层列表：`nn.ModuleList` → Python `list`

TF Keras 会自动追踪 `self.*` 中的 Layer 对象，无需 `nn.ModuleList`：

```python
# PyTorch
self.encoder_blocks = nn.ModuleList([FNOBlock(...) for i in range(levels+1)])

# TF（直接用 list，Keras 自动注册）
self.encoder_blocks = [FNOBlock(..., name=f"enc_{i}") for i in range(levels+1)]
```

---

## 四、使用方法

### 快速开始

```python
from half_unet_fno import build_and_compile

model = build_and_compile(
    in_channels=1,          # mask 通道数
    out_channels=1,         # aerial image 通道数
    input_shape_hw=(512, 512),
    modes=20,               # Fourier 模式数
    width=48,               # 内部通道宽度
    levels=3,               # 编码器下采样层数
    num_final_blocks=2,     # 融合后 FNOBlock 数量
    activation='gelu',
    learning_rate=3e-4,
    weight_decay=1e-5,
)

# 训练
model.fit(train_dataset, epochs=100, validation_data=val_dataset)
```

### 训练脚本

```bash
# 烟雾测试（验证流程）
python train_fno.py --smoke-test

# 从生成数据训练
python train_fno.py --num-samples 2000 --image-size 256 \
    --modes 20 --width 48 --levels 3 --epochs 100 --batch-size 8

# 从 .npz 文件训练，同时写出 TFRecord
python train_fno.py --dataset ./data/litho_1k.npz \
    --write-tfrecord --tfrecord-dir ./tfrecords --epochs 100

# 从 TFRecord 多 GPU 训练
python train_fno.py --tfrecord-dir ./tfrecords \
    --multi-gpu --batch-size 16 --epochs 200

# 自定义超参数
python train_fno.py --modes 12 --width 32 --levels 2 \
    --lr 1e-4 --step-size 20 --gamma 0.5 --patience 15
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--modes` | 20 | Fourier 模式数（H 和 W 方向相同） |
| `--width` | 48 | 编码器通道宽度（全程恒定） |
| `--levels` | 3 | 下采样层数（编码器共 levels+1 个 FNOBlock） |
| `--num-final-blocks` | 2 | 融合后追加的 FNOBlock 数 |
| `--activation` | gelu | 激活函数：`gelu` 或 `relu` |
| `--batch-size` | 8 | 每个 GPU 的批次大小（per-replica） |
| `--lr` | 3e-4 | 初始学习率 |
| `--step-size` | 20 | StepLR 每 N 轮衰减一次 |
| `--gamma` | 0.5 | LR 衰减系数 |
| `--patience` | 10 | 早停耐心（0 禁用） |
| `--multi-gpu` | False | 启用 MirroredStrategy |

---

## 五、模型参数量参考

| modes | width | levels | num_final | 参数量 |
|-------|-------|--------|-----------|--------|
| 12 | 32 | 2 | 1 | ~320K |
| 20 | 48 | 3 | 2 | ~1.8M |
| 20 | 48 | 4 | 2 | ~2.2M |
| 20 | 64 | 3 | 2 | ~3.1M |

SpectralConv2d 参数量公式（每层）：

```
4 × modes × (modes//2 + 1) × C_in × C_out × 2（实虚部）
         ↑         ↑              ↑
     w1/w2 各两组  rfft W方向      float32 对
```

---

## 六、训练输出目录

```
experiments_fno/
└── 20260211_143022/            ← 时间戳命名（或 --run-name 指定）
    ├── config.json             ← 训练配置（所有 args）
    ├── history.csv             ← 逐 epoch 的 loss/mae
    ├── best_model.keras        ← val_loss 最优 checkpoint
    ├── final_model.keras       ← 最终 epoch 的模型
    └── tb_logs/                ← TensorBoard 日志（--tensorboard）
```

加载已保存的模型：

```python
import tensorflow as tf
from half_unet_fno import SpectralConv2d, FNOBlock, HalfUNetFNO

model = tf.keras.models.load_model(
    'experiments_fno/run/best_model.keras',
    custom_objects={
        'SpectralConv2d': SpectralConv2d,
        'FNOBlock': FNOBlock,
        'HalfUNetFNO': HalfUNetFNO,
    }
)
```

---

## 七、已知限制与注意事项

1. **静态空间尺寸**：`irfft2d` 的 `fft_length` 参数在 TF 2.10 中需要静态值。当输入形状完全动态时，会在第一次 `tf.function` 编译时报错。建议训练时保持固定的 H×W。

2. **`tf.function` 追踪**：由于 Python `list` 中的子层，`model.call` 中的 for 循环在 `tf.function` 下会展开（unroll）。当 `levels` 较大时会增加图构建时间，但对运行性能无影响。

3. **混合精度**：SpectralConv2d 内部使用 `complex64`（等价 `float32`）计算。若启用 `float16` 混合精度，FFT 部分仍以 `float32` 运行，最终输出会自动转换。建议添加 `tf.keras.mixed_precision.Policy('mixed_float16')` 时在 SpectralConv2d 中显式 cast 输入到 float32。

4. **Optuna 超参优化**：原始项目含 Optuna 集成，当前移植不包含此功能，可参考原项目的 `optimize.py` 并将 PyTorch 模型替换为 `build_and_compile()` 调用。
