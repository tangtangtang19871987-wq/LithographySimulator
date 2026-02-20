# 计算光刻 ILT 频谱损失函数（Spectral Loss）实施指南

这份文档针对你提到的核心痛点：**频谱底噪 + 数量级爆炸 + DC 主导梯度**，给出可落地方案。

配套代码：`ilt_spectral_loss.py`（纯 Python，无第三方依赖）。

---

## 1) 问题复盘：为什么直接 `log(|FFT|+eps)` 容易翻车

在典型版图（如 metal）中，低频/DC 与线端/角点高频的能量差可达 `1e4 ~ 1e6`。

如果你直接做：

\[
L = ||\log(P_{pred}+\epsilon) - \log(P_{tgt}+\epsilon)||_2^2
\]

当 `eps` 极小（如 `1e-16`）时：
- 频谱底噪位置会得到极大负值；
- 这些“本应无意义”的背景点反而可能在损失中占高权重；
- 梯度被 DC/低频牵引，模型更难学到 CD/line-end 的高频细节。

---

## 2) 推荐：物理加权归一化（Physical-Normalized Weighting）

核心思想：
1. 先把目标频谱能量归一化，避免绝对量级主导。
2. 再构造权重矩阵，压制 DC，适度提升关键中高频。
3. 对谱取 log 前使用**相对地板值**（relative floor），而不是固定超小 eps。

### 2.1 建议的频域损失

设：
- `P_pred(u,v), P_tgt(u,v)` 为功率谱（通常先 `fftshift`）；
- `\hat P_tgt = P_tgt / max(P_tgt)`；
- `r(u,v)` 为归一化径向频率（中心低频=0，角落高频≈1）。

定义权重：

\[
W(u,v) = (\alpha + \hat P_{tgt}(u,v))^{-\gamma} \cdot (1 + \beta r(u,v))
\]

定义稳定 log 谱：

\[
S(P)=\log(\max(P,\eta\cdot\max(P)))
\]

最终损失：

\[
L_{spec}=\frac{\sum_{u,v} W(u,v)\,\big(S(P_{pred})-S(P_{tgt})\big)^2}{\sum_{u,v}W(u,v)}
\]

推荐初始参数：
- `alpha=0.03~0.1`
- `gamma=0.5~0.9`
- `beta(hf_boost)=0.3~1.0`
- `eta(floor_ratio)=1e-6~1e-4`

---

## 3) 工程实现要点（非常关键）

1. **不要只看总损失**：记录 DC 环、mid-band、high-band 的分区损失。
2. **别让权重失控**：可对 `W` 做 clip，例如 `[w_min,w_max]`。
3. **和空间域目标联合训练**：
   \[
   L = \lambda_{pix}L_{pix}+\lambda_{edge}L_{edge}+\lambda_{spec}L_{spec}
   \]
4. **频域 mask 要与工艺目标匹配**：CD/line-end 常落在中高频，不应被统一低权。
5. **调参顺序**：先固定 `eta` 与 `alpha` 保数值稳定，再调 `gamma/beta` 控梯度重心。

---

## 4) 配套代码接口

`ilt_spectral_loss.py` 提供：
- `dft2_power(image)`: 2D 功率谱
- `fftshift2(m)`: 频谱中心化
- `normalized_log_spectrum(power, floor_ratio)`: 稳定 log 变换
- `physical_normalized_weight(target_power_shifted, alpha, gamma, hf_boost)`: 物理权重
- `spectral_loss(pred, target, weight, floor_ratio)`: 计算加权谱损失
- `demo_metrics()`: 小案例输出对比指标

运行：

```bash
python ilt_spectral_loss.py
python test_ilt_spectral_loss.py
```

---

## 5) 我对你给出的思路的补充看法

你的方向完全正确，尤其是“不要直接硬上 `log+tiny eps`，而是做物理归一化加权”。

我建议再加两点：

1. **分频段 curriculum**：训练早期先弱化高频，待模型稳定后逐步提升 `hf_boost`，收敛更稳。
2. **工艺敏感方向加权**：如果某些方向（如 line-end 轴向）更关键，可引入方向性权重（各向异性频域权重）。

这两点在 ILT 场景通常比单纯增大高频权重更稳健。
