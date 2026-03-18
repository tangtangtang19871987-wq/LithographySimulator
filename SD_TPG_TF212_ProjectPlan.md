# 基于 Stable Diffusion 的 Text-to-Computational-Lithography Test Pattern Generator（TF 2.12）项目方案

## 1. 联网调研结论（面向本项目的可落地方向）

### 1.1 Stable Diffusion / LDM 可借鉴点
- LDM（Latent Diffusion Models）在**潜空间**中做扩散，能显著降低训练/采样成本，并通过 cross-attention 融合文本条件，适合“文本到版图模式”的可控生成。  
  来源：Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*（CVPR 2022 / arXiv:2112.10752）。
- CFG（Classifier-Free Guidance）可在不引入外部分类器的前提下提升条件一致性，适合在“图形可制造性”与“文本语义对齐”之间调平。  
  来源：Ho & Salimans, *Classifier-Free Diffusion Guidance*（arXiv:2207.12598）。

### 1.2 计算光刻侧可借鉴点
- OpenILT 将 ILT 流程拆成仿真、初始化、优化、评估等模块，强调可替换组件和快速评估，这与我们做“生成器 + 可微仿真器 + 多目标损失”高度一致。  
  来源：OpenOPC/OpenILT GitHub。
- 产业端（如 cuLitho）验证了计算光刻是高计算负载任务，说明项目工程上需要重视 GPU 并行、混合精度、流水线化。  
  来源：NVIDIA cuLitho Developer 页面。

### 1.3 TF 2.12 工程要点
- TF 官方 mixed precision 指南明确了在支持 Tensor Core 的 GPU 上可显著提速，并给出 Keras mixed precision 的标准用法（`tf.keras.mixed_precision.Policy`）。
- 本项目若坚持 TF 2.12，可采用：`tensorflow==2.12.* + keras-cv + xformers替代方案（若仅TF则用原生attention优化）`。

---

## 2. 项目目标与边界

### 2.1 目标
输入自然语言（如“dense horizontal lines with 40nm half-pitch and corner serifs”），输出：
1) 目标 mask/test pattern（二维灰度或二值）；
2) 对应曝光后 wafer 预测图（由内置可微光刻仿真器计算）；
3) 可选 OPC/ILT 修正建议层（迭代 refinement）。

### 2.2 非目标（V1 阶段）
- 不直接替代全流程 sign-off EDA 工具；
- 不覆盖所有工艺节点，仅先支持单一工艺窗口（固定 NA、σ、dose/focus 范围）。

---

## 3. 总体技术架构（Stable-Diffusion-Style）

1. **Text Encoder**（可用 CLIP text encoder 或领域微调 tokenizer+Transformer）
2. **VAE（版图潜空间编码器）**：将高分辨率 pattern 压缩到 latent（例如 256×256 -> 32×32×4）
3. **Conditional U-Net Diffusion**：在 latent 空间去噪，cross-attention 注入文本条件
4. **Differentiable Lithography Simulator（DLS）**：
   - Hopkins/Abbe 成像近似 + resist threshold/简化化学模型；
   - 支持反向传播，把“印刷误差”梯度传回 U-Net。
5. **Multi-objective Loss**：
   - 扩散噪声预测损失（`L_diff`）
   - 几何保真损失（`L_geom`: IoU/Dice/边缘距离）
   - 光刻可制造损失（`L_litho`: CD/EPE/PVBand surrogate）
   - 文本一致性损失（`L_text`: text-image alignment）

总损失：
`L = λ1*L_diff + λ2*L_geom + λ3*L_litho + λ4*L_text`

---

## 4. 数据方案（核心）

### 4.1 数据来源
- **规则程序化生成**（首选）：line/space、contact/via、L/U/S/zigzag、2D logic clips。
- **公开 benchmark 转换**：可参考 OpenILT / ICCAD contest 模式库。
- **工艺仿真增强**：对每个 mask 采样 dose/focus/blur/噪声，生成 wafer 分布标签。

### 4.2 文本标注策略
- 使用模板化 prompt：
  - 几何属性：pitch、CD、角点半径、方向、密度；
  - 工艺属性：dose bias、focus offset、resist threshold；
  - 目标属性：min EPE、maximize process window。
- 例：
  - "Generate sparse contact holes, 60nm CD, high process-window robustness"
  - "Curvilinear assist features around L-shape corner for better edge fidelity"

### 4.3 数据规模建议
- V1：20万~50万样本（程序生成成本低，重点是分布覆盖）
- Train/Val/Test = 8:1:1，且按 pattern family 分层切分，避免泄漏。

---

## 5. TF 2.12 实现方案

## 5.1 环境建议
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "tensorflow==2.12.*" keras-cv matplotlib numpy scipy pandas pyyaml tqdm opencv-python
```
可选：
- `tensorflow-addons`（若用到特定 ops）
- `numba/cupy`（用于快速仿真算子原型）

### 5.2 代码结构建议
```text
project/
  configs/
  data/
    generators/
    prompts/
  models/
    text_encoder.py
    vae.py
    unet_diffusion.py
    schedulers.py
  litho/
    optics.py
    resist.py
    metrics.py
  train/
    train_diffusion.py
    train_joint.py
  eval/
    eval_printability.py
    eval_text_alignment.py
  infer/
    sample_from_text.py
```

### 5.3 训练流程（两阶段）
- **Stage A：预训练生成能力**
  - 先训练 VAE + diffusion（纯几何/文本一致性）；
- **Stage B：引入光刻可微约束**
  - 冻结部分 encoder，联训 U-Net + DLS loss；
  - 逐步升高 `λ3`（curriculum）防止 early collapse。

### 5.4 关键实现细节
- 使用 `tf.data` + cache/prefetch；
- 使用 `mixed_float16`（若 GPU 支持）；
- Checkpoint EMA（指数滑动平均参数）用于更稳的采样质量；
- 采用 DDIM/DPM-Solver 风格快速采样（可自行在 TF 中实现 scheduler）。

---

## 6. 验证方案（必须可量化）

### 6.1 离线生成质量
1. **Geometry Fidelity**：IoU / Dice / Hausdorff distance
2. **Diversity**：LPIPS 或特征空间覆盖度（可用自建 encoder feature distance）
3. **Text-Condition Consistency**：
   - 模板属性解析准确率（例如从图中反推 pitch/CD 与 prompt 对比）

### 6.2 光刻可制造性验证（核心KPI）
1. **CD Error**：`|CD_print - CD_target|` 的均值/95分位
2. **EPE（Edge Placement Error）**：均值、最大值、违规率
3. **PV Band / Process Window Surrogate**：在 dose-focus 网格上的可印刷稳定性
4. **Hotspot Rate**：规则/学习型 hotspot 检出率

### 6.3 对比实验（Ablation）
- Baseline-1：无文本条件（只做 unconditional）
- Baseline-2：无 `L_litho`（只追求视觉几何）
- Baseline-3：GAN/VAE-only 生成器
- Ours：LDM + CFG + DLS loss

预期结论方向：
- 有 `L_litho` 时，EPE/CD 指标显著优于纯视觉生成；
- CFG 在合理区间提升文本约束满足率，但过大 guidance 会降低图形多样性。

### 6.4 在线/工程验证
- 小规模接入一条真实 OPC review 流程：
  - 人工评审“可用候选率”；
  - 与传统规则模板库生成速度对比（TAT）；
  - fail case 分类（角点桥连、孤立线断裂、SRAF误触发）。

---

## 7. 里程碑计划（12周样例）
- W1-W2：数据生成器与 prompt DSL
- W3-W4：VAE + 文本条件 U-Net 跑通（TF2.12）
- W5-W6：引入 DLS 与 `L_litho`，完成 joint training
- W7-W8：评估框架（CD/EPE/PVBand）和 ablation
- W9-W10：推理加速 + CFG 扫描 + 失败案例分析
- W11-W12：Demo、文档、可复现实验脚本

---

## 8. 风险与缓解

1. **仿真器与真实工艺 gap**
   - 缓解：先校准 surrogate（用少量 sign-off 数据拟合）。
2. **文本与版图语义错位**
   - 缓解：限制 prompt DSL + 属性解析器闭环训练。
3. **训练不稳定/模式坍缩**
   - 缓解：EMA、梯度裁剪、`λ3` 课程学习、分阶段训练。
4. **TF2.12 生态限制**
   - 缓解：优先使用原生 Keras 层，避免重度依赖只支持新版本的插件。

---

## 9. 最小可行产品（MVP）定义

交付一个命令行工具：
```bash
python infer/sample_from_text.py \
  --prompt "dense vertical lines, 45nm half-pitch, low EPE priority" \
  --num_samples 16 \
  --guidance_scale 5.0
```
输出：
- `mask_*.png`（生成版图）
- `wafer_*.png`（仿真成像）
- `report.json`（CD/EPE/PVBand 指标）

MVP 达标阈值示例：
- 平均 CD error <= 10% target CD
- EPE 违规率较 baseline 降低 >= 20%
- 文本属性命中率 >= 85%
