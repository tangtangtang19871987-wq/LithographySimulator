# Geometry-driven ILT Feature Extraction
## 复杂光刻规则到可执行几何表达式的方法论报告（分章节交付）

> 交付策略：严格按“章节逐步确认”执行。本文档先提交**顶层设计 + 第1章**。收到你的确认后，我再继续第2章。

---

## 0. 顶层设计（Top-level Design）

### 0.1 报告目标
将自然语言/工艺文档中的复杂规则，系统地改写为可执行、可验证、可并行的几何操作序列，且仅使用基础闭包操作：

- 基元：`Polygon`, `Edge`, `Vertex`
- 操作：`EDGE`, `GROW`, `SHRINK`, `AND`, `OR`, `NOT`, `INTERACT`, `FILTER`

### 0.2 统一转换流水线
每条规则都必须经过如下 6 个步骤（不可跳步）：

1. **Spec Parsing**：将自然语言规则分解为对象、方向、阈值、关系。
2. **Manhattan Rewrite**：将任意方向表达重写为 H/V 方向可处理形式。
3. **Distance Rewrite**：将“距离判断”重写为 `GROW/SHRINK + INTERACT`。
4. **Operation Graph**：形成有向无环图（DAG），节点为操作，边为几何对象流。
5. **DSL Emission**：落地为 DSL 代码块（可执行）。
6. **Validation**：检查阈值方向、对称性、边界条件、tile 一致性。

### 0.3 全文结构计划（满足原始 spec）
- 第1章：Manhattan 几何策略（本次交付）
- 第2章：Distance → Sizing 变换（含数学等价与 6+ 例）
- 第3章：几何操作闭包（15+ 分解例）
- 第4章：Spec → Graph → DSL（15+ 完整案例）
- 第5章：训练手册（30+ 题含解答与错误模式）
- 第6章：Agent 约束与评估清单

### 0.4 本章节交付边界
本次仅完成：
- 第1章全部内容
- 至少 5 个完整改写示例
- 每个示例均给出“规则 → H/V 分解 → 操作化表达”

---

# 第1章：Manhattan Geometry in Lithography

## 1.1 为什么优先使用 Manhattan（H/V）表达

### 1.1.1 问题背景
在 ILT/DRC/OPC 流程中，设计规则原文常写成“距离小于 d”“端点对端点间隔不足”“局部桥连风险”等语义。直接在连续欧氏平面上执行这些规则会面临三类问题：

1. **计算复杂度上升**：任意角度边的最近距离计算需要更复杂的几何核。
2. **API 映射困难**：主流 EDA 几何 API 对 H/V 边的布尔和尺寸化操作高度优化。
3. **并行切片不稳定**：tile 边界处任意角度几何更容易触发不一致。

Manhattan 策略的核心是：
- 先把规则投影为“水平/垂直主导”的约束语义；
- 再用轴对齐操作（grow/shrink/boolean/interact）执行。

### 1.1.2 Manhattan 的 5 个工程优势

#### 优势 A：几何复杂度下降
任意方向线段最近距离问题，转换为 x/y 轴方向的区间重叠与扩张相交判断。计算对象从“连续角度”降为“离散方向类别 + 轴向阈值”。

#### 优势 B：可直接映射到 EDA 操作原语
- `GROW(dx, dy)`、`SHRINK(dx, dy)`天然轴向；
- `EDGE(layer, orientation=H|V)`天然支持边方向筛选；
- `INTERACT(A, B)`可表达“经过尺寸化后的触碰/重叠”。

#### 优势 C：并行与分块友好
tile 处理时，H/V 扩张半径易计算 halo，边界补偿策略统一，减少跨 tile 漏检。

#### 优势 D：规则可组合性更好
复杂规则可拆成“方向分支”：
- 水平分支：`R_H`
- 垂直分支：`R_V`
最后 `OR(R_H, R_V)` 回收整体。

#### 优势 E：可验证性强
每一步变换都可检查：方向集合是否完整、阈值是否被正确映射到 dx/dy、是否引入额外候选。

## 1.2 边方向分类（Edge Orientation Classification）

### 1.2.1 基本分类
给定边 `e = (p1, p2)`：
- 若 `|y2 - y1| = 0`，则 `e ∈ H`（水平边）；
- 若 `|x2 - x1| = 0`，则 `e ∈ V`（垂直边）；
- 否则 `e ∈ N`（非 Manhattan 边，后续第1.6处理）。

### 1.2.2 规则执行中的方向索引
在执行层不直接遍历“所有边”，而是建立方向索引：
- `E_H = EDGE(M, orientation=H)`
- `E_V = EDGE(M, orientation=V)`

随后将规则分支化执行，避免不必要的候选对比。

## 1.3 水平/垂直规则重写模板

### 1.3.1 通用模板
自然语言规则 `R`：
> “图形集合 `M` 满足条件 `C`”

重写为：
- `R_H := C` 在 `E_H` 上成立
- `R_V := C` 在 `E_V` 上成立
- `R := OR(R_H, R_V)` 或按规则语义做 `AND/组合`

### 1.3.2 方向耦合规则
若规则本质依赖“正交关系”（例如端点朝向、line-end 对 sidewall），则必须显式写出交叉分支：
- `R_HV := f(E_H, E_V)`
- `R_VH := f(E_V, E_H)`
- `R := OR(R_HV, R_VH)`

## 1.4 示例（≥5）

> 每个示例均包含：
> 1) 原始规则
> 2) Manhattan 改写
> 3) 可执行操作化表达（不使用模糊词）

---

### 示例1：最小间距（同层）

**原始规则**：`M1` 任意两处图形间距不得小于 `s_min`。

**Manhattan 改写**：
- 水平主导：检查 x 方向相邻关系（侧墙对侧墙）。
- 垂直主导：检查 y 方向相邻关系。

**操作化表达**：
1. `A = GROW(M1, dx=s_min/2, dy=0)`
2. `B = GROW(M1, dx=0, dy=s_min/2)`
3. `Vx = INTERACT(A, A) - SELF_PAIR`
4. `Vy = INTERACT(B, B) - SELF_PAIR`
5. `Violation = OR(Vx, Vy)`

说明：通过轴向半扩张将最小间距判定改写为“扩张后相交”。

---

### 示例2：最小线宽

**原始规则**：`M1` 线宽不得小于 `w_min`。

**Manhattan 改写**：
- 对水平线段，宽度是 y 向厚度；
- 对垂直线段，宽度是 x 向厚度。

**操作化表达**：
1. `H = EDGE(M1, H)`
2. `V = EDGE(M1, V)`
3. `Mh = FILTER(M1, supported_by=H)`
4. `Mv = FILTER(M1, supported_by=V)`
5. `Bad_h = NOT(SHRINK(Mh, dx=0, dy=w_min/2))`  (退化表示厚度不足)
6. `Bad_v = NOT(SHRINK(Mv, dx=w_min/2, dy=0))`
7. `Violation = OR(Bad_h, Bad_v)`

---

### 示例3：线端间距（tip-to-tip）

**原始规则**：相对朝向的线端之间距离必须 ≥ `t_min`。

**Manhattan 改写**：
- 水平端点对：沿 x 轴对向；
- 垂直端点对：沿 y 轴对向。

**操作化表达**：
1. `Ends = FILTER(EDGE(M1), type=LINE_END)`
2. `Ends_H = FILTER(Ends, axis=X)`
3. `Ends_V = FILTER(Ends, axis=Y)`
4. `Ex = GROW(Ends_H, dx=t_min/2, dy=0)`
5. `Ey = GROW(Ends_V, dx=0, dy=t_min/2)`
6. `Vx = INTERACT(Ex, Ex) - SELF_PAIR`
7. `Vy = INTERACT(Ey, Ey) - SELF_PAIR`
8. `Violation = OR(Vx, Vy)`

---

### 示例4：平行长边邻近风险

**原始规则**：若两条平行边长度均 > `L0` 且间距 < `s0`，标记热点。

**Manhattan 改写**：
- 长水平边对（y 向间距）；
- 长垂直边对（x 向间距）。

**操作化表达**：
1. `Long_H = FILTER(EDGE(M1,H), length > L0)`
2. `Long_V = FILTER(EDGE(M1,V), length > L0)`
3. `Hx = GROW(Long_H, dx=0, dy=s0/2)`
4. `Vx = GROW(Long_V, dx=s0/2, dy=0)`
5. `Risk_H = INTERACT(Hx, Hx) - SELF_PAIR`
6. `Risk_V = INTERACT(Vx, Vx) - SELF_PAIR`
7. `Hotspot = OR(Risk_H, Risk_V)`

---

### 示例5：桥连风险（窄缝 + 对向凸出）

**原始规则**：若两个图形在局部形成窄缝，且两侧均存在对向凸出，则存在桥连风险。

**Manhattan 改写**：
- 先按 H/V 检测窄缝候选；
- 再检测正交方向是否存在对向 edge-end 支持。

**操作化表达**：
1. `Gap_H = INTERACT(GROW(M1, dx=g0/2, dy=0), GROW(M1, dx=g0/2, dy=0)) - SELF_PAIR`
2. `Gap_V = INTERACT(GROW(M1, dx=0, dy=g0/2), GROW(M1, dx=0, dy=g0/2)) - SELF_PAIR`
3. `Prot_H = FILTER(EDGE(M1,H), protrusion >= p0)`
4. `Prot_V = FILTER(EDGE(M1,V), protrusion >= p0)`
5. `Risk1 = AND(Gap_H, INTERACT(Gap_H, Prot_V))`
6. `Risk2 = AND(Gap_V, INTERACT(Gap_V, Prot_H))`
7. `BridgeRisk = OR(Risk1, Risk2)`

---

### 示例6：包围（enclosure）不足

**原始规则**：`VIA` 必须被 `M1` 在四向包围，包围量 ≥ `e0`。

**Manhattan 改写**：
四向包围拆成 x/y 双轴要求：
- x 向至少 `e0`
- y 向至少 `e0`

**操作化表达**：
1. `Need = GROW(VIA, dx=e0, dy=e0)`
2. `Miss = NOT(AND(Need, M1))` （Need 中未被 M1 覆盖的部分）
3. `Violation = INTERACT(Miss, VIA)`

## 1.5 与 EDA API、Tiling、并行执行的兼容性

### 1.5.1 API 兼容性映射
Manhattan 重写后，规则主要依赖下列高稳定 API 模式：
- 方向提取：`EDGE(..., orientation)`
- 尺寸变换：`GROW/SHRINK(dx, dy)`
- 布尔组合：`AND/OR/NOT`
- 拓扑关系：`INTERACT`

这些模式在主流 DRC/版图引擎中属于高频路径，通常具备成熟优化。

### 1.5.2 Tiling 一致性
对于阈值 `d` 的规则，tile halo 可统一设为 `≥ d`（根据操作链取最大扩张半径）。
Manhattan 下 halo 计算直接取 `max(dx,dy)` 组合，不需处理任意角投影误差。

### 1.5.3 并行分发
按 tile 并行后，各 tile 的结果通过几何并集合并，再做一次边界去重。由于操作均为闭包原语，结果可重复、可追溯。

## 1.6 Manhattan 策略的限制与补救

### 1.6.1 限制：非 Manhattan 图形
45°、曲线、SRAF 曲边会落入 `N` 类边，直接按 H/V 处理会引入近似误差。

### 1.6.2 补救策略
1. **分层处理**：`M = M_HV ∪ M_N`，先精确处理 `M_HV`。
2. **局部栅格化/分段化**：将 `M_N` 细分为短段后再投影到近似 H/V 规则。
3. **误差预算**：给定 `ε_geo`，在最终阈值中保守收紧（例如 `d_eff = d - ε_geo`）。
4. **二阶段验证**：初筛用 Manhattan，复筛用高精度核，仅在候选热点区域运行。

## 1.7 本章小结（可执行结论）

1. Manhattan 重写不是“风格偏好”，而是将复杂几何问题映射到**稳定可执行原语**的必要步骤。
2. 方向分解（H/V）是规则工程化的第一层标准化。
3. 任何规则必须能写成“方向分支 + 布尔回收”的结构。
4. 非 Manhattan 图形应进入“近似 + 复核”双阶段流程，避免全局高开销。

---

## 下一章预告（待你确认后继续）
第2章将完成“Distance → Sizing Transformation”的严格推导：
- 证明“最小距离约束”与“对称/非对称 grow 后相交判定”的等价条件；
- 解释何时必须使用非对称扩张（方向性规则、遮罩偏置、工艺窗口）；
- 给出 6+ 个详细示例（spacing / tip-to-tip / bridge / enclosure 等）。

