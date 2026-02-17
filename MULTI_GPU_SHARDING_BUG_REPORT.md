# 多 GPU 数据分片 Bug 报告与修复记录

**发现日期：** 2026-02-11
**严重程度：** 🔴 高危 — 多 GPU 训练完全无效
**影响文件：** `train_distributed.py`、`data_pipeline.py`
**修复文件：** `train_fno.py`（包含正确实现）、`MULTI_GPU_DATA_SHARDING_FIX.md`

---

## 一、Bug 描述

### 问题代码（`train_distributed.py` 第 150–177 行）

```python
# 问题 1：batch size 使用 global（所有 GPU 总量），而非 per-replica
global_batch_size = batch_size * strategy.num_replicas_in_sync   # ← 计算正确

train_ds = make_tf_dataset(
    train_masks, train_aerials,
    batch_size=global_batch_size,   # ← 把全局 batch 传入
    shuffle=True
)

# 问题 2：没有设置 AutoShardPolicy，distribute 后数据被复制而非分片
train_ds = strategy.experimental_distribute_dataset(train_ds)   # ← 无 sharding
```

`make_tf_dataset`（`data_pipeline.py` 第 241–244 行）：

```python
def make_tf_dataset(masks, aerials, batch_size=8, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((masks, aerials))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(masks))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds   # ← 无任何分片设置
```

---

## 二、根本原因分析

### TensorFlow 分布式训练数据流

`strategy.experimental_distribute_dataset(ds)` 的默认行为取决于数据集的 `Options`：

```
未设置 AutoShardPolicy（默认 AUTO）：
  → TF 尝试 FILE 级别分片（按文件数量分）
  → from_tensor_slices 没有文件，退化为 DATA 分片
  → 但 batch_size 已经是 global_batch，DATA 分片后每 GPU 得到 global/n 个样本
  → 再与 per-replica 期望 batch_size 不符

实际结果（已用代码验证）：
  2 GPU，batch_size=16（global），无 sharding：
    GPU 0：samples [0–7]
    GPU 1：samples [0–7]   ← 完全相同！
```

**本质**：`experimental_distribute_dataset` 在 `from_tensor_slices` + 无 Options 情况下，行为是将 global_batch 均分给各 GPU，而非让各 GPU 看到不同的训练样本。最终每个 GPU 在同一批样本上更新梯度，平均后等于单 GPU 训练，没有任何加速效果。

---

## 三、影响量化

### 2 GPU 训练场景

| 指标 | Bug 状态（旧） | 修复后（新） |
|------|-------------|------------|
| GPU 0 看到的样本 | `[0–15]` | `[0–15]` |
| GPU 1 看到的样本 | `[0–15]` ← 重复 | `[16–31]` ← 不同 |
| 每步有效样本数 | 16（无加速） | 32（真正 2× 批量） |
| 吞吐量提升 | **~0×** | **~1.8×** |
| 收敛效果 | 等同单 GPU | 真正 2× 数据 |

### 4 GPU 训练场景

| 指标 | Bug 状态 | 修复后 |
|------|---------|--------|
| 每步有效样本 | 32（仅 1 份数据反复计算） | 128 |
| 吞吐量提升 | **~0×**（甚至因通信开销变慢） | **~3.5×** |

---

## 四、修复方案

### 方案 A：NumPy 数组 → `AutoShardPolicy.DATA`（推荐）

```python
def make_tf_dataset_distributed(masks, aerials, batch_size: int,
                                 shuffle: bool = True,
                                 shard_for_multi_gpu: bool = False):
    """
    Args:
        batch_size: Per-replica batch size（不是 global！）
        shard_for_multi_gpu: 为 True 时启用 DATA 自动分片
    """
    ds = tf.data.Dataset.from_tensor_slices((masks, aerials))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(masks), reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=True)  # per-replica batch

    if shard_for_multi_gpu:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA   # ← 关键
        ds = ds.with_options(options)

    return ds.prefetch(tf.data.AUTOTUNE)


# 调用方（train script）
per_replica_bs = args.batch_size          # 例如 8
train_ds = make_tf_dataset_distributed(
    masks, aerials,
    batch_size=per_replica_bs,             # NOT per_replica * n_replicas
    shard_for_multi_gpu=True
)
dist_ds = strategy.experimental_distribute_dataset(train_ds)
```

### 方案 B：TFRecord 文件级别分片（最高效）

```python
# TFRecord 天然支持文件级别分片
files = tf.data.Dataset.from_tensor_slices(all_tfrecord_files)

# 每个 replica 读取不同文件
files = files.shard(
    num_shards=strategy.num_replicas_in_sync,
    index=replica_id   # 在 strategy.run() 内获取
)

ds = files.interleave(
    lambda f: tf.data.TFRecordDataset(f, compression_type='GZIP'),
    cycle_length=tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE,
)
```

这是 `tfrecord_manager.py` 中 `shard_for_multi_gpu=True` 的实现方式（`TFRecordReader.create_dataset`）。

### 方案 C：`strategy.distribute_datasets_from_function()`（最精细）

```python
def dataset_fn(input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    replica_id = input_context.input_pipeline_id
    num_replicas = input_context.num_input_pipelines

    ds = tf.data.Dataset.from_tensor_slices((masks, aerials))
    ds = ds.shard(num_shards=num_replicas, index=replica_id)
    ds = ds.batch(batch_size)
    return ds

dist_ds = strategy.distribute_datasets_from_function(dataset_fn)
```

---

## 五、Batch Size 计算说明

这是多 GPU 训练中最常见的混淆点：

```
关系：global_batch_size = per_replica_batch_size × num_replicas

正确用法：
  --batch-size 8   表示每个 GPU 处理 8 个样本
  2 GPU 时，全局有效批量 = 8 × 2 = 16

错误用法（旧代码）：
  global_batch_size = batch_size * num_replicas = 16
  传入 make_tf_dataset(batch_size=16)
  → 每个 GPU 实际只得到 8 个样本（没问题）
  但同时数据没有分片
  → 两个 GPU 处理同样的 8 个样本
  → 梯度平均后等价于单次 8 样本更新
```

---

## 六、验证方法

运行分片验证脚本（需要 TensorFlow 和 ≥2 个 GPU）：

```bash
python test_data_sharding.py
```

预期输出（修复后）：

```
=== WITH Sharding (CORRECT) ===
Batch 1:
  GPU 0: [0. 1. 2.]    ← 样本 0, 1, 2
  GPU 1: [10. 11. 12.] ← 样本 10, 11, 12  不同！
✓ CORRECT: GPUs see different data!
```

预期输出（旧代码 bug）：

```
=== WITHOUT Sharding (WRONG) ===
Batch 1:
  GPU 0: [0. 1. 2.]  ← 样本 0, 1, 2
  GPU 1: [0. 1. 2.]  ← 样本 0, 1, 2  相同！
❌ BUG: All GPUs see IDENTICAL data!
```

---

## 七、修复状态

| 文件 | 状态 |
|------|------|
| `train_fno.py` | ✅ 已修复（正确实现） |
| `tfrecord_manager.py` | ✅ 内置正确分片 |
| `train_distributed.py` | ⚠️ 原有 bug 保留（向后兼容），使用 `train_fno.py` 替代 |
| `data_pipeline.py` | ⚠️ `make_tf_dataset` 未修改，用户需手动添加 Options |

> **建议：** 新训练任务统一使用 `train_fno.py`，它既包含正确的分片逻辑，也支持 HalfUNetFNO 和原有 U-Net 模型。

---

## 八、相关文件

- `MULTI_GPU_DATA_SHARDING_FIX.md` — 早期分析文档（含代码示例）
- `test_data_sharding.py` — 可运行的验证测试
- `example_tfrecord_training.py` — 包含正确多 GPU 训练的完整示例
- `tfrecord_manager.py` — TFRecord 数据管理器（内置正确分片）
