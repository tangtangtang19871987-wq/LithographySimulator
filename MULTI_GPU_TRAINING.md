# Multi-GPU Training Guide

This guide covers the enhanced training capabilities for the LithographySimulator, including multi-GPU training, mixed precision, data augmentation, and NCCL troubleshooting.

## Table of Contents

1. [Quick Start](#quick-start)
2. [New Features](#new-features)
3. [Multi-GPU Training](#multi-gpu-training)
4. [NCCL Troubleshooting](#nccl-troubleshooting)
5. [Performance Optimization](#performance-optimization)
6. [Advanced Usage](#advanced-usage)
7. [Known Issues and Workarounds](#known-issues-and-workarounds)

## Quick Start

### Single GPU with All Enhancements
```bash
python train_distributed.py \
    --epochs 50 \
    --batch-size 16 \
    --mixed-precision \
    --augmentation
```

### Multi-GPU Training
```bash
python train_distributed.py \
    --multi-gpu \
    --epochs 100 \
    --batch-size 16 \
    --mixed-precision \
    --augmentation
```

### Multi-GPU with NCCL Workarounds (for stability)
```bash
python train_distributed.py \
    --multi-gpu \
    --nccl-workarounds \
    --epochs 100 \
    --batch-size 16
```

## New Features

### 1. Multi-GPU Training (`train_distributed.py`)

**Benefits:**
- Linear speedup with multiple GPUs
- Automatic data distribution across devices
- Synchronized gradient updates
- Fallback mechanisms for stability

**Implementation:**
- Uses TensorFlow's `MirroredStrategy` for data parallelism
- Supports NCCL and HierarchicalCopy communication backends
- Automatic detection and distribution across available GPUs

**Usage:**
```bash
# Enable multi-GPU
python train_distributed.py --multi-gpu

# Specify communication backend
python train_distributed.py --multi-gpu --communication-backend nccl

# Apply NCCL workarounds for Docker/container environments
python train_distributed.py --multi-gpu --nccl-workarounds
```

### 2. Mixed Precision Training

**Benefits:**
- 2-3x faster training on modern GPUs (Volta, Turing, Ampere)
- ~2x less memory usage
- Same model accuracy with proper loss scaling

**Requirements:**
- GPU with Tensor Cores (compute capability >= 7.0)
- NVIDIA Volta (V100), Turing (RTX 20xx), Ampere (A100, RTX 30xx) or newer

**Usage:**
```bash
python train_distributed.py --mixed-precision
```

### 3. Data Augmentation

**Augmentations applied:**
- Random 90° rotations (preserves shift equivariance)
- Random horizontal/vertical flips
- Small brightness/contrast adjustments

**Benefits:**
- Better generalization
- Reduces overfitting
- Increases effective dataset size

**Usage:**
```bash
python train_distributed.py --augmentation
```

### 4. TensorBoard Integration

**Features:**
- Real-time training visualization
- Loss/metric curves
- Histogram tracking
- Learning rate monitoring

**Usage:**
```bash
# TensorBoard enabled by default
python train_distributed.py --epochs 50

# View in browser
tensorboard --logdir experiments/[run_name]/tensorboard

# Disable TensorBoard
python train_distributed.py --no-tensorboard
```

### 5. Advanced Learning Rate Scheduling

**Warmup + Cosine Decay:**
- Linear warmup for stable training start
- Cosine decay for better convergence
- Prevents learning rate spikes

**Usage:**
```bash
# 5-epoch warmup with cosine decay
python train_distributed.py --warmup-epochs 5 --epochs 100 --lr 1e-3
```

### 6. Checkpoint Resumption

**Features:**
- Resume interrupted training
- Automatic checkpoint discovery
- Preserves optimizer state

**Usage:**
```bash
# Save checkpoints every 10 epochs
python train_distributed.py --checkpoint-every 10 --run-name my_experiment

# Resume from previous run
python train_distributed.py --resume experiments/my_experiment
```

## Multi-GPU Training

### How It Works

The training script uses TensorFlow's `MirroredStrategy`:

1. **Model replication**: Each GPU gets a complete copy of the model
2. **Data distribution**: Each GPU processes a different batch
3. **Gradient synchronization**: Gradients are averaged across GPUs using all-reduce
4. **Weight updates**: All GPUs update their models synchronously

### Batch Size Considerations

With multi-GPU training, the **effective batch size** is multiplied:

```python
# Example with 4 GPUs
--batch-size 16  # Per-GPU batch size
# Effective global batch size = 16 × 4 = 64
```

**Recommendations:**
- Start with same per-GPU batch size as single-GPU training
- Increase learning rate proportionally to global batch size (optional)
- Monitor validation loss to ensure convergence quality

### Communication Backends

**NCCL (NVIDIA Collective Communications Library):**
- **Fastest** for multi-GPU on same node
- Optimized for NVIDIA GPUs
- Can have stability issues in containers

**HierarchicalCopy:**
- **More stable** in problematic environments
- Slightly slower than NCCL
- Better compatibility with Docker

**Auto (default):**
- TensorFlow chooses automatically
- Usually selects NCCL on NVIDIA GPUs

**Usage:**
```bash
# Force NCCL
python train_distributed.py --multi-gpu --communication-backend nccl

# Force HierarchicalCopy for stability
python train_distributed.py --multi-gpu --communication-backend hierarchical_copy
```

## NCCL Troubleshooting

### Common Issues

#### 1. **Hangs During Initialization**

**Symptoms:**
- Training hangs before first epoch
- No error messages
- CPU usage at 100%

**Solutions:**
```bash
# Apply NCCL workarounds
python train_distributed.py --multi-gpu --nccl-workarounds

# Or manually set environment variables:
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN
python train_distributed.py --multi-gpu
```

#### 2. **Docker Container Issues**

**Symptoms:**
- "Failed to initialize NCCL"
- Hangs or crashes in Docker

**Solutions:**
```bash
# Disable InfiniBand and P2P in containers
python train_distributed.py --multi-gpu --nccl-workarounds

# Or set manually:
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0,lo
python train_distributed.py --multi-gpu
```

#### 3. **Communication Timeouts**

**Symptoms:**
- "NCCL operation timed out"
- Random hangs during training

**Solutions:**
```bash
# Increase timeout
export NCCL_TIMEOUT=3600  # 1 hour
python train_distributed.py --multi-gpu

# Or use HierarchicalCopy backend
python train_distributed.py --multi-gpu --communication-backend hierarchical_copy
```

#### 4. **GPU Visibility Issues**

**Symptoms:**
- Not all GPUs detected
- Training uses fewer GPUs than available

**Solutions:**
```bash
# Check GPU visibility
nvidia-smi

# Verify TensorFlow sees GPUs
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Manually set visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_distributed.py --multi-gpu
```

### NCCL Environment Variables Reference

The `--nccl-workarounds` flag automatically sets these:

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_TIMEOUT` | 1800 | Timeout in seconds (30 min) |
| `NCCL_BLOCKING_WAIT` | 1 | Better error messages |
| `NCCL_IB_DISABLE` | 1 | Disable InfiniBand (container safety) |
| `NCCL_P2P_DISABLE` | 1 | Disable peer-to-peer (stability) |
| `NCCL_SHM_DISABLE` | 0 | Keep shared memory enabled |
| `NCCL_SOCKET_IFNAME` | eth0,lo | Use ethernet and loopback |
| `NCCL_DEBUG` | WARN | Show warnings only |
| `NCCL_NSOCKS_PERTHREAD` | 4 | More sockets per thread |
| `NCCL_SOCKET_NTHREADS` | 2 | More threads per connection |

### Fallback Strategy

The script tries multiple approaches in order:

1. **MirroredStrategy with NCCL** (fastest)
2. **MirroredStrategy with HierarchicalCopy** (more stable)
3. **MirroredStrategy with auto** (TF chooses)
4. **Single device** (fallback if all fail)

This ensures training always proceeds, even if multi-GPU setup fails.

## Performance Optimization

### Optimal Configuration for Different Scenarios

#### **Maximum Speed (Modern GPUs, Stable Environment)**
```bash
python train_distributed.py \
    --multi-gpu \
    --mixed-precision \
    --augmentation \
    --communication-backend nccl \
    --batch-size 32 \
    --gradient-clip-norm 1.0
```

#### **Maximum Stability (Docker, Cloud, Older GPUs)**
```bash
python train_distributed.py \
    --multi-gpu \
    --nccl-workarounds \
    --communication-backend hierarchical_copy \
    --batch-size 16
```

#### **Best Accuracy (Full Training)**
```bash
python train_distributed.py \
    --multi-gpu \
    --mixed-precision \
    --augmentation \
    --epochs 200 \
    --batch-size 16 \
    --warmup-epochs 10 \
    --lr 1e-3 \
    --gradient-clip-norm 1.0 \
    --checkpoint-every 10
```

### Performance Tips

1. **Batch Size:**
   - Larger batches = faster training
   - Too large = worse generalization
   - Start with 16-32 per GPU

2. **Mixed Precision:**
   - Free 2-3x speedup on modern GPUs
   - Minimal accuracy impact
   - Always enable if available

3. **Data Pipeline:**
   - Augmentation adds ~10-20% overhead
   - Prefetch and parallel loading already optimized
   - Use SSD for dataset storage

4. **Learning Rate:**
   - Scale LR with global batch size
   - Use warmup for large batches
   - Monitor validation loss

## Advanced Usage

### Custom Training Loop with Multi-GPU

```python
from train_utils import create_distribution_strategy, DistributedStrategyConfig

# Setup strategy
config = DistributedStrategyConfig(
    strategy_type='mirrored',
    cross_device_ops='nccl',
    enable_nccl_workarounds=True
)
strategy, num_replicas = create_distribution_strategy(config)

# Build and compile model in strategy scope
with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='mse')

# Distribute datasets
train_ds = strategy.experimental_distribute_dataset(train_ds)

# Train
model.fit(train_ds, epochs=50)
```

### Monitoring Training

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir experiments/

# Check logs
tail -f experiments/[run_name]/training_log.csv
tail -f experiments/[run_name]/epoch_metrics.jsonl
```

### Hyperparameter Search

```bash
# Example: Grid search over learning rates
for lr in 1e-4 1e-3 1e-2; do
    python train_distributed.py \
        --multi-gpu \
        --mixed-precision \
        --lr $lr \
        --run-name "lr_${lr}" \
        --epochs 100
done
```

## Known Issues and Workarounds

### Issue: Out of Memory (OOM)

**Cause:** Batch size too large or mixed precision disabled

**Solutions:**
- Reduce `--batch-size`
- Enable `--mixed-precision`
- Use gradient accumulation (not implemented, but can be added)

### Issue: Training Slower on Multiple GPUs

**Cause:** Communication overhead, small batch size, or CPU bottleneck

**Solutions:**
- Increase batch size
- Enable mixed precision
- Use NCCL backend
- Ensure data is on fast storage (SSD)
- Check CPU utilization

### Issue: Different Results Between Single and Multi-GPU

**Cause:** Different effective batch size changes training dynamics

**Solutions:**
- Adjust learning rate proportionally
- Use longer warmup
- Ensure same random seed
- This is expected behavior; validate on same test set

### Issue: NCCL Hangs on First Epoch

**Cause:** NCCL initialization issues

**Solutions:**
1. Try `--nccl-workarounds`
2. Try `--communication-backend hierarchical_copy`
3. Check `nvidia-smi` for GPU health
4. Restart NCCL daemon: `sudo systemctl restart nvidia-persistenced`

## Comparison: Original vs Enhanced Training

| Feature | `train.py` | `train_distributed.py` |
|---------|------------|------------------------|
| Multi-GPU | ❌ | ✅ |
| Mixed Precision | ❌ | ✅ |
| Data Augmentation | ❌ | ✅ |
| TensorBoard | ❌ | ✅ |
| Advanced LR Schedules | ❌ | ✅ |
| Checkpoint Resume | ❌ | ✅ |
| NCCL Workarounds | ❌ | ✅ |
| Gradient Clipping | ❌ | ✅ |
| Docker-safe mode | ✅ | ✅ |

## Summary

The enhanced training system provides:

✅ **Multi-GPU support** with automatic fallbacks
✅ **NCCL stability** through comprehensive workarounds
✅ **Mixed precision** for 2-3x speedup
✅ **Data augmentation** for better generalization
✅ **TensorBoard** for visualization
✅ **Advanced scheduling** for better convergence
✅ **Checkpoint resumption** for interrupted training

Use `train_distributed.py` for production training and `train.py` for simple experiments or environments without GPU support.
