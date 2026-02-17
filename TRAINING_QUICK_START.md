# Training Quick Start Guide

## TL;DR - Common Commands

### Simple Training (CPU/Single GPU)
```bash
# Original training script (simple, stable)
python train.py --epochs 50 --batch-size 8

# Enhanced training (more features, single GPU)
python train_distributed.py --epochs 50 --batch-size 16 --mixed-precision
```

### Multi-GPU Training
```bash
# Fast training on multiple GPUs
python train_distributed.py --multi-gpu --mixed-precision --epochs 100 --batch-size 16

# Stable multi-GPU (with NCCL workarounds)
python train_distributed.py --multi-gpu --nccl-workarounds --epochs 100
```

### Production Training
```bash
# Full featured training run
python train_distributed.py \
    --multi-gpu \
    --mixed-precision \
    --augmentation \
    --epochs 200 \
    --batch-size 16 \
    --warmup-epochs 10 \
    --gradient-clip-norm 1.0 \
    --checkpoint-every 10 \
    --run-name production_run_v1
```

## Feature Comparison

| Feature | train.py | train_distributed.py |
|---------|----------|---------------------|
| Multi-GPU | ❌ No | ✅ Yes |
| Mixed Precision | ❌ No | ✅ Yes (2-3x speedup) |
| Data Augmentation | ❌ No | ✅ Yes |
| TensorBoard | ❌ No | ✅ Yes |
| Warmup LR Schedule | ❌ No | ✅ Yes |
| Checkpoint Resume | ❌ No | ✅ Yes |
| Gradient Clipping | ❌ No | ✅ Yes |
| NCCL Workarounds | ❌ No | ✅ Yes |

## When to Use Which Script?

### Use `train.py` when:
- ✓ Quick experiments
- ✓ Limited resources (CPU only, low memory)
- ✓ Docker containers with strict thread limits
- ✓ You need stability over speed
- ✓ Simple baseline training

### Use `train_distributed.py` when:
- ✓ You have GPU(s) available
- ✓ Production training runs
- ✓ You need maximum performance
- ✓ Long training experiments
- ✓ You want advanced features (TensorBoard, augmentation, etc.)

## Common Scenarios

### Scenario 1: Quick Experiment (5 minutes)
```bash
python train.py --num-samples 50 --epochs 10 --batch-size 4
```

### Scenario 2: Full Training on Single GPU (1-2 hours)
```bash
python train_distributed.py \
    --num-samples 1000 \
    --epochs 100 \
    --batch-size 16 \
    --mixed-precision \
    --augmentation
```

### Scenario 3: Large-Scale Training on 4 GPUs (30 minutes)
```bash
python train_distributed.py \
    --multi-gpu \
    --mixed-precision \
    --augmentation \
    --num-samples 5000 \
    --epochs 200 \
    --batch-size 32 \
    --warmup-epochs 10
```

### Scenario 4: Resume Interrupted Training
```bash
# First run
python train_distributed.py --run-name my_experiment --checkpoint-every 10

# Resume later
python train_distributed.py --resume experiments/my_experiment
```

### Scenario 5: Hyperparameter Search
```bash
# Try different learning rates
for lr in 1e-4 5e-4 1e-3; do
    python train_distributed.py \
        --lr $lr \
        --run-name "lr_search_${lr}" \
        --epochs 50
done
```

## Troubleshooting

### Training is Slow
```bash
# Enable mixed precision for 2-3x speedup
python train_distributed.py --mixed-precision

# Use multi-GPU if available
python train_distributed.py --multi-gpu --mixed-precision

# Increase batch size (if memory allows)
python train_distributed.py --batch-size 32 --mixed-precision
```

### Out of Memory
```bash
# Reduce batch size
python train_distributed.py --batch-size 4

# Enable mixed precision (uses less memory)
python train_distributed.py --batch-size 8 --mixed-precision
```

### Multi-GPU Hangs
```bash
# Apply NCCL workarounds
python train_distributed.py --multi-gpu --nccl-workarounds

# Use different communication backend
python train_distributed.py --multi-gpu --communication-backend hierarchical_copy

# Check GPU status
nvidia-smi
```

### Different Results Each Run
```bash
# Set random seed
python train_distributed.py --seed 42

# Note: Some variation is normal in deep learning
```

## Monitoring Training

### View TensorBoard (Real-time)
```bash
# Start training
python train_distributed.py --epochs 100 --run-name my_run

# In another terminal, start TensorBoard
tensorboard --logdir experiments/my_run/tensorboard

# Open browser to http://localhost:6006
```

### View Training Logs
```bash
# CSV format (easy to plot)
cat experiments/my_run/training_log.csv

# JSONL format (easy to parse)
cat experiments/my_run/epoch_metrics.jsonl

# Live monitoring
tail -f experiments/my_run/training_log.csv
```

### Check GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv -l 1
```

## Performance Tips

1. **Always enable mixed precision on modern GPUs** (free 2-3x speedup)
   ```bash
   python train_distributed.py --mixed-precision
   ```

2. **Use data augmentation for better generalization**
   ```bash
   python train_distributed.py --augmentation
   ```

3. **Scale batch size with number of GPUs**
   ```bash
   # 1 GPU: batch-size 16
   # 2 GPUs: batch-size 16 (per GPU) = 32 global
   # 4 GPUs: batch-size 16 (per GPU) = 64 global
   ```

4. **Use warmup for large batch sizes**
   ```bash
   python train_distributed.py --batch-size 32 --warmup-epochs 10
   ```

5. **Enable gradient clipping for stability**
   ```bash
   python train_distributed.py --gradient-clip-norm 1.0
   ```

## Example Outputs

After training completes, you'll find in `experiments/[run_name]/`:

```
experiments/my_run/
├── run_config.json              # Configuration used
├── run_summary.json             # Training results summary
├── training_log.csv             # Epoch-by-epoch metrics (CSV)
├── epoch_metrics.jsonl          # Epoch-by-epoch metrics (JSONL)
├── training_history.png         # Loss curves plot
├── predictions.png              # Sample predictions visualization
├── litho_model_final.keras      # Final model
├── litho_model_best.keras       # Best model (by val_loss)
├── litho_dataset.npz            # Generated dataset (if not provided)
├── tensorboard/                 # TensorBoard logs
│   └── events.out.tfevents.*
└── checkpoints/                 # Periodic checkpoints
    ├── litho_model_epoch0010.keras
    ├── litho_model_epoch0020.keras
    └── ...
```

## Getting Help

- **Full documentation**: See [MULTI_GPU_TRAINING.md](MULTI_GPU_TRAINING.md)
- **NCCL issues**: See [MULTI_GPU_TRAINING.md#nccl-troubleshooting](MULTI_GPU_TRAINING.md#nccl-troubleshooting)
- **Code**: Check `train_utils.py` for implementation details
- **Test setup**: Run `python test_multi_gpu.py` to verify installation

## What's New?

### Enhanced Training Features (train_distributed.py)

1. **Multi-GPU Support** 🚀
   - Automatic GPU detection and distribution
   - Linear speedup with multiple GPUs
   - NCCL and HierarchicalCopy backends

2. **Mixed Precision Training** ⚡
   - 2-3x faster training on modern GPUs
   - ~50% memory reduction
   - Automatic loss scaling

3. **Data Augmentation** 🔄
   - Random rotations (90°, 180°, 270°)
   - Random flips (horizontal/vertical)
   - Brightness/contrast adjustments
   - Preserves shift equivariance

4. **Advanced Callbacks** 📊
   - TensorBoard integration
   - Warmup + Cosine decay LR schedule
   - Gradient clipping
   - Automatic best model saving
   - Periodic checkpoints

5. **Robustness** 🛡️
   - NCCL fallback mechanisms
   - Signal handling (Ctrl+C)
   - Checkpoint resumption
   - Comprehensive error handling

## Next Steps

1. **Verify setup**: `python test_multi_gpu.py`
2. **Quick test**: `python train_distributed.py --smoke-test`
3. **Read docs**: See [MULTI_GPU_TRAINING.md](MULTI_GPU_TRAINING.md)
4. **Start training**: Choose a command from above and run!
