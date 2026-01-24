# Quick Start Guide

## For First-Time Setup on Nebius

### 1. Connect to Nebius
```bash
ssh root@login.slurm-XXXXX.backbone-XXXXX.msp.eu-north1.nebius.cloud -i ~/.ssh/id_ed25519
```

### 2. Clone Repository
```bash
mkdir -p /shared
cd /shared
git clone https://github.com/smilenaderi/llm_fine_tune.git llm-fine-tune
cd llm-fine-tune
```

### 3. Run Setup Script
```bash
bash scripts/setup_nebius.sh
```

The setup script will prompt you for Hugging Face authentication. This is **optional** - only needed for gated models. See [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md) for details.

---

## For Training (After Setup)

### 1. Choose Your Configuration

Edit `config.yaml` and select a preset:

#### Fast Demo (15-30 min) ⭐ RECOMMENDED FOR POC
```yaml
model:
  model_id: "Qwen/Qwen2.5-7B-Instruct"
dataset:
  max_samples: 20000
training:
  num_train_epochs: 1
```

#### Quick Test (5-15 min)
```yaml
model:
  model_id: "Qwen/Qwen2.5-1.5B-Instruct"
dataset:
  max_samples: 10000
training:
  num_train_epochs: 1
  per_device_train_batch_size: 16
lora:
  r: 8
```

#### Production (2-4 hours)
```yaml
model:
  model_id: "Qwen/Qwen2.5-7B-Instruct"
dataset:
  max_samples: 60000
training:
  num_train_epochs: 3
lora:
  r: 32
```

### 2. Prepare Data

```bash
source envs/llm-env/bin/activate
python scripts/prepare_data.py
```

### 3. Train

```bash
sbatch scripts/submit_job.sh
```

---

## Monitor Training

```bash
# Load helpers
source scripts/monitor.sh

# Watch logs
watch_latest

# Monitor GPUs
watch_gpu

# Monitor CPU/memory
watch_htop

# Check status
job_status
```

### View Training Metrics (TensorBoard)

**On the Nebius login node:**
```bash
source scripts/monitor.sh

# Compare all training runs (recommended)
start_tensorboard

# Or view specific job only
start_tensorboard_job 12345
```

**Access from your local machine:**

Open a new terminal on your local machine and create an SSH tunnel:
```bash
# Replace XXXXX with your cluster ID
ssh -L 6006:localhost:6006 root@login.slurm-XXXXX.backbone-XXXXX.msp.eu-north1.nebius.cloud -i ~/.ssh/id_ed25519
```

Then open your browser to: **http://localhost:6006**

**In TensorBoard Web UI:**
- Left sidebar shows all jobs (job_12345, job_12346, etc.)
- Check/uncheck boxes to compare specific runs
- Hover over graphs to see exact values
- Use smoothing slider to reduce noise

**Metrics available:**
- **Training:** loss, learning rate, progress
- **Performance:** tokens/sec, steps/sec, elapsed time
- **Per-GPU:** utilization %, memory GB, temperature °C, power W
- **Aggregate:** avg utilization, total memory, avg temp, total power

**Log Organization:**
```
logs/
├── job_12345/
│   ├── config.yaml            # Snapshot of config used for this job ⭐
│   ├── slurm.out              # SLURM stdout
│   ├── slurm.err              # SLURM stderr  
│   ├── tensorboard/           # TensorBoard events
│   ├── validation_results.json # Validation results
│   └── benchmark_results.json  # Performance metrics
└── job_12346/
    └── ...

checkpoints/
├── job_12345/
│   ├── checkpoint-100/
│   └── final_adapter/
└── job_12346/
    └── ...
```

**Important:** Each job saves a snapshot of `config.yaml` at training start. This ensures validation always uses the exact same model and settings that were used during training, even if you modify the main `config.yaml` later. This prevents model mismatch errors when loading adapters.

---

## View Results

```bash
# Performance metrics
cat logs/benchmark_results.json

# Validation results
cat logs/validation_results.json

# Test inference
python scripts/inference.py
```

---

## Model Options

| Model | Size | Time | Quality | Best For |
|-------|------|------|---------|----------|
| Qwen 1.5B | 1.5B | Fast | Good | Quick tests |
| Qwen 7B ⭐ | 7B | Medium | Excellent | PoC demos |
| Qwen 14B | 14B | Slow | Excellent | Production |
| Llama 8B | 8B | Medium | Excellent | Production |

See [MODEL_ALTERNATIVES.md](MODEL_ALTERNATIVES.md) for full list.

---

## Dataset Options

| Dataset | Size | Domain | Best For |
|---------|------|--------|----------|
| xLAM 60k ⭐ | 60k | Function Calling | PoC |
| Glaive v2 | 113k | Function Calling | Production |
| Hermes v1 | 115k | Function Calling | Research |

---

## Common Commands

```bash
# View config summary
python scripts/config_loader.py

# Check storage
df -h /shared /mnt/network-disk

# Clean checkpoints
source scripts/monitor.sh && clean_checkpoints

# Start TensorBoard
source scripts/monitor.sh && start_tensorboard

# Stop TensorBoard
pkill -f tensorboard

# Cancel job
scancel <JOB_ID>

# Check job queue
squeue --me

# View logs
ls -lh logs/
```

---

## Troubleshooting

**Out of memory?**
```yaml
training:
  per_device_train_batch_size: 4  # Reduce from 8
lora:
  r: 8  # Reduce from 16
```

**Training too slow?**
```yaml
dataset:
  max_samples: 10000  # Reduce from 20000
model:
  model_id: "Qwen/Qwen2.5-1.5B-Instruct"  # Use smaller model
```

**Need better quality?**
```yaml
training:
  num_train_epochs: 3  # Increase from 1
dataset:
  max_samples: 60000  # Use full dataset
lora:
  r: 32  # Increase from 16
```

**Environment not activated?**
```bash
source envs/llm-env/bin/activate
```

**Hugging Face authentication needed?**
```bash
huggingface-cli login
```
See [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md) for details. Only required for gated models.

**Check GPU availability:**
```bash
sinfo -N -l
```

---

## Documentation

- **README.md** - Complete guide
- **MODEL_ALTERNATIVES.md** - Model and dataset options
- **SPLITS_EXPLAINED.md** - Understanding dataset splits
- **STORAGE.md** - Storage configuration
- **COMMANDS.md** - Command reference
- **config.yaml** - Configuration file

---

## Support

1. Check logs: `tail -f logs/train_*.err`
2. View config: `python scripts/config_loader.py`
3. Check storage: `df -h`
4. Review [README.md](README.md)

---

**Ready to start?** Just run:
```bash
nano config.yaml          # Choose your preset
python scripts/prepare_data.py
sbatch scripts/submit_job.sh
```
