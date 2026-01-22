# LLM Fine-Tuning with Qwen2.5-7B

Fine-tune Qwen2.5-7B-Instruct on function-calling data using LoRA adapters and distributed training on Nebius AI Cloud.

## Features

- ✅ **Configuration-driven**: Single YAML file for all settings
- ✅ **LoRA-based fine-tuning**: Memory-efficient training
- ✅ **Multi-GPU support**: FSDP for distributed training
- ✅ **Flash Attention 2**: Optimized for H100/H200 GPUs
- ✅ **Automatic validation**: Test function-calling capabilities
- ✅ **Performance benchmarking**: Track throughput and GPU utilization
- ✅ **Error handling**: Robust error recovery and logging
- ✅ **Storage optimization**: Efficient use of 2TB SSD resources
- ✅ **Real dataset**: xLAM function-calling dataset (60k samples)

## Quick Start (3 Steps)

### 1. Configure Your Training

Edit `config.yaml` to customize your setup:

```yaml
cluster:
  nodes: 1              # Number of nodes
  gpus_per_node: 4      # GPUs per node (PoC: 4x H200)

dataset:
  max_samples: 20000    # 20k for fast PoC, 60000 for full dataset

training:
  num_train_epochs: 1
  learning_rate: 2.0e-4
  per_device_train_batch_size: 8

lora:
  r: 16                 # LoRA rank (8/16/32/64)
  lora_alpha: 32
```

See [Configuration Guide](#configuration) for all options.

### 2. Prepare Data

```bash
source envs/llm-env/bin/activate
python scripts/prepare_data.py
```

### 3. Submit Training Job

```bash
sbatch scripts/submit_job.sh
```

Monitor training:
```bash
source scripts/monitor.sh
watch_latest    # View logs
watch_gpu       # Monitor GPU usage
```

---

## Configuration

All training parameters are configured in `config.yaml`. No need to edit Python scripts!

### Key Configuration Sections

#### Cluster Resources
```yaml
cluster:
  nodes: 1                    # Number of compute nodes
  gpus_per_node: 4           # GPUs per node
```

#### Storage Paths
```yaml
storage:
  shared_fs: "/shared/llm-fine-tune"      # Shared filesystem (2TB)
  network_disk: "/mnt/network-disk"       # Network disk (2TB)
```

See [STORAGE.md](STORAGE.md) for detailed storage configuration.

#### Dataset Configuration
```yaml
dataset:
  name: "Beryex/xlam-function-calling-60k-sharegpt"
  max_samples: 20000          # null = use all 60k samples
  streaming: true
```

#### Training Hyperparameters
```yaml
training:
  num_train_epochs: 1
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  max_seq_length: 2048
```

#### LoRA Configuration
```yaml
lora:
  r: 16                      # LoRA rank
  lora_alpha: 32             # Scaling factor
  target_modules:            # Modules to apply LoRA
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
```

#### Distributed Training
```yaml
distributed:
  strategy: "fsdp"           # "fsdp" or "ddp"
  fsdp:
    enabled: true
    sharding_strategy: "full_shard"
```

### Hyperparameter Tuning Guide

**Learning Rate:**
- `2e-4`: Recommended starting point
- `5e-4`: Faster convergence (may be unstable)
- `1e-4`: More stable, slower convergence

**LoRA Rank (r):**
- `r=8`: Fast, low memory, simple tasks
- `r=16`: Balanced (recommended)
- `r=32`: Better performance, more memory
- `r=64`: Best performance, highest memory

**Batch Size:**
- Effective batch size = `per_device_batch_size × gradient_accumulation × num_gpus`
- Recommended: 32-128 for most tasks
- Increase `per_device_batch_size` if you have GPU memory
- Adjust `gradient_accumulation_steps` to maintain effective batch size

**Epochs:**
- `1 epoch`: Quick PoC (~15-30 min with 20k samples)
- `2-3 epochs`: Recommended for production
- `5+ epochs`: Risk of overfitting

**Dataset Size:**
- `20k samples`: Fast PoC testing (~15-30 min)
- `60k samples`: Full dataset (~45-90 min)

See `config.yaml` for detailed hyperparameter tuning guide.

---

## Deployment on Nebius AI Cloud

### 1. Create SLURM Cluster

1. Go to [Nebius Console](https://console.nebius.ai/)
2. Navigate to **Compute** → **SLURM Operators**
3. Click **Create SLURM Operator**
4. Configure your cluster:
   - **Nodes**: 1+ compute nodes
   - **GPUs**: 4x H200 per node (PoC allocation)
   - **Storage**: 2TB shared filesystem + 2TB network disk
5. Wait for cluster provisioning to complete

### 2. Setup SSH Access

Generate SSH key (if you don't have one):
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Copy your public key:
```bash
pbcopy < ~/.ssh/id_ed25519.pub
# On Linux: xclip -sel clip < ~/.ssh/id_ed25519.pub
```

Add the public key to your Nebius SLURM operator in the console.

### 3. Connect to Login Node

Get the SSH connection command from Nebius Console, then connect:
```bash
ssh root@login.slurm-XXXXX.backbone-XXXXX.msp.eu-north1.nebius.cloud -i ~/.ssh/id_ed25519
```

### 4. Upload Project Files

**Option A: Clone from GitHub (Recommended)**
```bash
# On the login node
cd /shared
git clone https://github.com/smilenaderi/llm_fine_tune.git llm-fine-tune
cd llm-fine-tune
```

**Option B: Upload via SCP**
```bash
# From your local machine
scp -i ~/.ssh/id_ed25519 -r . root@login.slurm-XXXXX.backbone-XXXXX.msp.eu-north1.nebius.cloud:/shared/llm-fine-tune/
```

### 5. Setup Environment

**Automated Setup (Recommended):**
```bash
cd /shared/llm-fine-tune
bash scripts/setup_nebius.sh
```

This script will:
- Create directory structure
- Setup Python virtual environment
- Install PyTorch with CUDA 12.1
- Install Flash Attention 2 for H100/H200 optimization
- Install all ML dependencies (transformers, peft, trl, etc.)
- Install monitoring tools (nvitop)
- Prompt for Hugging Face authentication (optional)

See [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md) for authentication details.

### 6. Configure Storage

Verify storage mounts:
```bash
df -h | grep -E "shared|network"
```

Expected output:
```
/dev/sda1       2.0T  100G  1.9T   5% /shared
nfs-server:/    2.0T   50G  1.9T   3% /mnt/network-disk
```

See [STORAGE.md](STORAGE.md) for detailed storage configuration.

### 7. Customize Configuration

Edit `config.yaml` to match your PoC resources:

```bash
nano config.yaml
```

Key settings to verify:
- `cluster.nodes`: 1 (for 4 GPU PoC)
- `cluster.gpus_per_node`: 4
- `dataset.max_samples`: 20000 (fast) or 60000 (full)
- `training.num_train_epochs`: 1-3

### 8. Prepare Data

```bash
source envs/llm-env/bin/activate
python scripts/prepare_data.py
```

### 9. Submit Training Job

```bash
sbatch scripts/submit_job.sh
```

### 10. Monitor Training

**Source the monitoring helpers:**
```bash
source scripts/monitor.sh
```

**Check job status:**
```bash
job_status
```

**Watch logs:**
```bash
watch_latest
```

**Monitor GPU usage:**
```bash
watch_gpu
```

### 11. View Results

After training completes, check the benchmark results:

```bash
cat logs/benchmark_results.json
```

Example output:
```json
{
  "total_training_time_seconds": 1234.56,
  "tokens_per_second": 15000,
  "max_gpu_memory_gb": 45.2,
  "final_loss": 0.7123
}
```

View validation results:
```bash
cat logs/validation_results.json
```

### 12. Run Inference

Test the fine-tuned model:

```bash
python scripts/inference.py
```

Or submit as a job:
```bash
sbatch scripts/run_inf.sh
```

---

## Performance Benchmarking

The solution automatically tracks:
- **Training time**: Total and per-step
- **Throughput**: Tokens per second
- **GPU utilization**: Memory usage
- **Model quality**: Loss and validation metrics

Results are saved to `logs/benchmark_results.json`.

### Expected Performance (4x H200)

| Dataset Size | Epochs | Training Time | Throughput |
|--------------|--------|---------------|------------|
| 20k samples  | 1      | ~15-30 min    | ~12-15k tokens/s |
| 60k samples  | 1      | ~45-90 min    | ~12-15k tokens/s |
| 60k samples  | 3      | ~2-4 hours    | ~12-15k tokens/s |

*Actual performance may vary based on configuration and hardware.*

---

## Validation

The solution includes automatic validation of function-calling capabilities:

```bash
python scripts/validate_model.py
```

Tests include:
- Flight booking
- Calendar reminders
- Restaurant search
- Weather queries
- Email composition

Validation score >60% indicates successful fine-tuning.

---

## Project Structure

```
llm-fine-tune/
├── config.yaml              # Main configuration file
├── scripts/
│   ├── config_loader.py     # Configuration loader
│   ├── fine_tune.py         # Main training script
│   ├── prepare_data.py      # Dataset preparation
│   ├── validate_model.py    # Model validation
│   ├── inference.py         # Inference testing
│   ├── submit_job.sh        # SLURM job submission
│   ├── run_inf.sh           # Inference job
│   ├── monitor.sh           # Monitoring helpers
│   └── setup_nebius.sh      # Environment setup
├── data/                    # Training data (generated)
├── checkpoints/             # Model checkpoints (generated)
├── logs/                    # Training logs (generated)
├── cache/                   # Temporary cache
└── envs/                    # Python virtual environment
```

---

## Useful Commands

**View configuration summary:**
```bash
python scripts/config_loader.py
```

**Check job queue:**
```bash
squeue --me
```

**Cancel a job:**
```bash
scancel JOB_ID
```

**View all logs:**
```bash
ls -lh logs/
```

**Check storage usage:**
```bash
du -sh checkpoints/ logs/ cache/
```

**Clean old checkpoints:**
```bash
source scripts/monitor.sh
clean_checkpoints
```

---

## Troubleshooting

### Issue: Job fails to start

**Check resource availability:**
```bash
sinfo -N -l
```

**Verify configuration:**
```bash
python scripts/config_loader.py
```

### Issue: Out of memory

**Solution 1: Reduce batch size**
```yaml
training:
  per_device_train_batch_size: 4  # Reduce from 8
```

**Solution 2: Enable CPU offload**
```yaml
distributed:
  fsdp:
    cpu_offload: true
```

**Solution 3: Reduce LoRA rank**
```yaml
lora:
  r: 8  # Reduce from 16
```

### Issue: Slow training

**Solution 1: Increase batch size**
```yaml
training:
  per_device_train_batch_size: 16  # If you have memory
```

**Solution 2: Reduce dataset size**
```yaml
dataset:
  max_samples: 10000  # For quick testing
```

**Solution 3: Check GPU utilization**
```bash
watch_gpu
```

### Issue: Low validation score

**Solution 1: Train longer**
```yaml
training:
  num_train_epochs: 3
```

**Solution 2: Use full dataset**
```yaml
dataset:
  max_samples: null  # Use all 60k samples
```

**Solution 3: Tune learning rate**
```yaml
training:
  learning_rate: 1.0e-4  # Lower for stability
```

---

## Requirements

### Hardware
- CUDA-capable GPU (H100/H200 recommended)
- Multi-GPU setup (4+ GPUs for PoC)
- Shared filesystem (2TB SSD)
- Network disk (2TB SSD)

### Software
- Python 3.8+
- CUDA 12.1+
- PyTorch 2.5+
- Transformers 4.57+
- TRL 0.27+
- Flash Attention 2 (for H100/H200)

---

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [STORAGE.md](STORAGE.md) - Storage configuration
- [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md) - HuggingFace authentication
- [config.yaml](config.yaml) - Full configuration reference

---

## Support

For issues or questions:
1. Check logs: `tail -f logs/train_*.err`
2. Review configuration: `python scripts/config_loader.py`
3. Check storage: `df -h`
4. Contact Nebius support with your cluster ID

---

## License

MIT

---

**Last Updated**: January 2026  
**Version**: 2.0
