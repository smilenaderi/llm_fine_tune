# Quick Command Reference

## Setup Commands

```bash
# Clone repository
cd /shared
git clone https://github.com/smilenaderi/llm_fine_tune.git llm-fine-tune
cd llm-fine-tune

# Run automated setup
bash scripts/setup_nebius.sh

# Activate environment
source envs/llm-env/bin/activate
```

## Configuration Commands

```bash
# View configuration summary
python scripts/config_loader.py

# Edit configuration
nano config.yaml

# Validate configuration
python -c "from scripts.config_loader import load_config; load_config().print_summary()"
```


## Training Commands

```bash
# Submit training job
sbatch scripts/submit_job.sh

# Check job status
squeue --me

# Cancel job
scancel <JOB_ID>

# View job details
scontrol show job <JOB_ID>
```

## Monitoring Commands

```bash
# Load monitoring helpers
source scripts/monitor.sh

# Watch latest logs
watch_latest

# Monitor GPU usage
watch_gpu

# Check job status
job_status

# View specific log
tail -f logs/train_<JOB_ID>.out
tail -f logs/train_<JOB_ID>.err
```

## Results Commands

```bash
# View benchmark results
cat logs/benchmark_results.json
python -m json.tool logs/benchmark_results.json

# View validation results
cat logs/validation_results.json
python -m json.tool logs/validation_results.json

# List all checkpoints
ls -lh checkpoints/

# Check checkpoint size
du -sh checkpoints/*
```

## Inference Commands

```bash
# Run inference (interactive)
python scripts/inference.py

# Submit inference job
sbatch scripts/run_inf.sh

# Run validation
python scripts/validate_model.py
```

## Storage Commands

```bash
# Check disk space
df -h /shared
df -h /mnt/network-disk

# Check directory sizes
du -sh checkpoints/
du -sh logs/
du -sh /mnt/network-disk/model_cache/

# Clean old checkpoints
source scripts/monitor.sh
clean_checkpoints

# Manual cleanup
rm -rf checkpoints/checkpoint-*
```

## Debugging Commands

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Check GPU status
nvidia-smi
watch -n 1 nvidia-smi

# Check Python packages
pip list | grep -E "torch|transformers|peft|trl"

# Test imports
python -c "from scripts.config_loader import load_config; print('OK')"
python -c "import transformers, peft, trl; print('OK')"

# Check SLURM configuration
sinfo
sinfo -N -l
scontrol show partition
```

## Log Analysis Commands

```bash
# Find errors in logs
grep -i error logs/*.err
grep -i "failed\|exception" logs/*.err

# Check training progress
grep "loss" logs/train_*.out | tail -20

# Count completed steps
grep -c "%" logs/train_*.out

# View last 50 lines
tail -50 logs/train_*.out
```

## Performance Analysis

```bash
# Extract training time
jq '.total_training_time_seconds' logs/benchmark_results.json

# Extract throughput
jq '.tokens_per_second' logs/benchmark_results.json

# Extract GPU memory
jq '.max_gpu_memory_gb' logs/benchmark_results.json

# View all metrics
jq '.' logs/benchmark_results.json
```

## Backup Commands

```bash
# Backup final model
cp -r checkpoints/final_adapter /mnt/network-disk/backups/final_adapter_$(date +%Y%m%d)

# Backup configuration
cp config.yaml /mnt/network-disk/backups/config_$(date +%Y%m%d).yaml

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
mv logs_backup_*.tar.gz /mnt/network-disk/backups/
```

## Cleanup Commands

```bash
# Remove old logs (>7 days)
find logs/ -name "*.out" -mtime +7 -delete
find logs/ -name "*.err" -mtime +7 -delete

# Remove all checkpoints except final
cd checkpoints
ls | grep -v "final_adapter" | xargs rm -rf

# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/*

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```


## Troubleshooting Commands

```bash
# Check if environment is activated
echo $VIRTUAL_ENV

# Reactivate environment
source envs/llm-env/bin/activate

# Check Python version
python --version

# Check CUDA version
nvcc --version

# Check storage mounts
mount | grep -E "shared|network"

# Check file permissions
ls -la checkpoints/
ls -la logs/

# Test SLURM
srun --nodes=1 --gpus=1 --pty bash
```

## Quick Workflows

### Fast PoC Test (15-30 min)
```bash
# 1. Configure
nano config.yaml  # Set max_samples: 10000, epochs: 1

# 2. Prepare
python scripts/prepare_data.py

# 3. Train
sbatch scripts/submit_job.sh

# 4. Monitor
source scripts/monitor.sh && watch_latest
```

### Full Production Run
```bash
# 1. Configure
nano config.yaml  # Set max_samples: 60000, epochs: 3

# 2. Prepare
python scripts/prepare_data.py

# 3. Train
sbatch scripts/submit_job.sh

# 4. Monitor
watch -n 60 'squeue --me && tail -20 logs/train_*.out'

# 5. Results
cat logs/benchmark_results.json
cat logs/validation_results.json
```

### Hyperparameter Sweep
```bash
# Create configs
for lr in 1e-4 2e-4 5e-4; do
  cp config.yaml config_lr${lr}.yaml
  sed -i "s/learning_rate: .*/learning_rate: ${lr}/" config_lr${lr}.yaml
done

# Run experiments
for config in config_lr*.yaml; do
  python scripts/fine_tune.py --config $config
done

# Compare results
for result in logs/benchmark_results_*.json; do
  echo "=== $result ==="
  jq '.final_loss, .tokens_per_second' $result
done
```

## Useful Aliases

Add to `~/.bashrc`:

```bash
# LLM Fine-tuning aliases
alias llm='cd /shared/llm-fine-tune && source envs/llm-env/bin/activate'
alias llm-config='nano /shared/llm-fine-tune/config.yaml'
alias llm-train='cd /shared/llm-fine-tune && sbatch scripts/submit_job.sh'
alias llm-status='squeue --me'
alias llm-logs='cd /shared/llm-fine-tune && source scripts/monitor.sh && watch_latest'
alias llm-gpu='cd /shared/llm-fine-tune && source scripts/monitor.sh && watch_gpu'
alias llm-results='cat /shared/llm-fine-tune/logs/benchmark_results.json | python -m json.tool'
```

Then use:
```bash
llm              # Go to project and activate env
llm-config       # Edit configuration
llm-train        # Submit training job
llm-status       # Check job status
llm-logs         # Watch logs
llm-gpu          # Monitor GPUs
llm-results      # View results
```

---

**Tip**: Bookmark this file for quick reference during demos and development!
