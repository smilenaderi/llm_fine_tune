# Quick Start Guide - Nebius Deployment

## 1. Connect to Nebius
```bash
ssh root@login.slurm-XXXXX.backbone-XXXXX.msp.eu-north1.nebius.cloud -i ~/.ssh/id_ed25519
```

## 2. Clone Repository
```bash
cd /shared
git clone https://github.com/smilenaderi/llm_fine_tune.git llm-fine-tune
cd llm-fine-tune
```

## 3. Run Setup Script
```bash
bash scripts/setup_nebius.sh
```

The setup script will prompt you for Hugging Face authentication. This is **optional** - only needed for gated models. See [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md) for details.

## 4. Prepare Data
```bash
source envs/llm-env/bin/activate
python scripts/prepare_data.py
```

## 5. Submit Training Job
```bash
sbatch scripts/submit_job.sh
```

## 6. Monitor Training

Load monitoring helpers:
```bash
source scripts/monitor.sh
```

Available commands:
```bash
watch_latest    # Watch logs
watch_gpu       # Monitor GPU usage
job_status      # Check job status
```

### Monitoring Function Definitions

If you prefer to use these without sourcing the script:

```bash
# Watch the latest error log
watch_latest() {
    latest_file=$(ls -t logs/*.err | head -n 1)
    echo "Tailing: $latest_file"
    tail -f "$latest_file"
}

# Monitor GPU usage for running job
watch_gpu() {
    JOB_ID=$(squeue --me --states=RUNNING -h -o %A | head -n 1)
    if [ -z "$JOB_ID" ]; then
        echo "❌ No running jobs found! Submit one first."
        return
    fi
    echo "👀 Found Job $JOB_ID. Launching nvitop..."
    srun --jobid=$JOB_ID --overlap --pty bash -c "source envs/llm-env/bin/activate && nvitop"
}

# Check job status
job_status() {
    squeue --me
}
```

## 7. Run Inference

After training completes, submit inference as a job:
```bash
sbatch scripts/run_inf.sh
```

Or run directly (if you have GPU access on login node):
```bash
source envs/llm-env/bin/activate
python scripts/inference.py
```

## Useful Commands

```bash
# Check job queue
squeue --me

# Cancel job
scancel JOB_ID

# View logs
ls -lh logs/

# Clean checkpoints
source scripts/monitor.sh
clean_checkpoints
```

## Troubleshooting

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

For detailed documentation, see [README.md](README.md)
