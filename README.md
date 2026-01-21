# LLM Fine-Tuning with Qwen2.5-7B

Fine-tune Qwen2.5-7B-Instruct on function-calling data using LoRA adapters and distributed training.

## Features

- LoRA-based fine-tuning for efficient training
- Multi-node distributed training support (SLURM)
- Flash Attention 2 optimization for H100/H200 GPUs
- Checkpoint management and resume capability
- Real function-calling dataset (xLAM)

## Deployment on Nebius AI Cloud

### 1. Create SLURM Cluster

1. Go to [Nebius Console](https://console.nebius.ai/)
2. Navigate to **Compute** → **SLURM Operators**
3. Click **Create SLURM Operator**
4. Configure your cluster (recommended: 2+ nodes with H100/H200 GPUs)
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
- Prompt for Hugging Face authentication (optional - only needed for gated models)

See [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md) for authentication details.

**Manual Setup:**
```bash
cd /shared/llm-fine-tune

# Create virtual environment
python3 -m venv envs/llm-env
source envs/llm-env/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install build dependencies
pip install packaging ninja wheel

# Install ML libraries
pip install transformers datasets peft accelerate trl bitsandbytes

# Install Flash Attention 2 (H100/H200 optimization)
pip install flash-attn --no-build-isolation

# Install monitoring tools
pip install nvitop

# Login to Hugging Face (OPTIONAL - only for gated models)
# See HUGGINGFACE_SETUP.md for details
huggingface-cli login
```

### 6. Prepare Data

```bash
python scripts/prepare_data.py
```

### 7. Submit Training Job

```bash
sbatch scripts/submit_job.sh
```

### 8. Monitor Training

**Source the monitoring helpers:**
```bash
source scripts/monitor.sh
```

**Check job status:**
```bash
squeue --me
# Or use helper:
job_status
```

**Watch latest error log:**
```bash
watch_latest
```

**Monitor GPU usage:**
```bash
watch_gpu
```

Or monitor specific job:
```bash
srun --jobid=YOUR_JOB_ID --overlap --pty nvitop
```

### 9. Run Inference

After training completes, submit inference as a SLURM job:
```bash
sbatch scripts/run_inf.sh
```

Or run directly (if you have GPU access on login node):
```bash
source envs/llm-env/bin/activate
python scripts/inference.py
```

## Useful Commands

**View monitoring helpers:**
```bash
source scripts/monitor.sh
```

**Cancel a job:**
```bash
scancel JOB_ID
```

**Clean checkpoints (use helper):**
```bash
source scripts/monitor.sh
clean_checkpoints
```

**View all logs:**
```bash
ls -lh logs/
```

**Combine all files for debugging:**
```bash
find . -type f ! -name "combined_files.txt" -exec bash -c 'echo -e "\n\n=========================================\nFILE PATH: {}\n========================================="; cat "{}"' \; > combined_files.txt
```

## Quick Start (Local Development)

If you're running on a local machine with GPU:

```bash
# 1. Setup
python -m venv envs/llm-env
source envs/llm-env/bin/activate
pip install -r requirements.txt

# 2. Prepare data
python scripts/prepare_data.py

# 3. Train (single GPU)
python scripts/fine_tune.py

# 4. Run inference
python scripts/inference.py
```

## Project Structure

```
llm-fine-tune/
├── scripts/
│   ├── fine_tune.py      # Main training script
│   ├── inference.py      # Test trained model
│   ├── prepare_data.py   # Download and format data
│   └── submit_job.sh     # SLURM job submission
├── data/                 # Training data (generated)
├── checkpoints/          # Model checkpoints (generated)
├── logs/                 # Training logs (generated)
└── envs/                 # Python virtual environment
```

## Configuration

Edit `scripts/fine_tune.py` to adjust:
- `model_id`: Base model to fine-tune
- `num_train_epochs`: Training epochs
- `per_device_train_batch_size`: Batch size per GPU
- `learning_rate`: Learning rate
- LoRA parameters (r, alpha, target_modules)

## Requirements

- Python 3.8+
- CUDA-capable GPU (H100/H200 recommended)
- PyTorch 2.5+
- Transformers 4.57+
- TRL 0.27+

## Requirements

### Hardware
- CUDA-capable GPU (H100/H200 recommended for Flash Attention 2)
- Multi-node SLURM cluster (for distributed training)
- Shared filesystem (e.g., `/shared` on Nebius)

### Software
- Python 3.8+
- CUDA 12.1+
- PyTorch 2.5+
- Transformers 4.57+
- TRL 0.27+
- Flash Attention 2 (for H100/H200 optimization)
- Hugging Face account with access token (optional - only for gated models)

## Development

### Running Tests

```bash
pip install pytest flake8
pytest tests/
```

### Linting

```bash
flake8 scripts/
```

## CI/CD

GitHub Actions automatically runs tests on every push and pull request:
- Python syntax validation
- Import checks
- Project structure validation
- SLURM script syntax check

## License

MIT
