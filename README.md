# LLM Fine-Tuning with Qwen2.5-7B

Fine-tune Qwen2.5-7B-Instruct on function-calling data using LoRA adapters and distributed training.

## Features

- LoRA-based fine-tuning for efficient training
- Multi-node distributed training support (SLURM)
- Flash Attention 2 optimization for H100/H200 GPUs
- Checkpoint management and resume capability
- Real function-calling dataset (xLAM)

## Quick Start

### 1. Setup Environment

```bash
python -m venv envs/llm-env
source envs/llm-env/bin/activate
pip install torch transformers datasets peft trl accelerate
```

### 2. Prepare Data

```bash
python scripts/prepare_data.py
```

This downloads 20k function-calling examples from the xLAM dataset.

### 3. Train

**Single GPU:**
```bash
python scripts/fine_tune.py
```

**Multi-Node (SLURM):**
```bash
sbatch scripts/submit_job.sh
```

### 4. Run Inference

```bash
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

## License

MIT
