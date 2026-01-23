# LLM Fine-Tuning with FSDP and Flash Attention 2

A fully configurable program for fine-tuning large language models from Hugging Face using function-calling datasets. Built for distributed training on GPU clusters with FSDP (Fully Sharded Data Parallel) and Flash Attention 2 optimization.

## What Is This?

This project provides a production-ready pipeline for fine-tuning LLMs on function-calling tasks. Everything is controlled through a single `config.yaml` file - select your model, dataset, and training parameters without touching code. The system handles distributed training, checkpointing, validation, and benchmarking automatically.

## How It Works

1. **Configure** - Edit `config.yaml` to select model, dataset, LoRA parameters, and training settings
2. **Setup** - Run setup script to create environment and install dependencies
3. **Train** - Submit SLURM job that distributes training across GPUs using FSDP
4. **Validate** - Automatic validation tests model on function-calling tasks
5. **Deploy** - Fine-tuned model saved with LoRA adapters ready for inference

The pipeline uses LoRA for efficient parameter updates, FSDP for distributed training, and Flash Attention 2 for H100/H200 optimization.

## Project Structure

```
llm-fine-tune/
├── config.yaml              # Main configuration file
├── requirements.txt         # Python dependencies
├── scripts/
│   ├── setup_nebius.sh     # Environment setup
│   ├── submit_job.sh       # SLURM job submission
│   ├── fine_tune.py        # Main training script
│   ├── config_loader.py    # Configuration parser
│   ├── validate_model.py   # Model validation tests
│   ├── inference.py        # Inference script
│   ├── monitor.sh          # Monitoring utilities
│   └── generate_summary.py # Training summary generator
├── tests/
│   └── test_scripts.py     # Basic project tests
├── data/                   # Dataset cache
├── checkpoints/            # Model checkpoints
└── logs/                   # Training logs and results
```

## Scripts Explained

### Setup & Deployment

**`scripts/setup_nebius.sh`**
- Creates directory structure on SLURM cluster
- Sets up Python virtual environment
- Installs PyTorch with CUDA 12.1 support
- Installs ML libraries (transformers, PEFT, TRL, accelerate)
- Compiles Flash Attention 2 for H100/H200 optimization
- Optionally configures Hugging Face authentication

**`scripts/submit_job.sh`**
- Reads cluster configuration from `config.yaml`
- Generates SLURM job script dynamically
- Submits distributed training job with correct node/GPU allocation
- Configures torchrun for multi-node training
- Runs validation and summary generation after training

### Training & Validation

**`scripts/fine_tune.py`**
- Main training script with full FSDP support
- Loads model and dataset based on `config.yaml`
- Configures LoRA adapters for efficient fine-tuning
- Implements BenchmarkCallback for performance tracking
- Handles checkpointing and resumption
- Supports both streaming and regular datasets
- Validates environment (CUDA, Flash Attention compatibility)

**`scripts/config_loader.py`**
- Parses and validates `config.yaml`
- Resolves path variables and creates directories
- Provides dot-notation access to config values
- Calculates effective batch size across GPUs
- Prints configuration summary before training

**`scripts/validate_model.py`**
- Tests fine-tuned model on 5 function-calling scenarios
- Evaluates responses for expected keywords
- Generates validation score (0-100%)
- Saves detailed results to JSON
- Provides quality assessment (Excellent/Good/Fair/Poor)

**`scripts/inference.py`**
- Loads base model and fine-tuned LoRA adapter
- Runs inference on test prompts from config
- Demonstrates how to use the fine-tuned model
- Useful for quick testing after training

### Monitoring

**`scripts/monitor.sh`**
- `watch_latest` - Displays most recent training logs
- `watch_gpu` - Live GPU monitoring with nvitop
- `job_status` - Shows SLURM job queue and history
- `clean_checkpoints` - Removes old checkpoint files

Usage: `source scripts/monitor.sh` then call functions directly

## Tests

**`tests/test_scripts.py`**
- Validates project structure (directories exist)
- Checks required files are present
- Tests script imports work correctly
- Verifies SLURM script has valid bash syntax

Run tests: `pytest tests/`

## Configuration

All settings are in `config.yaml`:

- **Cluster**: Nodes, GPUs per node, partition
- **Model**: Model ID, dtype, Flash Attention settings
- **Dataset**: Dataset name, split, sample limit, text field
- **LoRA**: Rank, alpha, dropout, target modules
- **Training**: Epochs, batch size, learning rate, optimizer
- **Distributed**: FSDP/DDP strategy, sharding, CPU offload
- **Checkpointing**: Save frequency, resume settings
- **Validation**: Split ratio, evaluation frequency, metrics

See `config.yaml` for detailed documentation and presets.

## Performance

On 4x H200 GPUs:
- 20k samples, 1 epoch: 15-30 minutes @ 12-15k tok/s
- 60k samples, 3 epochs: 2-4 hours @ 12-15k tok/s

## Getting Started

See [QUICK_START.md](QUICK_START.md) for detailed setup instructions and deployment guide.

## Documentation

- [QUICK_START.md](QUICK_START.md) - Setup and usage guide
- [COMMANDS.md](COMMANDS.md) - Full list of available commands
- [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md) - Hugging Face authentication
- [config.yaml](config.yaml) - Complete configuration reference

## License

MIT
