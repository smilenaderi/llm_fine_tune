#!/bin/bash
#SBATCH --job-name=llm-finetune
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00

# This script reads configuration from config.yaml
# To customize: edit config.yaml instead of this file

echo "=========================================="
echo "LLM Fine-Tuning Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# 1. Environment Setup
echo "Setting up environment..."
source envs/llm-env/bin/activate
export OMP_NUM_THREADS=1

# 2. Load configuration
echo "Loading configuration from config.yaml..."
NODES=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['cluster']['nodes'])")
GPUS_PER_NODE=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['cluster']['gpus_per_node'])")

echo "Configuration:"
echo "  Nodes: $NODES"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Total GPUs: $((NODES * GPUS_PER_NODE))"

# 3. Network Setup
head_node_ip=$(hostname --ip-address)
echo "Head node IP: $head_node_ip"

# 4. Launch Training
echo "=========================================="
echo "Starting distributed training..."
echo "=========================================="

srun torchrun \
    --nnodes=$NODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    scripts/fine_tune.py

TRAIN_EXIT_CODE=$?

echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully"
    echo "=========================================="
    
    # Run validation
    echo "Running model validation..."
    python scripts/validate_model.py
    
    VALIDATION_EXIT_CODE=$?
    if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
        echo "✅ Validation passed"
    else
        echo "⚠️  Validation completed with warnings"
    fi
else
    echo "❌ Training failed with exit code $TRAIN_EXIT_CODE"
    echo "=========================================="
    exit $TRAIN_EXIT_CODE
fi

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
