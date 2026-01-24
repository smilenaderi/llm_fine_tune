#!/bin/bash

# This script reads configuration from config.yaml and submits the job
# To customize: edit config.yaml instead of this file

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: config.yaml not found"
    exit 1
fi

# Read configuration from config.yaml using simple parsing
NODES=$(grep "^  nodes:" config.yaml | head -1 | awk '{print $2}')
GPUS_PER_NODE=$(grep "^  gpus_per_node:" config.yaml | head -1 | awk '{print $2}')
PARTITION=$(grep "^  partition:" config.yaml | head -1 | awk '{print $2}' | tr -d '"')

echo "=========================================="
echo "Submitting LLM Fine-Tuning Job"
echo "=========================================="
echo "Configuration from config.yaml:"
echo "  Nodes: $NODES"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Total GPUs: $((NODES * GPUS_PER_NODE))"
echo "  Partition: $PARTITION"
echo "=========================================="

# Create the actual job script
cat > /tmp/llm_finetune_job_$.sh << 'EOFSCRIPT'
#!/bin/bash

# Create job-specific log directory
JOB_LOG_DIR="logs/job_${SLURM_JOB_ID}"
mkdir -p "$JOB_LOG_DIR"

echo "=========================================="
echo "LLM Fine-Tuning Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Log Directory: $JOB_LOG_DIR"
echo "=========================================="

# 1. Environment Setup
echo "Setting up environment..."
source envs/llm-env/bin/activate
export OMP_NUM_THREADS=1
export HF_HUB_DISABLE_TORCH_LOAD_SECURITY_CHECK=1

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
echo "All outputs saved to: $JOB_LOG_DIR"
echo "=========================================="
EOFSCRIPT

# Submit the job with dynamic parameters and job-specific output files
sbatch \
    --job-name=llm-finetune \
    --partition=$PARTITION \
    --nodes=$NODES \
    --gpus-per-node=$GPUS_PER_NODE \
    --output=logs/job_%j/slurm.out \
    --error=logs/job_%j/slurm.err \
    --time=04:00:00 \
    /tmp/llm_finetune_job_$.sh

# Clean up temp file after a delay (in background)
(sleep 5 && rm -f /tmp/llm_finetune_job_$.sh) &
