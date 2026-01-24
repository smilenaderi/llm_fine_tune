#!/bin/bash
#SBATCH --job-name=llm-inference
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --time=00:30:00

# Usage:
#   sbatch scripts/run_inf.sh                              # Use latest job, default prompt
#   sbatch scripts/run_inf.sh 8                            # Use job 8, default prompt
#   sbatch scripts/run_inf.sh 8 "Your prompt"             # Use job 8, custom prompt
#   sbatch scripts/run_inf.sh 8 --prompts prompts.txt     # Use job 8, multiple prompts from file

echo "=========================================="
echo "LLM Inference Job"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Environment
source envs/llm-env/bin/activate
export HF_HUB_DISABLE_TORCH_LOAD_SECURITY_CHECK=1

# Get arguments
JOB_ID=${1:-""}
shift || true
ARGS="$@"

# Run inference
if [ -n "$JOB_ID" ]; then
    echo "Using job ID: $JOB_ID"
    if [ -n "$ARGS" ]; then
        echo "Additional args: $ARGS"
        python scripts/inference.py --job-id "$JOB_ID" $ARGS
    else
        python scripts/inference.py --job-id "$JOB_ID"
    fi
else
    echo "Using latest trained model"
    if [ -n "$ARGS" ]; then
        python scripts/inference.py $ARGS
    else
        python scripts/inference.py
    fi
fi

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Inference completed successfully"
else
    echo "❌ Inference failed with exit code $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
