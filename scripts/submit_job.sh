#!/bin/bash
#SBATCH --job-name=llm-finetune
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# 1. Environment
source envs/llm-env/bin/activate
export OMP_NUM_THREADS=1

# 2. Network (Simpler for single node, but we keep the logic to prevent errors)
head_node_ip=$(hostname --ip-address)
echo "Running on Single Node IP: $head_node_ip"

# 3. Launch Training
# Note: nproc_per_node is set to 1 because you only have 1 GPU
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    scripts/fine_tune.py