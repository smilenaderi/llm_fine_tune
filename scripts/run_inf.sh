#!/bin/bash
#SBATCH --job-name=llm-inference
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# Environment
source envs/llm-env/bin/activate

# Run inference
python scripts/inference.py
