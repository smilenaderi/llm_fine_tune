#!/bin/bash

# List all available trained models

echo "=========================================="
echo "Available Trained Models"
echo "=========================================="

CHECKPOINT_DIR="checkpoints"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Find all job directories with final adapters
found=0
for job_dir in "$CHECKPOINT_DIR"/job_*; do
    if [ -d "$job_dir" ]; then
        job_id=$(basename "$job_dir" | sed 's/job_//')
        adapter_path="$job_dir/final_adapter"
        
        if [ -d "$adapter_path" ]; then
            found=1
            
            # Get model info from adapter config (most reliable)
            adapter_config_file="$adapter_path/adapter_config.json"
            if [ -f "$adapter_config_file" ]; then
                model_name=$(python3 -c "import json; print(json.load(open('$adapter_config_file'))['base_model_name_or_path'])" 2>/dev/null || echo "Unknown")
            else
                model_name="Unknown"
            fi
            
            # Try to get training info from job config
            config_file="logs/job_${job_id}/config.yaml"
            if [ -f "$config_file" ]; then
                epochs=$(grep "num_train_epochs:" "$config_file" | head -1 | awk '{print $2}')
                samples=$(grep "max_samples:" "$config_file" | head -1 | awk '{print $2}')
            else
                epochs="?"
                samples="?"
            fi
            
            # Get adapter size
            adapter_size=$(du -sh "$adapter_path" 2>/dev/null | awk '{print $1}')
            
            # Get training completion time
            if [ -f "logs/job_${job_id}/slurm.out" ]; then
                completion=$(grep "Training completed" "logs/job_${job_id}/slurm.out" | tail -1)
                if [ -n "$completion" ]; then
                    status="✅ Completed"
                else
                    status="⚠️  Incomplete"
                fi
            else
                status="❓ Unknown"
            fi
            
            echo ""
            echo "Job ID: $job_id"
            echo "  Status: $status"
            echo "  Model: $model_name"
            echo "  Training: $epochs epochs, $samples samples"
            echo "  Adapter Size: $adapter_size"
            echo "  Path: $adapter_path"
            
            # Show validation score if available
            validation_file="logs/job_${job_id}/validation_results.json"
            if [ -f "$validation_file" ]; then
                score=$(grep '"score"' "$validation_file" | awk '{print $2}' | tr -d ',')
                if [ -n "$score" ]; then
                    echo "  Validation Score: ${score}%"
                fi
            fi
        fi
    fi
done

if [ $found -eq 0 ]; then
    echo ""
    echo "❌ No trained models found"
    echo ""
    echo "💡 To train a model:"
    echo "   sbatch scripts/submit_job.sh"
    echo ""
fi

echo ""
echo "=========================================="
echo "Usage:"
echo "  sbatch scripts/run_inf.sh <job_id>"
echo "  python scripts/inference.py --job-id <job_id>"
echo "=========================================="
