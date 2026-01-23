#!/bin/bash
# Helper scripts for monitoring SLURM jobs

# Watch the latest log (both output and errors)
watch_latest() {
    latest_err=$(ls -t logs/*.err 2>/dev/null | head -n 1)
    
    if [ -z "$latest_err" ]; then
        echo "❌ No log files found in logs/"
        return 1
    fi
    
    # Get corresponding .out file
    latest_out="${latest_err%.err}.out"
    
    echo "📋 Watching: $latest_err and $latest_out"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Show output file first (has success messages)
    if [ -f "$latest_out" ]; then
        echo "📤 OUTPUT:"
        cat "$latest_out"
        echo ""
    fi
    
    # Show errors/warnings
    if [ -f "$latest_err" ]; then
        echo "⚠️  STDERR:"
        cat "$latest_err"
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check for success indicators
    if grep -q "✅" "$latest_out" 2>/dev/null; then
        echo "✅ JOB COMPLETED SUCCESSFULLY!"
    elif grep -q "Error\|Failed\|Traceback" "$latest_err" 2>/dev/null; then
        echo "❌ JOB FAILED - Check errors above"
    else
        echo "⏳ Job may still be running..."
    fi
    
    echo ""
    echo "💡 Tip: Use 'tail -f $latest_out' to follow live output"
}

# Monitor GPU usage for running job
watch_gpu() {
    # Automatically grab the Job ID of your first RUNNING job
    JOB_ID=$(squeue --me --states=RUNNING -h -o %A | head -n 1)
    
    if [ -z "$JOB_ID" ]; then
        echo "❌ No running jobs found! Submit one first."
        echo "💡 Tip: Run 'sbatch scripts/submit_job.sh' to start training"
        return 1
    fi
    
    echo "👀 Found Job $JOB_ID. Launching nvitop..."
    srun --jobid=$JOB_ID --overlap --pty bash -c "source envs/llm-env/bin/activate && nvitop"
}

# Show job status
job_status() {
    echo "📊 Your SLURM Jobs:"
    squeue --me
    echo ""
    echo "📈 Recent completions:"
    sacct --format=JobID,JobName,State,ExitCode,Elapsed -j $(squeue --me -h -o %A | tr '\n' ',' | sed 's/,$//')
}

# Clean old checkpoints
clean_checkpoints() {
    read -p "⚠️  This will delete all checkpoints. Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf checkpoints/*
        echo "✅ Checkpoints cleaned"
    else
        echo "❌ Cancelled"
    fi
}

# Start TensorBoard server (shows all jobs for comparison)
start_tensorboard() {
    if [ ! -d "logs" ] || [ -z "$(ls -A logs/job_* 2>/dev/null)" ]; then
        echo "❌ No job logs found in logs/"
        echo "💡 Train a model first to generate logs"
        return 1
    fi
    
    echo "🚀 Starting TensorBoard..."
    echo "📊 Comparing all training runs"
    echo ""
    echo "Available jobs:"
    for job_dir in logs/job_*; do
        if [ -d "$job_dir" ]; then
            job_id=$(basename "$job_dir" | sed 's/job_//')
            echo "  - Job $job_id: $job_dir"
        fi
    done
    echo ""
    echo "Access TensorBoard at: http://localhost:6006"
    echo "💡 Use the web UI to select/compare specific runs"
    echo "Press Ctrl+C to stop"
    echo ""
    
    source envs/llm-env/bin/activate
    tensorboard --logdir=logs --host=0.0.0.0 --port=6006
}

# Start TensorBoard for specific job
start_tensorboard_job() {
    if [ -z "$1" ]; then
        echo "❌ Please provide a job ID"
        echo "Usage: start_tensorboard_job <JOB_ID>"
        return 1
    fi
    
    job_dir="logs/job_$1"
    tensorboard_dir="$job_dir/tensorboard"
    
    if [ ! -d "$tensorboard_dir" ]; then
        echo "❌ No TensorBoard logs found for job $1"
        echo "Available jobs:"
        ls -d logs/job_* 2>/dev/null | sed 's/logs\/job_/  - /'
        return 1
    fi
    
    echo "🚀 Starting TensorBoard for Job $1..."
    echo "📊 Logs directory: $tensorboard_dir"
    echo ""
    echo "Access TensorBoard at: http://localhost:6006"
    echo "Press Ctrl+C to stop"
    echo ""
    
    source envs/llm-env/bin/activate
    tensorboard --logdir="$tensorboard_dir" --host=0.0.0.0 --port=6006
}

# Show TensorBoard status
tensorboard_status() {
    if pgrep -f "tensorboard.*logs/runs" > /dev/null; then
        echo "✅ TensorBoard is running"
        echo "📊 Access at: http://localhost:6006"
        echo ""
        echo "To stop: pkill -f tensorboard"
    else
        echo "❌ TensorBoard is not running"
        echo "💡 Start with: start_tensorboard"
    fi
}

# Show help
show_help() {
    echo "🛠️  LLM Fine-Tuning Monitor Helper"
    echo ""
    echo "Usage: source scripts/monitor.sh"
    echo ""
    echo "Available functions:"
    echo "  watch_latest           - Tail the most recent error log"
    echo "  watch_gpu              - Monitor GPU usage for running job"
    echo "  job_status             - Show current and recent job status"
    echo "  clean_checkpoints      - Remove all checkpoint files"
    echo "  start_tensorboard      - Start TensorBoard (compare all jobs)"
    echo "  start_tensorboard_job  - Start TensorBoard for specific job ID"
    echo "  tensorboard_status     - Check if TensorBoard is running"
    echo ""
    echo "Example:"
    echo "  $ watch_latest"
    echo "  $ watch_gpu"
    echo "  $ start_tensorboard              # Compare all runs"
    echo "  $ start_tensorboard_job 12345    # View specific job"
}

# If script is sourced, show help
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    show_help
fi
