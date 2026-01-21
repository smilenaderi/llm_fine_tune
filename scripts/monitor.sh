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

# Show help
show_help() {
    echo "🛠️  LLM Fine-Tuning Monitor Helper"
    echo ""
    echo "Usage: source scripts/monitor.sh"
    echo ""
    echo "Available functions:"
    echo "  watch_latest       - Tail the most recent error log"
    echo "  watch_gpu          - Monitor GPU usage for running job"
    echo "  job_status         - Show current and recent job status"
    echo "  clean_checkpoints  - Remove all checkpoint files"
    echo ""
    echo "Example:"
    echo "  $ watch_latest"
    echo "  $ watch_gpu"
}

# If script is sourced, show help
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    show_help
fi
