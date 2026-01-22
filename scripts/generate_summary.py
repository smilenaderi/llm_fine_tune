#!/usr/bin/env python3
"""
Generate a comprehensive training summary report
"""
import json
import os
from datetime import datetime

def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def main():
    # Get job ID from environment or find latest
    job_id = os.environ.get('SLURM_JOB_ID', None)
    
    # Load benchmark results
    if job_id:
        benchmark_file = f"logs/benchmark_results_{job_id}.json"
        validation_file = f"logs/validation_results_{job_id}.json"
    else:
        # Find the latest benchmark file
        import glob
        benchmark_files = glob.glob("logs/benchmark_results_*.json")
        if not benchmark_files:
            benchmark_file = "logs/benchmark_results.json"
            validation_file = "logs/validation_results.json"
        else:
            benchmark_file = max(benchmark_files, key=os.path.getmtime)
            # Get corresponding validation file
            job_id_from_file = benchmark_file.split('_')[-1].replace('.json', '')
            validation_file = f"logs/validation_results_{job_id_from_file}.json"
    
    if not os.path.exists(benchmark_file):
        print("❌ No benchmark results found. Training may not have completed.")
        return
    
    with open(benchmark_file, 'r') as f:
        benchmark = json.load(f)
    
    validation = None
    if os.path.exists(validation_file):
        with open(validation_file, 'r') as f:
            validation = json.load(f)
    
    # Generate summary
    summary = []
    summary.append("=" * 80)
    summary.append("🎯 LLM FINE-TUNING SUMMARY REPORT")
    summary.append("=" * 80)
    summary.append("")
    
    # Job ID
    if 'job_id' in benchmark:
        summary.append(f"🆔 Job ID: {benchmark['job_id']}")
    
    # Timestamp
    timestamp = datetime.fromisoformat(benchmark['timestamp'])
    summary.append(f"📅 Completed: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # Model Information
    if 'model' in benchmark:
        summary.append("🤖 MODEL")
        summary.append("-" * 80)
        model_info = benchmark['model']
        summary.append(f"  Model:                {model_info.get('name', 'N/A')}")
        summary.append(f"  LoRA Rank:            {model_info.get('lora_rank', 'N/A')}")
        summary.append(f"  LoRA Alpha:           {model_info.get('lora_alpha', 'N/A')}")
        summary.append("")
    
    # Dataset Information
    if 'dataset' in benchmark:
        summary.append("📚 DATASET")
        summary.append("-" * 80)
        dataset_info = benchmark['dataset']
        summary.append(f"  Dataset:              {dataset_info.get('name', 'N/A')}")
        summary.append(f"  Split:                {dataset_info.get('split', 'N/A')}")
        max_samples = dataset_info.get('max_samples')
        summary.append(f"  Max Samples:          {max_samples if max_samples else 'All'}")
        summary.append("")
    
    # Configuration
    summary.append("⚙️  CONFIGURATION")
    summary.append("-" * 80)
    config = benchmark['config']
    summary.append(f"  Cluster:              {config['nodes']} nodes × {config['gpus_per_node']} GPUs = {config['nodes'] * config['gpus_per_node']} total GPUs")
    summary.append(f"  Batch Size:           {config['batch_size']} per device")
    summary.append(f"  Gradient Accumulation: {config['gradient_accumulation']} steps")
    summary.append(f"  Effective Batch Size: {config['effective_batch_size']}")
    summary.append(f"  Learning Rate:        {config['learning_rate']}")
    summary.append(f"  Epochs:               {config['epochs']}")
    summary.append("")
    
    # Performance Metrics
    summary.append("⚡ PERFORMANCE METRICS")
    summary.append("-" * 80)
    summary.append(f"  Training Time:        {format_time(benchmark['total_training_time_seconds'])}")
    summary.append(f"  Total Steps:          {benchmark['total_steps']}")
    summary.append(f"  Avg Time per Step:    {benchmark['avg_time_per_step_seconds']:.2f}s")
    summary.append(f"  Throughput:           {benchmark['tokens_per_second']:,.0f} tokens/second")
    summary.append(f"  Total Tokens:         {benchmark['total_tokens']:,.0f}")
    summary.append("")
    
    # GPU Metrics
    summary.append("🖥️  GPU METRICS")
    summary.append("-" * 80)
    summary.append(f"  Max GPU Memory:       {benchmark['max_gpu_memory_gb']:.2f} GB")
    summary.append(f"  Memory per GPU:       {benchmark['max_gpu_memory_gb'] / (config['nodes'] * config['gpus_per_node']):.2f} GB")
    summary.append("")
    
    # Training Loss
    summary.append("📊 TRAINING METRICS")
    summary.append("-" * 80)
    summary.append(f"  Final Loss:           {benchmark['final_loss']:.4f}")
    summary.append("")
    
    # Validation Results
    if validation:
        summary.append("✅ VALIDATION RESULTS")
        summary.append("-" * 80)
        summary.append(f"  Average Score:        {validation['average_score']:.1%}")
        summary.append(f"  Total Tests:          {validation['total_tests']}")
        summary.append("")
        summary.append("  Test Breakdown:")
        for result in validation['results']:
            summary.append(f"    • {result['test_name']:20s} {result['score']:.1%}")
        summary.append("")
    
    # Efficiency Metrics
    summary.append("📈 EFFICIENCY METRICS")
    summary.append("-" * 80)
    tokens_per_gpu = benchmark['tokens_per_second'] / (config['nodes'] * config['gpus_per_node'])
    summary.append(f"  Tokens/sec per GPU:   {tokens_per_gpu:,.0f}")
    
    # Calculate cost efficiency (if training time is known)
    training_hours = benchmark['total_training_time_seconds'] / 3600
    summary.append(f"  GPU Hours Used:       {training_hours * config['nodes'] * config['gpus_per_node']:.2f}")
    summary.append("")
    
    # Recommendations
    summary.append("💡 RECOMMENDATIONS")
    summary.append("-" * 80)
    
    if validation and validation['average_score'] < 0.7:
        summary.append("  ⚠️  Validation score is low. Consider:")
        summary.append("     - Training for more epochs")
        summary.append("     - Using more training data")
        summary.append("     - Adjusting learning rate")
    elif validation and validation['average_score'] >= 0.8:
        summary.append("  ✅ Excellent validation score! Model is performing well.")
    
    if benchmark['tokens_per_second'] < 5000:
        summary.append("  ⚠️  Throughput is low. Consider:")
        summary.append("     - Increasing batch size")
        summary.append("     - Checking GPU utilization")
        summary.append("     - Reducing sequence length")
    elif benchmark['tokens_per_second'] >= 10000:
        summary.append("  ✅ Excellent throughput! Training is efficient.")
    
    summary.append("")
    summary.append("=" * 80)
    
    # Print to console
    report = "\n".join(summary)
    print(report)
    
    # Save to file
    job_suffix = f"_{benchmark.get('job_id', 'latest')}" if 'job_id' in benchmark else ""
    output_file = f"logs/training_summary{job_suffix}.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Summary saved to: {output_file}")

if __name__ == "__main__":
    main()
