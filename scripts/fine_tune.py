import os
import sys
import json
import time
import torch
import logging
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Import configuration loader
from config_loader import load_config

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Auto-detect Flash Attention 2 availability
FLASH_ATTENTION_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTENTION_AVAILABLE = True
    logger.info(f"✓ Flash Attention 2 detected (version {flash_attn.__version__})")
except ImportError:
    logger.info("ℹ️  Flash Attention 2 not installed - using standard attention")
    logger.info("   Install with: pip install flash-attn --no-build-isolation")


class BenchmarkCallback(TrainerCallback):
    """Callback to track performance metrics during training"""
    
    def __init__(self, config):
        self.config = config
        self.start_time = None
        self.step_times = []
        self.metrics = {
            'throughput': [],
            'gpu_memory': [],
            'training_time': 0
        }
        self.last_log_time = None
        
        # Initialize GPU monitoring
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.gpu_metrics = {i: {'utilization': [], 'memory_used': [], 'temperature': []} 
                           for i in range(self.num_gpus)}
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        logger.info("🚀 Training started")
        
        # Log initial GPU state
        if torch.cuda.is_available():
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    def _get_gpu_metrics(self):
        """Get current GPU metrics for all devices"""
        metrics = {}
        if not torch.cuda.is_available():
            return metrics
        
        try:
            import subprocess
            # Use nvidia-smi to get detailed GPU metrics
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            gpu_id = int(parts[0])
                            metrics[gpu_id] = {
                                'utilization': float(parts[1]) if parts[1] != '[N/A]' else 0,
                                'memory_used_mb': float(parts[2]) if parts[2] != '[N/A]' else 0,
                                'memory_total_mb': float(parts[3]) if parts[3] != '[N/A]' else 0,
                                'temperature': float(parts[4]) if parts[4] != '[N/A]' else 0,
                                'power_draw': float(parts[5]) if parts[5] != '[N/A]' else 0,
                            }
        except Exception as e:
            # Fallback to basic PyTorch metrics
            pass
        
        # Add PyTorch memory metrics
        for i in range(self.num_gpus):
            if i not in metrics:
                metrics[i] = {}
            metrics[i]['memory_allocated_gb'] = torch.cuda.memory_allocated(i) / 1024**3
            metrics[i]['memory_reserved_gb'] = torch.cuda.memory_reserved(i) / 1024**3
            metrics[i]['memory_max_allocated_gb'] = torch.cuda.max_memory_allocated(i) / 1024**3
        
        return metrics
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log additional metrics to TensorBoard"""
        if logs is None:
            return
        
        # Only log from main process to avoid duplicates
        if not state.is_world_process_zero:
            return
        
        current_time = time.time()
        
        # Calculate throughput (steps per second)
        if self.last_log_time is not None:
            time_diff = current_time - self.last_log_time
            if time_diff > 0:
                steps_per_sec = args.logging_steps / time_diff
                logs['performance/steps_per_second'] = steps_per_sec
        
        self.last_log_time = current_time
        
        # Get comprehensive GPU metrics
        gpu_metrics = self._get_gpu_metrics()
        
        # Log per-GPU metrics
        for gpu_id, metrics in gpu_metrics.items():
            prefix = f'gpu_{gpu_id}'
            
            if 'utilization' in metrics:
                logs[f'{prefix}/utilization_percent'] = metrics['utilization']
            
            if 'memory_used_mb' in metrics and 'memory_total_mb' in metrics:
                logs[f'{prefix}/memory_used_gb'] = metrics['memory_used_mb'] / 1024
                logs[f'{prefix}/memory_total_gb'] = metrics['memory_total_mb'] / 1024
                logs[f'{prefix}/memory_usage_percent'] = (metrics['memory_used_mb'] / metrics['memory_total_mb'] * 100) if metrics['memory_total_mb'] > 0 else 0
            
            if 'temperature' in metrics:
                logs[f'{prefix}/temperature_celsius'] = metrics['temperature']
            
            if 'power_draw' in metrics:
                logs[f'{prefix}/power_draw_watts'] = metrics['power_draw']
            
            if 'memory_allocated_gb' in metrics:
                logs[f'{prefix}/memory_allocated_gb'] = metrics['memory_allocated_gb']
                logs[f'{prefix}/memory_reserved_gb'] = metrics['memory_reserved_gb']
                logs[f'{prefix}/memory_max_allocated_gb'] = metrics['memory_max_allocated_gb']
        
        # Aggregate GPU metrics across all GPUs
        if gpu_metrics:
            avg_utilization = sum(m.get('utilization', 0) for m in gpu_metrics.values()) / len(gpu_metrics)
            total_memory_used = sum(m.get('memory_used_mb', 0) for m in gpu_metrics.values()) / 1024
            avg_temperature = sum(m.get('temperature', 0) for m in gpu_metrics.values()) / len(gpu_metrics)
            total_power = sum(m.get('power_draw', 0) for m in gpu_metrics.values())
            
            logs['gpu_aggregate/avg_utilization_percent'] = avg_utilization
            logs['gpu_aggregate/total_memory_used_gb'] = total_memory_used
            logs['gpu_aggregate/avg_temperature_celsius'] = avg_temperature
            logs['gpu_aggregate/total_power_watts'] = total_power
        
        # Calculate tokens per second if available
        if 'train_samples_per_second' in logs:
            # Estimate tokens (assuming avg 512 tokens per sample)
            avg_tokens_per_sample = self.config.get('training.max_seq_length', 2048) * 0.5
            logs['performance/tokens_per_second'] = logs['train_samples_per_second'] * avg_tokens_per_sample
        
        # Add training progress percentage
        if state.max_steps > 0:
            logs['performance/progress_percent'] = (state.global_step / state.max_steps) * 100
        
        # Add elapsed time
        logs['performance/elapsed_time_minutes'] = (current_time - self.start_time) / 60
        
        # Debug: Log what we're adding (only first time)
        if state.global_step == args.logging_steps:
            logger.info(f"📊 Logging {len([k for k in logs.keys() if '/' in k])} custom metrics to TensorBoard")
        
        # Note: The logs dict is automatically written to TensorBoard by the Trainer
        # All keys with '/' will be grouped in TensorBoard UI
    
    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time()
        self.step_times.append(step_time)
        
        # Track GPU memory
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.metrics['gpu_memory'].append(memory_allocated)
    
    def on_train_end(self, args, state, control, **kwargs):
        self.metrics['training_time'] = time.time() - self.start_time
        
        # Calculate throughput
        if state.global_step > 0:
            avg_time_per_step = self.metrics['training_time'] / state.global_step
            self.metrics['avg_time_per_step'] = avg_time_per_step
        
        logger.info(f"✅ Training completed in {self.metrics['training_time']:.2f} seconds")
        
        # Save benchmark results
        if self.config.get('benchmarking.enabled'):
            self._save_benchmark_results(state)
    
    def _save_benchmark_results(self, state):
        """Save benchmark results to file"""
        # Get job ID from environment
        job_id = os.environ.get('SLURM_JOB_ID', f'local_{int(time.time())}')
        
        # Save to job-specific directory
        job_log_dir = os.path.join(self.config.get('storage.log_dir'), f'job_{job_id}')
        os.makedirs(job_log_dir, exist_ok=True)
        benchmark_file = os.path.join(job_log_dir, 'benchmark_results.json')
        
        # Get final loss from train_loss in the last log entry
        final_loss = 0
        if state.log_history:
            # Look for train_loss in the last entries
            for log in reversed(state.log_history):
                if 'train_loss' in log:
                    final_loss = log['train_loss']
                    break
                elif 'loss' in log:
                    final_loss = log['loss']
        
        results = {
            'job_id': job_id,
            'timestamp': datetime.now().isoformat(),
            'total_training_time_seconds': self.metrics['training_time'],
            'total_steps': state.global_step,
            'avg_time_per_step_seconds': self.metrics.get('avg_time_per_step', 0),
            'max_gpu_memory_gb': max(self.metrics['gpu_memory']) if self.metrics['gpu_memory'] else 0,
            'final_loss': final_loss,
            'model': {
                'name': self.config.get('model.model_id'),
                'lora_rank': self.config.get('lora.r'),
                'lora_alpha': self.config.get('lora.lora_alpha')
            },
            'dataset': {
                'name': self.config.get('dataset.name'),
                'max_samples': self.config.get('dataset.max_samples'),
                'split': self.config.get('dataset.split')
            },
            'config': {
                'nodes': self.config.get('cluster.nodes'),
                'gpus_per_node': self.config.get('cluster.gpus_per_node'),
                'batch_size': self.config.get('training.per_device_train_batch_size'),
                'gradient_accumulation': self.config.get('training.gradient_accumulation_steps'),
                'effective_batch_size': self.config.get_effective_batch_size(),
                'learning_rate': self.config.get('training.learning_rate'),
                'epochs': self.config.get('training.num_train_epochs')
            }
        }
        
        # Calculate tokens per second if available
        if state.log_history:
            total_tokens = sum(log.get('num_tokens', 0) for log in state.log_history)
            if total_tokens > 0:
                results['total_tokens'] = total_tokens
                results['tokens_per_second'] = total_tokens / self.metrics['training_time']
        
        os.makedirs(os.path.dirname(benchmark_file), exist_ok=True)
        with open(benchmark_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📊 Benchmark results saved to {benchmark_file}")


def validate_environment():
    """Validate that the environment is properly configured"""
    logger.info("🔍 Validating environment...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU training requires CUDA.")
    
    logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    logger.info(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    
    # Check GPU compute capability for Flash Attention
    compute_capability = torch.cuda.get_device_capability()[0]
    if compute_capability >= 8:
        logger.info(f"✓ GPU compute capability {compute_capability}.x - Flash Attention 2 supported")
    else:
        logger.warning(f"⚠ GPU compute capability {compute_capability}.x - Flash Attention 2 may not be optimal")


def load_and_prepare_dataset(config):
    """Load and prepare the training dataset with error handling"""
    logger.info("📥 Loading dataset...")
    
    dataset_config = config.get_dataset_config()
    
    try:
        # Load dataset
        dataset = load_dataset(
            dataset_config['name'],
            split=dataset_config['split'],
            streaming=dataset_config.get('streaming', True)
        )
        
        # Limit samples if specified
        max_samples = dataset_config.get('max_samples')
        if max_samples:
            logger.info(f"📊 Limiting dataset to {max_samples} samples")
            if dataset_config.get('streaming'):
                dataset = dataset.take(max_samples)
            else:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Convert streaming dataset to regular dataset for training
        if dataset_config.get('streaming'):
            logger.info("🔄 Converting streaming dataset to regular dataset...")
            dataset = list(dataset)
            from datasets import Dataset
            dataset = Dataset.from_list(dataset)
        
        logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        raise


def prepare_model_and_tokenizer(config):
    """Load model and tokenizer with error handling"""
    logger.info("🤖 Loading model and tokenizer...")
    
    model_config = config.get_model_config()
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['model_id'],
            cache_dir=model_config.get('cache_dir'),
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("✓ Tokenizer loaded")
        
        # Determine attention implementation
        compute_capability = torch.cuda.get_device_capability()[0]
        gpu_supports_flash = compute_capability >= 8
        config_wants_flash = model_config.get('use_flash_attention', False)
        
        # Use Flash Attention only if: installed AND GPU supports it AND (config enables it OR auto-detect)
        use_flash = FLASH_ATTENTION_AVAILABLE and gpu_supports_flash and (config_wants_flash or not config_wants_flash)
        # Simplified: if flash-attn is installed and GPU supports it, use it
        use_flash = FLASH_ATTENTION_AVAILABLE and gpu_supports_flash
        
        attn_implementation = "flash_attention_2" if use_flash else "eager"
        
        if use_flash:
            logger.info("⚡ Using Flash Attention 2 (auto-detected)")
        else:
            if not FLASH_ATTENTION_AVAILABLE:
                logger.info("📝 Using standard attention (flash-attn not installed)")
            elif not gpu_supports_flash:
                logger.info(f"📝 Using standard attention (GPU compute capability {compute_capability}.x < 8.0)")
            else:
                logger.info("📝 Using standard attention")
        
        # Load model
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32
        }
        dtype = dtype_map.get(model_config.get('torch_dtype', 'bfloat16'), torch.bfloat16)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_config['model_id'],
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            use_cache=False,
            cache_dir=model_config.get('cache_dir'),
            trust_remote_code=True
        )
        logger.info("✓ Model loaded")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def create_lora_config(config):
    """Create LoRA configuration"""
    lora_config = config.get_lora_config()
    
    return LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=lora_config['target_modules'],
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        bias=lora_config.get('bias', 'none'),
        task_type="CAUSAL_LM"
    )


def create_training_args(config):
    """Create training arguments from config"""
    training_config = config.get_training_config()
    storage_config = config.get_storage_config()
    checkpointing_config = config.get_checkpointing_config()
    logging_config = config.get_logging_config()
    dataset_config = config.get_dataset_config()
    distributed_config = config.get_distributed_config()
    
    # Get job ID for organizing outputs
    job_id = os.environ.get('SLURM_JOB_ID', f'local_{int(time.time())}')
    
    # Create job-specific directories
    job_log_dir = os.path.join(storage_config['log_dir'], f'job_{job_id}')
    job_checkpoint_dir = os.path.join(storage_config['checkpoint_dir'], f'job_{job_id}')
    os.makedirs(job_log_dir, exist_ok=True)
    os.makedirs(job_checkpoint_dir, exist_ok=True)
    
    logger.info(f"📁 Job ID: {job_id}")
    logger.info(f"📁 Logs: {job_log_dir}")
    logger.info(f"📁 Checkpoints: {job_checkpoint_dir}")
    
    # Base arguments
    args = {
        'output_dir': job_checkpoint_dir,
        'num_train_epochs': training_config['num_train_epochs'],
        'per_device_train_batch_size': training_config['per_device_train_batch_size'],
        'gradient_accumulation_steps': training_config['gradient_accumulation_steps'],
        'learning_rate': training_config['learning_rate'],
        'lr_scheduler_type': training_config.get('lr_scheduler_type', 'linear'),
        'warmup_ratio': training_config.get('warmup_ratio', 0.03),
        'optim': training_config.get('optim', 'adamw_torch'),
        'weight_decay': training_config.get('weight_decay', 0.01),
        'max_grad_norm': training_config.get('max_grad_norm', 1.0),
        'bf16': training_config.get('bf16', True),
        'fp16': training_config.get('fp16', False),
        'logging_steps': logging_config.get('logging_steps', 1),
        'save_strategy': checkpointing_config.get('save_strategy', 'steps'),
        'save_steps': checkpointing_config.get('save_steps', 25),
        'save_total_limit': checkpointing_config.get('save_total_limit', 3),
        'report_to': logging_config.get('report_to', 'none'),
        'dataset_text_field': dataset_config.get('text_field', 'messages'),
        'packing': training_config.get('packing', False),
        'max_length': training_config.get('max_seq_length', 2048),
    }
    
    # Add TensorBoard logging directory if enabled
    if logging_config.get('report_to') == 'tensorboard' or 'tensorboard' in logging_config.get('report_to', ''):
        tensorboard_config = logging_config.get('tensorboard', {})
        if tensorboard_config.get('enabled', True):
            tensorboard_dir = os.path.join(job_log_dir, 'tensorboard')
            args['logging_dir'] = tensorboard_dir
            logger.info(f"✓ TensorBoard logging enabled: {tensorboard_dir}")
    
    # Add FSDP configuration if enabled
    if distributed_config.get('strategy') == 'fsdp' and distributed_config['fsdp'].get('enabled'):
        fsdp_config = distributed_config['fsdp']
        args['fsdp'] = fsdp_config.get('sharding_strategy', 'full_shard')
        args['fsdp_config'] = {
            'backward_prefetch': fsdp_config.get('backward_prefetch', 'backward_pre'),
            'cpu_offload': fsdp_config.get('cpu_offload', False)
        }
        logger.info("✓ FSDP enabled for distributed training")
    elif distributed_config.get('strategy') == 'ddp':
        ddp_config = distributed_config.get('ddp', {})
        args['ddp_find_unused_parameters'] = ddp_config.get('find_unused_parameters', False)
        logger.info("✓ DDP enabled for distributed training")
    
    # Add validation configuration if enabled
    validation_config = config.get_validation_config()
    if validation_config.get('enabled'):
        args['eval_strategy'] = validation_config.get('eval_strategy', 'steps')
        args['eval_steps'] = validation_config.get('eval_steps', 25)
    
    return SFTConfig(**args)


def main():
    try:
        # Load configuration
        logger.info("📋 Loading configuration...")
        config = load_config()
        config.print_summary()
        
        # Validate environment
        validate_environment()
        
        # Load dataset
        dataset = load_and_prepare_dataset(config)
        
        # Split dataset for validation if enabled
        validation_config = config.get_validation_config()
        train_dataset = dataset
        eval_dataset = None
        
        if validation_config.get('enabled') and validation_config.get('validation_split', 0) > 0:
            split_ratio = validation_config['validation_split']
            split_idx = int(len(dataset) * (1 - split_ratio))
            train_dataset = dataset.select(range(split_idx))
            eval_dataset = dataset.select(range(split_idx, len(dataset)))
            logger.info(f"✓ Dataset split: {len(train_dataset)} train, {len(eval_dataset)} validation")
        
        # Load model and tokenizer
        model, tokenizer = prepare_model_and_tokenizer(config)
        
        # Create LoRA config
        peft_config = create_lora_config(config)
        logger.info(f"✓ LoRA config created (r={peft_config.r}, alpha={peft_config.lora_alpha})")
        
        # Create training arguments
        training_args = create_training_args(config)
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            args=training_args,
            callbacks=[BenchmarkCallback(config)] if config.get('benchmarking.enabled') else []
        )
        
        # Check for existing checkpoints
        checkpointing_config = config.get_checkpointing_config()
        last_checkpoint = None
        job_id = os.environ.get('SLURM_JOB_ID', f'local_{int(time.time())}')
        output_dir = os.path.join(config.get('storage.checkpoint_dir'), f'job_{job_id}')
        
        if checkpointing_config.get('resume_from_checkpoint') and os.path.isdir(output_dir):
            from transformers.trainer_utils import get_last_checkpoint
            last_checkpoint = get_last_checkpoint(output_dir)
            if last_checkpoint and trainer.accelerator.is_main_process:
                logger.info(f"🔄 Resuming from checkpoint: {last_checkpoint}")
        
        # Start training
        if trainer.accelerator.is_main_process:
            logger.info("=" * 70)
            logger.info("🚀 STARTING TRAINING")
            logger.info("=" * 70)
        
        trainer.train(resume_from_checkpoint=last_checkpoint)
        
        # Save final model
        final_path = os.path.join(output_dir, "final_adapter")
        trainer.save_model(final_path)
        
        if trainer.accelerator.is_main_process:
            logger.info("=" * 70)
            logger.info(f"✅ TRAINING COMPLETE")
            logger.info(f"📁 Model saved to: {final_path}")
            logger.info("=" * 70)
            
            # Print benchmark summary
            if config.get('benchmarking.enabled'):
                benchmark_file = config.get('benchmarking.output_file')
                if os.path.exists(benchmark_file):
                    with open(benchmark_file, 'r') as f:
                        results = json.load(f)
                    
                    print("\n" + "=" * 70)
                    print("📊 PERFORMANCE SUMMARY")
                    print("=" * 70)
                    print(f"Training Time: {results['total_training_time_seconds']:.2f} seconds")
                    print(f"Total Steps: {results['total_steps']}")
                    print(f"Avg Time/Step: {results['avg_time_per_step_seconds']:.3f} seconds")
                    print(f"Max GPU Memory: {results['max_gpu_memory_gb']:.2f} GB")
                    if 'tokens_per_second' in results:
                        print(f"Throughput: {results['tokens_per_second']:.0f} tokens/second")
                    print(f"Final Loss: {results['final_loss']:.4f}")
                    print("=" * 70)
    
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
