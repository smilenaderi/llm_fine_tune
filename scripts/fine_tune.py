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
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        logger.info("🚀 Training started")
    
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
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        benchmark_file = self.config.get('benchmarking.output_file')
        
        # Add job ID to filename
        base_name = os.path.splitext(benchmark_file)[0]
        ext = os.path.splitext(benchmark_file)[1]
        benchmark_file = f"{base_name}_{job_id}{ext}"
        
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
        use_flash = model_config.get('use_flash_attention', True) and compute_capability >= 8
        attn_implementation = "flash_attention_2" if use_flash else "eager"
        
        if use_flash:
            logger.info("⚡ Using Flash Attention 2")
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
    
    # Base arguments
    args = {
        'output_dir': storage_config['checkpoint_dir'],
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
        output_dir = config.get('storage.checkpoint_dir')
        
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
