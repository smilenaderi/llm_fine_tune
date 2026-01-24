import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config_loader import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_job(checkpoint_dir):
    """Find the most recent job with a final_adapter"""
    job_dirs = [d for d in Path(checkpoint_dir).glob("job_*") if d.is_dir()]
    
    if not job_dirs:
        return None
    
    # Sort by job number (extract number from job_XX)
    job_dirs.sort(key=lambda x: int(x.name.split('_')[1]), reverse=True)
    
    # Find first job with final_adapter
    for job_dir in job_dirs:
        adapter_path = job_dir / "final_adapter"
        if adapter_path.exists():
            return job_dir.name.split('_')[1]  # Return just the number
    
    return None


def find_adapter_path(checkpoint_dir, job_id=None):
    """Find adapter path for a specific job or latest job"""
    if job_id:
        # Use specific job
        adapter_path = os.path.join(checkpoint_dir, f"job_{job_id}", "final_adapter")
        if os.path.exists(adapter_path):
            return adapter_path, job_id
        else:
            logger.error(f"❌ Adapter not found for job {job_id}")
            logger.error(f"   Expected: {adapter_path}")
            return None, None
    else:
        # Find latest job
        latest_job = find_latest_job(checkpoint_dir)
        if latest_job:
            adapter_path = os.path.join(checkpoint_dir, f"job_{latest_job}", "final_adapter")
            return adapter_path, latest_job
        else:
            # Fallback to old location
            adapter_path = os.path.join(checkpoint_dir, "final_adapter")
            if os.path.exists(adapter_path):
                return adapter_path, "legacy"
            return None, None


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned model')
    parser.add_argument('--job-id', type=str, help='Job ID to use (e.g., 36)', default=None)
    parser.add_argument('--prompt', type=str, help='Custom prompt for inference', default=None)
    parser.add_argument('--prompts-file', type=str, help='File with multiple prompts (one per line)', default=None)
    parser.add_argument('--config', type=str, help='Path to config file', default='config.yaml')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode - enter prompts one by one')
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info("📋 Loading configuration...")
        
        # If job-id is specified, try to load config from that job's snapshot
        config_path = args.config
        if args.job_id:
            job_config_path = f"logs/job_{args.job_id}/config.yaml"
            if os.path.exists(job_config_path):
                config_path = job_config_path
                logger.info(f"✓ Using config snapshot from job {args.job_id}")
            else:
                logger.warning(f"⚠️  Config snapshot not found for job {args.job_id}, using current config.yaml")
        
        config = load_config(config_path)
        
        model_config = config.get_model_config()
        storage_config = config.get_storage_config()
        inference_config = config.get_inference_config()
        
        # Find adapter path
        adapter_path, job_id = find_adapter_path(storage_config['checkpoint_dir'], args.job_id)
        
        if not adapter_path:
            logger.error("❌ No trained adapter found!")
            logger.error("💡 Available options:")
            logger.error("   1. Run training: sbatch scripts/submit_job.sh")
            logger.error("   2. Specify job ID: sbatch scripts/run_inf.sh <job_id>")
            
            # List available jobs
            checkpoint_dir = Path(storage_config['checkpoint_dir'])
            job_dirs = sorted([d for d in checkpoint_dir.glob("job_*") if d.is_dir()])
            if job_dirs:
                logger.error("\n   Available trained jobs:")
                for job_dir in job_dirs:
                    adapter = job_dir / "final_adapter"
                    if adapter.exists():
                        logger.error(f"     - {job_dir.name.split('_')[1]}")
            sys.exit(1)
        
        logger.info(f"✓ Using adapter from job {job_id}: {adapter_path}")
        
        # Read the actual base model from adapter config (most reliable)
        adapter_config_file = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config_file):
            import json
            with open(adapter_config_file, 'r') as f:
                adapter_config_data = json.load(f)
            base_model_id = adapter_config_data.get('base_model_name_or_path')
            logger.info(f"✓ Detected base model from adapter: {base_model_id}")
        else:
            # Fallback to config
            base_model_id = model_config['model_id']
            logger.warning(f"⚠️  Using model from config.yaml: {base_model_id}")
            logger.warning("   This may cause mismatch errors if adapter was trained with different model")
        
        logger.info(f"📥 Loading base model: {base_model_id}")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                cache_dir=model_config.get('cache_dir'),
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {e}")
            sys.exit(1)
        
        # Load Base Model
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=model_config.get('cache_dir'),
                trust_remote_code=True
            )
            logger.info("✓ Base model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load base model: {e}")
            sys.exit(1)
        
        # Load Trained Adapter
        logger.info(f"📥 Loading adapter from: {adapter_path}")
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            logger.info("✅ Adapter loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load adapter: {e}")
            sys.exit(1)
        
        # Determine prompts to process
        prompts_to_run = []
        
        if args.prompts_file:
            # Load prompts from file
            logger.info(f"📄 Loading prompts from: {args.prompts_file}")
            with open(args.prompts_file, 'r') as f:
                prompts_to_run = [line.strip() for line in f if line.strip()]
            logger.info(f"✓ Loaded {len(prompts_to_run)} prompts")
        elif args.prompt:
            # Single custom prompt
            prompts_to_run = [args.prompt]
        elif args.interactive:
            # Interactive mode
            logger.info("🎮 Interactive mode - enter prompts (Ctrl+D or empty line to finish)")
            while True:
                try:
                    prompt = input("\nPrompt: ").strip()
                    if not prompt:
                        break
                    prompts_to_run.append(prompt)
                except (EOFError, KeyboardInterrupt):
                    break
            if not prompts_to_run:
                logger.error("❌ No prompts provided")
                sys.exit(1)
        else:
            # Use default from config
            test_prompts = inference_config.get('test_prompts', [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Book a flight from New York to San Francisco on March 15th for 2 passengers."}
            ])
            # Extract just the user prompt
            prompts_to_run = [test_prompts[-1]['content']]
        
        # Process each prompt
        for idx, user_prompt in enumerate(prompts_to_run, 1):
            if len(prompts_to_run) > 1:
                logger.info(f"\n{'='*70}")
                logger.info(f"Processing prompt {idx}/{len(prompts_to_run)}")
                logger.info(f"{'='*70}")
            
            # Prepare messages
            test_prompts = [
                {"role": "system", "content": "You are a helpful AI assistant with function calling capabilities."},
                {"role": "user", "content": user_prompt}
            ]
            
            # Prepare Input (Chat Format)
            logger.info("🤖 Generating response...")
            text = tokenizer.apply_chat_template(
                test_prompts,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(base_model.device)
            
            # Generate with fallback for cache compatibility issues
            generation_kwargs = {
                'max_new_tokens': inference_config.get('max_new_tokens', 256),
                'temperature': inference_config.get('temperature', 0.7),
                'top_p': inference_config.get('top_p', 0.9),
                'do_sample': inference_config.get('do_sample', True)
            }
            
            try:
                with torch.no_grad():
                    # Try with cache enabled (default behavior)
                    generated_ids = model.generate(**inputs, **generation_kwargs)
            except AttributeError as e:
                if 'seen_tokens' in str(e) or 'DynamicCache' in str(e):
                    # Fallback: disable cache for models with cache compatibility issues
                    logger.warning(f"⚠️  Cache compatibility issue detected, disabling cache: {e}")
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, **generation_kwargs, use_cache=False)
                else:
                    raise
            except Exception as e:
                logger.error(f"❌ Generation failed: {e}")
                continue
            
            # Decode
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Print results
            print("\n" + "=" * 70)
            print("INFERENCE RESULT")
            print("=" * 70)
            print(f"User: {user_prompt}")
            print(f"\nAssistant: {response}")
            print("=" * 70)
        
        logger.info("✅ Inference completed successfully")
    
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()