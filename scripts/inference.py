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


def find_latest_run(checkpoint_dir):
    """Find the most recent run with a final_adapter"""
    run_dirs = [d for d in Path(checkpoint_dir).iterdir() if d.is_dir()]
    
    if not run_dirs:
        return None
    
    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Find first run with final_adapter
    for run_dir in run_dirs:
        adapter_path = run_dir / "final_adapter"
        if adapter_path.exists():
            return run_dir.name
    
    return None


def find_adapter_path(checkpoint_dir, run_id=None):
    """Find adapter path for a specific run or latest run"""
    if run_id:
        # Try with run_id directly first
        adapter_path = os.path.join(checkpoint_dir, run_id, "final_adapter")
        if os.path.exists(adapter_path):
            return adapter_path, run_id
        
        # Try with job_ prefix for backward compatibility
        adapter_path = os.path.join(checkpoint_dir, f"job_{run_id}", "final_adapter")
        if os.path.exists(adapter_path):
            return adapter_path, f"job_{run_id}"
        
        logger.error(f"❌ Adapter not found for run: {run_id}")
        logger.error(f"   Tried: {checkpoint_dir}/{run_id}/final_adapter")
        logger.error(f"   Tried: {checkpoint_dir}/job_{run_id}/final_adapter")
        return None, None
    else:
        # Find latest run
        latest_run = find_latest_run(checkpoint_dir)
        if latest_run:
            adapter_path = os.path.join(checkpoint_dir, latest_run, "final_adapter")
            return adapter_path, latest_run
        else:
            # Fallback to old location
            adapter_path = os.path.join(checkpoint_dir, "final_adapter")
            if os.path.exists(adapter_path):
                return adapter_path, "legacy"
            return None, None


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned model')
    parser.add_argument('--run-id', type=str, help='Training run ID to use (e.g., "qwen32b_exp1" or "job_36")', default=None)
    parser.add_argument('--job-id', type=str, help='[Deprecated] Use --run-id instead', default=None)
    parser.add_argument('--prompt', type=str, help='Custom prompt for inference', default=None)
    parser.add_argument('--prompts-file', type=str, help='File with multiple prompts (one per line)', default=None)
    parser.add_argument('--config', type=str, help='Path to config file', default='config.yaml')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode - enter prompts one by one')
    args = parser.parse_args()
    
    # Handle backward compatibility
    if args.job_id and not args.run_id:
        args.run_id = args.job_id
        logger.warning("⚠️  --job-id is deprecated, use --run-id instead")
    
    try:
        # Load configuration
        logger.info("📋 Loading configuration...")
        
        # If run-id is specified, try to load config from that run's snapshot
        config_path = args.config
        if args.run_id:
            run_config_path = f"logs/{args.run_id}/config.yaml"
            if os.path.exists(run_config_path):
                config_path = run_config_path
                logger.info(f"✓ Using config snapshot from run: {args.run_id}")
            else:
                logger.warning(f"⚠️  Config snapshot not found for run {args.run_id}, using current config.yaml")
        
        config = load_config(config_path)
        
        model_config = config.get_model_config()
        storage_config = config.get_storage_config()
        inference_config = config.get_inference_config()
        
        # Find adapter path
        adapter_path, run_id = find_adapter_path(storage_config['checkpoint_dir'], args.run_id)
        
        if not adapter_path:
            logger.error("❌ No trained adapter found!")
            logger.error("💡 Available options:")
            logger.error("   1. Run training: sbatch scripts/submit_job.sh")
            logger.error("   2. Specify run ID: python scripts/inference.py --run-id <run_id>")
            
            # List available runs
            checkpoint_dir = Path(storage_config['checkpoint_dir'])
            run_dirs = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir()])
            if run_dirs:
                logger.error("\n   Available trained runs:")
                for run_dir in run_dirs:
                    adapter = run_dir / "final_adapter"
                    if adapter.exists():
                        logger.error(f"     - {run_dir.name}")
            sys.exit(1)
        
        logger.info(f"✓ Using adapter from run: {run_id}")
        logger.info(f"📁 Path: {adapter_path}")
        
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