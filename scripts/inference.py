import os
import sys
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config_loader import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    try:
        # Load configuration
        logger.info("📋 Loading configuration...")
        config = load_config()
        
        model_config = config.get_model_config()
        storage_config = config.get_storage_config()
        inference_config = config.get_inference_config()
        
        # Paths
        base_model_id = model_config['model_id']
        adapter_path = os.path.join(storage_config['checkpoint_dir'], "final_adapter")
        
        # Check if adapter exists
        if not os.path.exists(adapter_path):
            logger.error(f"❌ Adapter not found at: {adapter_path}")
            logger.error("💡 Run training first: sbatch scripts/submit_job.sh")
            sys.exit(1)
        
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
        
        # Get test prompts from config
        test_prompts = inference_config.get('test_prompts', [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain the benefits of Nebius AI Cloud in one sentence."}
        ])
        
        # Prepare Input (Chat Format)
        logger.info("🤖 Generating response...")
        text = tokenizer.apply_chat_template(
            test_prompts,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(base_model.device)
        
        # Generate
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=inference_config.get('max_new_tokens', 256),
                    temperature=inference_config.get('temperature', 0.7),
                    top_p=inference_config.get('top_p', 0.9),
                    do_sample=inference_config.get('do_sample', True)
                )
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            sys.exit(1)
        
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
        print(f"User: {test_prompts[-1]['content']}")
        print(f"\nAssistant: {response}")
        print("=" * 70)
        
        logger.info("✅ Inference completed successfully")
    
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()