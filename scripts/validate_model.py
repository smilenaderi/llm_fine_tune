"""
Model validation script
Tests the fine-tuned model on function calling tasks
"""
import os
import sys
import json
import time
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


# Function calling test cases
TEST_CASES = [
    {
        "name": "Flight Booking",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant with function calling capabilities."},
            {"role": "user", "content": "Book a flight from New York to San Francisco on March 15th for 2 passengers in business class."}
        ],
        "expected_keywords": ["flight", "book", "NYC", "SF", "March", "passenger"]
    },
    {
        "name": "Calendar Reminder",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant with function calling capabilities."},
            {"role": "user", "content": "Set a reminder for my dentist appointment tomorrow at 2 PM."}
        ],
        "expected_keywords": ["reminder", "dentist", "tomorrow", "2 PM"]
    },
    {
        "name": "Restaurant Search",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant with function calling capabilities."},
            {"role": "user", "content": "Find Italian restaurants near me that are open now and have outdoor seating."}
        ],
        "expected_keywords": ["restaurant", "Italian", "near", "open", "outdoor"]
    },
    {
        "name": "Weather Query",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant with function calling capabilities."},
            {"role": "user", "content": "What's the weather forecast for London next week?"}
        ],
        "expected_keywords": ["weather", "forecast", "London", "week"]
    },
    {
        "name": "Email Composition",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant with function calling capabilities."},
            {"role": "user", "content": "Send an email to john@example.com with subject 'Meeting Tomorrow' and tell him I'll be 15 minutes late."}
        ],
        "expected_keywords": ["email", "send", "john", "meeting", "late"]
    }
]


def load_model_and_tokenizer(config):
    """Load base model, tokenizer, and adapter
    
    Note: This function uses the model_id from the config that was passed in.
    The main() function ensures we load the job-specific config snapshot
    that was saved during training, so we always use the correct base model.
    """
    model_config = config.get_model_config()
    storage_config = config.get_storage_config()
    
    base_model_id = model_config['model_id']
    
    # Get job ID and construct path to adapter
    job_id = os.environ.get('SLURM_JOB_ID', f'local_{int(time.time())}')
    job_checkpoint_dir = os.path.join(storage_config['checkpoint_dir'], f'job_{job_id}')
    adapter_path = os.path.join(job_checkpoint_dir, "final_adapter")
    
    logger.info(f"📥 Loading base model: {base_model_id}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            cache_dir=model_config.get('cache_dir'),
            trust_remote_code=True
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=model_config.get('cache_dir'),
            trust_remote_code=True
        )
        
        # Load adapter
        logger.info(f"📥 Loading adapter from: {adapter_path}")
        if not os.path.exists(adapter_path):
            logger.error(f"❌ Adapter not found at {adapter_path}")
            logger.error(f"💡 Expected location: {job_checkpoint_dir}/final_adapter")
            logger.error("💡 Run training first: sbatch scripts/submit_job.sh")
            sys.exit(1)
        
        model = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("✅ Model and adapter loaded successfully")
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        sys.exit(1)


def generate_response(model, tokenizer, messages, config):
    """Generate response for given messages"""
    inference_config = config.get_inference_config()
    
    # Format messages using chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=inference_config.get('max_new_tokens', 256),
            temperature=inference_config.get('temperature', 0.7),
            top_p=inference_config.get('top_p', 0.9),
            do_sample=inference_config.get('do_sample', True)
        )
    
    # Decode
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response


def validate_response(response, expected_keywords):
    """Check if response contains expected keywords"""
    response_lower = response.lower()
    found_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
    score = len(found_keywords) / len(expected_keywords)
    return score, found_keywords


def run_validation(model, tokenizer, config):
    """Run validation on all test cases"""
    logger.info("=" * 70)
    logger.info("🧪 RUNNING VALIDATION TESTS")
    logger.info("=" * 70)
    
    results = []
    total_score = 0
    
    for i, test_case in enumerate(TEST_CASES, 1):
        logger.info(f"\n📝 Test {i}/{len(TEST_CASES)}: {test_case['name']}")
        logger.info(f"   User: {test_case['messages'][-1]['content']}")
        
        # Generate response
        response = generate_response(model, tokenizer, test_case['messages'], config)
        
        # Validate response
        score, found_keywords = validate_response(response, test_case['expected_keywords'])
        
        # Store results
        result = {
            'test_name': test_case['name'],
            'user_input': test_case['messages'][-1]['content'],
            'response': response,
            'expected_keywords': test_case['expected_keywords'],
            'found_keywords': found_keywords,
            'score': score
        }
        results.append(result)
        total_score += score
        
        # Print results
        logger.info(f"   Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        logger.info(f"   Score: {score:.1%} ({len(found_keywords)}/{len(test_case['expected_keywords'])} keywords found)")
        
        if score < 0.5:
            logger.warning(f"   ⚠️  Low score - model may need more training")
    
    # Calculate average score
    avg_score = total_score / len(TEST_CASES)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total Tests: {len(TEST_CASES)}")
    logger.info(f"Average Score: {avg_score:.1%}")
    
    if avg_score >= 0.8:
        logger.info("✅ EXCELLENT - Model performs well on function calling tasks")
    elif avg_score >= 0.6:
        logger.info("✓ GOOD - Model shows reasonable function calling capability")
    elif avg_score >= 0.4:
        logger.info("⚠️  FAIR - Model needs improvement")
    else:
        logger.info("❌ POOR - Model requires more training")
    
    logger.info("=" * 70)
    
    # Save results
    storage_config = config.get_storage_config()
    job_id = os.environ.get('SLURM_JOB_ID', f'local_{int(time.time())}')
    job_log_dir = os.path.join(storage_config['log_dir'], f'job_{job_id}')
    os.makedirs(job_log_dir, exist_ok=True)
    results_file = os.path.join(job_log_dir, 'validation_results.json')
    
    with open(results_file, 'w') as f:
        json.dump({
            'job_id': job_id,
            'average_score': avg_score,
            'total_tests': len(TEST_CASES),
            'results': results
        }, f, indent=2)
    
    logger.info(f"📁 Detailed results saved to: {results_file}")
    
    return avg_score


def main():
    try:
        # Load configuration from job-specific copy
        logger.info("📋 Loading configuration...")
        
        # Get job ID and construct path to job-specific config
        job_id = os.environ.get('SLURM_JOB_ID', f'local_{int(time.time())}')
        storage_config_path = 'config.yaml'
        
        # Try to load from job-specific config first (saved during training)
        job_config_path = f'logs/job_{job_id}/config.yaml'
        if os.path.exists(job_config_path):
            logger.info(f"✓ Using job-specific config: {job_config_path}")
            config = load_config(job_config_path)
        else:
            logger.warning(f"⚠️  Job-specific config not found at {job_config_path}")
            logger.info(f"   Using main config: {storage_config_path}")
            config = load_config(storage_config_path)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Run validation
        avg_score = run_validation(model, tokenizer, config)
        
        # Exit with appropriate code
        if avg_score >= 0.6:
            sys.exit(0)
        else:
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
