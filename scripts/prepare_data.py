import json
import os
import sys
import logging
from datasets import load_dataset
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
        
        dataset_config = config.get_dataset_config()
        storage_config = config.get_storage_config()
        
        dataset_name = dataset_config['name']
        split = dataset_config['split']
        max_samples = dataset_config.get('max_samples')
        streaming = dataset_config.get('streaming', True)
        
        logger.info(f"⬇️  Loading dataset: {dataset_name}")
        logger.info(f"📊 Max samples: {max_samples if max_samples else 'all (60k)'}")
        
        # Load dataset
        try:
            ds = load_dataset(dataset_name, split=split, streaming=streaming)
        except Exception as e:
            logger.error(f"❌ Failed to load dataset from HuggingFace: {e}")
            logger.error("💡 Check your internet connection and HuggingFace access")
            sys.exit(1)
        
        # Prepare output path
        output_dir = storage_config['data_dir']
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "real_train.jsonl")
        
        # Check disk space
        stat = os.statvfs(output_dir)
        free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        logger.info(f"💾 Available disk space: {free_space_gb:.2f} GB")
        
        if free_space_gb < 1:
            logger.warning("⚠️  Low disk space! Dataset preparation may fail.")
        
        # Process dataset
        count = 0
        errors = 0
        
        logger.info(f"🔄 Processing dataset...")
        
        try:
            with open(output_path, "w") as f:
                for row in ds:
                    if max_samples and count >= max_samples:
                        break
                    
                    try:
                        # The dataset has a 'conversations' column (ShareGPT format)
                        # We standardize it to 'messages' for our Trainer
                        # Format: [{"from": "human", "value": "..."}] -> [{"role": "user", "content": "..."}]
                        messages = []
                        for msg in row['conversations']:
                            role = "user" if msg['from'] == "human" else "assistant"
                            # System prompts are often embedded in the first human message in ShareGPT
                            if msg['from'] == "system":
                                role = "system"
                            
                            messages.append({"role": role, "content": msg['value']})
                        
                        # Write to JSONL
                        f.write(json.dumps({"messages": messages}) + "\n")
                        count += 1
                        
                        # Progress indicator
                        if count % 1000 == 0:
                            logger.info(f"  Processed {count} samples...")
                    
                    except Exception as e:
                        errors += 1
                        if errors < 10:  # Only log first 10 errors
                            logger.warning(f"⚠️  Error processing row {count}: {e}")
                        continue
        
        except KeyboardInterrupt:
            logger.warning("⚠️  Interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Failed to write dataset: {e}")
            sys.exit(1)
        
        # Summary
        logger.info("=" * 70)
        logger.info("✅ DATASET PREPARATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"📁 Output: {output_path}")
        logger.info(f"📊 Samples: {count}")
        if errors > 0:
            logger.warning(f"⚠️  Errors: {errors} samples skipped")
        
        # File size
        file_size_mb = os.path.getsize(output_path) / (1024**2)
        logger.info(f"💾 File size: {file_size_mb:.2f} MB")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()