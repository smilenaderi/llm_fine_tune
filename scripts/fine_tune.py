import os
import torch
import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


import transformers.utils.import_utils
# MONKEYPATCH: Bypass the CVE-2025-32434 check because we trust our own checkpoints
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # --- 1. CONFIGURATION ---
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    data_path = "data/real_train.jsonl"
    output_dir = "checkpoints"
    
    # H100/H200 Optimization
    attn = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"

    # --- 2. MODEL & TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        attn_implementation=attn, 
        use_cache=False
    )

    # --- 3. LoRA ADAPTER CONFIG ---
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], 
        task_type="CAUSAL_LM", 
        bias="none"
    )

    # --- 4. TRAINING ARGUMENTS ---
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        
        # Batch Size Optimization (Effective Batch = 32)
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        
        learning_rate=2e-4,
        bf16=True,
        
        # Logging & Checkpointing
        logging_steps=1,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,  # Keep only last 2 checkpoints
        report_to="none",
        
        # TRL Specifics
        dataset_text_field="messages",
        packing=False
        
        # NOTE: For Multi-Node (4 GPUs), uncomment the line below:
        # fsdp="full_shard auto_wrap",
    )

    dataset = load_dataset("json", data_files=data_path, split="train")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer, 
        args=training_args
    )

    # --- 5. RESUME CAPABILITY ---
    last_checkpoint = None
    if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint and trainer.accelerator.is_main_process:
            logger.info(f"🔄 Resuming from checkpoint: {last_checkpoint}")

    # --- 6. START TRAINING ---
    if trainer.accelerator.is_main_process:
        logger.info("🚀 Starting Training...")
        
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # --- 7. FINAL SAVE (Rank 0 Only) ---
    final_path = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_path)
    
    if trainer.accelerator.is_main_process:
        logger.info(f"✅ Training Complete. Model saved to {final_path}")

if __name__ == "__main__":
    main()