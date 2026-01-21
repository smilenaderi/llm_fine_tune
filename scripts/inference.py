import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    # Paths
    base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = "checkpoints/final_adapter"

    print(f"Loading base model: {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Load Your Trained Adapter
    print(f"Loading adapter from: {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ Adapter loaded successfully!")
    except Exception as e:
        print(f"❌ Could not load adapter. Did you train yet? Error: {e}")
        return

    # Prepare Input (Chat Format)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the benefits of Nebius AI Cloud in one sentence."}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(base_model.device)

    # Generate
    print("🤖 Generating response...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        
    # Decode
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n" + "="*50)
    print(f"OUTPUT:\n{response}")
    print("="*50)

if __name__ == "__main__":
    main()