import json
from datasets import load_dataset

def main():
    print("⬇️  Streaming real function-calling data (xLAM)...")
    
    # We use a community mirror of the Salesforce xLAM dataset that is already formatted 
    # compatible with HuggingFace 'messages' (ShareGPT style).
    # Source: Beryex/xlam-function-calling-60k-sharegpt
    ds = load_dataset("Beryex/xlam-function-calling-60k-sharegpt", split="train", streaming=True)

    output_path = "data/real_train.jsonl"
    count = 0
    limit = 20000  # We only take 20k rows for the PoC to be fast

    print(f"🔄 Processing {limit} rows...")
    with open(output_path, "w") as f:
        for row in ds:
            if count >= limit:
                break
            
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

    print(f"✅ Success! Saved {count} real examples to {output_path}")

if __name__ == "__main__":
    main()