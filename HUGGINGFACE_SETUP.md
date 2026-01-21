# Hugging Face Setup (Optional)

## Why You Might Need This

Hugging Face authentication is only required if:
- You're downloading gated models (e.g., Llama, Mistral)
- You want to push models to Hugging Face Hub
- You're using private models from your account

For public models like Qwen2.5-7B-Instruct, authentication is **optional**.

## Setup Instructions

### 1. Get Your Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "nebius-training")
4. Select permissions:
   - **Read** - for downloading models
   - **Write** - if you want to push models
5. Copy the token (starts with `hf_...`)

### 2. Login on Nebius

```bash
# Activate your environment
source envs/llm-env/bin/activate

# Login to Hugging Face
huggingface-cli login

# Paste your token when prompted
# Choose whether to add token to git credentials (recommended: yes)
```

### 3. Verify Authentication

```bash
huggingface-cli whoami
```

## Alternative: Environment Variable

Instead of interactive login, you can set an environment variable:

```bash
export HF_TOKEN="hf_your_token_here"
```

Add this to your `~/.bashrc` or job scripts for persistence.

## Troubleshooting

**"Repository not found" error:**
- Check if the model requires authentication
- Verify your token has the correct permissions
- Ensure you've accepted any model license agreements on Hugging Face

**Token not persisting:**
- Make sure you chose "yes" when asked to add to git credentials
- Or use the environment variable method above
