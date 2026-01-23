#!/bin/bash
# Setup script for Nebius SLURM cluster
# Run this on the login node after first connection

set -e  # Exit on error

echo "🚀 Setting up LLM Fine-Tuning Environment on Nebius"
echo "=================================================="

# 1. Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p /shared/llm-fine-tune/scripts
mkdir -p /shared/llm-fine-tune/data
mkdir -p /shared/llm-fine-tune/logs
mkdir -p /shared/llm-fine-tune/checkpoints
mkdir -p /shared/llm-fine-tune/envs

cd /shared/llm-fine-tune

# 2. Create Virtual Environment
echo ""
echo "🐍 Creating Python virtual environment..."
python3 -m venv envs/llm-env
source envs/llm-env/bin/activate

# 3. Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# 4. Install PyTorch with CUDA 12.1 support
echo ""
echo "🔥 Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install build dependencies
echo ""
echo "🔨 Installing build dependencies..."
pip install packaging ninja wheel

# 6. Install ML libraries
echo ""
echo "📚 Installing ML libraries..."
pip install transformers datasets peft accelerate trl bitsandbytes pyyaml

# 7. Install Flash Attention (H100/H200 optimization)
echo ""
echo "⚡ Installing Flash Attention 2..."
# Set pip cache and temp to /shared to avoid cross-device link errors
export PIP_CACHE_DIR=/shared/.pip-cache
export TMPDIR=/shared/tmp
mkdir -p $PIP_CACHE_DIR $TMPDIR

# Try installing with prebuilt wheel first
if ! pip install flash-attn --no-build-isolation 2>/dev/null; then
    echo "⚠️  Prebuilt wheel failed, trying direct wheel download..."
    # Fallback: Download and install wheel directly
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    wget -q $WHEEL_URL -O /shared/tmp/flash_attn.whl
    pip install /shared/tmp/flash_attn.whl
    rm -f /shared/tmp/flash_attn.whl
fi

# 8. Install monitoring tools
echo ""
echo "📊 Installing monitoring tools..."
pip install nvitop tensorboard

# 9. Hugging Face login
echo ""
echo "🤗 Hugging Face Authentication (Optional)"
echo "Only required for gated models (e.g., Llama, Mistral)."
echo "Qwen2.5-7B-Instruct is public and doesn't require authentication."
echo "Get your token from: https://huggingface.co/settings/tokens"
echo ""
read -p "Do you want to login to Hugging Face now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    huggingface-cli login
else
    echo "⚠️  Skipping Hugging Face login."
    echo "   If needed later, run: source envs/llm-env/bin/activate && huggingface-cli login"
    echo "   See HUGGINGFACE_SETUP.md for more details."
fi

# 10. Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"

# Check Flash Attention (optional)
if python -c "import flash_attn" 2>/dev/null; then
    python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
else
    echo "⚠️  Flash Attention not installed (optional optimization)"
fi

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml to choose your preset"
echo ""
echo "2. Submit training job:"
echo "   sbatch scripts/submit_job.sh"
echo ""
echo "3. Monitor training:"
echo "   source scripts/monitor.sh"
echo "   watch_latest"
echo "=================================================="
