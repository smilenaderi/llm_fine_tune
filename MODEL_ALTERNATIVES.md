# Model and Dataset Alternatives

This guide helps you choose the right model and dataset for your use case.

## Model Options

### Small Models (1-3B parameters)

#### Qwen/Qwen2.5-1.5B-Instruct
- **Size**: 1.5B parameters (~3GB)
- **VRAM**: 6-8GB per GPU
- **Training Time**: 2-3x faster than 7B
- **Best For**: Quick PoC, testing, resource-constrained environments
- **Quality**: Good for simple tasks
- **HuggingFace Auth**: Not required

```yaml
model:
  model_id: "Qwen/Qwen2.5-1.5B-Instruct"
training:
  per_device_train_batch_size: 16
lora:
  r: 8
```

#### microsoft/Phi-3-mini-4k-instruct
- **Size**: 3.8B parameters (~8GB)
- **VRAM**: 10-12GB per GPU
- **Training Time**: 1.5-2x faster than 7B
- **Best For**: High quality with small size
- **Quality**: Excellent reasoning capabilities
- **HuggingFace Auth**: Not required

```yaml
model:
  model_id: "microsoft/Phi-3-mini-4k-instruct"
training:
  per_device_train_batch_size: 12
lora:
  r: 16
```

#### google/gemma-2-2b-it
- **Size**: 2B parameters (~4GB)
- **VRAM**: 8-10GB per GPU
- **Training Time**: 2-3x faster than 7B
- **Best For**: Google ecosystem, instruction following
- **Quality**: Good for chat and instruction tasks
- **HuggingFace Auth**: Required (accept license)

```yaml
model:
  model_id: "google/gemma-2-2b-it"
training:
  per_device_train_batch_size: 16
lora:
  r: 8
```

---

### Medium Models (7-8B parameters) - RECOMMENDED

#### Qwen/Qwen2.5-7B-Instruct ⭐ RECOMMENDED
- **Size**: 7B parameters (~14GB)
- **VRAM**: 16-20GB per GPU
- **Training Time**: Baseline
- **Best For**: Balanced quality and speed
- **Quality**: Excellent for most tasks
- **HuggingFace Auth**: Not required

```yaml
model:
  model_id: "Qwen/Qwen2.5-7B-Instruct"
training:
  per_device_train_batch_size: 8
lora:
  r: 16
```

#### meta-llama/Llama-3.1-8B-Instruct
- **Size**: 8B parameters (~16GB)
- **VRAM**: 18-24GB per GPU
- **Training Time**: Similar to Qwen 7B
- **Best For**: Meta ecosystem, strong reasoning
- **Quality**: Excellent, industry standard
- **HuggingFace Auth**: Required (accept license)

```yaml
model:
  model_id: "meta-llama/Llama-3.1-8B-Instruct"
training:
  per_device_train_batch_size: 8
lora:
  r: 16
```

#### mistralai/Mistral-7B-Instruct-v0.3
- **Size**: 7B parameters (~14GB)
- **VRAM**: 16-20GB per GPU
- **Training Time**: Similar to Qwen 7B
- **Best For**: European AI, strong performance
- **Quality**: Excellent, competitive with Llama
- **HuggingFace Auth**: Required (accept license)

```yaml
model:
  model_id: "mistralai/Mistral-7B-Instruct-v0.3"
training:
  per_device_train_batch_size: 8
lora:
  r: 16
```

---

### Large Models (14-70B parameters)

#### Qwen/Qwen2.5-14B-Instruct
- **Size**: 14B parameters (~28GB)
- **VRAM**: 32-40GB per GPU
- **Training Time**: 2x slower than 7B
- **Best For**: High quality requirements
- **Quality**: Excellent, near GPT-3.5 level
- **HuggingFace Auth**: Not required

```yaml
model:
  model_id: "Qwen/Qwen2.5-14B-Instruct"
training:
  per_device_train_batch_size: 4
lora:
  r: 32
distributed:
  fsdp:
    enabled: true
```

#### meta-llama/Llama-3.1-70B-Instruct
- **Size**: 70B parameters (~140GB)
- **VRAM**: 80GB per GPU (requires H100/H200)
- **Training Time**: 10x slower than 7B
- **Best For**: Maximum quality, production systems
- **Quality**: GPT-4 level performance
- **HuggingFace Auth**: Required (accept license)

```yaml
cluster:
  nodes: 2
  gpus_per_node: 4
model:
  model_id: "meta-llama/Llama-3.1-70B-Instruct"
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
lora:
  r: 64
distributed:
  fsdp:
    enabled: true
    cpu_offload: true
```

#### mistralai/Mixtral-8x7B-Instruct-v0.1
- **Size**: 47B parameters (8x7B MoE, ~94GB)
- **VRAM**: 60-80GB per GPU
- **Training Time**: 4-5x slower than 7B
- **Best For**: Mixture of Experts architecture
- **Quality**: Excellent, efficient inference
- **HuggingFace Auth**: Required (accept license)

```yaml
cluster:
  nodes: 1
  gpus_per_node: 4
model:
  model_id: "mistralai/Mixtral-8x7B-Instruct-v0.1"
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
lora:
  r: 32
distributed:
  fsdp:
    enabled: true
```

---

## Dataset Options

### Function Calling Datasets

#### Beryex/xlam-function-calling-60k-sharegpt ⭐ RECOMMENDED
- **Size**: 60,000 samples
- **Format**: ShareGPT (conversations)
- **Quality**: High quality, curated
- **Best For**: Function calling, tool use
- **Training Time**: 45-90 min (full dataset)

```yaml
dataset:
  name: "Beryex/xlam-function-calling-60k-sharegpt"
  max_samples: 60000  # or 20000 for fast PoC
```

#### glaiveai/glaive-function-calling-v2
- **Size**: 113,000 samples
- **Format**: Messages format
- **Quality**: Very high quality, diverse
- **Best For**: Comprehensive function calling
- **Training Time**: 90-180 min (full dataset)

```yaml
dataset:
  name: "glaiveai/glaive-function-calling-v2"
  max_samples: 60000  # Recommended subset
```

#### NousResearch/hermes-function-calling-v1
- **Size**: 115,000 samples
- **Format**: Messages format
- **Quality**: High quality, research-grade
- **Best For**: Advanced function calling
- **Training Time**: 90-180 min (full dataset)

```yaml
dataset:
  name: "NousResearch/hermes-function-calling-v1"
  max_samples: 60000  # Recommended subset
```

---

### General Instruction Datasets

#### teknium/GPTeacher-General-Instruct
- **Size**: 50,000 samples
- **Format**: Instruction-response pairs
- **Quality**: High quality, GPT-4 generated
- **Best For**: General instruction following
- **Training Time**: 40-80 min (full dataset)

```yaml
dataset:
  name: "teknium/GPTeacher-General-Instruct"
  max_samples: 50000
```

#### HuggingFaceH4/ultrachat_200k
- **Size**: 200,000 samples
- **Format**: Multi-turn conversations
- **Quality**: High quality, diverse topics
- **Best For**: Conversational AI
- **Training Time**: 3-6 hours (full dataset)

```yaml
dataset:
  name: "HuggingFaceH4/ultrachat_200k"
  max_samples: 60000  # Recommended subset
```

#### Open-Orca/OpenOrca
- **Size**: 4,200,000 samples
- **Format**: Instruction-response pairs
- **Quality**: Very high quality, comprehensive
- **Best For**: Large-scale training
- **Training Time**: 50-100 hours (full dataset)

```yaml
dataset:
  name: "Open-Orca/OpenOrca"
  max_samples: 100000  # Recommended subset
  streaming: true  # Required for large datasets
```

---

## Configuration Presets

### Preset 1: Fast PoC (5-15 minutes)
**Use Case**: Quick testing, demo preparation

```yaml
cluster:
  nodes: 1
  gpus_per_node: 4

model:
  model_id: "Qwen/Qwen2.5-1.5B-Instruct"

dataset:
  name: "Beryex/xlam-function-calling-60k-sharegpt"
  max_samples: 10000

training:
  num_train_epochs: 1
  per_device_train_batch_size: 16
  learning_rate: 2.0e-4

lora:
  r: 8
  lora_alpha: 16
```

**Expected Results**:
- Training Time: 5-15 minutes
- GPU Memory: ~8GB per GPU
- Quality: Good for simple tasks

---

### Preset 2: Balanced PoC (15-30 minutes) ⭐ RECOMMENDED
**Use Case**: Standard PoC, customer demos

```yaml
cluster:
  nodes: 1
  gpus_per_node: 4

model:
  model_id: "Qwen/Qwen2.5-7B-Instruct"

dataset:
  name: "Beryex/xlam-function-calling-60k-sharegpt"
  max_samples: 20000

training:
  num_train_epochs: 1
  per_device_train_batch_size: 8
  learning_rate: 2.0e-4

lora:
  r: 16
  lora_alpha: 32
```

**Expected Results**:
- Training Time: 15-30 minutes
- GPU Memory: ~18GB per GPU
- Quality: Excellent for most tasks

---

### Preset 3: Production Quality (2-4 hours)
**Use Case**: Production deployment, high quality requirements

```yaml
cluster:
  nodes: 1
  gpus_per_node: 4

model:
  model_id: "Qwen/Qwen2.5-7B-Instruct"

dataset:
  name: "Beryex/xlam-function-calling-60k-sharegpt"
  max_samples: 60000

training:
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 2.0e-4

lora:
  r: 32
  lora_alpha: 64

validation:
  enabled: true
  validation_split: 0.05
```

**Expected Results**:
- Training Time: 2-4 hours
- GPU Memory: ~20GB per GPU
- Quality: Production-ready

---

### Preset 4: Maximum Quality (4-8 hours)
**Use Case**: Best possible quality, research

```yaml
cluster:
  nodes: 1
  gpus_per_node: 4

model:
  model_id: "Qwen/Qwen2.5-14B-Instruct"

dataset:
  name: "glaiveai/glaive-function-calling-v2"
  max_samples: null  # Use all data

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 1.0e-4

lora:
  r: 64
  lora_alpha: 128
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

validation:
  enabled: true
  validation_split: 0.05
```

**Expected Results**:
- Training Time: 4-8 hours
- GPU Memory: ~35GB per GPU
- Quality: State-of-the-art

---

## Quick Comparison Table

| Model | Size | VRAM | Speed | Quality | Auth Required |
|-------|------|------|-------|---------|---------------|
| Qwen 1.5B | 1.5B | 8GB | Fast | Good | No |
| Phi-3 Mini | 3.8B | 12GB | Fast | Excellent | No |
| Qwen 7B ⭐ | 7B | 18GB | Medium | Excellent | No |
| Llama 3.1 8B | 8B | 20GB | Medium | Excellent | Yes |
| Mistral 7B | 7B | 18GB | Medium | Excellent | Yes |
| Qwen 14B | 14B | 35GB | Slow | Excellent | No |
| Llama 3.1 70B | 70B | 80GB | Very Slow | Best | Yes |

| Dataset | Size | Domain | Quality | Time (20k) |
|---------|------|--------|---------|------------|
| xLAM 60k ⭐ | 60k | Function Calling | High | 15-30 min |
| Glaive v2 | 113k | Function Calling | Very High | 15-30 min |
| Hermes v1 | 115k | Function Calling | High | 15-30 min |
| GPTeacher | 50k | General | High | 15-30 min |
| UltraChat | 200k | Conversation | High | 15-30 min |

---

## Recommendations by Use Case

### Quick Demo (< 30 minutes)
- **Model**: Qwen 2.5-7B-Instruct
- **Dataset**: xLAM 60k (20k samples)
- **Config**: Preset 2

### Production Deployment
- **Model**: Qwen 2.5-7B-Instruct or Llama 3.1-8B
- **Dataset**: xLAM 60k or Glaive v2 (full)
- **Config**: Preset 3

### Research / Maximum Quality
- **Model**: Qwen 2.5-14B-Instruct or Llama 3.1-70B
- **Dataset**: Glaive v2 or Hermes v1 (full)
- **Config**: Preset 4

### Resource Constrained
- **Model**: Qwen 2.5-1.5B-Instruct
- **Dataset**: xLAM 60k (10k samples)
- **Config**: Preset 1

---

## How to Switch Models/Datasets

Simply edit `config.yaml`:

```bash
nano config.yaml
```

Change the model:
```yaml
model:
  model_id: "Qwen/Qwen2.5-14B-Instruct"  # Change this line
```

Change the dataset:
```yaml
dataset:
  name: "glaiveai/glaive-function-calling-v2"  # Change this line
  max_samples: 60000  # Adjust sample count
```

Then run training as usual:
```bash
python scripts/prepare_data.py
sbatch scripts/submit_job.sh
```

---

**Note**: Models requiring HuggingFace authentication need you to accept their license on HuggingFace and run `huggingface-cli login`. See [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md) for details.
