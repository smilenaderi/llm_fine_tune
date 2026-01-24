# Inference Guide

## Quick Start

```bash
# List available models
bash scripts/list_models.sh

# Run with your own prompt
sbatch scripts/run_inf.sh 22 "Book a flight from NYC to Paris on March 15th"

# Run with multiple prompts from file
sbatch scripts/run_inf.sh 22 --prompts-file example_prompts.txt

# Check status and view results
squeue --me
tail -f logs/inference_*.out
```

---

## Your Available Models

| Job | Model | Size | Best For |
|-----|-------|------|----------|
| 22 | Qwen 7B | 7B | ⭐ Recommended - Fast & Good |
| 8 | Qwen 14B | 14B | Better quality |
| 20 | Qwen 72B | 72B | Best quality (needs 2-4 GPUs) |
| 10 | Phi-3 mini | 4B | Fastest |

---

## Usage Options

### Single Prompt
```bash
sbatch scripts/run_inf.sh 8 "Your prompt here"
```

### Multiple Prompts (File)
```bash
# Create prompts file (one per line)
cat > my_prompts.txt << EOF
Book a flight from New York to Tokyo on March 20th
Find Italian restaurants in San Francisco
Schedule a meeting tomorrow at 3pm
EOF

# Run inference
sbatch scripts/run_inf.sh 8 --prompts-file my_prompts.txt
```

Example file provided: `example_prompts.txt`

### Default Prompt
```bash
sbatch scripts/run_inf.sh 22  # Uses prompt from config.yaml
```

---

## Configuration

Edit `config.yaml` to adjust settings:

```yaml
inference:
  max_new_tokens: 256      # Max response length
  temperature: 0.7         # Creativity (0.1-1.0)
  top_p: 0.9              # Nucleus sampling
  do_sample: true         # Enable sampling
```

---

## Monitoring

```bash
# Check job status
squeue --me

# View output (live)
tail -f logs/inference_*.out

# View errors
tail -f logs/inference_*.err

# List recent jobs
ls -lht logs/inference_*.out | head -5
```

---

## Common Issues

**Model mismatch error?**
- Fixed! Model is auto-detected from adapter config

**No GPU on login node?**
- Always use `sbatch`, not `python` directly

**Job stuck in queue?**
```bash
squeue --me
sinfo -N -l  # Check GPU availability
```

**Out of memory?**
- Use smaller model (10, 22 instead of 20)
- Or edit `scripts/run_inf.sh`: `#SBATCH --gpus-per-node=2`

---

## Examples

```bash
# Function calling
sbatch scripts/run_inf.sh 22 "Book a hotel in Paris for 3 nights starting March 15th"

# Information retrieval
sbatch scripts/run_inf.sh 22 "What are the best restaurants in Tokyo?"

# Task planning
sbatch scripts/run_inf.sh 8 "Plan a 5-day trip to Italy including flights and hotels"
```

---

## Important Notes

✅ Always use `sbatch` (login node has no GPU)
✅ Model is auto-detected from adapter config
✅ No need to edit config.yaml for different jobs
✅ Start with job 22 for best balance of speed and quality
