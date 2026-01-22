# LLM Fine-Tuning with Qwen2.5-7B

Fine-tune Qwen2.5-7B-Instruct on function-calling data using LoRA and distributed training on Nebius AI Cloud.

## Quick Start

1. **Setup Environment**
```bash
bash scripts/setup_nebius.sh
```

2. **Configure** - Edit `config.yaml`:
```yaml
cluster:
  nodes: 1
  gpus_per_node: 4

dataset:
  max_samples: 20000  # 20k for PoC, 60000 for full

training:
  num_train_epochs: 1
  learning_rate: 2.0e-4
  per_device_train_batch_size: 8

lora:
  r: 16
  lora_alpha: 32
```

3. **Prepare Data**
```bash
source envs/llm-env/bin/activate
python scripts/prepare_data.py
```

4. **Train**
```bash
sbatch scripts/submit_job.sh
```

5. **Monitor**
```bash
source scripts/monitor.sh
watch_latest  # View logs
watch_gpu     # GPU usage
```

## Key Features

- Configuration-driven (single YAML file)
- LoRA fine-tuning with FSDP
- Flash Attention 2 for H100/H200
- Automatic validation & benchmarking
- xLAM dataset (60k function-calling samples)

## Deployment on Nebius

1. Create SLURM cluster at [console.nebius.ai](https://console.nebius.ai/)
2. Add SSH key to cluster
3. Connect: `ssh root@login.slurm-XXXXX...`
4. Clone repo: `git clone https://github.com/smilenaderi/llm_fine_tune.git /shared/llm-fine-tune`
5. Run setup: `bash scripts/setup_nebius.sh`
6. Follow Quick Start steps above

## Performance (4x H200)

| Dataset | Epochs | Time | Throughput |
|---------|--------|------|------------|
| 20k     | 1      | 15-30 min | 12-15k tok/s |
| 60k     | 1      | 45-90 min | 12-15k tok/s |
| 60k     | 3      | 2-4 hours | 12-15k tok/s |

## Troubleshooting

**Out of memory?** Reduce batch size or LoRA rank in `config.yaml`

**Slow training?** Increase batch size or reduce dataset size

**Low validation score?** Train longer or use full dataset

See logs: `tail -f logs/train_*.err`

## Documentation

- [QUICK_START.md](QUICK_START.md) - Detailed setup
- [STORAGE.md](STORAGE.md) - Storage config
- [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md) - HF auth
- [config.yaml](config.yaml) - Full reference

## License

MIT
