# Storage Configuration Guide

This document explains how to configure and use the storage resources allocated for your PoC.

## PoC Storage Allocation

Your Nebius PoC includes:
- **2TB SSD Shared Filesystem**: For code, checkpoints, and logs
- **2TB SSD Network Disk**: For datasets and model cache

## Storage Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Filesystem (2TB)                   │
│                    /shared/llm-fine-tune                     │
├─────────────────────────────────────────────────────────────┤
│  ├── scripts/          # Training scripts                    │
│  ├── checkpoints/      # Model checkpoints (auto-saved)      │
│  ├── logs/             # Training and job logs               │
│  ├── cache/            # Temporary cache                     │
│  ├── config.yaml       # Configuration file                  │
│  └── envs/             # Python virtual environment          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Network Disk (2TB)                        │
│                    /mnt/network-disk                         │
├─────────────────────────────────────────────────────────────┤
│  ├── model_cache/      # HuggingFace model cache            │
│  ├── datasets/         # Raw datasets (optional)            │
│  └── backups/          # Checkpoint backups (optional)       │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Edit `config.yaml` to customize storage paths:

```yaml
storage:
  # Shared filesystem - for code, checkpoints, logs
  shared_fs: "/shared/llm-fine-tune"
  
  # Network disk - for datasets and cache
  network_disk: "/mnt/network-disk"
  
  # Relative paths (under shared_fs)
  data_dir: "data"
  checkpoint_dir: "checkpoints"
  log_dir: "logs"
  cache_dir: "cache"
```

## Storage Usage Guidelines

### Shared Filesystem (`/shared`)
**Best for:**
- Source code and scripts
- Training checkpoints (frequent writes)
- Training logs
- Configuration files

**Characteristics:**
- High-speed SSD
- Shared across all nodes
- Optimized for small, frequent I/O operations

### Network Disk (`/mnt/network-disk`)
**Best for:**
- Large model downloads (7B+ models)
- Dataset storage
- Long-term checkpoint backups
- Model cache

**Characteristics:**
- High-capacity SSD
- Network-attached storage
- Optimized for large sequential reads

## Mounting Network Disk

### Option 1: Automatic Mount (Recommended)

If using Nebius SLURM Operator, the network disk is automatically mounted at `/mnt/network-disk`.

Verify mount:
```bash
df -h | grep network-disk
```

### Option 2: Manual Mount

If the network disk is not automatically mounted:

```bash
# Create mount point
sudo mkdir -p /mnt/network-disk

# Mount the disk (replace with your disk ID)
sudo mount -t nfs <DISK_ID>.fs.eu-north1.nebius.cloud:/ /mnt/network-disk

# Verify
ls -la /mnt/network-disk
```

### Option 3: Persistent Mount

Add to `/etc/fstab` for automatic mounting on boot:

```bash
<DISK_ID>.fs.eu-north1.nebius.cloud:/ /mnt/network-disk nfs defaults 0 0
```

## Storage Optimization Tips

### 1. Model Cache Location

Set HuggingFace cache to network disk to avoid re-downloading:

```yaml
model:
  cache_dir: "${storage.network_disk}/model_cache"
```

Or set environment variable:
```bash
export HF_HOME=/mnt/network-disk/model_cache
```

### 2. Checkpoint Management

Keep only recent checkpoints to save space:

```yaml
checkpointing:
  save_total_limit: 3  # Keep only last 3 checkpoints
```

### 3. Dataset Preparation

Prepare datasets on network disk, then copy to shared filesystem if needed:

```bash
# Prepare on network disk
python scripts/prepare_data.py

# Data is automatically saved to configured location
```

### 4. Log Rotation

Regularly clean old logs:

```bash
# Remove logs older than 7 days
find logs/ -name "*.out" -mtime +7 -delete
find logs/ -name "*.err" -mtime +7 -delete
```

### 5. Checkpoint Backup

Backup important checkpoints to network disk:

```bash
# Backup final model
cp -r checkpoints/final_adapter /mnt/network-disk/backups/final_adapter_$(date +%Y%m%d)
```

## Monitoring Storage Usage

### Check Disk Space

```bash
# Shared filesystem
df -h /shared

# Network disk
df -h /mnt/network-disk
```

### Check Directory Sizes

```bash
# Checkpoint size
du -sh checkpoints/

# Model cache size
du -sh /mnt/network-disk/model_cache/

# Log size
du -sh logs/
```

### Automated Monitoring

Add to your monitoring script:

```bash
#!/bin/bash
echo "Storage Usage Report - $(date)"
echo "================================"
echo "Shared Filesystem:"
df -h /shared | tail -1
echo ""
echo "Network Disk:"
df -h /mnt/network-disk | tail -1
echo ""
echo "Checkpoint Size:"
du -sh checkpoints/
echo ""
echo "Model Cache Size:"
du -sh /mnt/network-disk/model_cache/
```

## Troubleshooting

### Issue: "No space left on device"

**Solution 1: Clean old checkpoints**
```bash
# Remove all but the latest checkpoint
cd checkpoints
ls -t | tail -n +2 | xargs rm -rf
```

**Solution 2: Move cache to network disk**
```bash
# Update config.yaml
model:
  cache_dir: "/mnt/network-disk/model_cache"
```

**Solution 3: Clean HuggingFace cache**
```bash
rm -rf ~/.cache/huggingface/*
```

### Issue: "Network disk not mounted"

**Check mount status:**
```bash
mount | grep network-disk
```

**Remount:**
```bash
sudo mount -a
```

### Issue: "Permission denied"

**Fix permissions:**
```bash
sudo chown -R $USER:$USER /mnt/network-disk
sudo chmod -R 755 /mnt/network-disk
```

## Storage Best Practices

1. **Use network disk for model cache** - Avoid re-downloading large models
2. **Keep checkpoints on shared filesystem** - Faster access during training
3. **Backup final models to network disk** - Long-term storage
4. **Clean logs regularly** - Prevent disk space issues
5. **Monitor disk usage** - Set up alerts for low space
6. **Use checkpoint rotation** - Don't keep all checkpoints
7. **Compress old checkpoints** - Save space for archived models

## Storage Cost Optimization

### Checkpoint Compression

Compress old checkpoints to save space:

```bash
# Compress checkpoint
tar -czf checkpoint-25.tar.gz checkpoints/checkpoint-25/
rm -rf checkpoints/checkpoint-25/

# Decompress when needed
tar -xzf checkpoint-25.tar.gz
```

### Selective Model Caching

Only cache models you frequently use:

```bash
# Clear unused models from cache
huggingface-cli delete-cache
```

### Data Deduplication

If using multiple datasets, deduplicate to save space:

```bash
# Use symbolic links for shared data
ln -s /mnt/network-disk/datasets/common data/common
```

## Quick Reference

| Path | Purpose | Size Limit | Speed | Shared |
|------|---------|------------|-------|--------|
| `/shared/llm-fine-tune` | Code, checkpoints, logs | 2TB | High | Yes |
| `/mnt/network-disk` | Model cache, datasets | 2TB | Medium | Yes |
| `checkpoints/` | Training checkpoints | ~50GB | High | Yes |
| `logs/` | Training logs | ~1GB | High | Yes |
| `model_cache/` | HuggingFace models | ~30GB | Medium | Yes |

## Support

For storage-related issues:
1. Check disk space: `df -h`
2. Check mount status: `mount | grep network`
3. Review logs: `tail -f logs/train_*.err`
4. Contact Nebius support with your cluster ID

---

**Last Updated**: January 2026  
**Version**: 1.0
