# Understanding Dataset Splits

This guide explains how dataset splits and validation splits work together.

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HuggingFace Dataset                          │
│                  (e.g., 60,000 total samples)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐  ┌──────────┐  ┌──────────┐        │
│  │   "train" split      │  │ "valid"  │  │  "test"  │        │
│  │   (48,000 samples)   │  │ (6,000)  │  │ (6,000)  │        │
│  └──────────────────────┘  └──────────┘  └──────────┘        │
│           ↓                                                     │
│  You select this with: split: "train"                          │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              After Loading (with max_samples: 20000)            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐      │
│  │         Loaded Data (20,000 samples)                 │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│        After Validation Split (validation_split: 0.05)          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐  ┌─────────────┐      │
│  │   Training Data (19,000 samples)    │  │ Validation  │      │
│  │   Used to train the model           │  │ (1,000)     │      │
│  └─────────────────────────────────────┘  └─────────────┘      │
│                                             Used to evaluate    │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Process

### Step 1: Dataset Split Selection
```yaml
dataset:
  name: "Beryex/xlam-function-calling-60k-sharegpt"
  split: "train"  # ← Selects which portion from HuggingFace
```

**What happens**: Loads the "train" portion of the dataset from HuggingFace.

**Result**: 60,000 samples loaded (or whatever the "train" split contains)

---

### Step 2: Sample Limiting (Optional)
```yaml
dataset:
  max_samples: 20000  # ← Limits how many samples to use
```

**What happens**: Takes only the first 20,000 samples from the loaded data.

**Result**: 20,000 samples ready for training

---

### Step 3: Validation Split (Optional)
```yaml
validation:
  enabled: true
  validation_split: 0.05  # ← Reserves 5% for validation
```

**What happens**: Splits the 20,000 samples into:
- Training: 19,000 samples (95%)
- Validation: 1,000 samples (5%)

**Result**: 
- Model trains on 19,000 samples
- Model evaluates on 1,000 samples (never seen during training)

---

## Common Split Options

### Option 1: Standard "train" Split (Most Common)
```yaml
dataset:
  split: "train"
```
✅ Use this for 99% of cases

---

### Option 2: Use Validation Split
```yaml
dataset:
  split: "validation"  # or "valid"
```
📊 Use this to evaluate on official validation data

---

### Option 3: Use Test Split
```yaml
dataset:
  split: "test"
```
🧪 Use this for final evaluation only

---

### Option 4: Slice the Split
```yaml
# First 80% of train split
dataset:
  split: "train[:80%]"

# Last 20% of train split
dataset:
  split: "train[80%:]"

# First 10,000 samples
dataset:
  split: "train[:10000]"

# Samples 5,000 to 15,000
dataset:
  split: "train[5000:15000]"
```
✂️ Use this for custom data splitting

---

### Option 5: Combine Splits
```yaml
# Combine train and validation
dataset:
  split: "train+validation"

# Combine all splits
dataset:
  split: "train+validation+test"
```
🔗 Use this to train on all available data

---

## Practical Examples

### Example 1: Quick PoC (Fast)
```yaml
dataset:
  name: "Beryex/xlam-function-calling-60k-sharegpt"
  split: "train"
  max_samples: 10000

validation:
  enabled: false  # Skip validation for speed
```

**Result**: 
- Training: 10,000 samples
- Validation: None
- Time: ~10-15 minutes

---

### Example 2: Standard PoC (Recommended)
```yaml
dataset:
  name: "Beryex/xlam-function-calling-60k-sharegpt"
  split: "train"
  max_samples: 20000

validation:
  enabled: true
  validation_split: 0.05
```

**Result**:
- Training: 19,000 samples
- Validation: 1,000 samples
- Time: ~15-30 minutes

---

### Example 3: Production Training
```yaml
dataset:
  name: "Beryex/xlam-function-calling-60k-sharegpt"
  split: "train"
  max_samples: 60000

validation:
  enabled: true
  validation_split: 0.05
```

**Result**:
- Training: 57,000 samples
- Validation: 3,000 samples
- Time: ~45-90 minutes

---

### Example 4: Use Official Validation Split
```yaml
# For training
dataset:
  name: "HuggingFaceH4/ultrachat_200k"
  split: "train_sft"
  max_samples: 50000

validation:
  enabled: true
  dataset: "HuggingFaceH4/ultrachat_200k"
  # This would use the official validation split
```

**Result**:
- Training: 50,000 samples from train_sft
- Validation: Official validation split
- Time: ~40-80 minutes

---

## How to Check Available Splits

### Method 1: Check HuggingFace Website
Visit: `https://huggingface.co/datasets/[DATASET_NAME]`

Example: https://huggingface.co/datasets/Beryex/xlam-function-calling-60k-sharegpt

Look for the "Dataset Splits" section.

---

### Method 2: Check Programmatically
```python
from datasets import load_dataset

# Load dataset info
dataset = load_dataset("Beryex/xlam-function-calling-60k-sharegpt")

# Print available splits
print("Available splits:", list(dataset.keys()))
# Output: Available splits: ['train']

# Check split sizes
for split_name, split_data in dataset.items():
    print(f"{split_name}: {len(split_data)} samples")
# Output: train: 59,985 samples
```

---

### Method 3: Check in Config
```bash
# Run this to see dataset info
python -c "
from datasets import load_dataset
ds = load_dataset('Beryex/xlam-function-calling-60k-sharegpt')
print('Splits:', list(ds.keys()))
for name, data in ds.items():
    print(f'{name}: {len(data)} samples')
"
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Using Non-Existent Split
```yaml
dataset:
  split: "validation"  # Dataset only has "train"
```
**Error**: `ValueError: Split 'validation' not found`

**Fix**: Check available splits first, use `"train"`

---

### ❌ Mistake 2: Confusing Dataset Split with Validation Split
```yaml
# WRONG: Trying to create validation with dataset split
dataset:
  split: "train[:80%]"  # This is NOT validation

# RIGHT: Use validation_split parameter
dataset:
  split: "train"
validation:
  validation_split: 0.2  # This creates validation
```

---

### ❌ Mistake 3: Double Limiting
```yaml
# WRONG: Both limit samples
dataset:
  split: "train[:10000]"  # Limits to 10k
  max_samples: 20000      # Tries to get 20k (but only 10k available)

# RIGHT: Use one or the other
dataset:
  split: "train"
  max_samples: 10000
```

---

## Quick Reference

| Parameter | Location | Purpose | Example |
|-----------|----------|---------|---------|
| `split` | `dataset.split` | Select HuggingFace split | `"train"` |
| `max_samples` | `dataset.max_samples` | Limit total samples | `20000` |
| `validation_split` | `validation.validation_split` | Reserve for validation | `0.05` |

---

## Decision Tree

```
Do you know which split to use?
│
├─ No → Use "train" (works 99% of the time)
│
└─ Yes → Do you need validation?
    │
    ├─ No → Set validation.enabled: false
    │
    └─ Yes → Does dataset have official validation split?
        │
        ├─ No → Use validation_split: 0.05
        │
        └─ Yes → Use validation.dataset: "dataset_name"
```

---

## Summary

**For most PoC cases, use this configuration:**

```yaml
dataset:
  name: "Beryex/xlam-function-calling-60k-sharegpt"
  split: "train"              # ← Standard choice
  max_samples: 20000          # ← Adjust for speed
  
validation:
  enabled: true               # ← Recommended
  validation_split: 0.05      # ← 5% for validation
```

This gives you:
- ✅ Simple and standard
- ✅ Fast training (20k samples)
- ✅ Validation for quality check
- ✅ Works with any dataset

---

**Questions?** Check:
- `config.yaml` - Full configuration with comments
- `MODEL_ALTERNATIVES.md` - Dataset options
- `README.md` - Complete guide
