# Preprocessing Checkpoint System - Summary

## Changes Made to `preprocessing/diy_config.py`

### 1. Added Checkpoint Support

**Modified function signature:**
```python
def get_dataset(paths_config, preprocess_config, use_checkpoint=True):
```

**Added checkpoint directory:**
```python
checkpoint_dir = os.path.join(".", output_dir, "checkpoints")
checkpoint_sp_before_loc = os.path.join(checkpoint_dir, "sp_before_location_clustering.pk")
```

**Checkpoint loading logic:**
- If checkpoint exists → Load staypoints directly (skip 165M row processing)
- If checkpoint doesn't exist → Process from raw data and save checkpoint

**Checkpoint saving:**
- Saved after: quality filtering, activity flag creation
- Saved before: location clustering (epsilon parameter)

### 2. Added CLI Argument

```bash
--no-checkpoint  # Force reprocessing from raw data, ignore existing checkpoint
```

## Benefits

### Time Savings
- **Before**: 30-60 minutes every time you change epsilon
- **After**: 5-10 minutes (only location clustering + downstream)
- **Speedup**: ~6-12x faster for parameter tuning

### Use Cases
You can now quickly experiment with:
- Different `epsilon` values (20, 50, 100, etc.)
- Different merging strategies (`max_time_gap`)
- Different sequence lengths (`previous_days`)
- Different train/val/test splits

### Storage
- Checkpoint file: ~100-500MB
- Raw CSV: ~10GB+
- Tradeoff: Small disk space for huge time savings

## How to Use

### First Run (Creates Checkpoint)
```bash
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# Processes all data, saves checkpoint at data/diy/checkpoints/
```

### Subsequent Runs (Uses Checkpoint)
```bash
# Edit epsilon in config: 20 → 50
vim configs/preprocessing/diy.yaml

# Rerun - will use checkpoint!
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# Much faster! Skips raw data processing
```

### Force Full Reprocessing
```bash
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml --no-checkpoint
```

## Technical Details

**Checkpoint saves:**
- `sp` (staypoints GeoDataFrame)
- After quality filtering
- After activity flag creation
- Before location clustering

**Checkpoint preserves:**
- All geometry information
- User IDs (filtered)
- Timestamps
- Activity flags
- Staypoint boundaries

**Checkpoint location:**
```
data/diy/checkpoints/sp_before_location_clustering.pk
```

## When to Delete Checkpoint

Delete checkpoint if you change these parameters:
```bash
rm data/diy/checkpoints/sp_before_location_clustering.pk
```

Parameters requiring checkpoint deletion:
- `staypoints.dist_threshold`
- `staypoints.time_threshold`
- `staypoints.gap_threshold`
- `activity_flag.time_threshold`
- `user_quality.*` (any quality filter)

## Example Workflow: Testing Different Epsilon Values

```bash
# Initial run with epsilon=20
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# → 45 minutes, creates checkpoint

# Try epsilon=50
sed -i 's/epsilon: 20/epsilon: 50/' configs/preprocessing/diy.yaml
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# → 7 minutes (uses checkpoint!)

# Try epsilon=100
sed -i 's/epsilon: 50/epsilon: 100/' configs/preprocessing/diy.yaml
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# → 7 minutes (uses checkpoint!)

# Compare results
python view_results.py --dataset diy
```

## Verification

To check if checkpoint exists and is valid:
```bash
python << EOF
import os, pickle
ckpt = "./data/diy/checkpoints/sp_before_location_clustering.pk"
if os.path.exists(ckpt):
    with open(ckpt, 'rb') as f:
        sp = pickle.load(f)
    print(f"✓ Checkpoint exists with {len(sp)} staypoints")
else:
    print("✗ No checkpoint found")
EOF
```

## Documentation

See `preprocessing/CHECKPOINT_USAGE.md` for detailed usage guide.
