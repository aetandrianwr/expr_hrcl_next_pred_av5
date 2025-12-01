# Preprocessing Checkpoint System

## Overview

The DIY preprocessing pipeline now supports checkpoints to avoid reprocessing the entire 165M raw data rows when you only need to change downstream parameters like `epsilon` for location clustering.

## Checkpoint Location

Checkpoints are saved in: `data/diy/checkpoints/sp_before_location_clustering.pk`

This checkpoint contains:
- All staypoints after quality filtering
- Activity flags
- Valid users only
- **Before** location clustering (no location_id yet)

## What You Can Change Without Reprocessing

When a checkpoint exists, you can modify these parameters and rerun quickly:

### Location Clustering Parameters (`epsilon` and related)
```yaml
locations:
  epsilon: 50  # Changed from 20 - no reprocessing needed!
  num_samples: 1
  distance_metric: 'haversine'
  agg_level: 'user'
  n_jobs: -1
```

### Staypoint Merging Parameters
```yaml
staypoint_merging:
  max_time_gap: '4h'  # Can be changed with checkpoint
```

### Sequence Generation Parameters
```yaml
sequence_generation:
  previous_days: [7]  # Can be changed with checkpoint
```

### Dataset Split Ratios
```yaml
dataset_split:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
```

## What Requires Full Reprocessing

If you change these parameters, you **must** reprocess from scratch:

### Staypoint Generation
```yaml
staypoints:
  gap_threshold: 15
  dist_threshold: 100  # Changing this = reprocess
  time_threshold: 5    # Changing this = reprocess
```

### Activity Flag
```yaml
activity_flag:
  time_threshold: 5  # Changing this = reprocess
```

### User Quality Filtering
```yaml
user_quality:
  day_filter: 28
  window_size: 7
  min_thres: 5  # Changing these = reprocess
  mean_thres: 1
```

## Usage

### Normal Run (with checkpoint)
```bash
# First run - creates checkpoint
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml

# Change epsilon to 50 in config, then rerun
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# This will load checkpoint and skip raw data processing!
```

### Force Reprocessing from Raw Data
```bash
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml --no-checkpoint
```

### Using Different Configs with Checkpoints
```bash
# Each config can have its own checkpoint
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# Creates: data/diy/checkpoints/sp_before_location_clustering.pk

python preprocessing/diy_config.py --config configs/preprocessing/diy_skip_first_part.yaml
# Uses same checkpoint if output_dir is the same
```

## Example Workflow

```bash
# 1. Initial run (processes all 165M rows - takes time)
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# Output: epsilon=20, ~30-60 minutes

# 2. Edit configs/preprocessing/diy.yaml
#    Change: epsilon: 20 → epsilon: 50

# 3. Rerun with checkpoint (much faster!)
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# Output: epsilon=50, ~5-10 minutes (skips raw data processing)

# 4. Try different epsilon values
#    Change: epsilon: 50 → epsilon: 100
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# Output: epsilon=100, ~5-10 minutes
```

## Time Savings

- **Without checkpoint**: 30-60 minutes (read 165M rows + all processing)
- **With checkpoint**: 5-10 minutes (only location clustering + downstream)
- **Speedup**: ~6-12x faster

## Troubleshooting

### "Need to reprocess from beginning"
If you changed staypoint/quality parameters, delete the checkpoint:
```bash
rm data/diy/checkpoints/sp_before_location_clustering.pk
```

### Different sample sizes
Checkpoints are compatible with `--sample` parameter, but each sample size creates its own processing:
```bash
# Creates checkpoint for full dataset
python preprocessing/diy_config.py

# Won't use checkpoint (different sample size)
python preprocessing/diy_config.py --sample 10000
```

### Storage
The checkpoint file is ~100-500MB (depending on data size), much smaller than the 165M raw CSV.
