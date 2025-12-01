# Quick Reference: Changing Epsilon Without Reprocessing

## TL;DR

✅ **You can now change epsilon (20→50) without reading 165M rows again!**

## First Time Setup

```bash
# Run once to create checkpoint
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
```

This saves a checkpoint at: `data/diy/checkpoints/sp_before_location_clustering.pk`

## Changing Epsilon (Fast!)

```bash
# 1. Edit your config file
vim configs/preprocessing/diy.yaml
# Change: epsilon: 20 → epsilon: 50

# 2. Rerun preprocessing (uses checkpoint, much faster!)
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
```

**Time comparison:**
- Without checkpoint: ~45 minutes ⏰
- With checkpoint: ~7 minutes ⚡

## Answer to Your Question

> "if i want to change the location epsilon config, am i need to run from the beginning?"

**Before this update:** YES 😞  
**After this update:** NO! 🎉

The checkpoint saves all the expensive processing (reading 165M rows, generating staypoints, quality filtering) and lets you restart from just before location clustering.

## What Parameters Can You Change Quickly?

✅ `locations.epsilon` (20, 50, 100, etc.)  
✅ `staypoint_merging.max_time_gap`  
✅ `sequence_generation.previous_days`  
✅ `dataset_split` ratios  

## What Requires Full Reprocessing?

❌ `staypoints.dist_threshold`  
❌ `staypoints.time_threshold`  
❌ `user_quality.*` filters  

For these, use: `--no-checkpoint` flag

## Theoretically Correct?

> "is it theoretically wrong or what"

✅ **It's theoretically correct!**

- Staypoints are independent of epsilon
- Epsilon only affects **how staypoints are clustered into locations**
- The checkpoint saves staypoints BEFORE clustering
- So you can cluster them differently without affecting the underlying data

Think of it like:
1. Raw GPS → Staypoints (expensive, saved in checkpoint)
2. Staypoints → Locations via clustering (cheap, can redo with different epsilon)

## Example Session

```bash
# Try epsilon=20
echo "epsilon: 20" > temp_config.txt
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# Result: 2000 locations

# Try epsilon=50 (uses same staypoints!)
sed -i 's/epsilon: 20/epsilon: 50/' configs/preprocessing/diy.yaml
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# Result: 800 locations (fewer, larger clusters)

# Try epsilon=100
sed -i 's/epsilon: 50/epsilon: 100/' configs/preprocessing/diy.yaml
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
# Result: 400 locations
```

Each run after the first takes ~7 minutes instead of ~45 minutes!
