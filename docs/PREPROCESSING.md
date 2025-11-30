# Data Preprocessing Pipeline

This document describes the complete data preprocessing workflow for the next-location prediction project.

## Quick Start

### Process Geolife Dataset

```bash
# Option 1: Use master script
python preprocessing/run_preprocessing.py --dataset geolife

# Option 2: Run steps manually
python preprocessing/preprocess_geolife.py --config configs/preprocessing/geolife.yaml
python preprocessing/generate_transformer_data.py --config configs/preprocessing/geolife.yaml
```

### Process DIY Dataset

```bash
# Note: DIY dataset is very large (165M rows) and requires significant processing time
python preprocessing/run_preprocessing.py --dataset diy

# Or manually:
python preprocessing/preprocess_diy.py --config configs/preprocessing/diy.yaml
python preprocessing/generate_transformer_data.py --config configs/preprocessing/diy.yaml
```

## Configuration Files

All preprocessing parameters are stored in YAML files in `configs/preprocessing/`:

### Geolife Configuration (`geolife.yaml`)
- **Dataset**: Geolife (Beijing, China)
- **Timezone**: UTC
- **Staypoint threshold**: 200m, 30min
- **User quality**: 50 days minimum
- **Location clustering**: epsilon=50m

### DIY Configuration (`diy.yaml`)
- **Dataset**: DIY Mobility (Yogyakarta, Indonesia)  
- **Timezone**: Asia/Jakarta
- **Staypoint threshold**: 100m, 30min
- **User quality**: 60 days, quality thresholds 0.6/0.7
- **Location clustering**: epsilon=50m

## Pipeline Steps

### 1. Raw Data Processing
- **Geolife**: Read from trackintel format
- **DIY**: Convert CSV (user_id, lat, lon, timestamp) to trackintel positionfixes

### 2. Staypoint Generation
- Detect stationary periods using:
  - Distance threshold (100-200m)
  - Time threshold (30 minutes)
  - Gap threshold (24 hours)
- Create activity flags (time threshold: 25 minutes)

### 3. User Quality Filtering
- Calculate temporal tracking quality
- Filter users based on:
  - Minimum tracking days
  - Sliding window quality metrics
  - Consistency thresholds (for DIY)

### 4. Location Generation
- Cluster staypoints into locations using DBSCAN
- Parameters: epsilon (50m), min_samples (2)
- Filter noise staypoints

### 5. Data Enrichment
- Merge consecutive staypoints (within 1 minute)
- Add temporal features:
  - Day of tracking
  - Time of day
  - Weekday
  - Duration

### 6. Sequence Generation
- Create valid sequences with 7-day history
- Require minimum 3 historical records
- Encode location and user IDs

### 7. Dataset Splitting
- Train: 60% (first 60% of each user's timeline)
- Validation: 20% (next 20%)
- Test: 20% (last 20%)

## Output Files

For each dataset in `data/{dataset}/`:

### Preprocessing Outputs
- `dataSet_{dataset}.csv`: Main preprocessed dataset
- `locations_{dataset}.csv`: Location catalog with coordinates
- `sp_time_temp_{dataset}.csv`: Intermediate staypoint data
- `valid_ids_{dataset}.pk`: Valid sequence IDs
- `quality/{dataset}_slide_filtered.csv`: User quality results

### Model Input Files
- `{dataset}_transformer_7_train.pk`: Training sequences
- `{dataset}_transformer_7_validation.pk`: Validation sequences
- `{dataset}_transformer_7_test.pk`: Test sequences

Each pickle file contains a list of dictionaries:
```python
{
    'X': array([loc1, loc2, ...]),       # Historical location IDs
    'user_X': array([user, user, ...]),  # User ID repeated
    'weekday_X': array([0-6, ...]),      # Weekday values
    'start_min_X': array([mins, ...]),   # Minutes since midnight
    'dur_X': array([duration, ...]),     # Durations in minutes
    'diff': array([days, ...]),          # Days from current
    'Y': int                              # Next location (target)
}
```

## Verification

### Test Preprocessing Worked
After preprocessing, verify the output:

```python
import pickle

# Load generated data
with open('data/geolife/geolife_transformer_7_train.pk', 'rb') as f:
    train_data = pickle.load(f)

print(f"Training samples: {len(train_data)}")
print(f"First sample: {train_data[0]}")
```

### Test Model Training
Train the model for a few epochs to verify everything works:

```bash
# Create quick test config (5 epochs)
python -c "
import yaml
with open('configs/geolife_default.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['training']['num_epochs'] = 5
config['experiment']['name'] = 'geolife_test'
with open('configs/geolife_test.yaml', 'w') as f:
    yaml.dump(config, f)
"

# Run training
python train_model.py --config configs/geolife_test.yaml --seed 42
```

Expected test accuracy for Geolife: ~47-48% Acc@1

## Reproducing Baseline Results

The preprocessing pipeline is designed to match the baseline preprocessing from `/content/location-prediction-baseline-ori`:

1. Same staypoint generation parameters
2. Same user quality filtering logic
3. Same location clustering approach
4. Same sequence generation rules

Minor differences may occur due to:
- Pandas/NumPy version differences
- Floating-point precision
- Tie-breaking in clustering

## Troubleshooting

### Out of Memory Errors
For large datasets like DIY (165M rows):
- Process in chunks
- Use systems with sufficient RAM (>32GB recommended)
- Consider sampling for initial testing

### Index Out of Bounds
If you see CUDA errors during training:
- Verify `num_locations` and `num_users` in model config
- Check max location/user IDs in the data:
  ```python
  import pickle
  with open('data/geolife/geolife_transformer_7_train.pk', 'rb') as f:
      data = pickle.load(f)
  max_loc = max([s['Y'] for s in data])
  max_user = max([s['user_X'][0] for s in data])
  print(f"Need: num_locations={max_loc+1}, num_users={max_user+1}")
  ```

### Different Results from Baseline
Small differences are expected due to:
- Random seed differences in DBSCAN
- Different preprocessing order
- Minor library version differences

Large differences indicate configuration mismatch - verify YAML parameters match baseline.

## Performance Tips

1. **Use parallel processing**: Set `n_jobs=-1` in configs (uses all CPU cores)
2. **Monitor progress**: All scripts show progress bars
3. **Save intermediate results**: Pipeline saves checkpoints for recovery
4. **Test on small sample first**: Use head -n 10000 for testing

## Next Steps

After preprocessing:
1. Verify output files exist and have expected sizes
2. Test model training with quick config (5 epochs)
3. Run full training with default config (120 epochs)
4. Evaluate results and compare with baseline

See `preprocessing/README.md` for more details on the preprocessing pipeline.
