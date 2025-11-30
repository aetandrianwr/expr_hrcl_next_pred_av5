# Data Preprocessing Documentation

This document provides comprehensive documentation for the data preprocessing pipeline used in this project.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Prerequisites](#prerequisites)
4. [Preprocessing Scripts](#preprocessing-scripts)
5. [Configuration Files](#configuration-files)
6. [Usage Guide](#usage-guide)
7. [Output Files](#output-files)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The preprocessing pipeline transforms raw GPS trajectory data into structured sequences suitable for training location prediction models. The pipeline is fully **config-driven**, with all parameters specified in YAML files for reproducibility.

### Supported Datasets

| Dataset | Location | Data Type | Raw Format |
|---------|----------|-----------|------------|
| **Geolife** | Beijing, China | GPS trajectories | Folder structure |
| **DIY** | Yogyakarta, Indonesia | GPS points | CSV file |

### Pipeline Stages

```
Raw Data → Staypoints → User Quality Filter → Locations (DBSCAN) 
→ Time Enrichment → Dataset Split → Location Encoding → Sequences
```

---

## Directory Structure

```
preprocessing/
├── geolife_config.py          # Geolife preprocessing (config-driven)
├── diy_config.py              # DIY preprocessing (config-driven)
├── generate_transformer_data.py  # Generate .pk files for training
├── utils.py                   # Shared utility functions
└── __init__.py

configs/preprocessing/
├── geolife.yaml               # Geolife parameters
└── diy.yaml                   # DIY parameters

data/
├── geolife/
│   ├── Data/                  # Raw GPS trajectories
│   ├── quality/               # User quality assessment
│   ├── locations_geolife.csv  # Generated locations
│   ├── dataSet_geolife.csv    # Preprocessed dataset
│   ├── valid_ids_geolife.pk   # Valid sequence IDs
│   ├── geolife_transformer_7_train.pk
│   ├── geolife_transformer_7_validation.pk
│   └── geolife_transformer_7_test.pk
└── diy/
    ├── raw/
    │   └── raw_diy_mobility_dataset.csv
    ├── quality/
    ├── locations_diy.csv
    ├── dataSet_diy.csv
    ├── valid_ids_diy.pk
    ├── diy_transformer_7_train.pk
    ├── diy_transformer_7_validation.pk
    └── diy_transformer_7_test.pk
```

---

## Prerequisites

### Required Libraries

```bash
pip install pandas numpy geopandas scikit-learn pyyaml tqdm trackintel shapely
```

### Configuration File

Create `paths.json` in project root:

```json
{
    "raw_geolife": "./data/geolife/Data",
    "raw_diy": "./data/diy/raw"
}
```

---

## Preprocessing Scripts

### 1. `geolife_config.py`

**Purpose:** Preprocess Geolife GPS trajectory data.

**Key Features:**
- Config-driven parameter management
- Built-in Geolife data reader from trackintel
- User quality filtering based on tracking days
- DBSCAN location clustering with epsilon=20m

**Algorithm:**
1. Read GPS trajectories using `trackintel.read_geolife()`
2. Generate staypoints (dist=200m, time=30min, gap=24h)
3. Create activity flags (time_threshold=25min)
4. Filter users by tracking quality (day_filter=50)
5. Generate locations using DBSCAN (epsilon=20m)
6. Merge nearby staypoints (max_time_gap=1min)
7. Enrich with temporal features
8. Save preprocessed dataset

### 2. `diy_config.py`

**Purpose:** Preprocess DIY GPS point data.

**Key Features:**
- Config-driven parameter management
- Manual CSV parsing with timezone conversion
- Stricter user quality filtering (min_thres/mean_thres)
- DBSCAN location clustering with epsilon=50m
- Supports sampling for testing large datasets

**Algorithm:**
1. Read CSV and convert to GeoDataFrame
2. Parse timestamps and convert to Asia/Jakarta timezone
3. Convert to trackintel positionfixes format
4. Generate staypoints (dist=100m, time=30min, gap=24h)
5. Create activity flags (time_threshold=25min)
6. Generate triplegs for quality assessment
7. Filter users by tracking quality (day_filter=60, min_thres=0.6, mean_thres=0.7)
8. Generate locations using DBSCAN (epsilon=50m)
9. Merge nearby staypoints (max_time_gap=1min)
10. Enrich with temporal features
11. Save preprocessed dataset

**Sample Mode:**
```bash
# Test with first 100K rows
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml --sample 100000
```

### 3. `generate_transformer_data.py`

**Purpose:** Generate train/validation/test .pk files from preprocessed data.

**Key Features:**
- Splits dataset (60% train, 20% val, 20% test)
- Encodes locations based on training set only
- Generates sequences with 7-day history
- Filters valid sequences (must have enough history)

**Algorithm:**
1. Load preprocessed dataset and valid IDs
2. Split by user into train/val/test
3. Encode locations (0=padding, 1=unknown, 2-N=train locations)
4. Generate sequences with previous 7 days of history
5. Filter sequences with insufficient history
6. Save as pickle files

### 4. `utils.py`

**Purpose:** Shared utility functions.

**Key Functions:**
- `calculate_user_quality()` - Assess tracking quality using sliding window
- `enrich_time_info()` - Add temporal features (weekday, hour, etc.)
- `split_dataset()` - Split data by user (60/20/20)
- `get_valid_sequence()` - Filter sequences with sufficient history

---

## Configuration Files

### Geolife Configuration (`configs/preprocessing/geolife.yaml`)

```yaml
dataset:
  name: "geolife"
  output_dir: "data/geolife"
  timezone: "UTC"

staypoints:
  method: "sliding"
  distance_metric: "haversine"
  dist_threshold: 200        # meters - Beijing is dense, larger threshold
  time_threshold: 30         # minutes
  gap_threshold: 1440        # 24 hours
  include_last: true
  print_progress: true
  n_jobs: -1

activity_flag:
  method: "time_threshold"
  time_threshold: 25         # minutes

user_quality:
  day_filter: 50             # minimum tracking days
  window_size: 10            # weeks for sliding window
  min_thres: null            # not used for Geolife
  mean_thres: null           # not used for Geolife

locations:
  epsilon: 20                # CRITICAL: 20m for dense Beijing
  num_samples: 2             # minimum samples for DBSCAN
  distance_metric: "haversine"
  agg_level: "dataset"
  n_jobs: -1

staypoint_merging:
  max_time_gap: "1min"

sequence_generation:
  previous_days: [7]         # use 7 days of history
  min_sequence_length: 3

dataset_split:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2

seed: 42
```

### DIY Configuration (`configs/preprocessing/diy.yaml`)

```yaml
dataset:
  name: "diy"
  output_dir: "data/diy"
  timezone: "Asia/Jakarta"   # Indonesia timezone

staypoints:
  method: "sliding"
  distance_metric: "haversine"
  dist_threshold: 100        # meters - Yogyakarta, smaller threshold
  time_threshold: 30         # minutes
  gap_threshold: 1440        # 24 hours
  include_last: true
  print_progress: true
  n_jobs: -1

activity_flag:
  method: "time_threshold"
  time_threshold: 25         # minutes

user_quality:
  day_filter: 60             # minimum tracking days
  window_size: 10            # weeks for sliding window
  min_thres: 0.6             # quality threshold (crowdsourced data)
  mean_thres: 0.7            # mean quality threshold

locations:
  epsilon: 50                # 50m for Yogyakarta (less dense than Beijing)
  num_samples: 2
  distance_metric: "haversive"
  agg_level: "dataset"
  n_jobs: -1

staypoint_merging:
  max_time_gap: "1min"

sequence_generation:
  previous_days: [7]
  min_sequence_length: 3

dataset_split:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2

seed: 42
```

---

## Usage Guide

### Preprocessing Geolife Dataset

**Step 1: Ensure raw data is in place**
```bash
ls data/geolife/Data/  # Should show folders 000, 001, ..., 181
```

**Step 2: Run preprocessing**
```bash
cd /path/to/project
python preprocessing/geolife_config.py --config configs/preprocessing/geolife.yaml
```

**Expected output:**
```
Using epsilon=20 for location clustering
...
After filter non-location staypoints: 19584
Location size: 2049 2049
After staypoints merging: 19191
User size: 91
Max location id:1186, unique location id:1185
Final user size: 45
```

**Step 3: Generate transformer data**
```bash
python preprocessing/generate_transformer_data.py --config configs/preprocessing/geolife.yaml
```

**Expected output:**
```
Max location ID: 1186
Train sequences: 7424
Validation sequences: 3334
Test sequences: 3502
```

**Verification:**
- Check `data/geolife/geolife_transformer_7_train.pk` exists
- Should have 7424 training sequences
- Max location ID should be 1186
- Max user ID should be 45

### Preprocessing DIY Dataset

**Step 1: Ensure raw data is in place**
```bash
ls -lh data/diy/raw/raw_diy_mobility_dataset.csv  # ~14GB file
```

**Step 2: Test with sample (recommended for first run)**
```bash
python preprocessing/diy_config.py \
    --config configs/preprocessing/diy.yaml \
    --sample 100000
```

**Step 3: Run full preprocessing (takes several hours)**
```bash
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml
```

**Step 4: Generate transformer data**
```bash
python preprocessing/generate_transformer_data.py --config configs/preprocessing/diy.yaml
```

---

## Output Files

### Intermediate Files

| File | Description | Size |
|------|-------------|------|
| `quality/{dataset}_slide_filtered.csv` | Users passing quality filter | Small |
| `locations_{dataset}.csv` | DBSCAN-generated locations | Small-Medium |
| `sp_time_temp_{dataset}.csv` | Staypoints with time features | Medium |
| `dataSet_{dataset}.csv` | Main preprocessed dataset | Medium-Large |
| `valid_ids_{dataset}.pk` | Valid sequence IDs | Small |

### Final Training Files

| File | Description | Format |
|------|-------------|--------|
| `{dataset}_transformer_7_train.pk` | Training sequences | List of dicts |
| `{dataset}_transformer_7_validation.pk` | Validation sequences | List of dicts |
| `{dataset}_transformer_7_test.pk` | Test sequences | List of dicts |

### Sequence Format

Each sequence is a dictionary:
```python
{
    'X': [loc_id_1, loc_id_2, ..., loc_id_7],  # 7 days of history
    'user_X': [user_id] * 7,                    # User ID repeated
    'weekday_X': [0-6] * 7,                     # Day of week
    'start_min_X': [0-1439] * 7,                # Minutes since midnight
    'dur_X': [minutes] * 7,                     # Duration at location
    'diff': [minutes] * 7,                      # Time since last visit
    'Y': target_loc_id                          # Next location to predict
}
```

---

## Parameter Tuning Guide

### Staypoint Parameters

**`dist_threshold`** (meters)
- Higher → Fewer staypoints (merge distant points)
- Lower → More staypoints (separate close points)
- **Geolife (Beijing)**: 200m (dense city, need larger threshold)
- **DIY (Yogyakarta)**: 100m (less dense)

**`time_threshold`** (minutes)
- Higher → Only long stops are staypoints
- Lower → Short stops also count
- **Both datasets**: 30min (standard for mobility analysis)

**`gap_threshold`** (minutes)
- Maximum gap between GPS points to be same trajectory
- **Both datasets**: 1440min (24 hours)

### Location Clustering (DBSCAN)

**`epsilon`** (meters) - **MOST CRITICAL PARAMETER**
- Radius for clustering nearby staypoints into locations
- Too small → Too many locations, over-segmentation
- Too large → Too few locations, under-segmentation
- **Geolife (Beijing)**: 20m (dense megacity, tight POI clusters)
- **DIY (Yogyakarta)**: 50m (medium city, wider spread)
- **General rule**: Dense cities 20-30m, medium cities 50m, rural 100m+

**`num_samples`**
- Minimum staypoints to form a location
- **Both datasets**: 2 (standard DBSCAN minimum)

### User Quality Filtering

**`day_filter`** (days)
- Minimum tracking days to include user
- **Geolife**: 50 days (short-term study)
- **DIY**: 60 days (longer study period)

**`min_thres` / `mean_thres`** (0.0-1.0)
- Quality score thresholds
- **Geolife**: null (research-grade data, no need)
- **DIY**: 0.6/0.7 (crowdsourced data, needs filtering)
- Only used when data quality is variable

---

## Troubleshooting

### Problem: No users pass quality filter

**Symptoms:**
```
final selected user 0
TypeError: 'NoneType' object does not support item assignment
```

**Solutions:**
1. Reduce `day_filter` (e.g., from 60 to 30 days)
2. Use larger sample size (if using `--sample`)
3. Lower `min_thres` and `mean_thres` (e.g., from 0.6 to 0.4)
4. Check raw data actually covers sufficient time period

### Problem: Too many/few locations generated

**Symptoms:**
- Too many: Thousands of locations for small area
- Too few: Dozens of locations for large city

**Solutions:**
- Adjust `epsilon` parameter:
  - Too many locations → Increase epsilon (20→30→50)
  - Too few locations → Decrease epsilon (50→30→20)
- Adjust `num_samples`:
  - Too many → Increase num_samples (2→3→5)
  - Too few → Decrease num_samples (3→2)

### Problem: Out of memory during preprocessing

**Solutions:**
1. Use `--sample` parameter to process subset
2. Increase system RAM or use machine with more memory
3. Process in batches (modify script to process user chunks)
4. Reduce `n_jobs` parameter (e.g., from -1 to 4)

### Problem: Timezone issues

**Symptoms:**
- Incorrect hour of day features
- Weekday misalignment

**Solutions:**
- Verify `timezone` in config matches data collection location
- Geolife: UTC (Beijing time stored as UTC)
- DIY: Asia/Jakarta (Yogyakarta, Indonesia)
- Check input data timestamp format

---

## Expected Results

### Geolife Dataset

After preprocessing, you should get:

| Metric | Expected Value |
|--------|----------------|
| Final users | 45 |
| Total locations (raw) | 2049 |
| Encoded locations (train) | 1185 |
| Max location ID | 1186 |
| Train sequences | 7424 |
| Validation sequences | 3334 |
| Test sequences | 3502 |
| Model Test Acc@1 | ~47-48% |

### DIY Dataset

Results will vary based on full dataset characteristics. With 100K sample:

| Metric | Approximate Value |
|--------|-------------------|
| Final users | Depends on quality filter |
| Total locations | Depends on epsilon |
| Train sequences | Depends on users |

For full dataset (165M rows), expect much larger numbers.

---

## Best Practices

1. **Always version control your config files** - YAML files are small and track parameter changes

2. **Test with samples first** - Especially for large datasets like DIY
   ```bash
   --sample 10000   # Quick test
   --sample 100000  # Medium test
   # No sample flag for full run
   ```

3. **Document parameter choices** - Add comments in YAML files explaining why

4. **Verify intermediate outputs** - Check CSV files before running transformer data generation

5. **Keep preprocessing separate from training** - Don't mix preprocessing and model code

6. **Use consistent random seed** - `seed: 42` in all configs for reproducibility

7. **Save preprocessing logs** - Redirect output to log files
   ```bash
   python preprocessing/geolife_config.py --config ... 2>&1 | tee preprocess.log
   ```

---

## References

- **Trackintel Documentation**: https://trackintel.readthedocs.io/
- **Geolife Dataset**: https://www.microsoft.com/en-us/download/details.aspx?id=52367
- **DBSCAN Algorithm**: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

---

## Version History

- **v1.0** (2024-11-30): Initial config-driven preprocessing implementation
  - Geolife preprocessing validated (matches baseline exactly)
  - DIY preprocessing implemented
  - Full YAML configuration support
  - Comprehensive documentation

---

## Contact & Support

For issues or questions about preprocessing:
1. Check this documentation first
2. Review `docs/PREPROCESSING_COMPARISON.md` for implementation details
3. Check config files in `configs/preprocessing/`
4. Review preprocessing logs for error messages

---

**Last Updated:** November 30, 2024  
**Maintainer:** Project Team
