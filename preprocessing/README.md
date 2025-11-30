# Preprocessing Pipeline

This directory contains scripts for preprocessing mobility datasets for next-location prediction.

## Overview

The preprocessing pipeline converts raw mobility data into the format required by the HistoryCentricModel:
1. **Read raw data**: Position fixes (GPS coordinates with timestamps)
2. **Generate staypoints**: Detect stationary periods using spatial and temporal thresholds
3. **Filter users**: Select high-quality users based on tracking duration and consistency
4. **Generate locations**: Cluster staypoints into meaningful locations using DBSCAN
5. **Create sequences**: Generate historical sequences for model training
6. **Split dataset**: Divide into train/validation/test sets (60/20/20)

## Configuration

All preprocessing parameters are specified in YAML configuration files located in `configs/preprocessing/`:

- `geolife.yaml`: Configuration for Geolife dataset (Beijing, China)
- `diy.yaml`: Configuration for DIY dataset (Yogyakarta, Indonesia)

### Key Parameters

#### Staypoints Generation
- `dist_threshold`: Maximum distance (meters) for a staypoint
- `time_threshold`: Minimum time (minutes) to be considered a staypoint
- `gap_threshold`: Maximum gap (minutes) between position fixes

#### User Quality Filtering
- `day_filter`: Minimum tracking days required
- `window_size`: Sliding window size (weeks) for quality assessment
- `min_thres`: Minimum quality threshold (DIY only)
- `mean_thres`: Mean quality threshold (DIY only)

#### Location Generation
- `epsilon`: DBSCAN radius (meters) for clustering staypoints
- `num_samples`: Minimum samples for DBSCAN core points

## Usage

### Process Geolife Dataset

```bash
# Full pipeline (preprocessing + transformer data generation)
python preprocessing/preprocess_geolife.py --config configs/preprocessing/geolife.yaml
python preprocessing/generate_transformer_data.py --config configs/preprocessing/geolife.yaml
```

### Process DIY Dataset

```bash
# Full pipeline
python preprocessing/preprocess_diy.py --config configs/preprocessing/diy.yaml
python preprocessing/generate_transformer_data.py --config configs/preprocessing/diy.yaml
```

## Output Files

For each dataset, the following files are generated in `data/{dataset_name}/`:

### Intermediate Files
- `sp_time_temp_{dataset}.csv`: Staypoints with temporal information
- `locations_{dataset}.csv`: Detected locations with coordinates
- `dataSet_{dataset}.csv`: Final preprocessed dataset
- `valid_ids_{dataset}.pk`: Valid staypoint IDs for sequence generation
- `quality/`: User quality filtering results

### Final Model Input Files
- `{dataset}_transformer_7_train.pk`: Training sequences
- `{dataset}_transformer_7_validation.pk`: Validation sequences
- `{dataset}_transformer_7_test.pk`: Test sequences

## Data Format

### Pickle File Structure
Each `.pk` file contains a list of dictionaries with the following keys:
- `X`: Location IDs sequence (history)
- `user_X`: User IDs sequence
- `weekday_X`: Weekday sequence (0-6)
- `start_min_X`: Start time in minutes since midnight
- `dur_X`: Duration sequence in minutes
- `diff`: Days difference from current day
- `Y`: Target location ID (next location)

## Dataset Statistics

### Geolife (from Beijing, China)
- Raw users: 182
- Final users: 47
- Locations: ~960
- Training sequences: ~8,485
- Test Acc@1: ~48%

### DIY (from Yogyakarta, Indonesia)
- Raw data: 165M position fixes
- Processing time: Several hours (full dataset)
- Note: Use appropriate hardware for full processing

## Reproducibility

All preprocessing steps are controlled by the YAML configuration files and use seed=42 for reproducibility. The same configuration should produce identical results across runs.

## Dependencies

- pandas
- numpy
- geopandas
- trackintel
- scikit-learn
- shapely
- tqdm
- pyyaml
