# Preprocessing Module

Config-driven preprocessing pipeline for GPS trajectory data.

## Quick Start

### Geolife Dataset
```bash
# Preprocess raw data
python preprocessing/geolife_config.py --config configs/preprocessing/geolife.yaml

# Generate training files
python preprocessing/generate_transformer_data.py --config configs/preprocessing/geolife.yaml
```

### DIY Dataset
```bash
# Test with sample
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml --sample 100000

# Full preprocessing
python preprocessing/diy_config.py --config configs/preprocessing/diy.yaml

# Generate training files
python preprocessing/generate_transformer_data.py --config configs/preprocessing/diy.yaml
```

## Files

| File | Purpose |
|------|---------|
| `geolife_config.py` | Geolife preprocessing (config-driven) |
| `diy_config.py` | DIY preprocessing (config-driven) |
| `generate_transformer_data.py` | Generate .pk files for training |
| `utils.py` | Shared utility functions |

## Configuration

All parameters are in YAML files:
- `configs/preprocessing/geolife.yaml` - Geolife parameters
- `configs/preprocessing/diy.yaml` - DIY parameters

## Documentation

See **`docs/PREPROCESSING_DOCUMENTATION.md`** for complete documentation:
- Detailed usage guide
- Parameter tuning
- Troubleshooting
- Expected results

## Key Parameters

| Parameter | Geolife | DIY | Purpose |
|-----------|---------|-----|---------|
| **epsilon** | 20m | 50m | Location clustering radius |
| **dist_threshold** | 200m | 100m | Staypoint detection distance |
| **day_filter** | 50 | 60 | Minimum tracking days |
| **min_thres** | null | 0.6 | Quality threshold |
| **mean_thres** | null | 0.7 | Mean quality threshold |

## Output

Training files generated in `data/{dataset}/`:
- `{dataset}_transformer_7_train.pk`
- `{dataset}_transformer_7_validation.pk`
- `{dataset}_transformer_7_test.pk`

## Verified Results

**Geolife:**
- ✅ 7424 train sequences
- ✅ 1186 max location ID
- ✅ 45 users
- ✅ Test Acc@1: ~47%

---

For complete documentation, see `docs/PREPROCESSING_DOCUMENTATION.md`
