# DIY Test Configuration

This is a **test-only configuration** (`diy_test.yaml`) designed to verify that the preprocessing pipeline works from beginning to end with minimal data.

## Purpose

- **Quick testing**: Runs with small samples (e.g., 10,000 rows) to verify the code works
- **Debugging**: Helps identify issues in the preprocessing pipeline
- **Development**: Allows rapid iteration without processing the full 165M row dataset

## Key Differences from Production Config

### Very Loose Thresholds

1. **Quality Filtering**: SKIPPED entirely (`skip_check: true`)
   - No user quality requirements
   - Accepts all users regardless of tracking duration or quality

2. **Staypoint Generation**:
   - `dist_threshold: 200m` (vs 100m) - more lenient distance
   - `time_threshold: 5min` (vs 30min) - lower time requirement
   
3. **Activity Flag**:
   - `time_threshold: 5min` (vs 25min) - very low threshold

4. **Location Clustering**:
   - `epsilon: 100m` (vs 50m) - larger clustering radius
   - `num_samples: 1` (vs 2) - minimum DBSCAN requirement

5. **Sequence Generation**:
   - `min_sequence_length: 2` (vs 3) - fewer history records required

## Usage

```bash
# Test with 10,000 rows (fast, ~10 seconds)
python preprocessing/diy_config.py --config configs/preprocessing/diy_test.yaml --sample 10000

# Test with 100,000 rows (slower, may get some valid users)
python preprocessing/diy_config.py --config configs/preprocessing/diy_test.yaml --sample 100000

# Test with 1,000,000 rows (should get valid results)
python preprocessing/diy_config.py --config configs/preprocessing/diy_test.yaml --sample 1000000
```

## Expected Outcomes

### With 10,000 rows:
- ✅ Pipeline runs without errors
- ⚠️ May have 0 final users (too sparse)
- ✅ Verifies code correctness

### With 100,000+ rows:
- ✅ Should get some valid users
- ✅ Files are generated in `data/diy_test/`

## Output Files

- `data/diy_test/locations_diy.csv` - Discovered locations
- `data/diy_test/dataSet_diy.csv` - Processed staypoints
- `data/diy_test/sp_time_temp_diy.csv` - Temporary time-enriched data
- `data/diy_test/quality/` - Quality metrics (if not skipped)

## For Production

Use the regular `diy.yaml` configuration file with appropriate quality thresholds and run on the full dataset or large samples (millions of rows).

## Notes

- This configuration is **NOT suitable for model training**
- It sacrifices data quality for pipeline verification
- Use only for testing and development
