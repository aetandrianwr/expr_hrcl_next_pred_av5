# Hardcoded Parameters Audit Report

**Date:** 2025-12-01  
**Issue:** Positional encoding buffer was hardcoded to 60, but DIY dataset has max_seq_len=100  
**Status:** ✅ FIXED AND VERIFIED

## Summary

All models and data pipeline have been audited and fixed to properly handle variable `max_seq_len` from datasets.

## Fixed Issues

### 1. ✅ src/models/history_centric.py (FIXED)
- **Line 44:** Changed from hardcoded `pe = torch.zeros(60, 80)` 
- **Fix:** Now uses `max_len = getattr(config, 'max_seq_len', 100)`
- **Status:** ✅ Correct - dynamically infers from config

### 2. ✅ src/models/final_model.py (FIXED)
- **Line 50:** Changed from hardcoded `pe = torch.zeros(60, 96)`
- **Fix:** Now uses `max_len = getattr(config, 'max_seq_len', 100)`
- **Status:** ✅ Fixed in this audit

### 3. ✅ src/models/compact_transformer.py (FIXED)
- **Line 20:** Removed default `max_len=60` from `CompactPositionalEncoding.__init__`
- **Line 106:** Already correctly uses `config.max_seq_len` when instantiating
- **Status:** ✅ Fixed in this audit

### 4. ✅ src/models/transformer_model.py (VERIFIED)
- **Line 21:** Has default `max_len=100` in `SinusoidalPositionalEncoding`
- **Line 104:** Correctly uses `config.max_seq_len` when instantiating
- **Status:** ✅ Correct - default is fallback only

### 5. ✅ src/models/efficient_transformer.py (VERIFIED)
- **Line 21:** Has default `max_len=100` in `SinusoidalPositionalEncoding`
- **Line 133:** Correctly uses `config.max_seq_len` when instantiating
- **Status:** ✅ Correct - default is fallback only

## Data Pipeline Verification

### ✅ train_model.py (VERIFIED CORRECT)
- **Line 81:** Infers `max_seq_len` from training data: `infer_dataset_parameters(train_file_path)`
- **Line 97:** Uses inferred value: `max_seq_len = inferred_params['max_seq_len']`
- **Line 105, 112, 119:** Passes `max_seq_len` to all dataloaders
- **Line 136:** Passes `max_seq_len` to model config
- **Status:** ✅ Correct - fully dynamic based on dataset

### ✅ src/data/dataset.py (VERIFIED CORRECT)
- **Line 25:** Has default `max_seq_len=60` in `GeoLifeDataset.__init__`
- **Line 120:** Has default `max_seq_len=60` in `get_dataloader`
- **Status:** ✅ Correct - defaults are overridden by train_model.py passing inferred values

## Parameter Inference Flow

```
Dataset (*.pk file)
    ↓
infer_dataset_parameters() → extracts actual max_seq_len from data
    ↓
train_model.py → uses inferred max_seq_len
    ↓
├─→ get_dataloader(..., max_seq_len=inferred) → Dataset
    │
    └─→ ModelConfig(max_seq_len=inferred) → Model
            ↓
        Positional Encoding buffer sized correctly
```

## Other Hardcoded Values (Acceptable)

### Time Gap Embedding
- All models: `nn.Embedding(8, ...)` for time gap
- **Status:** ✅ Acceptable - diff values are binned to 0-7 during preprocessing

### Model Dimensions
- `history_centric.py`: `self.d_model = 80` (compact design)
- `final_model.py`: `self.d_model = 96` (compact design)
- **Status:** ✅ Acceptable - these are design choices, not data-dependent

### Time Conversion
- All models: `hours = start_min_seq / 60.0`
- **Status:** ✅ Acceptable - constant for minutes to hours conversion

## Testing Results

### GeoLife Dataset
- Max sequence length: 3
- Model config: Uses `max_seq_len=3`
- ✅ Training successful

### DIY Skip First Part Dataset
- Max sequence length: 100
- Model config: Uses `max_seq_len=100`
- ✅ Training successful after fixes

## Recommendations

1. ✅ **All positional encoding classes should accept max_len from config** - DONE
2. ✅ **Never hardcode buffer sizes that depend on data** - FIXED
3. ✅ **Always infer data-dependent parameters from actual dataset** - VERIFIED
4. ⚠️ **Consider adding assertion checks** - Future work

## Verification Commands

```bash
# Check all positional encoding definitions
grep -n "max_len" src/models/*.py

# Check how models use config.max_seq_len
grep -n "config.max_seq_len" src/models/*.py

# Verify data inference
grep -n "infer_dataset_parameters" train_model.py

# Check dataloader calls
grep -n "max_seq_len=max_seq_len" train_model.py
```

## Conclusion

✅ **ALL HARDCODED PARAMETER ISSUES RESOLVED**

The codebase now properly:
1. Infers `max_seq_len` from actual dataset
2. Passes it through entire pipeline (data → model)
3. Uses it to size positional encoding buffers correctly
4. Works with any dataset regardless of sequence length

No more hardcoded assumptions about sequence length!
