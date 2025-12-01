# Comprehensive Parameter Audit Report

**Date:** 2025-12-01  
**Scope:** ALL hardcoded parameters across entire codebase  
**Status:** 🔴 **CRITICAL ISSUES FOUND**

---

## Executive Summary

### Critical Issues (Must Fix)

1. 🔴 **history_centric.py** - Active model ignores ALL config embedding dimensions
2. 🔴 **final_model.py** - Ignores config embedding dimensions  
3. 🟡 **Hardcoded dropout rates** - Not configurable
4. 🟡 **Hardcoded layer dimensions** - Not configurable

### What Works Correctly ✅

- ✅ `num_locations`, `num_users`, `num_weekdays` - Inferred from dataset
- ✅ `max_seq_len` - Inferred from dataset (fixed earlier)
- ✅ Data pipeline - Properly uses inferred values
- ✅ Other models (transformer_model, efficient_transformer) - Use config correctly

---

## Detailed Findings

### 1. Critical: history_centric.py (CURRENTLY USED MODEL)

**Problem:** Completely ignores config parameters and uses hardcoded values

| Parameter | Config Value | Model Uses | Status |
|-----------|-------------|------------|--------|
| loc_emb_dim | 64 | **56** (hardcoded) | ❌ IGNORED |
| user_emb_dim | 16 | **12** (hardcoded) | ❌ IGNORED |
| d_model | 128 | **80** (hardcoded) | ❌ IGNORED |
| dropout | 0.3 | **0.35** (hardcoded) | ❌ IGNORED |
| nhead | 4 | **4** (hardcoded) | ❌ IGNORED |

**Code Evidence:**
```python
# Line 31: Ignores config.d_model
self.d_model = 80  # Compact

# Line 34-35: Ignores config.loc_emb_dim, config.user_emb_dim
self.loc_emb = nn.Embedding(config.num_locations, 56, padding_idx=0)
self.user_emb = nn.Embedding(config.num_users, 12, padding_idx=0)

# Line 53: Ignores config.nhead
self.attn = nn.MultiheadAttention(80, 4, dropout=0.35, batch_first=True)

# Line 62: Ignores config.dropout
self.dropout = nn.Dropout(0.35)
```

**Impact:** 
- Users cannot configure the model via YAML
- Config file is misleading (says one thing, model does another)
- Model architecture cannot be experimented with
- Training experiments are not reproducible from config alone

---

### 2. Critical: final_model.py

**Problem:** Same issue as history_centric.py

| Parameter | Config Value | Model Uses | Status |
|-----------|-------------|------------|--------|
| loc_emb_dim | 64 | **64** (hardcoded, coincidentally same) | ❌ IGNORED |
| user_emb_dim | 16 | **16** (hardcoded, coincidentally same) | ❌ IGNORED |
| d_model | 128 | **96** (hardcoded) | ❌ IGNORED |
| dropout | 0.3 | **0.4** (hardcoded) | ❌ IGNORED |

---

### 3. Models That Use Config Correctly ✅

#### efficient_transformer.py ✅
```python
self.loc_emb = nn.Embedding(config.num_locations, config.loc_emb_dim, padding_idx=0)
self.user_emb = nn.Embedding(config.num_users, config.user_emb_dim, padding_idx=0)
self.weekday_emb = nn.Embedding(config.num_weekdays, config.weekday_emb_dim)
```

#### transformer_model.py ✅
```python
self.loc_emb = nn.Embedding(config.num_locations, config.loc_emb_dim, padding_idx=0)
self.user_emb = nn.Embedding(config.num_users, config.user_emb_dim, padding_idx=0)
```

#### compact_transformer.py ✅
```python
self.d_model = config.d_model
self.loc_emb = nn.Embedding(config.num_locations, config.d_model // 2, padding_idx=0)
```

---

### 4. Data-Dependent Parameters (Correctly Handled) ✅

All properly inferred from dataset files:

| Parameter | Source | Flow |
|-----------|--------|------|
| num_locations | Dataset max location ID + 1 | `infer_dataset_parameters()` → ModelConfig |
| num_users | Dataset max user ID + 1 | `infer_dataset_parameters()` → ModelConfig |
| num_weekdays | Always 7 | `infer_dataset_parameters()` → ModelConfig |
| max_seq_len | Dataset max sequence length | `infer_dataset_parameters()` → ModelConfig |

**Evidence:**
```python
# train_model.py lines 81-89
inferred_params = infer_dataset_parameters(train_file_path)

# Lines 133-136
self.num_locations = inferred_params['num_locations']
self.num_users = inferred_params['num_users']
self.num_weekdays = inferred_params['num_weekdays']
self.max_seq_len = inferred_params['max_seq_len']
```

✅ **CORRECT** - These should be inferred, not configured

---

### 5. Hardcoded Layer Dimensions

All models have some hardcoded internal layer dimensions:

```python
# history_centric.py
self.temporal_proj = nn.Linear(6, 12)  # Input features hardcoded
self.ff = nn.Sequential(
    nn.Linear(80, 160),   # d_model * 2
    nn.Linear(160, 80)    # back to d_model
)

# final_model.py
self.temporal_proj = nn.Linear(6, 16)
self.ff = nn.Sequential(
    nn.Linear(96, 192),   # d_model * 2
    nn.Linear(192, 96)
)
```

**Status:** 🟡 **Acceptable** - These are derived from input features (6 temporal features) and are proportional to d_model. However, they break if d_model changes.

---

### 6. Acceptable Hardcoded Constants

#### Time Conversions ✅
```python
hours = start_min_seq / 60.0  # Minutes to hours
time_rad = (hours / 24.0) * 2 * math.pi  # Hours to radians
wd_rad = (weekday_seq.float() / 7.0) * 2 * math.pi  # Weekday to radians
```
**Status:** ✅ **Correct** - Mathematical constants

#### Gap Embedding Size ✅
```python
self.gap_emb = nn.Embedding(8, ...)  # 8 time gap bins
```
**Status:** ✅ **Correct** - Matches preprocessing that bins diff values to 0-7

#### Padding Index ✅
```python
nn.Embedding(..., padding_idx=0)
```
**Status:** ✅ **Correct** - Padding is always index 0 by convention

---

## Recommended Fixes

### Priority 1: Fix history_centric.py (Critical)

**Current:** Hardcoded dimensions
```python
self.d_model = 80
self.loc_emb = nn.Embedding(config.num_locations, 56, padding_idx=0)
self.user_emb = nn.Embedding(config.num_users, 12, padding_idx=0)
```

**Should be:**
```python
# Use config values OR keep compact design as separate config
self.d_model = getattr(config, 'd_model', 80)  # Allow override
self.loc_emb = nn.Embedding(config.num_locations, 
                            getattr(config, 'loc_emb_dim', 56), 
                            padding_idx=0)
self.user_emb = nn.Embedding(config.num_users, 
                            getattr(config, 'user_emb_dim', 12), 
                            padding_idx=0)
```

**OR:** Create a separate config file specifically for the compact design:
- `configs/diy_skip_first_part_compact.yaml` with matching dimensions

### Priority 2: Fix final_model.py

Same approach as history_centric.py

### Priority 3: Add Configuration Validation

Add checks in train_model.py:
```python
# Warn if config doesn't match model's expectations
if hasattr(model, 'd_model') and model.d_model != config.get('model.d_model'):
    logger.warning(f"Model d_model ({model.d_model}) != Config d_model ({config.get('model.d_model')})")
```

---

## Testing Matrix

| Dataset | max_seq_len | num_locations | num_users | Status |
|---------|-------------|---------------|-----------|--------|
| GeoLife | 3 | 1187 | 183 | ✅ Tested |
| DIY Skip First | 100 | 7038 | 693 | ✅ Tested |
| Future datasets | Variable | Variable | Variable | ✅ Will work (inferred) |

---

## Conclusion

### What Must Be Fixed Immediately 🔴

1. **history_centric.py** - Make it respect config parameters
2. **final_model.py** - Make it respect config parameters
3. **Config validation** - Add warnings for mismatches

### What's Working Correctly ✅

1. ✅ Dataset parameter inference (num_locations, num_users, max_seq_len)
2. ✅ Data pipeline (DataLoader, collation)
3. ✅ Other models (transformer_model, efficient_transformer)
4. ✅ Mathematical constants (time conversions, gap bins)

### Root Cause

The compact models (history_centric, final_model) were designed with **hardcoded dimensions as an optimization strategy** to stay under 500K parameters. However, this makes them:
- Non-configurable
- Misleading (config says one thing, model does another)
- Hard to experiment with
- Not reproducible from config alone

### Recommended Approach

**Option A:** Make models respect config (more flexible)  
**Option B:** Create separate "compact" config files that match the hardcoded values  
**Option C:** Add model variants (e.g., HistoryCentricCompact vs HistoryCentricConfigurable)

My recommendation: **Option A** - Use config with sensible defaults, allow overrides
