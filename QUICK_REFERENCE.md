# Quick Reference - Configuration & Model Fixes

## ✅ What Was Fixed

1. **Config files** - Now match actual model architecture
2. **Models** - Now respect config parameters (with defaults)
3. **Validation** - Warns if config doesn't match model
4. **max_seq_len** - Now properly inferred from dataset

## 🚀 How to Use

### Standard Training
```bash
# DIY dataset
python train_model.py --config configs/diy_skip_first_part_default.yaml

# GeoLife dataset  
python train_model.py --config configs/geolife_default.yaml
```

### Custom Architecture
Create a new config file:
```yaml
model:
  loc_emb_dim: 64      # Increase from default 56
  user_emb_dim: 16     # Increase from default 12
  d_model: 128         # Increase from default 80
  nhead: 8             # More attention heads
  dropout: 0.2         # Less dropout
```

### New Dataset
```yaml
data:
  data_dir: "data/my_dataset"
  train_file: "my_train.pk"
  val_file: "my_val.pk"
  test_file: "my_test.pk"
  
# num_locations, num_users, max_seq_len auto-detected!
```

## 📊 Validation Output

### Good (Config Matches)
```
✓ d_model: 80
✓ loc_emb_dim: 56
✓ user_emb_dim: 12
✓ All config parameters match model architecture
```

### Warning (Mismatch)
```
⚠ MISMATCH: d_model: Config=128, Model=80
WARNING: Model architecture differs from config file!
```

## 📁 Documentation Files

- `COMPREHENSIVE_PARAMETER_AUDIT.md` - Full technical audit
- `CONFIG_MODEL_FIXES_SUMMARY.md` - Summary of all fixes
- `HARDCODED_PARAMETERS_AUDIT.md` - max_seq_len fix details

## 🔧 Architecture Constraints

**Important:** Ensure dimensions add up correctly:
```
loc_emb_dim + user_emb_dim + temporal_dim = d_model

Examples:
  56 + 12 + 12 = 80 ✓
  64 + 16 + 48 = 128 ✓
  128 + 32 + 96 = 256 ✓
```

## ⚙️ Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loc_emb_dim` | 56 | Location embedding size |
| `user_emb_dim` | 12 | User embedding size |
| `d_model` | 80 | Model dimension |
| `nhead` | 4 | Attention heads |
| `dim_feedforward` | 160 | FFN dimension |
| `dropout` | 0.35 | Dropout rate |

## 🎯 Auto-Inferred (Don't Configure)

These are **automatically detected** from your dataset:
- `num_locations` - Vocabulary size
- `num_users` - User count
- `max_seq_len` - Max sequence length
- `num_weekdays` - Always 7

## ✨ What's New

**Before:**
- Config said one thing, model did another ❌
- Could not tune architecture via config ❌
- No warnings about mismatches ❌

**After:**
- Config accurately reflects model ✅
- Fully configurable via YAML ✅
- Validation warns about issues ✅
- Backward compatible ✅

## 🔍 Quick Check

Verify everything works:
```bash
# Should show validation output
python train_model.py --config configs/diy_skip_first_part_default.yaml 2>&1 | grep -A 5 "VALIDATION"
```

Expected output:
```
MODEL CONFIGURATION VALIDATION
================================================================================
✓ d_model: 80
✓ loc_emb_dim: 56
✓ user_emb_dim: 12
✓ All config parameters match model architecture
```

## 💡 Tips

1. **Start with defaults** - Optimized for <500K params
2. **Check validation** - Always review validation output
3. **Monitor params** - Training will show total parameter count
4. **Experiment safely** - Models use defaults if config missing

## 🆘 Troubleshooting

**Validation warnings?**
- Check if config values are intentional
- Update config to match model or vice versa

**Parameter count too high?**
- Reduce `d_model`, `loc_emb_dim`, `user_emb_dim`
- Default values stay under 500K for most datasets

**Training crashes?**
- Ensure dimensions add up: `loc + user + temporal = d_model`
- Check `dim_feedforward` is appropriate for `d_model`

---

**Status:** ✅ All fixes verified and working  
**Date:** 2025-12-01
