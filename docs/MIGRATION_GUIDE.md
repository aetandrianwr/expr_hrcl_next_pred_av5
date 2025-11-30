# Migration Guide: v1.0 → v2.0

This guide helps users transition from the old codebase structure to the new production-level system.

## Quick Migration

### Old Way (v1.0)
```bash
# Training with hardcoded config
python src/train.py
```

### New Way (v2.0)
```bash
# Training with YAML config
python train_model.py --config configs/geolife_default.yaml --seed 42
```

## What Changed

### 1. Project Structure

**Before:**
```
project/
├── src/
│   ├── train.py
│   ├── configs/config.py  # Python config
│   ├── models/
│   └── ...
├── README.md
├── REPORT.md
└── ...
```

**After:**
```
project/
├── configs/               # YAML configs (NEW)
│   ├── geolife_default.yaml
│   └── template.yaml
├── docs/                  # All documentation (MOVED)
│   ├── README.md
│   └── ...
├── results/               # Global results (NEW)
│   └── benchmark_results.csv
├── runs/                  # Experiment runs (NEW)
│   └── {experiment}_{timestamp}/
├── src/
│   ├── utils/             # NEW: Production utilities
│   ├── training/
│   │   └── trainer_v3.py  # NEW: Production trainer
│   └── ...
├── train_model.py         # NEW: Main entry point
└── view_results.py        # NEW: Results viewer
```

### 2. Configuration System

**Before (config.py):**
```python
class Config:
    batch_size = 96
    learning_rate = 0.0025
    # ...
```

**After (YAML):**
```yaml
training:
  batch_size: 96
  learning_rate: 0.0025
  # ...
```

### 3. Running Experiments

**Before:**
```bash
# Edit src/configs/config.py manually
python src/train.py
```

**After:**
```bash
# Use YAML config
python train_model.py --config configs/geolife_default.yaml

# Override seed from command line
python train_model.py --config configs/geolife_default.yaml --seed 123
```

### 4. Results Tracking

**Before:**
- Manual result collection
- Scattered log files
- No automatic comparison

**After:**
- Automatic CSV logging (`results/benchmark_results.csv`)
- Unique run directories with full logs
- Easy comparison with `view_results.py`

```bash
# View all results
python view_results.py

# View top 10 by specific metric
python view_results.py --top 10 --metric test_acc@1
```

## Converting Old Configs to YAML

If you have old Python configs, convert them to YAML:

**Old (config.py):**
```python
class Config:
    batch_size = 96
    learning_rate = 0.0025
    d_model = 128
    num_layers = 2
```

**New (my_config.yaml):**
```yaml
experiment:
  name: "my_experiment"
  dataset: "geolife"

training:
  batch_size: 96
  learning_rate: 0.0025

model:
  d_model: 128
  num_layers: 2
```

## Using Legacy Code

The old training scripts still work but are deprecated:

```bash
# Old way (still works, but not recommended)
python src/train.py

# New way (recommended)
python train_model.py --config configs/geolife_default.yaml
```

## Benefits of Migration

1. **Better Organization**: All configs in one place (`configs/`)
2. **Experiment Tracking**: Automatic logging to CSV
3. **Reproducibility**: Every run saved with full configuration
4. **Multi-Dataset Support**: Easy dataset switching via configs
5. **Documentation**: Organized in `docs/` directory

## Adding New Datasets

**Before:** Manually edit code in multiple files

**After:** Copy template and modify:

```bash
# Copy template
cp configs/template.yaml configs/my_dataset.yaml

# Edit configuration
vim configs/my_dataset.yaml

# Run training
python train_model.py --config configs/my_dataset.yaml
```

## Experiment Comparison

**Before:** Manual tracking in Excel/spreadsheet

**After:** Automatic CSV with all experiments:

```python
import pandas as pd

# Load all experiments
df = pd.read_csv('results/benchmark_results.csv')

# Compare configurations
print(df[['experiment_name', 'test_acc@1', 'learning_rate', 'num_layers']])

# Find best experiment
best = df.loc[df['test_acc@1'].idxmax()]
print(f"Best config: {best['run_dir']}")
```

## Troubleshooting

### "Module not found: utils"
Make sure you're running from the project root:
```bash
cd /path/to/project
python train_model.py --config configs/geolife_default.yaml
```

### "Config file not found"
Check the config path is correct:
```bash
ls configs/  # List available configs
python train_model.py --config configs/geolife_default.yaml
```

### Results not saving
Make sure you have write permissions:
```bash
mkdir -p results runs
chmod -R u+w results runs
```

## Getting Help

1. Check documentation in `docs/README.md`
2. View example config: `configs/geolife_default.yaml`
3. Run with default settings: `python train_model.py`

## Quick Reference

| Task | Old Command | New Command |
|------|-------------|-------------|
| Train model | `python src/train.py` | `python train_model.py --config configs/geolife_default.yaml` |
| View results | Manual | `python view_results.py` |
| Change config | Edit `config.py` | Edit YAML or create new config |
| Add dataset | Edit multiple files | Copy `template.yaml`, edit, run |
| Track experiments | Manual spreadsheet | Automatic CSV logging |

---

For more details, see `docs/README.md`
