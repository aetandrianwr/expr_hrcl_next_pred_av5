# Next-Location Prediction: Production-Level Implementation

A production-ready implementation of hierarchical transformer-based next-location prediction with comprehensive experiment tracking and configuration management.

## 🎯 Key Features

- **YAML-based Configuration**: All experiments configured via YAML files
- **Automatic Experiment Tracking**: Every run creates a unique directory with logs and checkpoints
- **CSV Benchmarking**: Automatic logging of all experiments to CSV for easy comparison
- **Multi-Dataset Support**: Easy adaptation to new datasets through standardized configuration
- **Reproducible Results**: Fixed seeds and deterministic training
- **Production-Ready**: Proper logging, checkpointing, and error handling

## 📁 Project Structure

```
expr_hrcl_next_pred_av5/
├── configs/                    # YAML configuration files
│   ├── geolife_default.yaml   # Default GeoLife configuration
│   └── template.yaml          # Template for custom datasets
├── data/                       # Dataset storage
│   └── geolife/               # GeoLife dataset
│       ├── geolife_transformer_7_train.pk
│       ├── geolife_transformer_7_validation.pk
│       └── geolife_transformer_7_test.pk
├── docs/                       # Documentation
│   ├── README.md              # Main documentation
│   ├── INDEX.md               # Documentation index
│   ├── PROJECT_SUMMARY.md     # Project overview
│   ├── COMPREHENSIVE_REPORT.md
│   ├── VALIDATION_CERTIFICATE.md
│   └── COMPLETE_IMPLEMENTATION_GUIDE.md
├── results/                    # Experiment results
│   ├── benchmark_results.csv  # CSV log of all experiments
│   ├── checkpoints/           # Global checkpoint storage
│   ├── logs/                  # Global log storage
│   └── predictions/           # Prediction outputs
├── runs/                       # Individual experiment runs
│   └── {experiment_name}_{timestamp}/
│       ├── checkpoints/       # Run-specific checkpoints
│       ├── logs/              # Run-specific logs
│       ├── predictions/       # Run-specific predictions
│       └── config.yaml        # Saved configuration
├── src/                        # Source code
│   ├── data/                  # Data loading utilities
│   ├── evaluation/            # Metrics and evaluation
│   ├── models/                # Model architectures
│   ├── training/              # Training utilities
│   │   ├── trainer.py         # Legacy trainer
│   │   └── trainer_v3.py      # Production trainer
│   └── utils/                 # Utility functions
│       ├── config_manager.py  # Configuration management
│       ├── results_tracker.py # Results tracking
│       └── logger.py          # Logging utilities
└── train_model.py             # Main training script

```

## 🚀 Quick Start

### 1. Train with Default Configuration

```bash
# Train with default GeoLife configuration
python train_model.py --config configs/geolife_default.yaml

# Train with specific seed
python train_model.py --config configs/geolife_default.yaml --seed 42
```

### 2. View Results

All results are automatically logged to `results/benchmark_results.csv`:

```bash
# View all experiments
cat results/benchmark_results.csv

# Or use Python
import pandas as pd
df = pd.read_csv('results/benchmark_results.csv')
print(df[['experiment_name', 'test_acc@1', 'test_f1', 'test_mrr']])
```

### 3. Reproduce Baseline Results

```bash
# Reproduce baseline (seed 42 is default)
python train_model.py --config configs/geolife_default.yaml --seed 42
```

Expected results:
- Test Acc@1: ~47-49%
- Test F1: ~45-47%
- Test MRR: ~60-62%

## 📝 Configuration Guide

### Understanding YAML Configuration

Each YAML file contains all experiment settings:

```yaml
experiment:
  name: "my_experiment"
  description: "Description of experiment"
  dataset: "geolife"

data:
  dataset_name: "geolife"
  data_dir: "data/geolife"
  num_locations: 1187
  num_users: 46

model:
  d_model: 128
  nhead: 4
  num_layers: 2
  dropout: 0.3

training:
  batch_size: 96
  learning_rate: 0.0025
  num_epochs: 120
```

### Creating Custom Configuration

1. Copy template:
```bash
cp configs/template.yaml configs/my_experiment.yaml
```

2. Edit configuration:
```yaml
experiment:
  name: "my_experiment"
  description: "Testing larger model"

model:
  d_model: 256  # Increase model size
  num_layers: 3

training:
  learning_rate: 0.001  # Lower learning rate
```

3. Run experiment:
```bash
python train_model.py --config configs/my_experiment.yaml
```

## 🔄 Adding New Datasets

### Step 1: Prepare Data

Place your dataset files in `data/your_dataset/`:
```
data/
└── your_dataset/
    ├── train.pk
    ├── val.pk
    └── test.pk
```

### Step 2: Create Configuration

Copy and modify template:
```bash
cp configs/template.yaml configs/your_dataset.yaml
```

Update dataset-specific parameters:
```yaml
experiment:
  name: "your_dataset_baseline"
  dataset: "your_dataset"

data:
  dataset_name: "your_dataset"
  data_dir: "data/your_dataset"
  train_file: "train.pk"
  val_file: "val.pk"
  test_file: "test.pk"
  num_locations: YOUR_NUM_LOCATIONS
  num_users: YOUR_NUM_USERS
```

### Step 3: Run Training

```bash
python train_model.py --config configs/your_dataset.yaml
```

## 📊 Experiment Tracking

### Run Directory Structure

Each training run creates a unique directory:
```
runs/geolife_baseline_20241130_123456/
├── checkpoints/
│   └── best_model.pt
├── logs/
│   └── training_20241130_123456.log
├── predictions/
└── config.yaml
```

### Benchmark CSV Format

The `results/benchmark_results.csv` tracks all experiments:

| Column | Description |
|--------|-------------|
| timestamp | When experiment was run |
| experiment_name | Name from config |
| dataset | Dataset used |
| run_dir | Path to run directory |
| d_model, num_layers, etc. | Model hyperparameters |
| batch_size, learning_rate | Training hyperparameters |
| val_acc@1, val_f1, etc. | Validation metrics |
| test_acc@1, test_f1, etc. | Test metrics |
| training_time | Total training time (seconds) |

### Analyzing Results

```python
import pandas as pd

# Load all results
df = pd.read_csv('results/benchmark_results.csv')

# Find best experiment
best = df.loc[df['test_acc@1'].idxmax()]
print(f"Best experiment: {best['experiment_name']}")
print(f"Test Acc@1: {best['test_acc@1']:.2f}%")

# Compare experiments
experiments = ['baseline', 'large_model', 'high_lr']
comparison = df[df['experiment_name'].isin(experiments)]
print(comparison[['experiment_name', 'test_acc@1', 'test_f1']])

# Plot learning curves
import matplotlib.pyplot as plt
plt.plot(df['timestamp'], df['test_acc@1'])
plt.xlabel('Time')
plt.ylabel('Test Acc@1')
plt.show()
```

## 🧪 Metrics

All experiments track:
- **Accuracy@K**: Top-k prediction accuracy (k=1,3,5,10)
- **F1 Score**: Weighted F1 score
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain

## 🔧 Advanced Usage

### Override Config from Command Line

While not directly supported via CLI, you can modify configs programmatically:

```python
from src.utils.config_manager import ConfigManager

config = ConfigManager('configs/geolife_default.yaml', {
    'training.learning_rate': 0.001,
    'model.num_layers': 3
})
```

### Custom Training Loop

```python
from train_model import set_seed
from src.utils.config_manager import ConfigManager
from src.training.trainer_v3 import ProductionTrainer

config = ConfigManager('configs/geolife_default.yaml')
set_seed(config.get('system.seed'))

# Your custom training code here
trainer = ProductionTrainer(model, train_loader, val_loader, config)
results = trainer.train()
```

## 📚 Documentation

Comprehensive documentation available in `docs/`:
- `README.md`: Main documentation (this file)
- `PROJECT_SUMMARY.md`: Project overview and objectives
- `COMPREHENSIVE_REPORT.md`: Detailed technical report
- `VALIDATION_CERTIFICATE.md`: Results validation
- `COMPLETE_IMPLEMENTATION_GUIDE.md`: Implementation details

## ⚙️ System Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU training)
- 8GB+ RAM
- 2GB+ disk space for data and models

## 📦 Dependencies

Install required packages:
```bash
pip install torch numpy pandas scikit-learn pyyaml
```

## 🐛 Troubleshooting

### Out of Memory
Reduce batch size in config:
```yaml
training:
  batch_size: 64  # Reduce from 96
```

### Slow Training
Enable GPU:
```yaml
system:
  device: "cuda"
```

### Results Not Matching
Ensure seed is fixed:
```yaml
system:
  seed: 42
  deterministic: true
```

## 📄 License

This project is for academic research purposes.

## 🙏 Acknowledgments

- GeoLife dataset from Microsoft Research
- PyTorch team for the framework
- Research community for baseline methods

## 📧 Contact

For questions and support, please open an issue in the repository.
