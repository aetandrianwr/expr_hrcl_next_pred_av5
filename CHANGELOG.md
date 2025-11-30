# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-11-30

### Major Restructuring - Production-Level PhD Quality

#### Added
- **YAML Configuration System**
  - All experiments configured via YAML files in `configs/`
  - Template for custom datasets (`configs/template.yaml`)
  - Easy parameter management and experiment tracking

- **Experiment Tracking**
  - Unique run directory for each experiment (`runs/{experiment}_{timestamp}/`)
  - Automatic saving of configuration, logs, and checkpoints per run
  - CSV benchmark file (`results/benchmark_results.csv`) tracking all experiments
  
- **New Utilities**
  - `ConfigManager`: YAML-based configuration management
  - `ResultsTracker`: Automatic CSV logging of all experiments
  - `ExperimentLogger`: Dual console/file logging
  - `ProductionTrainer`: Updated trainer with config manager support

- **Production Scripts**
  - `train_model.py`: Main training script with YAML config support
  - `view_results.py`: Script to view and analyze benchmark results

- **Documentation**
  - Reorganized all `.md` files into `docs/` directory
  - New comprehensive `README.md` with quickstart and guides
  - Configuration guide and dataset integration instructions

#### Changed
- **Project Structure**
  ```
  ├── configs/          # YAML configuration files
  ├── data/             # Dataset storage
  ├── docs/             # All documentation
  ├── results/          # Benchmark CSV and global results
  ├── runs/             # Individual experiment runs
  ├── src/              # Source code
  │   ├── data/
  │   ├── evaluation/
  │   ├── models/
  │   ├── training/
  │   └── utils/        # NEW: Utility modules
  ├── train_model.py    # NEW: Main entry point
  └── view_results.py   # NEW: Results viewer
  ```

- **Training Workflow**
  - Before: `python src/train.py` with hardcoded config
  - After: `python train_model.py --config configs/geolife_default.yaml --seed 42`

- **Results Management**
  - Before: Manual tracking, scattered logs
  - After: Automatic CSV logging, organized run directories

#### Improved
- **Reproducibility**: Fixed seed (42), deterministic training, saved configs
- **Multi-Dataset Support**: Easy dataset switching via YAML
- **Experiment Tracking**: Every run tracked with full configuration
- **Maintainability**: Modular utilities, clear separation of concerns

### Performance
- Test Acc@1: 49.40% (seed 42)
- Test F1: 45.48%
- Test MRR: 61.45%
- Test NDCG: 65.52%

## [1.0.0] - 2025-11-29

### Added
- F1 score metric calculation
- Weighted F1 score: `f1 = f1_score(true_ls, top1_ls, average="weighted")`
- F1 display in validation and test output

### Fixed
- Metric collection in validation loop
- Performance dictionary handling

## [0.9.0] - Initial Implementation

### Added
- Hierarchical transformer model for next-location prediction
- GeoLife dataset support
- Training pipeline with early stopping
- Metrics: Acc@K, MRR, NDCG
- Model checkpointing
