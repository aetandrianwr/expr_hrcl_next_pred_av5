# Architectural Justification: Ablation Study Notebook

## 📓 File
`architecture_justification_ablation_study.ipynb`

## 🎯 Purpose
This notebook provides **rigorous, empirical justification** for the History-Centric Model architecture through comprehensive ablation studies and comparative experiments.

## 🔬 Research Questions Answered

1. **Why do we need the History Scoring Module?**
   - Evidence: ~84% of next locations appear in visit history
   - Analysis: History coverage study + baseline performance

2. **Why do we need the Transformer branch?**
   - Evidence: Learns complex temporal patterns beyond simple recency/frequency
   - Analysis: Transformer-Only vs. History-Only comparison

3. **Why do we need BOTH components together?**
   - Evidence: Hybrid achieves superior performance
   - Analysis: Ablation studies showing complementary strengths

## 📊 Models Evaluated

| Model | Description | Purpose |
|-------|-------------|---------|
| **History-Only** | Pure recency + frequency scoring | Upper bound of non-learned approach |
| **Transformer-Only** | Deep learning without history bias | Can DL alone match history-centric? |
| **History-Centric** | Full hybrid architecture | Our proposed solution |

## 🎓 Key Features

- ✅ **Self-Contained**: No external project dependencies
- ✅ **Reproducible**: Fixed seed (42), same data splits
- ✅ **Comprehensive**: 25 cells with detailed explanations
- ✅ **Executable**: Run top-to-bottom without errors
- ✅ **Educational**: Clear explanations at every step
- ✅ **Rigorous**: Evidence-based conclusions

## 🚀 How to Run

```bash
cd notebooks/
jupyter notebook architecture_justification_ablation_study.ipynb
```

Then:
1. Run all cells (Cell → Run All)
2. Review results and visualizations
3. Read conclusions

**Note:** Training takes ~10-20 minutes (20 epochs). For full results, increase to 120 epochs.

## 📈 Expected Results

| Metric | History-Only | Transformer-Only | History-Centric |
|--------|--------------|------------------|-----------------|
| Acc@1  | ~35-40%      | ~42-47%          | ~47-52% |
| Acc@5  | ~60-65%      | ~68-73%          | ~72-77% |
| MRR    | ~45-50%      | ~55-60%          | ~60-65% |

The History-Centric model outperforms both baselines, demonstrating the value of combining history scoring with transformer learning.

## 📚 Notebook Structure

1. **Executive Summary**: Research questions and design
2. **Setup**: Imports and environment
3. **Data Loading**: GeoLife dataset
4. **History Coverage Analysis**: ~84% coverage finding
5. **PyTorch Dataset**: Data preparation
6. **Model Implementations**: 3 model variants
7. **Evaluation Metrics**: Acc@K, MRR, F1
8. **Training Function**: Training loop
9. **Experiments**: Train and evaluate all models
10. **Results Comparison**: Tables and visualizations
11. **Conclusions**: Architectural justification

## ✅ Reproducibility

- Random seed: 42
- PyTorch version: ≥1.12
- Data: GeoLife preprocessed splits
- Hyperparameters: Documented in code

## 📝 Citation

If you use this analysis in your research, please cite the project and mention the ablation study notebook.

## 🤝 Contributing

This notebook is part of the History-Centric Next-Location Prediction project. For questions or improvements, please open an issue in the repository.

---

**Last Updated:** November 30, 2024
**Status:** ✅ Complete and Tested
**Location:** `/notebooks/architecture_justification_ablation_study.ipynb`
