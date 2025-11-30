# Comprehensive Mobility EDA Notebook

## Overview

**File**: `comprehensive_mobility_eda.ipynb`

This notebook provides comprehensive exploratory data analysis (EDA) for human mobility datasets, specifically designed to analyze the Geolife and DIY datasets with a focus on mobility patterns and behavioral metrics.

## Features

### Data Analysis Coverage

1. **Dataset Loading & Overview**
   - Geolife dataset (with 10K sampling option)
   - DIY dataset support
   - Basic statistics and dimensions

2. **Temporal Pattern Analysis**
   - Daily activity patterns (hourly distributions)
   - Weekly patterns (weekday vs weekend)
   - Duration distributions
   - Activity timing analysis

3. **Spatial Pattern Analysis**
   - Location visit frequencies
   - Spatial distributions
   - Location concentration metrics
   - Geographic patterns

4. **Human Mobility Metrics**
   - **Radius of Gyration**: Measures characteristic travel distance
   - **Location Entropy**: Quantifies movement predictability
   - **Return Time Analysis**: Analyzes routine behavior
   - **Transition Patterns**: Identifies common routes

5. **Visualizations**
   - Histograms and distributions
   - Time series plots
   - Statistical summaries
   - Spatial visualizations

### Self-Contained Design

✅ **No External Dependencies**: All logic is contained within the notebook  
✅ **Standard Libraries Only**: Uses pandas, numpy, matplotlib, seaborn, scipy, geopandas  
✅ **Reproducible**: Can be run independently without project scripts  
✅ **Well-Documented**: Comprehensive markdown explanations throughout  

## Usage

### Prerequisites

```bash
pip install pandas numpy geopandas matplotlib seaborn scipy shapely
```

### Required Data Files

- `../data/geolife/dataSet_geolife.csv`
- `../data/geolife/locations_geolife.csv`
- `../data/diy/dataSet_diy.csv` (optional)
- `../data/diy/locations_diy.csv` (optional)

### Running the Notebook

1. Open in Jupyter:
   ```bash
   jupyter notebook comprehensive_mobility_eda.ipynb
   ```

2. Run all cells sequentially (Cell → Run All)

3. The notebook will:
   - Load and sample the data (10K records from Geolife)
   - Perform comprehensive analysis
   - Generate visualizations
   - Display statistical summaries

## Notebook Structure

```
1. Setup & Data Loading
   └── Import libraries
   └── Load Geolife (10K sample)
   └── Load DIY dataset

2. Dataset Overview
   └── Basic statistics
   └── User activity distributions

3. Temporal Patterns
   └── Daily patterns
   └── Weekly patterns
   └── Duration analysis

4. Spatial Patterns
   └── Location frequencies
   └── Spatial distributions

5. Mobility Metrics
   └── Radius of Gyration
   └── Location Entropy
   └── Return Time Analysis

6. Transition Patterns
   └── Location transitions
   └── Common routes

7. Summary & Conclusions
```

## Key Metrics Explained

### Radius of Gyration

Measures the characteristic distance traveled by a user:

```
r_g = sqrt(1/n * Σ(r_i - r_cm)²)
```

Where:
- `r_i` = position of location i
- `r_cm` = center of mass of all locations
- `n` = number of visits

### Location Entropy

Quantifies the predictability of movement:

```
S = -Σ p_i * log₂(p_i)
```

Where:
- `p_i` = probability of visiting location i
- Higher entropy = more unpredictable movement

### Return Time

Time gap between consecutive visits to the same location, revealing:
- Daily routines (return time = 1 day)
- Weekly patterns (return time = 7 days)
- Occasional visits (return time > 7 days)

## Output

The notebook generates:
- Statistical summaries
- Distribution plots
- Box plots
- Bar charts
- Scatter plots
- Comprehensive textual analysis

## Specifications

- **Total Cells**: 41
- **Markdown Cells**: 19 (explanations and documentation)
- **Code Cells**: 22 (analysis and visualization)
- **File Size**: ~51 KB
- **Estimated Runtime**: 2-5 minutes (depending on data size)

## Use Cases

This notebook is suitable for:
- ✅ Research in human mobility patterns
- ✅ Urban planning analysis
- ✅ Transportation behavior studies
- ✅ Next location prediction model development
- ✅ Data quality assessment
- ✅ Educational purposes

## Version

- **Version**: 2.0
- **Last Updated**: November 2024
- **Status**: Complete and tested

## Repository

Part of the `expr_hrcl_next_pred_av5` project:
https://github.com/aetandrianwr/expr_hrcl_next_pred_av5

## License

Follow the project's license agreement.

---

**Note**: This notebook is designed to be completely self-contained and can be run without any external project scripts. All functions and computations are defined within the notebook itself.
