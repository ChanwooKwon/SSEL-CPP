# SSEL-CPP Filter Method Comparison

This directory contains the implementation of 4 filter methods used in the SSEL-CPP paper for systematic feature selection comparison.

## Filter Methods

### Method 1: Selection by |τ|, Ranking by |τ|
- **File**: `filters/feature_importance_1.py`
- **Selection criterion**: Absolute Kendall's tau for removing correlated features
- **Ranking criterion**: Absolute Kendall's tau for final ranking

### Method 2: Selection by τ, Ranking by |τ|
- **File**: `filters/feature_importance_2.py`
- **Selection criterion**: Kendall's tau for removing correlated features
- **Ranking criterion**: Absolute Kendall's tau for final ranking

### Method 3: Selection by τ, Ranking by τ
- **File**: `filters/feature_importance_3.py`
- **Selection criterion**: Kendall's tau for removing correlated features
- **Ranking criterion**: Kendall's tau for final ranking

### Method 4: Selection by |τ|, Ranking by τ (BEST)
- **File**: `filters/feature_importance_4.py`
- **Selection criterion**: Absolute Kendall's tau for removing correlated features
- **Ranking criterion**: Kendall's tau for final ranking

## Directory Structure

```
filter_methods/
├── filters/                    # Filter method implementations
│   ├── feature_importance_1.py
│   ├── feature_importance_2.py
│   ├── feature_importance_3.py
│   └── feature_importance_4.py
├── models/                     # Random Forest training
│   ├── RandomForest_tuning.py
│   └── RandomForest_combination_features.py
├── analysis/                   # Statistical analysis
│   ├── Combination_result_csv.py
│   └── statistical_analysis_percount_significance.py
├── data/                       # Generated feature rankings
│   └── feature_importance/
└── results/                    # Experiment results
    └── combination_results/
```

## Usage

### 1. Generate Feature Rankings
```bash
# Run each filter method
python filters/feature_importance_1.py
python filters/feature_importance_2.py
python filters/feature_importance_3.py
python filters/feature_importance_4.py
```

### 2. Hyperparameter Tuning
```bash
python models/RandomForest_tuning.py
```

### 3. Evaluate Combinations
```bash
python models/RandomForest_combination_features.py
```

### 4. Statistical Analysis
```bash
python analysis/Combination_result_csv.py
python analysis/statistical_analysis_percount_significance.py
```

## Requirements

- pandas
- numpy
- scipy
- scikit-learn
- multiprocessing

## Input Data

- `mordred2d_normalized.csv`: Normalized Mordred 2D descriptors
- Feature count evaluation: 50, 100, 150, 200, 250, 300

## Output

- **Feature rankings**: `data/feature_importance/feature_importance_*.csv`
- **Model parameters**: `data/best_rf_params_per_method.csv`
- **Results**: `results/combination_results/4x4_rf_combination_results_top*.csv`

## Key Findings

Filter Method 4 consistently demonstrated superior performance across different feature counts, leading to its selection for the SSEL-CPP two-stage feature selection pipeline.