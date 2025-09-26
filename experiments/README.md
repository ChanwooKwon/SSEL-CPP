# Cell-Penetrating Peptide Prediction Experiments

This directory contains all experimental components for cell-penetrating peptide prediction using machine learning approaches.

## Directory Structure

### 1. Performance Comparison (`performance_comparison/`)
Comparative analysis of different feature selection methods for peptide classification.

#### 1.1 All Features (`1_all_features/`)
Baseline experiments using all available molecular descriptors:
- **Models**: XGBoost, LightGBM, Random Forest, Gradient Boosting, Extra Trees, CatBoost
- **Cross-validation**: 5-fold StratifiedKFold
- **Features**: All 1613 Mordred molecular descriptors
- **Output**: Performance metrics (Accuracy, MCC, ROC-AUC) saved to individual `results/` directories

#### 1.2 Filter Method 4 (`2_filter_method4/`)
Feature selection using Filter Method 4 approach:
- **Feature Selection**: Based on statistical significance and importance scores
- **Models**: Same 6 models as baseline
- **Features**: Reduced feature set from filter method analysis
- **Output**: Comparative performance metrics

#### 1.3 SHAP Selected Features (`3_shap_selected/`)
Feature selection based on SHAP (SHapley Additive exPlanations) analysis:
- **Feature Selection**: Model-specific SHAP importance with curve optimization
- **Optimization Methods**: 
  - Lowess smoothing (XGB, LGBM, RF, GB, ERT)
  - Polynomial fitting (CatBoost)
- **Features**: Optimized feature sets per model
- **Output**: SHAP-based performance results

#### 1.4 Comparison Analysis (`4_comparison_analysis/`)
Statistical comparison between feature selection methods:
- `compare_all_vs_shap.py`: All Features vs SHAP Selected
- `compare_all_vs_filter.py`: All Features vs Filter Method 4
- `compare_filter_vs_shap.py`: Filter Method 4 vs SHAP Selected
- **Statistical Test**: Paired t-test for significance assessment
- **Output**: Detailed comparison reports with p-values and effect sizes

### 2. SHAP Feature Selection (`shap_feature_selection/`)
Complete SHAP analysis pipeline for feature importance and selection.

#### 2.1 Analysis (`analysis/`)
SHAP value computation and visualization:
- Model-specific SHAP analysis scripts
- Waterfall plots and summary visualizations
- Feature importance rankings

#### 2.2 Feature Optimization (`feature_optimization/`)
Curve fitting and feature selection optimization:
- **Lowess Method**: Non-parametric regression for smooth curves
- **Polynomial Method**: Parametric fitting for CatBoost
- **Output**: Optimized feature lists per model

### 3. Filter Methods (`filter_methods/`)
Statistical and univariate feature selection approaches.

#### 3.1 Method 4 (`method4/`)
Comprehensive filter-based feature selection:
- Statistical significance testing
- Feature importance scoring
- Correlation analysis
- Final feature ranking and selection

### 4. Model Training (`model_training/`)
Individual model training and hyperparameter optimization:
- Optuna-based hyperparameter tuning
- Cross-validation training
- Model persistence and evaluation

### 5. Ensemble Model Training (`ensemble_model_training/`)
Advanced ensemble methods and model combination strategies:
- Voting classifiers
- Stacking approaches
- Model combination optimization

### 6. Descriptor Analysis (`descriptor_analysis/`)
Comprehensive analysis of molecular descriptor characteristics and their effects.

#### 6.1 Feature Correlation (`feature_correlation/`)
Analysis of feature relationships and correlations:
- **Files**:
  - `Top5 features correlation_Pearson.py`: Pearson correlation analysis script
  - `correlation_heatmap_top5_features_pearson.png`: Correlation heatmap visualization
- **Purpose**: Understanding feature interdependencies and multicollinearity

#### 6.2 Effect Size Analysis (`effect_size_analysis/`)
Statistical analysis of descriptor effect sizes for different descriptor families:

##### AATSC - Averaged Autocorrelation of Topological Structure
- **Files**: `AATSC0i_mean_CI_MWU_CliffsDelta_m.png`, `AATSC_m.py`
- **Analysis**: Mean differences, confidence intervals, Mann-Whitney U test, Cliff's Delta effect size

##### ATSC2m - Autocorrelation of Topological Structure (2nd order, mass)
- **Files**: Multiple PNG visualizations for ATSC0m through ATSC8m, `ATSC2m_m.py`
- **Analysis**: Comprehensive autocorrelation analysis across different lag orders

##### BCUTc-1h - BCUT Descriptors (1st eigenvalue, hydrogen)
- **Files**: `BCUTc-1h_mean_CI_MWU_CliffsDelta_m.png`, `BCUTc-1h.py`
- **Analysis**: BCUT descriptor effect size analysis

##### BIC5 - Bond Information Content (5th order)
- **Files**: Multiple PNG files for BIC1 through BIC5, `BIC m.py`
- **Analysis**: Information content analysis across different bond orders

##### ETA_Epsilon_5 - Extended Topochemical Atom (5th order)
- **Files**: `ETA_epsilon_5_mean_CI_MWU_CliffsDelta_m.png`, `BCUTc m.py`
- **Analysis**: Extended topochemical atom descriptor effects

## Experimental Pipeline

1. **Data Preparation**: Mordred descriptor calculation and preprocessing
2. **Feature Selection**: Apply different selection strategies (All, Filter, SHAP)
3. **Model Training**: Train multiple ML models with hyperparameter optimization
4. **Performance Evaluation**: Cross-validation with multiple metrics
5. **Statistical Comparison**: Paired t-tests between different approaches
6. **Descriptor Analysis**: Effect size and correlation analysis
7. **Result Interpretation**: Comprehensive analysis of feature importance and model performance

## Key Features

- **Reproducible Experiments**: All scripts use relative paths and consistent random seeds
- **Statistical Rigor**: Proper cross-validation, significance testing, and effect size calculation
- **Comprehensive Evaluation**: Multiple metrics (Accuracy, MCC, ROC-AUC) and statistical tests
- **Feature Understanding**: SHAP analysis and descriptor effect size studies
- **Model Diversity**: Six different machine learning algorithms for robust comparison

## Usage

Each subdirectory contains ready-to-run Python scripts with proper path handling. Results are automatically saved to designated output directories with consistent naming conventions.

## Dependencies

- Python 3.8+
- scikit-learn
- XGBoost, LightGBM, CatBoost
- SHAP
- Optuna
- pandas, numpy
- matplotlib, seaborn
- Mordred (molecular descriptor calculation)

## Data Format

All experiments expect:
- Training data: `mordred2dfeature_knn.csv` (peptide sequences with Mordred descriptors)
- Features: 1613 molecular descriptors after preprocessing
- Target: Binary classification (cell-penetrating vs non-cell-penetrating peptides)