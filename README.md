# SSEL-CPP: Cell Penetrating Peptide Prediction Tool

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub release](https://img.shields.io/badge/release-v1.0.0-blue)](https://github.com/ChanwooKwon/SSELCPP/releases)

A machine learning-based tool for predicting Cell Penetrating Peptide (CPP) activity using soft-voting ensemble learning with comprehensive feature selection and molecular descriptor analysis.
<img width="648" height="454" alt="image" src="https://github.com/user-attachments/assets/656595b9-5605-4fa1-ad43-a0899ad65af7" />

## 🎯 Overview

SSEL-CPP is a state-of-the-art predictive model for identifying cell-penetrating peptides using:
- **Advanced Ensemble Learning**: XGBoost and LightGBM with soft voting
- **Multi-stage Feature Selection**: Filter methods and SHAP-based optimization
- **Comprehensive Molecular Descriptors**: 1613 Mordred descriptors with KNN imputation
- **Rigorous Validation**: 5-fold cross-validation with statistical significance testing

The model was trained and validated on the CPP1708 dataset with extensive hyperparameter optimization using Optuna.

## ✨ Key Features

- 🤖 **Ensemble Learning**: Combines multiple ML algorithms for robust predictions
- 🔬 **Feature Engineering**: Multi-stage selection process using statistical and SHAP methods
- 📁 **Flexible Input**: Single sequences, SMILES strings, or batch FASTA file processing
- 📊 **Detailed Analytics**: Probability scores, confidence intervals, and feature importance
- 🔄 **Reproducible Research**: Complete experimental pipeline with relative paths
- 📈 **Statistical Rigor**: Cross-validation, paired t-tests, and effect size analysis

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ChanwooKwon/SSELCPP.git
cd SSELCPP
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Basic Usage

**Predict a single peptide sequence:**
```bash
python test.py "KRRRRRRR" --sequence
```

**Predict from SMILES string:**
```bash
python test.py "CC(C)C[C@H](NC(=O)[C@H](CCCCN)NC(=O)CNC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCCN)N)C(=O)O" --smiles
```

**Predict from FASTA file:**
```bash
python test.py input_sequences.fasta --output results.csv
```

**Example output:**
```
Predicting CPP activity for sequence: KRRRRRRR
Prediction Results:
CPP Probability: 0.8542
Classification: CPP
```

## 📖 Usage

### Command Line Interface

```bash
python test.py [input] [options]
```

**Arguments:**
- `input`: FASTA file path, peptide sequence, or SMILES string
- `--output, -o`: Output file name (default: predictions.csv)
- `--sequence, -s`: Treat input as single peptide sequence instead of file
- `--smiles`: Treat input as single SMILES string instead of file

### Python API

```python
from src.ssel_cpp import SSELCPP

# Initialize model
model = SSELCPP()

# Predict single peptide sequence
result = model.predict_single("KRRRRRRR")
print(f"CPP Probability: {result['probability']:.4f}")

# Predict single SMILES string
result = model.predict_smiles("CC(C)C[C@H](NC(=O)[C@H](CCCCN)NC(=O)CNC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCCN)N)C(=O)O")
print(f"CPP Probability: {result['probability']:.4f}")

# Predict from FASTA file
results_df = model.predict_fasta_file("sequences.fasta")
```

## 🏗️ Model Architecture

### Feature Extraction Pipeline
- **Mordred Descriptors**: 1613 comprehensive 2D molecular descriptors
- **SMILES Generation**: Peptide sequences converted to SMILES representations
- **Preprocessing**: KNN imputation (k=5) and standardization
- **Quality Control**: Feature filtering and correlation analysis

### Multi-stage Feature Selection
1. **Filter Method 4**: Statistical significance and importance-based filtering
2. **SHAP Analysis**: Model-specific feature importance with curve optimization
   - Lowess smoothing for XGB, LGBM, RF, GB, ERT
   - Polynomial fitting for CatBoost
3. **Final Selection**: Optimized feature subsets per model (200-400 features)

### Ensemble Learning
- **Base Models**: XGBoost and LightGBM with Optuna hyperparameter optimization
- **Voting Strategy**: Soft voting for probability averaging
- **Cross-validation**: 5-fold StratifiedKFold for robust evaluation

### Performance Metrics
- **Accuracy**: Overall classification performance
- **MCC**: Matthews Correlation Coefficient for balanced evaluation
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Statistical Testing**: Paired t-tests for significance assessment

## 📊 Experimental Framework

The project includes comprehensive experiments comparing different approaches:

### Feature Selection Comparison
- **All Features**: Baseline using all 1613 descriptors
- **Filter Method 4**: Statistical filtering approach
- **SHAP Selected**: Model-specific SHAP-based selection

### Statistical Analysis
- **Cross-validation**: 5-fold stratified sampling
- **Significance Testing**: Paired t-tests between methods
- **Effect Size Analysis**: Comprehensive descriptor family analysis

### Descriptor Analysis
- **Correlation Studies**: Feature interdependency analysis
- **Effect Size Studies**: Statistical significance across descriptor families
- **Visualization**: Comprehensive plots and heatmaps

For detailed experimental results, see the `experiments/` directory.

## 📁 Project Structure

```
SSELCPP/
├── data/                              # Training and test datasets
│   ├── train.csv                      # Original training data
│   ├── test.csv                       # Original test data
│   ├── mordred2dfeature_knn.csv      # Processed training features
│   └── test_mordred2dfeature_knn.csv # Processed test features
├── src/                              # Source code modules
│   ├── ssel_cpp.py                   # Main prediction class
│   ├── feature_extraction.py         # Mordred feature extraction
│   └── utils/                        # Utility functions
├── models/                           # Pre-trained models and configurations
│   ├── xgb_model.pkl                 # XGBoost model
│   ├── lgb_model.pkl                 # LightGBM model
│   └── knn_imputer.pkl               # KNN imputation model
├── experiments/                       # Complete experimental pipeline
│   ├── performance_comparison/        # Feature selection comparisons
│   ├── shap_feature_selection/       # SHAP analysis and optimization
│   ├── filter_methods/               # Statistical filtering methods
│   ├── descriptor_analysis/          # Molecular descriptor studies
│   └── README.md                     # Detailed experiment documentation
├── notebooks/                        # Jupyter notebook examples
├── test.py                          # Command-line interface
├── requirements.txt                  # Python dependencies
└── README.md                        # Project documentation
```

## 🔧 Dependencies

- **Python**: 3.7+
- **Core Libraries**: numpy, pandas, scikit-learn
- **ML Models**: xgboost, lightgbm, catboost
- **Feature Extraction**: mordred, rdkit-pypi
- **Bioinformatics**: biopython
- **Analysis**: shap, optuna, scipy
- **Visualization**: matplotlib, seaborn

## 📈 Performance Metrics

The SSEL-CPP ensemble model (XGBoost + LightGBM) achieves excellent performance on the independent test set:

### Test Set Performance
- **Accuracy**: 81.97%
- **ROC-AUC Score**: 87.53%
- **Matthews Correlation Coefficient (MCC)**: 63.35%
- **Precision**: 77.78%
- **Recall (Sensitivity)**: 80.77%
- **F1-Score**: 79.25%
- **Specificity**: 82.86%

### Model Robustness
- **Ensemble Strategy**: Soft voting combination of XGBoost and LightGBM
- **Cross-validation**: 5-fold StratifiedKFold for training validation
- **Statistical Significance**: p < 0.05 in comparative feature selection studies
- **Balanced Performance**: High sensitivity (80.77%) and specificity (82.86%)

The model demonstrates strong predictive capability with balanced performance across both CPP and non-CPP classes, making it reliable for practical peptide screening applications.

## 🧪 Reproducing Results

To reproduce the experimental results:

1. **Run baseline experiments:**
```bash
cd experiments/performance_comparison/1_all_features/
python XGB_allfeatures.py
```

2. **Compare feature selection methods:**
```bash
cd experiments/performance_comparison/4_comparison_analysis/
python compare_all_vs_shap.py
```

3. **Analyze descriptor effects:**
```bash
cd experiments/descriptor_analysis/feature_correlation/
python "Top5 features correlation_Pearson.py"
```

## 📝 Citation

If you use SSEL-CPP in your research, please cite:

```bibtex
@software{sselcpp2024,
  title={SSEL-CPP: Cell Penetrating Peptide Prediction Tool},
  author={Chanwoo Kwon},
  year={2024},
  url={https://github.com/ChanwooKwon/SSELCPP}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

- **Author**: Chanwoo Kwon
- **Email**: cwkwon99@ajou.ac.kr
- **Institution**: Ajou University

## 🙏 Acknowledgments

- CPP1708 dataset contributors
- Mordred descriptor development team

- Open-source machine learning community


