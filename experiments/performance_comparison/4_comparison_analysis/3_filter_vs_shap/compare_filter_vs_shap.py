import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from scipy.stats import ttest_rel
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent.parent.parent.parent

# ── Path settings ─────────────────────────────────────────────
DATA_PATH = base_dir / "data" / "mordred2dfeature_knn.csv"
FILTER_FEATURE_PATH = base_dir / "filtermethod" / "filtermethod4" / "feature_importance_4.csv"
RESULTS_DIR = current_dir / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Best hyperparameters for FILTER vs SHAP comparison ────────────────
BEST_PARAMS = {
    "XGB": {
        "filter_method4": {
            "n_estimators": 759, "learning_rate": 0.056667824039790884, "max_depth": 5,
            "subsample": 0.8635155497163529, "colsample_bytree": 0.6828479436667787,
            "gamma": 2.3687790825240382, "reg_alpha": 0.0018902391653725092,
            "reg_lambda": 4.345184122564328, "random_state": 0, "n_jobs": 1,
            "use_label_encoder": False, "eval_metric": "logloss"
        },
        "shap_selected": {
            "n_estimators": 305, "learning_rate": 0.07117943143345946, "max_depth": 9,
            "subsample": 0.7532829289449027, "colsample_bytree": 0.6170818259192309,
            "gamma": 0.2869122602420824, "reg_alpha": 1.7925434240717297,
            "reg_lambda": 3.690400968762814, "random_state": 0, "n_jobs": 1,
            "use_label_encoder": False, "eval_metric": "logloss"
        },
        "shap_feature_path": base_dir / "shap_feature_selection" / "feature_optimization" / "XGB" / "curve" / "optimal_features_lowess0.2_accuracy_75.csv"
    },
    "RF": {
        "filter_method4": {
            "n_estimators": 359, "max_depth": 20, "min_samples_split": 2,
            "min_samples_leaf": 2, "max_features": "sqrt", "bootstrap": False,
            "random_state": 0, "n_jobs": 7
        },
        "shap_selected": {
            "n_estimators": 430, "max_depth": 9, "min_samples_split": 2,
            "min_samples_leaf": 1, "max_features": "log2", "bootstrap": False,
            "random_state": 0, "n_jobs": 7
        },
        "shap_feature_path": base_dir / "shap_feature_selection" / "feature_optimization" / "RF" / "curve" / "optimal_features_polyfit_rf_acc_152.csv"
    },
    "LGB": {
        "filter_method4": {
            "n_estimators": 750, "learning_rate": 0.030387024947067486, "max_depth": 7,
            "num_leaves": 101, "min_child_samples": 22, "subsample": 0.9385379896309153,
            "colsample_bytree": 0.8282781403363384, "reg_alpha": 0.6979474291344726,
            "reg_lambda": 9.08021145576311, "random_state": 0, "n_jobs": 7, "verbosity": -1
        },
        "shap_selected": {
            "n_estimators": 536, "learning_rate": 0.11905245682182887, "max_depth": 5,
            "num_leaves": 93, "min_child_samples": 23, "subsample": 0.7136208380400285,
            "colsample_bytree": 0.868716667619904, "reg_alpha": 1.0588580169133697,
            "reg_lambda": 8.33180434758218, "random_state": 0, "n_jobs": 7, "verbosity": -1
        },
        "shap_feature_path": base_dir / "shap_feature_selection" / "feature_optimization" / "LGB" / "curve" / "optimal_features_lowess_lgb_accuracy_95.csv"
    },
    "GB": {
        "filter_method4": {
            "n_estimators": 320, "learning_rate": 0.035049778272249005, "max_depth": 5,
            "subsample": 0.737171292043863, "min_samples_split": 8,
            "min_samples_leaf": 7, "random_state": 0
        },
        "shap_selected": {
            "n_estimators": 590, "learning_rate": 0.016075174922150864, "max_depth": 4,
            "subsample": 0.7614719689151963, "min_samples_split": 3,
            "min_samples_leaf": 10, "random_state": 0
        },
        "shap_feature_path": base_dir / "shap_feature_selection" / "feature_optimization" / "GB" / "curve" / "optimal_features_lowess0.3_gb_accuracy_125.csv"
    },
    "ERT": {
        "filter_method4": {
            "n_estimators": 464, "max_depth": 25, "min_samples_split": 3,
            "min_samples_leaf": 1, "max_features": "sqrt", "random_state": 0, "n_jobs": 7
        },
        "shap_selected": {
            "n_estimators": 461, "max_depth": 19, "min_samples_split": 3,
            "min_samples_leaf": 1, "max_features": "log2", "random_state": 0, "n_jobs": 7
        },
        "shap_feature_path": base_dir / "shap_feature_selection" / "feature_optimization" / "ERT" / "curve" / "optimal_features_lowess0.3_ert_accuracy_125.csv"
    },
    "CB": {
        "filter_method4": {
            "iterations": 749, "learning_rate": 0.1143714251818698, "depth": 8,
            "l2_leaf_reg": 9.367867600810548, "border_count": 160,
            "bagging_temperature": 0.41034102168715714, "random_strength": 1.8159010648408973,
            "random_seed": 0, "verbose": 0, "task_type": "CPU", "loss_function": "Logloss"
        },
        "shap_selected": {
            "iterations": 492, "learning_rate": 0.11737752626711202, "depth": 7,
            "l2_leaf_reg": 9.980224773739117, "border_count": 230,
            "bagging_temperature": 0.6793418231867362, "random_strength": 18.8425654535369,
            "random_seed": 0, "verbose": 0, "task_type": "CPU", "loss_function": "Logloss"
        },
        "shap_feature_path": base_dir / "shap_feature_selection" / "feature_optimization" / "CB" / "curve" / "optimal_features_lowess0.2_cb_accuracy_95.csv"
    }
}

# ── Model classes ─────────────────────────────────────────────
MODEL_CLASSES = {
    "XGB": XGBClassifier,
    "RF": RandomForestClassifier,
    "LGB": LGBMClassifier,
    "GB": GradientBoostingClassifier,
    "ERT": ExtraTreesClassifier,
    "CB": CatBoostClassifier
}

# ── Load data function ────────────────────────────────────────
def load_data(data_type, shap_feature_path=None):
    df = pd.read_csv(DATA_PATH)
    df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x else 0)
    
    if data_type == "filter":
        filter_features = pd.read_csv(FILTER_FEATURE_PATH)["Feature"].tolist()
        feature_cols = filter_features
    elif data_type == "shap":
        shap_features = pd.read_csv(shap_feature_path)["Feature"].tolist()
        feature_cols = shap_features
    
    X = df[feature_cols]
    y = df["Label"].values
    return X, y, len(feature_cols)

# ── Evaluation function ───────────────────────────────────────
def evaluate_model(X, y, model_class, params, cv_folds=5):
    accs, mccs, aucs = [], [], []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
    
    for train_idx, val_idx in skf.split(X, y):
        model = model_class(**params)
        model.fit(X.iloc[train_idx], y[train_idx])
        preds = model.predict(X.iloc[val_idx])
        probas = model.predict_proba(X.iloc[val_idx])[:, 1]
        
        accs.append(accuracy_score(y[val_idx], preds))
        mccs.append(matthews_corrcoef(y[val_idx], preds))
        aucs.append(roc_auc_score(y[val_idx], probas))
    
    return accs, mccs, aucs

# ── Statistical test function ─────────────────────────────────
def paired_t_test(metric1, metric2, name):
    stat, p = ttest_rel(metric1, metric2)
    diff = np.mean(metric2) - np.mean(metric1)
    significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    
    return {
        "metric": name,
        "before_mean": np.mean(metric1),
        "after_mean": np.mean(metric2),
        "difference": diff,
        "p_value": p,
        "significance": significance
    }

# ── Main comparison function ──────────────────────────────────
def compare_all_models(cv_folds=5):
    for model_name in BEST_PARAMS.keys():
        print(f"🔍 Evaluating {model_name}...")
        
        # Load data
        X_filter, y, n_features_filter = load_data("filter")
        X_shap, _, n_features_shap = load_data("shap", BEST_PARAMS[model_name]["shap_feature_path"])
        
        # Get model class and parameters
        model_class = MODEL_CLASSES[model_name]
        params_filter = BEST_PARAMS[model_name]["filter_method4"]
        params_shap = BEST_PARAMS[model_name]["shap_selected"]
        
        # Evaluate models
        print(f"  📊 Filter Method4 ({n_features_filter} features)...")
        accs_filter, mccs_filter, aucs_filter = evaluate_model(X_filter, y, model_class, params_filter, cv_folds)
        
        print(f"  🎯 SHAP Selected ({n_features_shap} features)...")
        accs_shap, mccs_shap, aucs_shap = evaluate_model(X_shap, y, model_class, params_shap, cv_folds)
        
        # Statistical tests
        acc_test = paired_t_test(accs_filter, accs_shap, "Accuracy")
        mcc_test = paired_t_test(mccs_filter, mccs_shap, "MCC")
        auc_test = paired_t_test(aucs_filter, aucs_shap, "ROC-AUC")
        
        # Save results to txt file
        save_results_txt(model_name, n_features_filter, n_features_shap, 
                        acc_test, mcc_test, auc_test, cv_folds)

# ── Save results to txt file ──────────────────────────────────
def save_results_txt(model_name, n_features_filter, n_features_shap, 
                    acc_test, mcc_test, auc_test, cv_folds):
    
    filename = RESULTS_DIR / f"{model_name}_filter_comparison_result_{cv_folds}fold.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{model_name} (Feature Filtering, {cv_folds}-fold CV + Paired t-test)\n")
        f.write("=" * 70 + "\n")
        
        # Brief feature information
        f.write(f"Filter Method4 Features: {n_features_filter:,}\n")
        f.write(f"SHAP Selected Features: {n_features_shap:,}\n")
        f.write(f"Feature Difference: {n_features_filter - n_features_shap:,}\n\n")
        
        for test_result in [acc_test, mcc_test, auc_test]:
            metric = test_result['metric']
            before = test_result['before_mean']
            after = test_result['after_mean']
            diff = test_result['difference']
            p_val = test_result['p_value']
            sig = test_result['significance']
            
            f.write(f"{metric}:\n")
            f.write(f"  Before = {before:.4f}, After = {after:.4f}\n")
            f.write(f"  Δ = {diff:+.4f}, p-value = {p_val:.4f} {sig}\n\n")

# ── Main execution ────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting Filter Method4 vs SHAP Selected Comparison...")
    print(f"📁 Results will be saved to: {RESULTS_DIR}")
    
    # Run comparisons for both 5-fold and 10-fold CV
    for cv_folds in [5, 10]:
        print(f"\n📊 Running {cv_folds}-fold Cross-Validation...")
        compare_all_models(cv_folds)
        print(f"✅ {cv_folds}-fold comparison completed!")
    
    print(f"\n🎯 All comparisons completed!")
    print(f"📁 Results saved in: {RESULTS_DIR}")
    print("\nGenerated files:")
    for model in BEST_PARAMS.keys():
        print(f"  📄 {model}_filter_comparison_result_5fold.txt")
        print(f"  📄 {model}_filter_comparison_result_10fold.txt")