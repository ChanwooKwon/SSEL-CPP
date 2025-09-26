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
RESULTS_DIR = current_dir / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Best hyperparameters for ALL vs SHAP comparison ──────────────────
BEST_PARAMS = {
    "XGB": {
        "all_features": {
            "n_estimators": 532, "learning_rate": 0.040362765286706355, "max_depth": 6,
            "subsample": 0.6169070498002484, "colsample_bytree": 0.921101602391001,
            "gamma": 2.1296880492289154, "reg_alpha": 0.022861232289593844,
            "reg_lambda": 1.104970375045134, "random_state": 0, "n_jobs": 1,
            "use_label_encoder": False, "eval_metric": "logloss"
        },
        "shap_selected": {
            "n_estimators": 305, "learning_rate": 0.07117943143345946, "max_depth": 9,
            "subsample": 0.7532829289449027, "colsample_bytree": 0.6170818259192309,
            "gamma": 0.2869122602420824, "reg_alpha": 1.7925434240717297,
            "reg_lambda": 3.690400968762814, "random_state": 0, "n_jobs": 1,
            "use_label_encoder": False, "eval_metric": "logloss"
        },
        "feature_path": base_dir / "shap_feature_selection" / "feature_optimization" / "XGB" / "curve" / "optimal_features_lowess0.2_accuracy_75.csv"
    },
    "RF": {
        "all_features": {
            "n_estimators": 598, "max_depth": 19, "min_samples_split": 7,
            "min_samples_leaf": 2, "max_features": "log2", "bootstrap": False,
            "random_state": 0, "n_jobs": 7
        },
        "shap_selected": {
            "n_estimators": 430, "max_depth": 9, "min_samples_split": 2,
            "min_samples_leaf": 1, "max_features": "log2", "bootstrap": False,
            "random_state": 0, "n_jobs": 7
        },
        "feature_path": base_dir / "shap_feature_selection" / "feature_optimization" / "RF" / "curve" / "optimal_features_polyfit_rf_acc_152.csv"
    },
    "LGB": {
        "all_features": {
            "n_estimators": 637, "learning_rate": 0.05166356972544993, "max_depth": 10,
            "num_leaves": 25, "min_child_samples": 7, "subsample": 0.626850496301491,
            "colsample_bytree": 0.995079942774497, "reg_alpha": 0.1022711962671985,
            "reg_lambda": 2.515467193222938, "random_state": 0, "n_jobs": 7, "verbosity": -1
        },
        "shap_selected": {
            "n_estimators": 536, "learning_rate": 0.11905245682182887, "max_depth": 5,
            "num_leaves": 93, "min_child_samples": 23, "subsample": 0.7136208380400285,
            "colsample_bytree": 0.868716667619904, "reg_alpha": 1.0588580169133697,
            "reg_lambda": 8.33180434758218, "random_state": 0, "n_jobs": 7, "verbosity": -1
        },
        "feature_path": base_dir / "shap_feature_selection" / "feature_optimization" / "LGB" / "curve" / "optimal_features_lowess_lgb_accuracy_95.csv"
    },
    "GB": {
        "all_features": {
            "n_estimators": 418, "learning_rate": 0.0977262891229071, "max_depth": 4,
            "subsample": 0.8605770209457032, "min_samples_split": 5,
            "min_samples_leaf": 5, "random_state": 0
        },
        "shap_selected": {
            "n_estimators": 590, "learning_rate": 0.016075174922150864, "max_depth": 4,
            "subsample": 0.7614719689151963, "min_samples_split": 3,
            "min_samples_leaf": 10, "random_state": 0
        },
        "feature_path": base_dir / "shap_feature_selection" / "feature_optimization" / "GB" / "curve" / "optimal_features_lowess0.3_gb_accuracy_125.csv"
    },
    "ERT": {
        "all_features": {
            "n_estimators": 728, "max_depth": 46, "min_samples_split": 6,
            "min_samples_leaf": 1, "max_features": "sqrt", "random_state": 0, "n_jobs": 7
        },
        "shap_selected": {
            "n_estimators": 461, "max_depth": 19, "min_samples_split": 3,
            "min_samples_leaf": 1, "max_features": "log2", "random_state": 0, "n_jobs": 7
        },
        "feature_path": base_dir / "shap_feature_selection" / "feature_optimization" / "ERT" / "curve" / "optimal_features_lowess0.3_ert_accuracy_125.csv"
    },
    "CB": {
        "all_features": {
            "iterations": 675, "learning_rate": 0.1175002857276872, "depth": 4,
            "l2_leaf_reg": 7.845919104345466, "border_count": 148,
            "bagging_temperature": 0.4491131985989324, "random_strength": 5.946747822912739,
            "random_seed": 0, "verbose": 0, "task_type": "CPU", "loss_function": "Logloss"
        },
        "shap_selected": {
            "iterations": 492, "learning_rate": 0.11737752626711202, "depth": 7,
            "l2_leaf_reg": 9.980224773739117, "border_count": 230,
            "bagging_temperature": 0.6793418231867362, "random_strength": 18.8425654535369,
            "random_seed": 0, "verbose": 0, "task_type": "CPU", "loss_function": "Logloss"
        },
        "feature_path": base_dir / "shap_feature_selection" / "feature_optimization" / "CB" / "curve" / "optimal_features_lowess0.2_cb_accuracy_95.csv"
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
def load_data(feature_csv=None):
    df = pd.read_csv(DATA_PATH)
    df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x else 0)
    
    if feature_csv is None or feature_csv == "all":
        feature_cols = [col for col in df.columns if col not in ["id", "Label"]]
    else:
        top_features = pd.read_csv(feature_csv)["Feature"].tolist()
        feature_cols = top_features
    
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
        X_all, y, n_features_all = load_data("all")
        X_shap, _, n_features_shap = load_data(BEST_PARAMS[model_name]["feature_path"])
        
        # Get model class and parameters
        model_class = MODEL_CLASSES[model_name]
        params_all = BEST_PARAMS[model_name]["all_features"]
        params_shap = BEST_PARAMS[model_name]["shap_selected"]
        
        # Evaluate models
        print(f"  📊 All Features ({n_features_all} features)...")
        accs_all, mccs_all, aucs_all = evaluate_model(X_all, y, model_class, params_all, cv_folds)
        
        print(f"  🎯 SHAP Selected ({n_features_shap} features)...")
        accs_shap, mccs_shap, aucs_shap = evaluate_model(X_shap, y, model_class, params_shap, cv_folds)
        
        # Statistical tests
        acc_test = paired_t_test(accs_all, accs_shap, "Accuracy")
        mcc_test = paired_t_test(mccs_all, mccs_shap, "MCC")
        auc_test = paired_t_test(aucs_all, aucs_shap, "ROC-AUC")
        
        # Save results to txt file
        save_results_txt(model_name, n_features_all, n_features_shap, 
                        acc_test, mcc_test, auc_test, cv_folds)

# ── Save results to txt file ──────────────────────────────────
def save_results_txt(model_name, n_features_all, n_features_shap, 
                    acc_test, mcc_test, auc_test, cv_folds):
    
    filename = RESULTS_DIR / f"{model_name}_compare_{cv_folds}fold_all_ss.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{model_name} Comparison: All Features vs SHAP Selected\n")
        f.write("=" * 60 + "\n")
        f.write(f"Cross-Validation: {cv_folds}-fold\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("FEATURE INFORMATION:\n")
        f.write(f"All Features: {n_features_all:,} features\n")
        f.write(f"SHAP Selected: {n_features_shap:,} features\n")
        f.write(f"Reduction: {((n_features_all - n_features_shap) / n_features_all * 100):.1f}%\n\n")
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-" * 40 + "\n")
        
        for test_result in [acc_test, mcc_test, auc_test]:
            metric = test_result['metric']
            before = test_result['before_mean']
            after = test_result['after_mean']
            diff = test_result['difference']
            p_val = test_result['p_value']
            sig = test_result['significance']
            
            f.write(f"\n{metric}:\n")
            f.write(f"  All Features:    {before:.6f}\n")
            f.write(f"  SHAP Selected:   {after:.6f}\n")
            f.write(f"  Difference:      {diff:+.6f}\n")
            f.write(f"  p-value:         {p_val:.6f} {sig}\n")
            
            if p_val < 0.05:
                direction = "improvement" if diff > 0 else "decrease"
                f.write(f"  Result:          Significant {direction}\n")
            else:
                f.write(f"  Result:          No significant change\n")
        
        f.write("\n" + "=" * 60 + "\n")

# ── Main execution ────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting All Features vs SHAP Selected Comparison...")
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
        print(f"  📄 {model}_compare_5fold_all_ss.txt")
        print(f"  📄 {model}_compare_10fold_all_ss.txt")