import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from scipy.stats import ttest_rel

#  Best params (manually entered)
params_before = {
    "iterations": 675,
    "learning_rate": 0.1175002857276872,
    "depth": 4,
    "l2_leaf_reg": 7.845919104345466,
    "border_count": 148,
    "bagging_temperature": 0.4491131985989324,
    "random_strength": 5.946747822912739,
    "random_seed": 0,
    "verbose": 0,
    "task_type": "CPU",
    "loss_function": "Logloss"
}

params_after = {
    "iterations": 492,
    "learning_rate": 0.11737752626711202,
    "depth": 7,
    "l2_leaf_reg": 9.980224773739117,
    "border_count": 230,
    "bagging_temperature": 0.6793418231867362,
    "random_strength": 18.8425654535369,
    "random_seed": 0,
    "verbose": 0,
    "task_type": "CPU",
    "loss_function": "Logloss"
}

#  Data paths
DATA_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
FEATURE_BEFORE_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Method4/Mordred/feature_importance_4.csv"
FEATURE_AFTER_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/CB/feature/optimal_features_lowess_cb_accuracy_95.csv"
RESULT_SAVE_PATH = "./catboost_filter_comparison_allss_result_5fold.txt"

#  Load data
def load_data(feature_csv, data_csv):
    df = pd.read_csv(data_csv)

    # Create Label
    df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x else 0)

    # Use all features
    if feature_csv in [None, "", "all"]:
        feature_cols = [col for col in df.columns if col not in ["id", "Label"]]
    else:
        top_features = pd.read_csv(feature_csv)["Feature"].tolist()
        feature_cols = top_features

    X = df[feature_cols]
    y = df["Label"].values
    return X, y

#  Evaluation function
def evaluate_model(X, y, params):
    accs, mccs, aucs = [], [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, val_idx in skf.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(X.iloc[train_idx], y[train_idx])
        preds = model.predict(X.iloc[val_idx])
        probas = model.predict_proba(X.iloc[val_idx])[:, 1]

        accs.append(accuracy_score(y[val_idx], preds))
        mccs.append(matthews_corrcoef(y[val_idx], preds))
        aucs.append(roc_auc_score(y[val_idx], probas))
    return accs, mccs, aucs

#  Paired t-test
def paired_t_test(metric1, metric2, name):
    stat, p = ttest_rel(metric1, metric2)
    diff = np.mean(metric2) - np.mean(metric1)
    return f"{name}:\n" \
           f"  Before = {np.mean(metric1):.4f}, After = {np.mean(metric2):.4f}\n" \
           f"  Δ = {diff:.4f}, p-value = {p:.4f} {'' if p < 0.05 else ''}\n"

#  Prepare data and evaluate
X_before, y = load_data("all", DATA_PATH)
X_after, _ = load_data(FEATURE_AFTER_PATH, DATA_PATH)

accs_before, mccs_before, aucs_before = evaluate_model(X_before, y, params_before)
accs_after, mccs_after, aucs_after = evaluate_model(X_after, y, params_after)

#  Save results
with open(RESULT_SAVE_PATH, "w") as f:
    f.write("CatBoost Performance Comparison (Before vs After Feature Filtering, 5-fold CV + Paired t-test)\n")
    f.write("=" * 70 + "\n")
    f.write(paired_t_test(accs_before, accs_after, "Accuracy"))
    f.write(paired_t_test(mccs_before, mccs_after, "MCC"))
    f.write(paired_t_test(aucs_before, aucs_after, "ROC-AUC"))

print(f" Results saved: {RESULT_SAVE_PATH}")
