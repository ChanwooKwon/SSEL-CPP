import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from scipy.stats import ttest_rel


params_before = {
    
    "iterations": 749,
    "learning_rate": 0.1143714251818698,
    "depth": 8,
    "l2_leaf_reg":9.367867600810548,
    "border_count": 160,
    "bagging_temperature": 0.41034102168715714,
    "random_strength": 1.8159010648408973,
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


DATA_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
FEATURE_BEFORE_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Method4/Mordred/feature_importance_4.csv"
FEATURE_AFTER_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/CB/feature/optimal_features_lowess_cb_accuracy_95.csv"
RESULT_SAVE_PATH = "./catboost_filter_comparison_result_5fold.txt"


def load_data(feature_csv, data_csv):
    top_features = pd.read_csv(feature_csv)["Feature"].tolist()
    df = pd.read_csv(data_csv)
    df_filtered = df[["id"] + top_features].copy()
    df_filtered["Label"] = df_filtered["id"].apply(lambda x: 1 if "positive" in x else 0)
    X = df_filtered.drop(columns=["id", "Label"])
    y = df_filtered["Label"].values
    return X, y


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


def paired_t_test(metric1, metric2, name):
    stat, p = ttest_rel(metric1, metric2)
    diff = np.mean(metric2) - np.mean(metric1)
    return f"{name}:\n" \
           f"  Before = {np.mean(metric1):.4f}, After = {np.mean(metric2):.4f}\n" \
           f"  Δ = {diff:.4f}, p-value = {p:.4f} {'' if p < 0.05 else ''}\n"


X_before, y = load_data(FEATURE_BEFORE_PATH, DATA_PATH)
X_after, _ = load_data(FEATURE_AFTER_PATH, DATA_PATH)

accs_before, mccs_before, aucs_before = evaluate_model(X_before, y, params_before)
accs_after, mccs_after, aucs_after = evaluate_model(X_after, y, params_after)


with open(RESULT_SAVE_PATH, "w") as f:
    f.write("CatBoost  (Feature Filtering, 5-fold CV + Paired t-test)\n")
    f.write("=" * 70 + "\n")
    f.write(paired_t_test(accs_before, accs_after, "Accuracy"))
    f.write(paired_t_test(mccs_before, mccs_after, "MCC"))
    f.write(paired_t_test(aucs_before, aucs_after, "ROC-AUC"))

print(f"{RESULT_SAVE_PATH}")



