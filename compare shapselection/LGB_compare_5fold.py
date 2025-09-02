import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from scipy.stats import ttest_rel


params_before = {
    "n_estimators": 750,
    "learning_rate": 0.030387024947067486,
    "max_depth": 7,
    "num_leaves": 101,
    "min_child_samples": 22,
    "subsample": 0.9385379896309153,
    "colsample_bytree": 0.8282781403363384,
    "reg_alpha": 0.6979474291344726,
    "reg_lambda": 9.08021145576311,
    "random_state": 0,
    "n_jobs": 7
}

params_after = {
    "n_estimators": 536,
    "learning_rate": 0.11905245682182887,
    "max_depth": 5,
    "num_leaves": 93,
    "min_child_samples": 23,
    "subsample": 0.7136208380400285,
    "colsample_bytree": 0.868716667619904,
    "reg_alpha": 1.0588580169133697,
    "reg_lambda": 8.33180434758218,
    "random_state": 0,
    "n_jobs": 7
}


DATA_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
FEATURE_BEFORE_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Method4/Mordred/feature_importance_4.csv"
FEATURE_AFTER_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/LGB/feature/optimal_features_lowess_lgb_accuracy_95.csv"
RESULT_SAVE_PATH = "./lgb_filter_comparison_result_5fold.txt"

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
        model = LGBMClassifier(**params)
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
    f.write("LGBM 성(Feature Filtering  5-fold CV + Paired t-test)\n")
    f.write("=" * 70 + "\n")
    f.write(paired_t_test(accs_before, accs_after, "Accuracy"))
    f.write(paired_t_test(mccs_before, mccs_after, "MCC"))
    f.write(paired_t_test(aucs_before, aucs_after, "ROC-AUC"))

print(f"  {RESULT_SAVE_PATH}")
