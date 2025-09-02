import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from scipy.stats import ttest_rel

#  Best params (manually entered)
params_before = {
    "n_estimators": 532,
    "learning_rate": 0.040362765286706355,
    "max_depth": 6,
    "subsample": 0.6169070498002484,
    "colsample_bytree": 0.921101602391001,
    "gamma": 2.1296880492289154,
    "reg_alpha": 0.022861232289593844,
    "reg_lambda": 1.104970375045134,
    "random_state": 0,
    "n_jobs": 1,
    "use_label_encoder": False,
    "eval_metric": "logloss"
}

params_after = {
    "n_estimators": 305,
    "learning_rate": 0.07117943143345946,
    "max_depth": 9,
    "subsample": 0.7532829289449027,
    "colsample_bytree": 0.6170818259192309,
    "gamma": 0.2869122602420824,
    "reg_alpha": 1.7925434240717297,
    "reg_lambda": 3.690400968762814,
    "random_state": 0,
    "n_jobs": 1,
    "use_label_encoder": False,
    "eval_metric": "logloss"
}

#  Data paths
DATA_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
FEATURE_BEFORE_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Method4/Mordred/feature_importance_4.csv"
FEATURE_AFTER_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/XGB/feature/optimal_features_lowess0.2_accuracy_75.csv"
RESULT_SAVE_PATH = "./xgb_filter_comparison_allss_result_5fold.txt"

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
        model = XGBClassifier(**params)
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
    f.write("XGBoost 성능 비교 (Feature Filtering 전후, 5-fold CV + Paired t-test)\n")
    f.write("=" * 70 + "\n")
    f.write(paired_t_test(accs_before, accs_after, "Accuracy"))
    f.write(paired_t_test(mccs_before, mccs_after, "MCC"))
    f.write(paired_t_test(aucs_before, aucs_after, "ROC-AUC"))

print(f" 결과 저장 완료: {RESULT_SAVE_PATH}")
