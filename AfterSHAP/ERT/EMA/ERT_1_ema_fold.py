import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, log_loss
)

#  Path settings
FEATURE_CSV_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/ERT/feature/optimal_features_ema_ert_acc_215.csv"
DATA_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
MODEL_SAVE_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/AfterSHAP/BestModel/ERT/EMA/ert_1_ema_best_model.pkl"
FOLD_METRICS_PATH = "./ert_1_ema_fold_metrics.csv"

#  Best hyperparameters (from XGB4 tuning reference)
best_params = {
    "n_estimators": 394,
    "max_depth": 24,
    "min_samples_split": 3,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": False,
    "n_jobs": 7
}

#  Data loading function
def load_data(feature_path, data_path):
    top_features = pd.read_csv(feature_path)["Feature"].tolist()
    df = pd.read_csv(data_path)
    df_filtered = df[["id"] + top_features].copy()
    df_filtered["Label"] = df_filtered["id"].apply(lambda x: 1 if "positive" in x else 0)
    X = df_filtered.drop(columns=["id", "Label"])
    y = df_filtered["Label"].values
    return X, y

#  Evaluation metrics function
def evaluate_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "LogLoss": log_loss(y_true, y_proba),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
    }

#  Prepare data
X, y = load_data(FEATURE_CSV_PATH, DATA_PATH)

#  5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = ExtraTreesClassifier(**best_params, random_state=0)
    model.fit(X.iloc[train_idx], y[train_idx])
    preds = model.predict(X.iloc[val_idx])
    probas = model.predict_proba(X.iloc[val_idx])[:, 1]
    metrics = evaluate_metrics(y[val_idx], preds, probas)
    metrics["Fold"] = fold
    fold_metrics.append(metrics)

#  Save fold metrics
fold_metrics_df = pd.DataFrame(fold_metrics)
fold_metrics_df.to_csv(FOLD_METRICS_PATH, index=False)

#  Print mean and standard deviation summary
summary_stats = fold_metrics_df.drop(columns="Fold").agg(["mean", "std"]).T
print(" Fold-level performance summary:\n")
print(summary_stats)
