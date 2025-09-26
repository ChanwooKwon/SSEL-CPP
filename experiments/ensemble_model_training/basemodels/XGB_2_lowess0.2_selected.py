import os
import numpy as np
import pandas as pd
import joblib
import optuna
import optuna.visualization as vis
import plotly.io as pio
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, log_loss
)
from optuna.samplers import TPESampler
from pathlib import Path

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent.parent.parent

#  Path settings
FEATURE_CSV_PATH = base_dir / "shap_feature_selection" / "feature_optimization" / "XGB" / "curve" / "optimal_features_lowess0.2_xgb_accuracy_75.csv"
DATA_PATH = base_dir / "data" / "mordred2dfeature_knn.csv"
SAVE_DIR = current_dir / "results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = SAVE_DIR / "xgb_2_lowess0.2_best_model.pkl"
PARAMS_TXT_PATH = SAVE_DIR / "xgb_2_lowess0.2_best_params.txt"
PROBA_SAVE_PATH = SAVE_DIR / "xgb_2_lowess0.2_oof_proba.npy"
LABEL_SAVE_PATH = SAVE_DIR / "xgb_2_lowess0.2_labels.npy"
METRIC_TXT_PATH = SAVE_DIR / "xgb_2_lowess0.2_metrics.txt"
PLOT_SAVE_PATH = SAVE_DIR / "xgb_2_lowess0.2_optimization_history.png"

#  Data loading
def load_data(feature_path, data_path):
    top_features = pd.read_csv(feature_path)["Feature"].tolist()
    df = pd.read_csv(data_path)
    df_filtered = df[["id"] + top_features].copy()
    df_filtered["Label"] = df_filtered["id"].apply(lambda x: 1 if "positive" in x else 0)
    X = df_filtered.drop(columns=["id", "Label"])
    y = df_filtered["Label"].values
    return X, y

#  Evaluation metrics
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

#  Optuna objective function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 250, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1),
        "max_depth": trial.suggest_int("max_depth", 7, 10),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 6.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 3.0),
        "random_state": 0,
        "n_jobs": 2,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params)
    y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=2)
    return accuracy_score(y, y_pred)

#  Prepare data
X, y = load_data(FEATURE_CSV_PATH, DATA_PATH)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

#  Run Optuna optimization
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=150)

#  Save optimization history plot
fig = vis.plot_optimization_history(study)
pio.write_image(fig, PLOT_SAVE_PATH, format="png", width=800, height=600, scale=2)

#  Save best parameters
best_params = study.best_params
with open(PARAMS_TXT_PATH, "w") as f:
    f.write("XGB2 Best Hyperparameters (Optuna)\n")
    f.write("=" * 40 + "\n")
    for key, val in best_params.items():
        f.write(f"{key}: {val}\n")

#  Train final model
best_params.update({
    "random_state": 0,
    "n_jobs": 2,
    "use_label_encoder": False,
    "eval_metric": "logloss"
})
final_model = XGBClassifier(**best_params)
oof_pred = cross_val_predict(final_model, X, y, cv=skf, n_jobs=2)
oof_proba = cross_val_predict(final_model, X, y, cv=skf, method="predict_proba", n_jobs=2)[:, 1]

#  Save trained model
final_model.fit(X, y)
joblib.dump(final_model, MODEL_SAVE_PATH)

#  Save OOF predictions
np.save(PROBA_SAVE_PATH, np.vstack([1 - oof_proba, oof_proba]).T)
np.save(LABEL_SAVE_PATH, y)

#  Save evaluation metrics
metrics = evaluate_metrics(y, oof_pred, oof_proba)
with open(METRIC_TXT_PATH, "w") as f:
    f.write("XGBoost (Optuna Tuned) CV Metrics\n")
    f.write("=" * 40 + "\n")
    for key, val in metrics.items():
        f.write(f"{key}: {val:.4f}\n")

print("XGBoost model, OOF predictions, metrics, and visualization saved successfully")