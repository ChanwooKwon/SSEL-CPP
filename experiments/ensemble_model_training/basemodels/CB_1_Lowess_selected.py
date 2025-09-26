import os
import numpy as np
import pandas as pd
import joblib
import optuna
import optuna.visualization as vis
import plotly.io as pio
from catboost import CatBoostClassifier, Pool
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
FEATURE_CSV_PATH = base_dir / "shap_feature_selection" / "feature_optimization" / "CB" / "curve" / "optimal_features_lowess_cb_accuracy_95.csv"
DATA_PATH = base_dir / "data" / "mordred2dfeature_knn.csv"
SAVE_DIR = current_dir / "results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = SAVE_DIR / "cb_1_lowess_best_model.cbm"
PARAMS_TXT_PATH = SAVE_DIR / "cb_1_lowess_best_params.txt"
PROBA_SAVE_PATH = SAVE_DIR / "cb_1_lowess_oof_proba.npy"
LABEL_SAVE_PATH = SAVE_DIR / "cb_1_lowess_labels.npy"
METRIC_TXT_PATH = SAVE_DIR / "cb_1_lowess_metrics.txt"
PLOT_SAVE_PATH = SAVE_DIR / "cb_1_lowess_optimization_history.png"

#  Load dataset with selected features
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
        "iterations": trial.suggest_int("iterations", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1.0, 20.0),
        "verbose": 0,
        "random_state": 0,
        "task_type": "CPU"
    }

    model = CatBoostClassifier(**params)
    y_pred = cross_val_predict(model, X, y, cv=skf, method="predict", n_jobs=1)
    return accuracy_score(y, y_pred)

#  Data preparation
X, y = load_data(FEATURE_CSV_PATH, DATA_PATH)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

#  Optuna tuning
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=150)

#  Save optimization history
fig = vis.plot_optimization_history(study)
pio.write_image(fig, PLOT_SAVE_PATH, format="png", width=800, height=600, scale=2)

#  Save best hyperparameters
best_params = study.best_params
with open(PARAMS_TXT_PATH, "w") as f:
    f.write("CatBoost Best Hyperparameters (Optuna)\n")
    f.write("=" * 40 + "\n")
    for key, val in best_params.items():
        f.write(f"{key}: {val}\n")

#  Train final model
best_params.update({
    "random_seed": 0,
    "verbose": 0,
    "task_type": "CPU"
})
final_model = CatBoostClassifier(**best_params)
oof_pred = cross_val_predict(final_model, X, y, cv=skf, method="predict", n_jobs=1)
oof_proba = cross_val_predict(final_model, X, y, cv=skf, method="predict_proba", n_jobs=1)[:, 1]

# Save model
final_model.fit(X, y)
final_model.save_model(MODEL_SAVE_PATH)

# Save OOF predictions
np.save(PROBA_SAVE_PATH, np.vstack([1 - oof_proba, oof_proba]).T)
np.save(LABEL_SAVE_PATH, y)

# Save metrics
metrics = evaluate_metrics(y, oof_pred, oof_proba)
with open(METRIC_TXT_PATH, "w") as f:
    f.write("CatBoost (Optuna Tuned) CV Metrics\n")
    f.write("=" * 40 + "\n")
    for key, val in metrics.items():
        f.write(f"{key}: {val:.4f}\n")

print("CatBoost model, OOF predictions, metrics, and visualization saved successfully")