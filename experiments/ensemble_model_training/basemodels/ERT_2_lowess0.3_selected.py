import os
import numpy as np
import pandas as pd
import joblib
import optuna
import optuna.visualization as vis
import plotly.io as pio
from sklearn.ensemble import ExtraTreesClassifier
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

#  Refined parameter ranges based on standards
param_ranges = {
    "n_estimators": (400, 600),
    "max_depth": (15, 25),
    "min_samples_split": (2, 6),
    "min_samples_leaf": (1, 3),
    "max_features": ["log2"]
}

#  Path settings
FEATURE_CSV_PATH = base_dir / "shap_feature_selection" / "feature_optimization" / "ERT" / "curve" / "optimal_features_lowess0.3_ert_accuracy_125.csv"
DATA_PATH = base_dir / "data" / "mordred2dfeature_knn.csv"
SAVE_DIR = current_dir / "results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = SAVE_DIR / "ert_2_lowess0.3_best_model.pkl"
PARAMS_TXT_PATH = SAVE_DIR / "ert_2_lowess0.3_best_params.txt"
PROBA_SAVE_PATH = SAVE_DIR / "ert_2_lowess0.3_oof_proba.npy"
LABEL_SAVE_PATH = SAVE_DIR / "ert_2_lowess0.3_labels.npy"
METRIC_TXT_PATH = SAVE_DIR / "ert_2_lowess0.3_metrics.txt"
PLOT_SAVE_PATH = SAVE_DIR / "ert_2_lowess0.3_optimization_history.png"

#  Load dataset
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
        "n_estimators": trial.suggest_int("n_estimators", 400, 600),
        "max_depth": trial.suggest_int("max_depth", 15, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
        "max_features": trial.suggest_categorical("max_features", ["log2"]),
        "bootstrap": False,
        "random_state": 0,
        "n_jobs": 2
    }
    model = ExtraTreesClassifier(**params)
    y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=2)
    return accuracy_score(y, y_pred)

#  Data preparation
X, y = load_data(FEATURE_CSV_PATH, DATA_PATH)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

#  Run Optuna hyperparameter tuning
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=150)

#  Save optimization history plot
fig = vis.plot_optimization_history(study)
pio.write_image(fig, PLOT_SAVE_PATH, format="png", width=800, height=600, scale=2)

#  Save best parameters
best_params = study.best_params
with open(PARAMS_TXT_PATH, "w") as f:
    f.write("ERT2 Best Hyperparameters (Optuna)\n")
    f.write("=" * 40 + "\n")
    for key, val in best_params.items():
        f.write(f"{key}: {val}\n")

#  Train final model
best_params.update({"random_state": 0, "n_jobs": 2})
final_model = ExtraTreesClassifier(**best_params)
oof_pred = cross_val_predict(final_model, X, y, cv=skf, n_jobs=2)
oof_proba = cross_val_predict(final_model, X, y, cv=skf, method="predict_proba", n_jobs=2)[:, 1]

final_model.fit(X, y)
joblib.dump(final_model, MODEL_SAVE_PATH)

#  Save OOF predictions
np.save(PROBA_SAVE_PATH, np.vstack([1 - oof_proba, oof_proba]).T)
np.save(LABEL_SAVE_PATH, y)

#  Save performance metrics
metrics = evaluate_metrics(y, oof_pred, oof_proba)
with open(METRIC_TXT_PATH, "w") as f:
    f.write("ExtraTrees (Optuna Tuned) CV Metrics\n")
    f.write("=" * 40 + "\n")
    for key, val in metrics.items():
        f.write(f"{key}: {val:.4f}\n")

print("ERT2 model, OOF, metrics, and optimization plot saved successfully")