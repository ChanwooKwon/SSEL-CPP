import os
import numpy as np
import pandas as pd
import joblib
import optuna
import optuna.visualization as vis
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
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

FEATURE_CSV_PATH = base_dir / "shap_feature_selection" / "feature_optimization" / "RF" / "curve" / "optimal_features_polyfit_rf_acc_152.csv"
DATA_PATH = base_dir / "data" / "mordred2dfeature_knn.csv"
SAVE_DIR = current_dir / "results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = SAVE_DIR / "rf_1_maxima__best_model.pkl"
PARAMS_TXT_PATH = SAVE_DIR / "rf_1_maxima__best_params.txt"
PROBA_SAVE_PATH = SAVE_DIR / "rf_1_maxima__oof_proba.npy"
LABEL_SAVE_PATH = SAVE_DIR / "rf_1_maxima__labels.npy"
METRIC_TXT_PATH = SAVE_DIR / "rf_1_maxima__metrics.txt"
PLOT_SAVE_PATH = SAVE_DIR / "rf_1_maxima__optimization_history.png"

def load_data(feature_path, data_path):
    top_features = pd.read_csv(feature_path)["Feature"].tolist()
    df = pd.read_csv(data_path)
    df_filtered = df[["id"] + top_features].copy()
    df_filtered["Label"] = df_filtered["id"].apply(lambda x: 1 if "positive" in x else 0)
    X = df_filtered.drop(columns=["id", "Label"])
    y = df_filtered["Label"].values
    return X, y

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

X, y = load_data(FEATURE_CSV_PATH, DATA_PATH)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 0,
        "n_jobs": 5
    }
    model = RandomForestClassifier(**params)
    y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=5)
    return accuracy_score(y, y_pred)

study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=150)

fig = vis.plot_optimization_history(study)
pio.write_image(fig, PLOT_SAVE_PATH, format="png", width=800, height=600, scale=2)

best_params = study.best_params
with open(PARAMS_TXT_PATH, "w") as f:
    f.write("RandomForest Best Hyperparameters (Optuna)\n")
    f.write("=" * 40 + "\n")
    for key, val in best_params.items():
        f.write(f"{key}: {val}\n")

best_params.update({
    "random_state": 0,
    "n_jobs": 7
})
final_model = RandomForestClassifier(**best_params)
oof_pred = cross_val_predict(final_model, X, y, cv=skf, n_jobs=7)
oof_proba = cross_val_predict(final_model, X, y, cv=skf, method="predict_proba", n_jobs=7)[:, 1]

final_model.fit(X, y)
joblib.dump(final_model, MODEL_SAVE_PATH)

np.save(PROBA_SAVE_PATH, np.vstack([1 - oof_proba, oof_proba]).T)
np.save(LABEL_SAVE_PATH, y)

metrics = evaluate_metrics(y, oof_pred, oof_proba)
with open(METRIC_TXT_PATH, "w") as f:
    f.write("RandomForest (Optuna Tuned) CV Metrics\n")
    f.write("=" * 40 + "\n")
    for key, val in metrics.items():
        f.write(f"{key}: {val:.4f}\n")

print("RandomForest model, OOF, metrics saved successfully")