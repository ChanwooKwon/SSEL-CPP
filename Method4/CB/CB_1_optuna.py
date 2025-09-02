import os
import numpy as np
import pandas as pd
import joblib
import optuna
import optuna.visualization as vis
import plotly.io as pio
import uuid
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, log_loss
)
from optuna.samplers import TPESampler

# ── Path settings ─────────────────────────────────────────────
FEATURE_CSV_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Filtermethod/Featureimportance/feature_importance_4.csv"
DATA_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
MODEL_SAVE_PATH = "./BestModel/CB/CB_1_optuna_best_model.cbm"
PARAMS_TXT_PATH = "./CB_1_optuna_best_params.txt"

PROBA_SAVE_PATH = "./CB_1_optuna_oof_proba.npy"
LABEL_SAVE_PATH = "./CB_1_optuna_labels.npy"
METRIC_TXT_PATH = "./CB_1_optuna_metrics.txt"
PLOT_SAVE_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Method4/CB/CB_1_optimization_history.png"

# ── Create output directories if they don’t exist ─────────────
for path in [MODEL_SAVE_PATH, PARAMS_TXT_PATH, PROBA_SAVE_PATH, LABEL_SAVE_PATH, METRIC_TXT_PATH]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ── Load dataset ─────────────────────────────────────────────
def load_data(feature_path, data_path):
    top_features = pd.read_csv(feature_path)["Feature"].tolist()
    df = pd.read_csv(data_path)
    df_filtered = df[["id"] + top_features].copy()
    df_filtered["Label"] = df_filtered["id"].apply(lambda x: 1 if "positive" in x else 0)
    X = df_filtered.drop(columns=["id", "Label"])
    y = df_filtered["Label"].values
    return X, y

# ── Evaluation metrics ───────────────────────────────────────
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

# ── Optuna objective function ─────────────────────────────────
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1.0, 20.0),
        "random_seed": 0,
        "verbose": 0,
        "task_type": "CPU",
        "loss_function": "Logloss",
        "eval_metric": "Logloss"
    }

    # Assign a unique train_dir to avoid conflicts
    unique_train_dir = f"./catboost_info_{uuid.uuid4()}"
    model = CatBoostClassifier(**params, allow_writing_files=False, train_dir=unique_train_dir)

    y_pred = cross_val_predict(model, X, y, cv=skf, method="predict", n_jobs=7)
    return accuracy_score(y, y_pred)

# ── Prepare data ─────────────────────────────────────────────
X, y = load_data(FEATURE_CSV_PATH, DATA_PATH)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# ── Run Optuna optimization ──────────────────────────────────
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100)

# ── Save optimization history plot ──────────────────────────
fig = vis.plot_optimization_history(study)
pio.write_image(fig, PLOT_SAVE_PATH, format="png", width=800, height=600, scale=2)

# ── Best hyperparameters ────────────────────────────────────
best_params = study.best_params

# Save hyperparameters
with open(PARAMS_TXT_PATH, "w") as f:
    f.write("CatBoost Best Hyperparameters (Optuna)\n")
    f.write("=" * 40 + "\n")
    for key, val in best_params.items():
        f.write(f"{key}: {val}\n")

# ── Train final model ───────────────────────────────────────
best_params.update({
    "random_seed": 0,
    "verbose": 0,
    "task_type": "CPU",
    "loss_function": "Logloss",
    "eval_metric": "Logloss"
})
final_model = CatBoostClassifier(**best_params)
final_model.fit(X, y)
final_model.save_model(MODEL_SAVE_PATH)

# ── OOF predictions ─────────────────────────────────────────
# Safe with already-trained model for parallelization
oof_pred = cross_val_predict(final_model, X, y, cv=skf, method="predict", n_jobs=7)
oof_proba = cross_val_predict(final_model, X, y, cv=skf, method="predict_proba", n_jobs=7)[:, 1]

# ── Save outputs ────────────────────────────────────────────
np.save(PROBA_SAVE_PATH, np.vstack([1 - oof_proba, oof_proba]).T)
np.save(LABEL_SAVE_PATH, y)

metrics = evaluate_metrics(y, oof_pred, oof_proba)
with open(METRIC_TXT_PATH, "w") as f:
    f.write("CatBoost (Optuna Tuned) CV Metrics\n")
    f.write("=" * 40 + "\n")
    for key, val in metrics.items():
        f.write(f"{key}: {val:.4f}\n")

print(" CatBoost model, OOF predictions, and metrics saved successfully.")
