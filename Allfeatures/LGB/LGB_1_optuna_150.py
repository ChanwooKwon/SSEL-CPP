import os
import numpy as np
import pandas as pd
import joblib
import optuna
import optuna.visualization as vis
import plotly.io as pio
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, log_loss
)
from optuna.samplers import TPESampler

#  Path settings (All features version)
DATA_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
MODEL_SAVE_PATH = "./Allfeatures/BestModel/LGB/LGB_1_optuna_best_model_all.pkl"
PARAMS_TXT_PATH = "./Allfeatures/LGB/LGB_1_optuna_best_params_all.txt"
PROBA_SAVE_PATH = "./Allfeatures/OOF_Proba/LGB_1_optuna_oof_proba_all.npy"
LABEL_SAVE_PATH = "./Allfeatures/OOF_Proba/LGB_1_optuna_labels_all.npy"
METRIC_TXT_PATH = "./Allfeatures/LGB/LGB_1_optuna_metrics_all.txt"
PLOT_SAVE_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Allfeatures/LGB/LGB_1_optimization_history_all.png"

#  Create save directories
for path in [MODEL_SAVE_PATH, PARAMS_TXT_PATH, PROBA_SAVE_PATH, LABEL_SAVE_PATH, METRIC_TXT_PATH]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

#  Load dataset (using all features)
def load_data(data_path):
    df = pd.read_csv(data_path)
    df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x else 0)
    X = df.drop(columns=["id", "Label"], errors="ignore")
    y = df["Label"].values
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
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0),
        "random_state": 0,
        "n_jobs": 7,
        "verbosity": -1
    }

    model = LGBMClassifier(**params)
    y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=7)
    return accuracy_score(y, y_pred)

#  Data preparation
X, y = load_data(DATA_PATH)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Optuna tuning
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=150)

#  Save optimization history plot
fig = vis.plot_optimization_history(study)
pio.write_image(fig, PLOT_SAVE_PATH, format="png", width=800, height=600, scale=2)

#  Save best parameters
best_params = study.best_params
with open(PARAMS_TXT_PATH, "w") as f:
    f.write("LightGBM Best Hyperparameters (Optuna, All Features)\n")
    f.write("=" * 40 + "\n")
    for key, val in best_params.items():
        f.write(f"{key}: {val}\n")

#  Build and train final model
best_params.update({
    "random_state": 0,
    "n_jobs": 7,
    "verbosity": -1
})
final_model = LGBMClassifier(**best_params)

# OOF predictions
oof_pred = cross_val_predict(final_model, X, y, cv=skf, n_jobs=7)
oof_proba = cross_val_predict(final_model, X, y, cv=skf, method="predict_proba", n_jobs=7)[:, 1]

# Train and save model
final_model.fit(X, y)
joblib.dump(final_model, MODEL_SAVE_PATH)

# Save results
np.save(PROBA_SAVE_PATH, np.vstack([1 - oof_proba, oof_proba]).T)
np.save(LABEL_SAVE_PATH, y)

metrics = evaluate_metrics(y, oof_pred, oof_proba)
with open(METRIC_TXT_PATH, "w") as f:
    f.write("LightGBM (Optuna Tuned, All Features) CV Metrics\n")
    f.write("=" * 40 + "\n")
    for key, val in metrics.items():
        f.write(f"{key}: {val:.4f}\n")

print(" LightGBM model using all features saved successfully")
