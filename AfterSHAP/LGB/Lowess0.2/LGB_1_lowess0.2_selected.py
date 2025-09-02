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

# 🔧 경로 설정
FEATURE_CSV_PATH = "C:/Users/LG_LAB/Desktop/SSCPP/SHAP_select/LGB/feature/optimal_features_lowess0.2_lgb_accuracy_115.csv"
DATA_PATH = "C:/Users/LG_LAB/Desktop/SSCPP/Dataset/mordred2dfeature_knn.csv"
MODEL_SAVE_PATH = "C:/Users/LG_LAB/Desktop/SSCPP/AfterSHAP/BestModel/LGB/Lowess0.2/lgb_1_lowess0.2_best_model.pkl"
PARAMS_TXT_PATH = "./lgb_1_lowess0.2_best_params.txt"
PROBA_SAVE_PATH = "./lgb_1_lowess0.2_oof_proba.npy"
LABEL_SAVE_PATH = "./lgb_1_lowess0.2_labels.npy"
METRIC_TXT_PATH = "./lgb_1_lowess0.2_metrics.txt"
PLOT_SAVE_PATH = "./lgb_1_lowess0.2_optimization_history.png"

# 🔧 저장 경로 생성
for path in [MODEL_SAVE_PATH, PROBA_SAVE_PATH, LABEL_SAVE_PATH, METRIC_TXT_PATH, PARAMS_TXT_PATH, PLOT_SAVE_PATH]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# 🔹 데이터 불러오기
def load_data(feature_path, data_path):
    top_features = pd.read_csv(feature_path)["Feature"].tolist()
    df = pd.read_csv(data_path)
    df_filtered = df[["id"] + top_features].copy()
    df_filtered["Label"] = df_filtered["id"].apply(lambda x: 1 if "positive" in x else 0)
    X = df_filtered.drop(columns=["id", "Label"])
    y = df_filtered["Label"].values
    return X, y

# 🔹 평가 지표 함수
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

# 🔹 Optuna 목적 함수
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

# 🔹 데이터 준비
X, y = load_data(FEATURE_CSV_PATH, DATA_PATH)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# 🔹 Optuna 튜닝 수행
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=150)

# 🔹 성능 수렴 그래프 저장
fig = vis.plot_optimization_history(study)
pio.write_image(fig, PLOT_SAVE_PATH, format="png", width=800, height=600, scale=2)

# 🔹 최적 파라미터 저장
best_params = study.best_params
with open(PARAMS_TXT_PATH, "w") as f:
    f.write("LGB Best Hyperparameters (Optuna)\n")
    f.write("=" * 40 + "\n")
    for key, val in best_params.items():
        f.write(f"{key}: {val}\n")

# 🔹 최종 모델 훈련
best_params.update({
    "random_state": 0,
    "n_jobs": 7
})
final_model = LGBMClassifier(**best_params)
oof_pred = cross_val_predict(final_model, X, y, cv=skf, n_jobs=7)
oof_proba = cross_val_predict(final_model, X, y, cv=skf, method="predict_proba", n_jobs=7)[:, 1]

# 🔹 모델 저장
final_model.fit(X, y)
joblib.dump(final_model, MODEL_SAVE_PATH)

# 🔹 OOF 결과 저장
np.save(PROBA_SAVE_PATH, np.vstack([1 - oof_proba, oof_proba]).T)
np.save(LABEL_SAVE_PATH, y)

# 🔹 성능 저장
metrics = evaluate_metrics(y, oof_pred, oof_proba)
with open(METRIC_TXT_PATH, "w") as f:
    f.write("LGB (Optuna Tuned) CV Metrics\n")
    f.write("=" * 40 + "\n")
    for key, val in metrics.items():
        f.write(f"{key}: {val:.4f}\n")

print("[✅] LGB 모델, OOF, metrics, 시각화 저장 완료")
