import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from pathlib import Path

# ─── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent.parent

# ─── Configuration (relative paths) ─────────────────────────────────────
MODEL_NAME = "LGB_1_optuna_best_model"
MODEL_DIR = base_dir / "model_training" / "saved_models" / "LGB"
DATA_PATH = base_dir / "data" / "mordred2dfeature_knn.csv"
SAVE_DIR = base_dir / "shap_feature_selection" / "results" / "LGB"
FEATURE_CSV_PATH = base_dir / "filter_methods" / "data" / "feature_importance" / "feature_importance_4.csv"

# ─── Ensure output directories exist ───────────────────────────────────
SAVE_DIR.mkdir(parents=True, exist_ok=True)
(SAVE_DIR / "folds").mkdir(parents=True, exist_ok=True)
(SAVE_DIR / "full_train").mkdir(parents=True, exist_ok=True)

# ─── Load Data ─────────────────────────────────────
def load_data(feature_path, data_path):
    top_features = pd.read_csv(feature_path)["Feature"].tolist()
    df = pd.read_csv(data_path)
    df_filtered = df[["id"] + top_features].copy()
    df_filtered["Label"] = df_filtered["id"].apply(lambda x: 1 if "positive" in x else 0)
    X = df_filtered.drop(columns=["id", "Label"])
    y = df_filtered["Label"].values
    return X, y, top_features

X, y, feature_names = load_data(FEATURE_CSV_PATH, DATA_PATH)

# ─── Best Hyperparameters (from Optuna) ─────────────────────────────
best_params = {
    'n_estimators': 799,
    'learning_rate': 0.2093349055172467,
    'max_depth': 6,
    'num_leaves': 110,
    'min_child_samples': 30,
    'subsample': 0.8122387609410768,
    'colsample_bytree': 0.9581329411908668,
    'reg_alpha': 1.363888870955132,
    'reg_lambda': 0.4345276021376643,
    'objective': 'binary',
    'random_state': 0,
    'n_jobs': -1
}

# ─── SHAP Visualization Function ───────────────────────────────
def save_shap_summary_plots(shap_values, X, feature_names, save_prefix):
    # Bar Plot
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_bar.png")
    plt.close()

    # Beeswarm/Dot Plot
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_summary.png")
    plt.close()

# ① SHAP per Fold
print("[SHAP per Fold]")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
fold_shap_values = []

for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X, y), total=5)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_val)[1]  # Class 1

    fold_shap_values.append(shap_vals)
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    shap_df = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_abs_shap})
    shap_df = shap_df.sort_values(by="MeanAbsSHAP", ascending=False)

    fold_dir = SAVE_DIR / "folds" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.save(fold_dir / "shap_values.npy", shap_vals)
    shap_df.to_csv(fold_dir / "mean_abs_shap.csv", index=False)
    shap_df["Feature"].to_csv(fold_dir / "ranked_features.csv", index=False, header=False)
    save_shap_summary_plots(shap_vals, X_val, feature_names, fold_dir / f"fold{fold}")

# ② SHAP Fold Average
print("[SHAP Fold Average]")
min_len = min([s.shape[0] for s in fold_shap_values])
fold_shap_values_trunc = [s[:min_len] for s in fold_shap_values]
mean_fold_shap = np.mean(np.abs(np.vstack(fold_shap_values_trunc)), axis=0)
mean_df = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_fold_shap}).sort_values(by="MeanAbsSHAP", ascending=False)
mean_df.to_csv(SAVE_DIR / "folds" / "fold_mean_abs_shap.csv", index=False)

# ③ SHAP Full Train
print("[SHAP Full Train]")
model_all = lgb.LGBMClassifier(**best_params)
model_all.fit(X, y)

explainer_all = shap.TreeExplainer(model_all)
shap_vals_all = explainer_all.shap_values(X)[1]

mean_abs_all = np.mean(np.abs(shap_vals_all), axis=0)
shap_df_all = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_abs_all}).sort_values(by="MeanAbsSHAP", ascending=False)

np.save(SAVE_DIR / "full_train" / "shap_values.npy", shap_vals_all)
shap_df_all.to_csv(SAVE_DIR / "full_train" / "mean_abs_shap.csv", index=False)
shap_df_all["Feature"].to_csv(SAVE_DIR / "full_train" / "ranked_features.csv", index=False, header=False)
save_shap_summary_plots(shap_vals_all, X, feature_names, SAVE_DIR / "full_train" / "full_train")

print("\n SHAP computation and saving complete")