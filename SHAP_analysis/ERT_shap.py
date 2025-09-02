import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ─── Settings ─────────────────────────────────────────────
MODEL_NAME = "ERT_1_optuna_best_model"
MODEL_DIR = "C:/Users/LG_LAB/Desktop/SSELCPP/Method4/BestModel/ERT"
PARAMS_TXT = "C:/Users/LG_LAB/Desktop/SSELCPP/Method4/ERT/ERT_1_optuna_best_params.txt"
DATA_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
SAVE_DIR = f"C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_Results/ERT"
FEATURE_CSV_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Filtermethod/Featureimportance/feature_importance_4.csv"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/folds", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/full_train", exist_ok=True)

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

# ─── Load Parameters ──────────────────────────────────
def load_ert_params(path):
    params = {}
    with open(path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":")
                key = key.strip()
                val = val.strip()
                if key in ["max_features"]:
                    params[key] = val
                elif key in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
                    params[key] = int(val)
    return params

best_params = load_ert_params(PARAMS_TXT)

# ─── SHAP Visualization Function ───────────────────────────────
def save_shap_summary_plots(shap_values, X, feature_names, save_prefix):
    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_bar.png")
    plt.close()

    # Beeswarm plot
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

    model = ExtraTreesClassifier(**best_params, random_state=0)
    model.fit(X_train, y_train)

    X_sample = shap.utils.sample(X_val, 100, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    shap_vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    fold_shap_values.append(shap_vals)
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    if len(mean_abs_shap.shape) > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=0)

    shap_df = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_abs_shap}).sort_values(by="MeanAbsSHAP", ascending=False)

    fold_dir = f"{SAVE_DIR}/folds/fold_{fold}"
    os.makedirs(fold_dir, exist_ok=True)
    np.save(f"{fold_dir}/shap_values.npy", shap_vals)
    shap_df.to_csv(f"{fold_dir}/mean_abs_shap.csv", index=False)
    shap_df["Feature"].to_csv(f"{fold_dir}/ranked_features.csv", index=False, header=False)
    save_shap_summary_plots(shap_vals, X_sample, feature_names, f"{fold_dir}/fold{fold}")

# ② Fold Average SHAP
print("[SHAP Fold Average]")
min_len = min([s.shape[0] for s in fold_shap_values])
fold_shap_values_trunc = [s[:min_len] for s in fold_shap_values]
mean_fold_shap = np.mean(np.abs(np.vstack(fold_shap_values_trunc)), axis=0)
if len(mean_fold_shap.shape) > 1:
    mean_fold_shap = mean_fold_shap.mean(axis=0)

mean_df = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_fold_shap}).sort_values(by="MeanAbsSHAP", ascending=False)
mean_df.to_csv(f"{SAVE_DIR}/folds/fold_mean_abs_shap.csv", index=False)

# ③ Full Training Set SHAP
print("[SHAP Full Train]")
model_all = joblib.load(os.path.join(MODEL_DIR, f"{MODEL_NAME}.pkl"))
X_sample_all = shap.utils.sample(X, 100, random_state=42)
explainer_all = shap.TreeExplainer(model_all)
shap_vals_all = explainer_all.shap_values(X_sample_all)
shap_vals_all = shap_vals_all[1] if isinstance(shap_vals_all, list) else shap_vals_all

mean_abs_all = np.mean(np.abs(shap_vals_all), axis=0)
if len(mean_abs_all.shape) > 1:
    mean_abs_all = mean_abs_all.mean(axis=0)

shap_df_all = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_abs_all}).sort_values(by="MeanAbsSHAP", ascending=False)

np.save(f"{SAVE_DIR}/full_train/shap_values.npy", shap_vals_all)
shap_df_all.to_csv(f"{SAVE_DIR}/full_train/mean_abs_shap.csv", index=False)
shap_df_all["Feature"].to_csv(f"{SAVE_DIR}/full_train/ranked_features.csv", index=False, header=False)
save_shap_summary_plots(shap_vals_all, X_sample_all, feature_names, f"{SAVE_DIR}/full_train/full_train")

print("\n SHAP computation and saving complete")
