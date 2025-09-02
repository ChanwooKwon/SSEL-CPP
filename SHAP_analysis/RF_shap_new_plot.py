import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ───── Configuration ─────
MODEL = "RF"
SAVE_DIR = f"C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_Results/{MODEL}"
FEATURE_CSV_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Filtermethod/Featureimportance/feature_importance_4.csv"
os.makedirs(SAVE_DIR, exist_ok=True)

# ───── Load Feature Names ─────
feature_names = pd.read_csv(FEATURE_CSV_PATH)["Feature"].tolist()

# ───── Visualization Function ─────
def save_shap_summary_plots(shap_values, X, feature_names, save_prefix):
    # ── Bar Plot ──
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, plot_type="bar")
    ax = plt.gca()
    ax.set_xlabel("mean(|SHAP value|)", fontsize=20)
    ax.set_ylabel("")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(22)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(20)
    plt.savefig(f"{save_prefix}_bar_new.png", dpi=1000, bbox_inches='tight')
    plt.close()

    # ── Dot Plot ──
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    ax = plt.gca()
    ax.set_xlabel("SHAP value", fontsize=20)
    ax.set_ylabel("")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(22)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(16)

    # Modify colorbar 
    colorbar = plt.gcf().axes[-1]
    colorbar.set_ylabel("Feature value", fontsize=24, labelpad=0.1)
    colorbar.tick_params(labelsize=22)

    plt.savefig(f"{save_prefix}_summary_new.png", dpi=1000, bbox_inches='tight')
    plt.close()

# ───── Load Original Data ─────
DATA_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
df = pd.read_csv(DATA_PATH)
df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x.lower() else 0)
X_all = df[["id"] + feature_names].drop(columns=["id"])

# ───── Re-generate Visualizations per Fold ─────
for fold in range(5):
    fold_dir = f"{SAVE_DIR}/folds/fold_{fold}"
    shap_vals = np.load(f"{fold_dir}/shap_values.npy")
    X_sample = shap.utils.sample(X_all, shap_vals.shape[0], random_state=42)
    save_shap_summary_plots(shap_vals, X_sample, feature_names, f"{fold_dir}/fold{fold}")
    print(f" Fold {fold} visualization complete")

# ───── Re-generate Visualization for Full Training Set ─────
shap_vals_all = np.load(f"{SAVE_DIR}/full_train/shap_values.npy")
X_sample_all = shap.utils.sample(X_all, shap_vals_all.shape[0], random_state=42)
save_shap_summary_plots(shap_vals_all, X_sample_all, feature_names, f"{SAVE_DIR}/full_train/full_train")
print(" Full train visualization complete")
