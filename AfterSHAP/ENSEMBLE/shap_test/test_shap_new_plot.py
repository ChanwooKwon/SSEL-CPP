import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ── Settings ─────────────────────────────────────────────
SHAP_DIR = "C:/Users/LG_LAB/Desktop/SSELCPP/AfterSHAP/ENSEMBLE/shap_test/"
FEATURE_DIR = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select"
TEST_DATA_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/test_mordred2dfeature_knn.csv"

model_names = ["XGB", "LGB"]

feature_files = {
    "XGB": os.path.join(FEATURE_DIR, "XGB", "feature", "optimal_features_lowess0.2_accuracy_75.csv"),
    "LGB": os.path.join(FEATURE_DIR, "LGB", "feature", "optimal_features_lowess_lgb_accuracy_95.csv")
}

# ── Load Test Data ───────────────────────────────────────
df_test = pd.read_csv(TEST_DATA_PATH)
X_test_full = df_test.drop(columns=["id"], errors="ignore")

# ── SHAP Visualization Function ──────────────────────────
def save_shap_summary_plots(shap_values, X, feature_names, save_prefix):
    # Bar Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, plot_type="bar")
    ax = plt.gca()
    ax.set_xlabel("mean(|SHAP value|)", fontsize=20)
    ax.set_ylabel("")
    for tick in ax.get_yticklabels(): tick.set_fontsize(22)
    for tick in ax.get_xticklabels(): tick.set_fontsize(20)
    plt.savefig(f"{save_prefix}_bar_new.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Dot Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    ax = plt.gca()
    ax.set_xlabel("SHAP value", fontsize=20)
    ax.set_ylabel("")
    for tick in ax.get_yticklabels(): tick.set_fontsize(22)
    for tick in ax.get_xticklabels(): tick.set_fontsize(16)
    colorbar = plt.gcf().axes[-1]
    colorbar.set_ylabel("Feature value", fontsize=24, labelpad=0.1)
    colorbar.tick_params(labelsize=22)
    plt.savefig(f"{save_prefix}_summary_new.png", dpi=300, bbox_inches='tight')
    plt.close()

# ── Run SHAP Visualization for Each Model ─────────────────
for model_name in model_names:
    shap_path = os.path.join(SHAP_DIR, f"shap_values_{model_name}.csv")
    if not os.path.exists(shap_path):
        print(f" {shap_path} not found - skipped")
        continue

    shap_values = pd.read_csv(shap_path).values
    feature_list = pd.read_csv(feature_files[model_name])["Feature"].tolist()
    X_selected = X_test_full[feature_list]

    if shap_values.shape[1] != X_selected.shape[1]:
        print(f" Feature count mismatch - skipped {model_name}")
        continue

    save_prefix = os.path.join(SHAP_DIR, f"shap_{model_name}")
    save_shap_summary_plots(shap_values, X_selected, feature_list, save_prefix)
    print(f" Saved successfully: {model_name}")

print(" SHAP visualization completed for all models")
