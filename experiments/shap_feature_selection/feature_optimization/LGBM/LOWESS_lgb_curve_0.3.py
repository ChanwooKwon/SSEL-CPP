import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import os
from pathlib import Path

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent.parent.parent

# 🔹 Path Settings
PERF_PATH = current_dir / "curve" / "lgb_performance_by_features.csv"
FEATURE_PATH = base_dir / "shap_feature_selection" / "results" / "LGBM" / "full_train" / "ranked_features.csv"
SAVE_DIR = current_dir / "curve"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 🔹 Load Data
if not PERF_PATH.exists():
    print(f"❌ Performance file not found: {PERF_PATH}")
    print("   Please run poly_ema_shap_curve.py first!")
    exit(1)

df_perf = pd.read_csv(PERF_PATH)
ranked_features = pd.read_csv(FEATURE_PATH, header=None)[0].tolist()

# 🔹 LOWESS smoothing function
def get_lowess_optimal(x_vals, y_vals, frac=0.3):
    lowess_result = lowess(y_vals, x_vals, frac=frac, return_sorted=False)
    max_idx = int(np.argmax(lowess_result))
    opt_feat = int(x_vals[max_idx])
    return lowess_result, opt_feat

# 🔹 Find the optimal number of features
x_vals = df_perf["Feature_Count"]
results = {}

for metric in ["Accuracy", "ROC_AUC", "MCC"]:
    y_vals = df_perf[metric]
    smooth, opt_feat = get_lowess_optimal(x_vals, y_vals)
    df_perf[f"{metric}_LOWESS"] = smooth
    results[metric] = opt_feat

# 🔹 Plot performance + LOWESS and save
plt.figure(figsize=(18, 5))

for i, metric in enumerate(["Accuracy", "ROC_AUC", "MCC"]):
    plt.subplot(1, 3, i + 1)
    plt.plot(x_vals, df_perf[metric], marker="o", label=metric)
    plt.plot(x_vals, df_perf[f"{metric}_LOWESS"], linestyle="--", label="LOWESS")
    plt.axvline(results[metric], color="red", linestyle=":", label=f"Optimal: {results[metric]}")

    # Axis and title font sizes
    plt.xlabel("Number of Features", fontsize=20)
    plt.ylabel(metric, fontsize=20)
    plt.title(f"LightGBM LOWESS0.3", fontsize=22)

    # Tick label font sizes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Legend in bottom-right with adjusted font size
    plt.legend(loc="lower right", fontsize=18)

    plt.grid(True)

plt.tight_layout()
plot_path = SAVE_DIR / "lgb_shap_lowess0.3_selection.png"
plt.savefig(plot_path, dpi=1000)
plt.show()
print(f"LOWESS performance graph saved: {plot_path}")

# 🔹 Save optimal feature list
for metric in ["Accuracy", "ROC_AUC", "MCC"]:
    opt_k = results[metric]
    feature_list = ranked_features[:opt_k]
    save_path = SAVE_DIR / f"optimal_features_lowess0.3_lgb_{metric.lower()}_{opt_k}.csv"
    pd.DataFrame(feature_list, columns=["Feature"]).to_csv(save_path, index=False)
    print(f"Optimal feature list saved based on {metric}: {save_path}")