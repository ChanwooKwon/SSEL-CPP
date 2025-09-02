import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import os

# 🔹 Path settings
PERF_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/ERT/curve/ert_performance_by_features.csv"
FEATURE_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_Results/ERT/full_train/ranked_features.csv"
SAVE_DIR = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/ERT/curve"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔹 Load data
df_perf = pd.read_csv(PERF_PATH)
ranked_features = pd.read_csv(FEATURE_PATH, header=None)[0].tolist()

# 🔹 LOWESS smoothing function
def get_lowess_optimal(x_vals, y_vals, frac=0.3):
    lowess_result = lowess(y_vals, x_vals, frac=frac, return_sorted=False)
    max_idx = int(np.argmax(lowess_result))
    opt_feat = int(x_vals[max_idx])
    return lowess_result, opt_feat

# 🔹 Find optimal number of features
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

    # Axis labels and title font size
    plt.xlabel("Number of Features", fontsize=20)
    plt.ylabel(metric, fontsize=20)
    plt.title(f"ERT LOWESS0.3", fontsize=22)

    # Tick label size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Legend in lower right corner
    plt.legend(loc="lower right", fontsize=18)

    plt.grid(True)

plt.tight_layout()
plot_path = os.path.join(SAVE_DIR, "ert_shap_lowess0.3_selection.png")
plt.savefig(plot_path, dpi=1000)
plt.show()
print(f"LOWESS performance plot saved: {plot_path}")

# 🔹 Save optimal feature lists
for metric in ["Accuracy", "ROC_AUC", "MCC"]:
    opt_k = results[metric]
    feature_list = ranked_features[:opt_k]
    save_path = os.path.join(SAVE_DIR, f"optimal_features_lowess0.3_ert_{metric.lower()}_{opt_k}.csv")
    pd.DataFrame(feature_list, columns=["Feature"]).to_csv(save_path, index=False)
    print(f"Optimal feature list saved for {metric}: {save_path}")
