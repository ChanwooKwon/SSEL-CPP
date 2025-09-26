from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent.parent.parent

# 🔹 Path settings
PERF_PATH = current_dir / "curve" / "ert_performance_by_features.csv"
FEATURE_PATH = base_dir / "shap_feature_selection" / "results" / "ERT" / "full_train" / "ranked_features.csv"
SAVE_DIR = current_dir / "curve"

# 🔹 Load Data
if not PERF_PATH.exists():
    print(f"❌ Performance file not found: {PERF_PATH}")
    print("   Please run poly_ema_shap_curve.py first!")
    exit(1)

df_perf = pd.read_csv(PERF_PATH)
ranked_features = pd.read_csv(FEATURE_PATH, header=None)[0].tolist()

# 🔹 LOWESS smoothing function
def get_lowess_optimal(x_vals, y_vals, frac=0.5):
    lowess_result = lowess(y_vals, x_vals, frac=frac, return_sorted=False)
    max_idx = int(np.argmax(lowess_result))
    opt_feat = int(x_vals[max_idx])
    return lowess_result, opt_feat

print("🔍 LOWESS 0.5 optimization...")
x_vals = df_perf["Feature_Count"]
lowess_results = {}

plt.figure(figsize=(18, 5))
for i, metric in enumerate(["Accuracy", "ROC_AUC", "MCC"]):
    y_vals = df_perf[metric]
    smooth, opt_feat = get_lowess_optimal(x_vals, y_vals, frac=0.5)
    df_perf[f"{metric}_LOWESS"] = smooth
    lowess_results[metric] = opt_feat

    plt.subplot(1, 3, i + 1)
    plt.plot(x_vals, y_vals, marker="o", label=metric, alpha=0.7)
    plt.plot(x_vals, smooth, linestyle="--", label="LOWESS", linewidth=2)
    plt.axvline(opt_feat, color="red", linestyle=":", label=f"Optimal: {opt_feat}", linewidth=2)

    plt.xlabel("Number of Features", fontsize=20)
    plt.ylabel(metric, fontsize=20)
    plt.title(f"ERT LOWESS0.5", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc="lower right", fontsize=18)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = SAVE_DIR / "ert_shap_lowess0.5_selection.png"
plt.savefig(plot_path, dpi=1000, bbox_inches='tight')
plt.close()
print(f"✅ LOWESS 0.5 plot saved: {plot_path}")

# 🔹 Save optimal feature lists
print("💾 Saving optimal feature lists...")
for metric in ["Accuracy", "ROC_AUC", "MCC"]:
    opt_k = lowess_results[metric]
    feature_list = ranked_features[:opt_k]
    save_path = SAVE_DIR / f"optimal_features_lowess0.5_ert_{metric.lower()}_{opt_k}.csv"
    pd.DataFrame(feature_list, columns=["Feature"]).to_csv(save_path, index=False)

# 🔹 Summary
print("\n📊 LOWESS 0.5 Optimization Results:")
for metric, opt_feat in lowess_results.items():
    print(f"  {metric}: {opt_feat} features")

# Save summary
lowess_summary = pd.DataFrame({
    'Metric': ['Accuracy', 'ROC_AUC', 'MCC'],
    'LOWESS_0.5_Optimal': [lowess_results['Accuracy'], lowess_results['ROC_AUC'], lowess_results['MCC']]
})
lowess_summary.to_csv(SAVE_DIR / "ert_lowess0.5_summary.csv", index=False)
print(f"✅ Summary saved: {SAVE_DIR / 'ert_lowess0.5_summary.csv'}")