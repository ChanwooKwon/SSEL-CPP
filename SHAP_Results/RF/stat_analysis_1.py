import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr
import matplotlib.pyplot as plt
import os

# ── Path Settings ───────────────────────────────────────────
MODEL = "RF"
BASE_DIR = f"C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_Results/{MODEL}"
fold_path = f"{BASE_DIR}/folds/fold_mean_abs_shap.csv"
full_path = f"{BASE_DIR}/full_train/mean_abs_shap.csv"
SAVE_TXT_PATH = f"{BASE_DIR}/{MODEL}_shap_comparison.txt"
SAVE_PNG_PATH = f"{BASE_DIR}/{MODEL}_shap_comparison_plot.png"

# ── Load Data ──────────────────────────────────────────────
fold_df = pd.read_csv(fold_path)
full_df = pd.read_csv(full_path)

# ── Merge Features and Align Rankings ──────────────────────
merged = pd.merge(fold_df, full_df, on="Feature", suffixes=('_fold', '_full'))
merged["Rank_fold"] = merged["MeanAbsSHAP_fold"].rank(ascending=False)
merged["Rank_full"] = merged["MeanAbsSHAP_full"].rank(ascending=False)

# ── Statistical Calculations ───────────────────────────────
spearman_corr, _ = spearmanr(merged["Rank_fold"], merged["Rank_full"])
kendall_corr, _ = kendalltau(merged["Rank_fold"], merged["Rank_full"])
pearson_corr, _ = pearsonr(merged["MeanAbsSHAP_fold"], merged["MeanAbsSHAP_full"])
mae = (merged["MeanAbsSHAP_fold"] - merged["MeanAbsSHAP_full"]).abs().mean()
mae_percent = (mae / merged["MeanAbsSHAP_full"].mean()) * 100

# ── Save Results ───────────────────────────────────────────
with open(SAVE_TXT_PATH, "w", encoding="utf-8") as f:
    f.write(f" {MODEL} Comparison Results\n")
    f.write(f" Pearson: {pearson_corr:.4f}\n")
    f.write(f" Spearman: {spearman_corr:.4f}\n")
    f.write(f" Kendall Tau: {kendall_corr:.4f}\n")
    f.write(f" MAE of SHAP Values: {mae:.6f}\n")
    f.write(f" Relative MAE (% of Full SHAP Mean): {mae_percent:.2f}%\n")

print(f" Saved: {SAVE_TXT_PATH}")

# ── Visualization ──────────────────────────────────────────
plt.figure(figsize=(10, 4))
x_vals = list(range(1, len(merged) + 1))
plt.plot(x_vals, merged["MeanAbsSHAP_fold"], label="Fold Mean SHAP")
plt.plot(x_vals, merged["MeanAbsSHAP_full"], label="Full Train SHAP")
plt.xticks(ticks=x_vals[::5], labels=x_vals[::5], fontsize=6, rotation=90)  # Adjust spacing
plt.title(f"{MODEL}: SHAP Comparison (Fold vs Full)")
plt.xlabel("Feature Index")
plt.ylabel("Mean Absolute SHAP")
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_PNG_PATH, dpi=300)
plt.close()

print(f" Visualization Saved: {SAVE_PNG_PATH}")
