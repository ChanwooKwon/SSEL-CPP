import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr
from pathlib import Path

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent

# ── Model List ──────────────────────────────────────────────────────────
MODELS = ["AB", "CB", "ERT", "RF", "GB", "LGBM", "XGB"]

# ── Function: Generate Individual Analysis ─────────────────────────────
def generate_individual_analysis(model):
    """Generate individual consistency analysis for a model"""
    print(f"Generating consistency analysis for {model}...")
    
    # Paths
    BASE_DIR = base_dir / "results" / model
    fold_path = BASE_DIR / "folds" / "fold_mean_abs_shap.csv"
    full_path = BASE_DIR / "full_train" / "mean_abs_shap.csv"
    SAVE_TXT_PATH = BASE_DIR / f"{model}_shap_comparison.txt"
    SAVE_PNG_PATH = BASE_DIR / f"{model}_shap_comparison_plot.png"
    
    try:
        # Load data
        fold_df = pd.read_csv(fold_path)
        full_df = pd.read_csv(full_path)
        
        # Merge and align
        merged = pd.merge(fold_df, full_df, on="Feature", suffixes=('_fold', '_full'))
        fold_shap = merged["MeanAbsSHAP_fold"].values
        full_shap = merged["MeanAbsSHAP_full"].values
        
        # Calculate metrics
        pearson_corr, pearson_p = pearsonr(fold_shap, full_shap)
        spearman_corr, spearman_p = spearmanr(fold_shap, full_shap)
        kendall_corr, kendall_p = kendalltau(fold_shap, full_shap)
        
        mae = np.mean(np.abs(fold_shap - full_shap))
        global_mean_shap = np.mean(full_shap)
        relative_mae = (mae / global_mean_shap) * 100
        
        # Save results to text file
        with open(SAVE_TXT_PATH, "w") as f:
            f.write(f"SHAP Consistency Analysis for {model}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Correlation Metrics:\n")
            f.write(f"Pearson Correlation: {pearson_corr:.4f} (p-value: {pearson_p:.6f})\n")
            f.write(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.6f})\n")
            f.write(f"Kendall's Tau: {kendall_corr:.4f} (p-value: {kendall_p:.6f})\n\n")
            
            f.write("Error Metrics:\n")
            f.write(f"MAE (|ΔSHAP|): {mae:.6f}\n")
            f.write(f"Global Mean |SHAP|: {global_mean_shap:.6f}\n")
            f.write(f"Relative MAE: {relative_mae:.2f}%\n\n")
            
            f.write("Data Summary:\n")
            f.write(f"Number of Features: {len(merged)}\n")
            f.write(f"Fold SHAP Range: [{fold_shap.min():.6f}, {fold_shap.max():.6f}]\n")
            f.write(f"Full SHAP Range: [{full_shap.min():.6f}, {full_shap.max():.6f}]\n")
        
        # Create comparison plot
        plt.figure(figsize=(10, 8))
        plt.scatter(fold_shap, full_shap, alpha=0.6, s=30)
        plt.plot([min(fold_shap.min(), full_shap.min()), max(fold_shap.max(), full_shap.max())],
                 [min(fold_shap.min(), full_shap.min()), max(fold_shap.max(), full_shap.max())],
                 'r--', lw=2, label='Perfect Agreement')
        
        plt.xlabel('Fold-wise Mean |SHAP|', fontsize=14)
        plt.ylabel('Full Train Mean |SHAP|', fontsize=14)
        plt.title(f'{model} SHAP Consistency\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}, Kendall: {kendall_corr:.3f}\nRelative MAE: {relative_mae:.1f}%', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(SAVE_PNG_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  {model}: Pearson={pearson_corr:.3f}, RelMAE={relative_mae:.1f}%")
        return True
        
    except FileNotFoundError as e:
        print(f"  Warning: Data not found for {model}: {e}")
        return False

# ── Generate Individual Analyses for All Models ───────────────────────
print("Generating individual consistency analyses for all models...")
print("=" * 60)

successful_models = []
failed_models = []

for model in MODELS:
    success = generate_individual_analysis(model)
    if success:
        successful_models.append(model)
    else:
        failed_models.append(model)

print("\n" + "=" * 60)
print(f"Successfully processed: {successful_models}")
if failed_models:
    print(f"Failed to process: {failed_models}")

print(f"\nIndividual analysis files saved in each model's result directory:")
print(f"  - {model}_shap_comparison.txt")
print(f"  - {model}_shap_comparison_plot.png")