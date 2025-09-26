from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent.parent.parent

# ───── Settings ─────
BASE_DIR = current_dir
PERF_PATH = current_dir / "curve" / "ert_performance_by_features.csv"
FEATURE_PATH = base_dir / "shap_feature_selection" / "results" / "ERT" / "full_train" / "ranked_features.csv"
SAVE_DIR = current_dir / "curve"

# ── Load Data ──────────────────────────────
if not PERF_PATH.exists():
    print(f"❌ Performance file not found: {PERF_PATH}")
    print("   Please run poly_ema_shap_curve.py first!")
    exit(1)

df_perf = pd.read_csv(PERF_PATH)
ranked_features = pd.read_csv(FEATURE_PATH, header=None)[0].tolist()
feature_counts = df_perf["Feature_Count"].tolist()

print("Performing EMA and Polynomial optimization...")

# ── Function for polynomial curve fitting and finding maxima ────────────────
def fit_curve_and_find_maxima(x_vals, y_vals, degree=3):
    x_vals = np.array(x_vals).reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x_vals)
    model = Ridge()
    model.fit(x_poly, y_vals)
    y_pred = model.predict(x_poly)

    coef = model.coef_
    a = 3 * coef[3]
    b = 2 * coef[2]
    c = coef[1]

    roots = np.roots([a, b, c])
    extrema = []

    for r in roots:
        if np.isreal(r):
            r = r.real
            if x_vals.min() <= r <= x_vals.max():
                second_derivative = 2 * coef[2] + 6 * coef[3] * r
                if second_derivative < 0:
                    extrema.append(r)

    if extrema:
        opt_x = round(min(extrema))
    else:
        opt_x = int(x_vals.max())

    return y_pred, opt_x

# ── EMA-based optimization ─────────────────────────────────────────────
print("🔍 EMA-based optimization...")
ema_span = 10
ema_results = {}

plt.figure(figsize=(18, 5))
for i, metric in enumerate(["Accuracy", "ROC_AUC", "MCC"]):
    y_vals = df_perf[metric].tolist()
    ema_series = pd.Series(y_vals).ewm(span=ema_span, adjust=False).mean()
    opt_feat = feature_counts[int(np.argmax(ema_series))]
    ema_results[metric] = opt_feat

    plt.subplot(1, 3, i + 1)
    plt.plot(feature_counts, y_vals, marker="o", label=metric, alpha=0.7)
    plt.plot(feature_counts, ema_series, linestyle="--", label="EMA", linewidth=2)
    plt.axvline(opt_feat, color="red", linestyle=":", label=f"EMA Opt: {opt_feat}", linewidth=2)

    plt.xlabel("Number of Features", fontsize=20)
    plt.ylabel(metric, fontsize=20)
    plt.title(f"ERT EMA", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18, loc="lower right")

plt.tight_layout()
ema_plot_path = SAVE_DIR / "ert_shap_ema_selection.png"
plt.savefig(ema_plot_path, dpi=1000, bbox_inches='tight')
plt.close()
print(f"✅ EMA plot saved: {ema_plot_path}")

# ── Polynomial-based optimization ──────────────────────────────────────
print("🔍 Polynomial-based optimization...")
poly_results = {}

plt.figure(figsize=(18, 5))
for i, metric in enumerate(["Accuracy", "ROC_AUC", "MCC"]):
    y_vals = df_perf[metric].tolist()
    poly_curve, opt_feat = fit_curve_and_find_maxima(feature_counts, y_vals)
    poly_results[metric] = opt_feat

    plt.subplot(1, 3, i + 1)
    plt.plot(feature_counts, y_vals, marker="o", label=metric, alpha=0.7)
    plt.plot(feature_counts, poly_curve, linestyle="-.", label="PolyFit", linewidth=2)
    plt.axvline(opt_feat, color="red", linestyle=":", label=f"Poly Opt: {opt_feat}", linewidth=2)

    plt.xlabel("Number of Features", fontsize=20)
    plt.ylabel(metric, fontsize=20)
    plt.title(f"ERT Polynomial", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18, loc="lower right")

plt.tight_layout()
poly_plot_path = SAVE_DIR / "ert_shap_polyfit_selection.png"
plt.savefig(poly_plot_path, dpi=1000, bbox_inches='tight')
plt.close()
print(f"✅ Polynomial plot saved: {poly_plot_path}")

# ── Save optimal feature lists ─────────────────────────────────────────
print("💾 Saving optimal feature lists...")

# EMA results
for metric in ["Accuracy", "ROC_AUC", "MCC"]:
    opt_k = ema_results[metric]
    feature_list = ranked_features[:opt_k]
    save_path = SAVE_DIR / f"optimal_features_ema_ert_{metric.lower()}_{opt_k}.csv"
    pd.DataFrame(feature_list, columns=["Feature"]).to_csv(save_path, index=False)

# Polynomial results  
for metric in ["Accuracy", "ROC_AUC", "MCC"]:
    opt_k = poly_results[metric]
    feature_list = ranked_features[:opt_k]
    save_path = SAVE_DIR / f"optimal_features_polyfit_ert_{metric.lower()}_{opt_k}.csv"
    pd.DataFrame(feature_list, columns=["Feature"]).to_csv(save_path, index=False)

# ── Summary ─────────────────────────────────────────────────────────────
print("\n📊 Optimization Results Summary:")
print("EMA Optimal Features:")
for metric, opt_feat in ema_results.items():
    print(f"  {metric}: {opt_feat} features")

print("\nPolynomial Optimal Features:")
for metric, opt_feat in poly_results.items():
    print(f"  {metric}: {opt_feat} features")

# Save summary
summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'ROC_AUC', 'MCC'],
    'EMA_Optimal': [ema_results['Accuracy'], ema_results['ROC_AUC'], ema_results['MCC']],
    'Polynomial_Optimal': [poly_results['Accuracy'], poly_results['ROC_AUC'], poly_results['MCC']]
})
summary_df.to_csv(SAVE_DIR / "ert_optimization_summary.csv", index=False)
print(f"✅ Summary saved: {SAVE_DIR / 'ert_optimization_summary.csv'}")