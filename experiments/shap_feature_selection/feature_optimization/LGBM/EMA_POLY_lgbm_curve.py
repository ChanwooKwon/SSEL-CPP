import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from pathlib import Path

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent.parent.parent

# ───── Settings ─────
SAVE_DIR = current_dir / "curve"
PERF_PATH = SAVE_DIR / "lgb_performance_by_features.csv"

# ───── Load Data ─────
if not PERF_PATH.exists():
    print(f"❌ Performance file not found: {PERF_PATH}")
    print("   Please run poly_ema_shap_curve.py first!")
    exit(1)

df_perf = pd.read_csv(PERF_PATH)
feature_counts = df_perf["Feature_Count"].tolist()

# ───── Polynomial Curve Fitting Function ─────
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

# ───── Save EMA-based Graph ─────
plt.figure(figsize=(18, 5))
ema_span = 10

for i, metric in enumerate(["Accuracy", "ROC_AUC", "MCC"]):
    y_vals = df_perf[metric].tolist()
    ema_series = pd.Series(y_vals).ewm(span=ema_span, adjust=False).mean()
    opt_feat = feature_counts[int(np.argmax(ema_series))]

    plt.subplot(1, 3, i + 1)
    plt.plot(feature_counts, y_vals, marker="o", label=metric)
    plt.plot(feature_counts, ema_series, linestyle="--", label="EMA")
    plt.axvline(opt_feat, color="red", linestyle=":", label=f"EMA Opt: {opt_feat}")

    plt.xlabel("Number of Features", fontsize=20)
    plt.ylabel(metric, fontsize=20)
    plt.title(f"LightGBM EMA", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18, loc="lower right")

plt.tight_layout()
ema_plot_path = SAVE_DIR / "lgb_shap_ema_selection.png"
plt.savefig(ema_plot_path, dpi=1000)
plt.show()
print(f"EMA graph saved: {ema_plot_path}")

# ───── Save Polynomial-based Graph ─────
plt.figure(figsize=(18, 5))

for i, metric in enumerate(["Accuracy", "ROC_AUC", "MCC"]):
    y_vals = df_perf[metric].tolist()
    poly_curve, opt_feat = fit_curve_and_find_maxima(feature_counts, y_vals)

    plt.subplot(1, 3, i + 1)
    plt.plot(feature_counts, y_vals, marker="o", label=metric)
    plt.plot(feature_counts, poly_curve, linestyle="-.", label="PolyFit")
    plt.axvline(opt_feat, color="red", linestyle=":", label=f"Poly Opt: {opt_feat}")

    plt.xlabel("Number of Features", fontsize=20)
    plt.ylabel(metric, fontsize=20)
    plt.title(f"LightGBM Polynomial", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18, loc="lower right")

plt.tight_layout()
poly_plot_path = SAVE_DIR / "lgb_shap_polyfit_selection.png"
plt.savefig(poly_plot_path, dpi=1000)
plt.show()
print(f"Polynomial graph saved: {poly_plot_path}")