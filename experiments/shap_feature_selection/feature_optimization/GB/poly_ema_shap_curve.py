import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
from pathlib import Path

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent.parent.parent

# ── Settings ────────────────────────────────────────────
SAVE_DIR = current_dir / "curve"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = base_dir / "data" / "mordred2dfeature_knn.csv"
RANKED_FEATURES_PATH = base_dir / "shap_feature_selection" / "results" / "GB" / "full_train" / "ranked_features.csv"
SCORE_CSV_PATH = SAVE_DIR / "gb_performance_by_features.csv"
PLOT_PATH = SAVE_DIR / "gb_feature_curve_plot.png"

# ── Load Data ───────────────────────────────────────────
ranked_features = pd.read_csv(RANKED_FEATURES_PATH, header=None).iloc[:, 0].tolist()
df = pd.read_csv(DATA_PATH)
X_full = df.drop(columns=["Label", "id"], errors="ignore")
y = df["id"].apply(lambda x: 1 if "positive" in x else 0)

# ── Initialize performance storage ──────────────────────
feature_counts = list(range(5, 318, 5))
acc_list, roc_list, mcc_list = [], [], []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for k in tqdm(feature_counts, desc="🔄 Feature Count"):
    subset = ranked_features[:k]
    X_sub = X_full[subset]

    acc_fold, roc_fold, mcc_fold = [], [], []

    for train_idx, val_idx in kf.split(X_sub, y):
        model = GradientBoostingClassifier(
            n_estimators=480,
            learning_rate=0.06770519235472558,
            max_depth=4,
            subsample=0.999340826942597,
            min_samples_split=8,
            min_samples_leaf=7,
            random_state=0,
        )
        model.fit(X_sub.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X_sub.iloc[val_idx])
        probs = model.predict_proba(X_sub.iloc[val_idx])[:, 1]

        acc_fold.append(accuracy_score(y.iloc[val_idx], preds))
        roc_fold.append(roc_auc_score(y.iloc[val_idx], probs))
        mcc_fold.append(matthews_corrcoef(y.iloc[val_idx], preds))

    acc_list.append(np.mean(acc_fold))
    roc_list.append(np.mean(roc_fold))
    mcc_list.append(np.mean(mcc_fold))

# ── Save Results ────────────────────────────────────────
score_df = pd.DataFrame({
    "Feature_Count": feature_counts,
    "Accuracy": acc_list,
    "ROC_AUC": roc_list,
    "MCC": mcc_list
})
score_df.to_csv(SCORE_CSV_PATH, index=False)
print(f"Performance results saved: {SCORE_CSV_PATH}")

# ── Polynomial Curve Fitting to Find Maxima ─────────────
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

# ── EMA-based Optimal Feature Count ─────────────────────
ema_span = 10
acc_ema = pd.Series(acc_list).ewm(span=ema_span, adjust=False).mean()
roc_ema = pd.Series(roc_list).ewm(span=ema_span, adjust=False).mean()
mcc_ema = pd.Series(mcc_list).ewm(span=ema_span, adjust=False).mean()

acc_ema_opt = feature_counts[int(np.argmax(acc_ema))]
roc_ema_opt = feature_counts[int(np.argmax(roc_ema))]
mcc_ema_opt = feature_counts[int(np.argmax(mcc_ema))]

# ── Polynomial-based Optimal Feature Count ──────────────
acc_poly_curve, acc_poly_opt = fit_curve_and_find_maxima(feature_counts, acc_list)
roc_poly_curve, roc_poly_opt = fit_curve_and_find_maxima(feature_counts, roc_list)
mcc_poly_curve, mcc_poly_opt = fit_curve_and_find_maxima(feature_counts, mcc_list)

# ── Visualization ──────────────────────────────────────
plt.figure(figsize=(18, 5))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(feature_counts, acc_list, label="Accuracy")
plt.plot(feature_counts, acc_ema, linestyle="--", label="EMA")
plt.plot(feature_counts, acc_poly_curve, linestyle="-.", label="PolyFit")
plt.axvline(acc_ema_opt, color="green", linestyle=":", label=f"EMA Opt: {acc_ema_opt}")
plt.axvline(acc_poly_opt, color="red", linestyle="--", label=f"Poly Opt: {acc_poly_opt}")
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Feature Count")
plt.grid(True)
plt.legend()

# ROC-AUC
plt.subplot(1, 3, 2)
plt.plot(feature_counts, roc_list, label="ROC-AUC")
plt.plot(feature_counts, roc_ema, linestyle="--", label="EMA")
plt.plot(feature_counts, roc_poly_curve, linestyle="-.", label="PolyFit")
plt.axvline(roc_ema_opt, color="green", linestyle=":", label=f"EMA Opt: {roc_ema_opt}")
plt.axvline(roc_poly_opt, color="red", linestyle="--", label=f"Poly Opt: {roc_poly_opt}")
plt.xlabel("Number of Features")
plt.ylabel("ROC-AUC")
plt.title("ROC-AUC vs Feature Count")
plt.grid(True)
plt.legend()

# MCC
plt.subplot(1, 3, 3)
plt.plot(feature_counts, mcc_list, label="MCC")
plt.plot(feature_counts, mcc_ema, linestyle="--", label="EMA")
plt.plot(feature_counts, mcc_poly_curve, linestyle="-.", label="PolyFit")
plt.axvline(mcc_ema_opt, color="green", linestyle=":", label=f"EMA Opt: {mcc_ema_opt}")
plt.axvline(mcc_poly_opt, color="red", linestyle="--", label=f"Poly Opt: {mcc_poly_opt}")
plt.xlabel("Number of Features")
plt.ylabel("MCC")
plt.title("MCC vs Feature Count")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
print(f"Optimal feature count visualization saved: {PLOT_PATH}")

# ── Save Optimal Feature Lists ──────────────────────────
# EMA
pd.DataFrame(ranked_features[:acc_ema_opt], columns=["Feature"]).to_csv(
    SAVE_DIR / f"optimal_features_ema_gb_acc_{acc_ema_opt}.csv", index=False)
pd.DataFrame(ranked_features[:roc_ema_opt], columns=["Feature"]).to_csv(
    SAVE_DIR / f"optimal_features_ema_gb_roc_{roc_ema_opt}.csv", index=False)
pd.DataFrame(ranked_features[:mcc_ema_opt], columns=["Feature"]).to_csv(
    SAVE_DIR / f"optimal_features_ema_gb_mcc_{mcc_ema_opt}.csv", index=False)

# PolyFit
pd.DataFrame(ranked_features[:acc_poly_opt], columns=["Feature"]).to_csv(
    SAVE_DIR / f"optimal_features_polyfit_gb_acc_{acc_poly_opt}.csv", index=False)
pd.DataFrame(ranked_features[:roc_poly_opt], columns=["Feature"]).to_csv(
    SAVE_DIR / f"optimal_features_polyfit_gb_roc_{roc_poly_opt}.csv", index=False)
pd.DataFrame(ranked_features[:mcc_poly_opt], columns=["Feature"]).to_csv(
    SAVE_DIR / f"optimal_features_polyfit_gb_mcc_{mcc_poly_opt}.csv", index=False)

print("All optimal feature lists saved (EMA + Polynomial)")