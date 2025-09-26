from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.linear_model import Ridge
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent.parent.parent

# ── Path Settings ──────────────────────────────
DATA_PATH = base_dir / "data" / "mordred2dfeature_knn.csv"
RANKED_FEATURES_PATH = base_dir / "shap_feature_selection" / "results" / "ERT" / "full_train" / "ranked_features.csv"
SAVE_DIR = current_dir / "curve"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SCORE_CSV_PATH = SAVE_DIR / "ert_performance_by_features.csv"
PLOT_PATH = SAVE_DIR / "ert_basic_performance_curve.png"

# ── Load Data ──────────────────────────────
ranked_features = pd.read_csv(RANKED_FEATURES_PATH, header=None).iloc[:, 0].tolist()
df = pd.read_csv(DATA_PATH)
df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x else 0)
X_full = df.drop(columns=["Label", "id"], errors="ignore")
y = df["Label"]

print(f"Dataset: {X_full.shape[0]} samples, {X_full.shape[1]} features")
print(f"SHAP-ranked features loaded: {len(ranked_features)}")

# ── Best ERT hyperparameters ─────────────────
best_params = {
    'n_estimators': 464,
    'max_depth': 25,
    'min_samples_split': 3,
    'min_samples_leaf': 1,
    'max_features': "sqrt",
    'bootstrap': False,
    'random_state': 0,
    'n_jobs': 7
}

# ── Initialize storage for performance ─────────────────────────
feature_counts = list(range(5, min(len(ranked_features), 320), 5))
acc_list, roc_list, mcc_list = [], [], []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

print("Evaluating performance across different feature counts...")
for k in tqdm(feature_counts, desc="Feature Count"):
    subset = ranked_features[:k]
    X_sub = X_full[subset]

    acc_fold, roc_fold, mcc_fold = [], [], []

    for train_idx, val_idx in kf.split(X_sub, y):
        model = ExtraTreesClassifier(**best_params)
        model.fit(X_sub.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X_sub.iloc[val_idx])
        probs = model.predict_proba(X_sub.iloc[val_idx])[:, 1]

        acc_fold.append(accuracy_score(y.iloc[val_idx], preds))
        roc_fold.append(roc_auc_score(y.iloc[val_idx], probs))
        mcc_fold.append(matthews_corrcoef(y.iloc[val_idx], preds))

    acc_list.append(np.mean(acc_fold))
    roc_list.append(np.mean(roc_fold))
    mcc_list.append(np.mean(mcc_fold))

# ── Save Results ───────────────────────────
score_df = pd.DataFrame({
    "Feature_Count": feature_counts,
    "Accuracy": acc_list,
    "ROC_AUC": roc_list,
    "MCC": mcc_list
})
score_df.to_csv(SCORE_CSV_PATH, index=False)
print(f"✅ Performance results saved: {SCORE_CSV_PATH}")

# ── Basic Visualization ─────────────────────────────────────────────
plt.figure(figsize=(15, 5))

for i, (metric, values) in enumerate([("Accuracy", acc_list), ("ROC_AUC", roc_list), ("MCC", mcc_list)]):
    plt.subplot(1, 3, i + 1)
    plt.plot(feature_counts, values, 'o-', linewidth=2, markersize=4)
    plt.xlabel("Number of Features", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f"ExtraTrees {metric} vs Feature Count", fontsize=14)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Basic performance plot saved: {PLOT_PATH}")

print("\n📊 Performance Summary:")
print(f"Max Accuracy: {max(acc_list):.4f} at {feature_counts[acc_list.index(max(acc_list))]} features")
print(f"Max ROC-AUC: {max(roc_list):.4f} at {feature_counts[roc_list.index(max(roc_list))]} features") 
print(f"Max MCC: {max(mcc_list):.4f} at {feature_counts[mcc_list.index(max(mcc_list))]} features")