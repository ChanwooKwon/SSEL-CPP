import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from scipy.stats import ttest_rel

# 🔧 경로 설정
DATA_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
FEATURE_AFTER_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Method4/Mordred/feature_importance_4.csv"
RESULT_SAVE_PATH = "./ERT_feature_selection_nof_comparison_5fold.txt"


params_before = {
    "n_estimators": 728,
    "max_depth": 46,
    "min_samples_split": 6,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": 0,
    "n_jobs": 7
}

params_after = {
    "n_estimators": 464,
    "max_depth": 25,
    "min_samples_split": 3,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": 0,
    "n_jobs": 7
}


def load_data(feature_csv, data_csv):
    df = pd.read_csv(data_csv)

    
    df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x else 0)

    
    if feature_csv in [None, "", "all"]:
        feature_cols = [col for col in df.columns if col not in ["id", "Label"]]
    else:
        top_features = pd.read_csv(feature_csv)["Feature"].tolist()
        feature_cols = top_features

    X = df[feature_cols]
    y = df["Label"].values
    return X, y



def evaluate_model(X, y, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs, mccs, aucs = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[val_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]

        accs.append(accuracy_score(y[val_idx], pred))
        mccs.append(matthews_corrcoef(y[val_idx], pred))
        aucs.append(roc_auc_score(y[val_idx], proba))

    return accs, mccs, aucs



def paired_t_test(metric1, metric2, name):
    stat, p = ttest_rel(metric1, metric2)
    diff = np.mean(metric2) - np.mean(metric1)
    sig = "" if p < 0.05 else ""
    return f"{name}:\n" \
           f"  Before = {np.mean(metric1):.4f}, After = {np.mean(metric2):.4f}\n" \
           f"  Δ = {diff:.4f}, p-value = {p:.4f} {sig}\n"


X_before, y = load_data("all", DATA_PATH)
X_after, _ = load_data(FEATURE_AFTER_PATH, DATA_PATH)


print(" Evaluating ERT (before feature selection)...")
accs_before, mccs_before, aucs_before = evaluate_model(X_before, y, ExtraTreesClassifier(**params_before))

print(" Evaluating ERT (after feature selection)...")
accs_after, mccs_after, aucs_after = evaluate_model(X_after, y, ExtraTreesClassifier(**params_after))


with open(RESULT_SAVE_PATH, "w", encoding="utf-8") as f:
    f.write("ERT before and after (Feature Filtering before and after, 5-fold CV + Paired t-test)\n")
    f.write("=" * 70 + "\n")
    f.write(paired_t_test(accs_before, accs_after, "Accuracy"))
    f.write(paired_t_test(mccs_before, mccs_after, "MCC"))
    f.write(paired_t_test(aucs_before, aucs_after, "ROC-AUC"))

print( RESULT_SAVE_PATH)
