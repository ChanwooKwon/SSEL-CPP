import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from scipy.stats import ttest_rel


params_before = {
    "n_estimators": 418,
    "learning_rate": 0.0977262891229071,
    "max_depth":4,
    "subsample": 0.8605770209457032,
    "min_samples_split": 5,
    "min_samples_leaf": 5,
    "random_state": 0
}

params_after = {
    "n_estimators": 320,
    "learning_rate": 0.035049778272249005,
    "max_depth": 5,
    "subsample": 0.737171292043863,
    "min_samples_split": 8,
    "min_samples_leaf": 7,
    "random_state": 0
}


DATA_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
FEATURE_AFTER_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Method4/Mordred/feature_importance_4.csv"
RESULT_SAVE_PATH = "./gb_filter_comparison_nof_result_5fold.txt"


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


def evaluate_model(X, y, params):
    accs, mccs, aucs = [], [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, val_idx in skf.split(X, y):
        model = GradientBoostingClassifier(**params)
        model.fit(X.iloc[train_idx], y[train_idx])
        preds = model.predict(X.iloc[val_idx])
        probas = model.predict_proba(X.iloc[val_idx])[:, 1]

        accs.append(accuracy_score(y[val_idx], preds))
        mccs.append(matthews_corrcoef(y[val_idx], preds))
        aucs.append(roc_auc_score(y[val_idx], probas))
    return accs, mccs, aucs


def paired_t_test(metric1, metric2, name):
    stat, p = ttest_rel(metric1, metric2)
    diff = np.mean(metric2) - np.mean(metric1)
    return f"{name}:\n" \
           f"  Before = {np.mean(metric1):.4f}, After = {np.mean(metric2):.4f}\n" \
           f"  Δ = {diff:.4f}, p-value = {p:.4f} {'' if p < 0.05 else ''}\n"


X_before, y = load_data("all", DATA_PATH)
X_after, _ = load_data(FEATURE_AFTER_PATH, DATA_PATH)

accs_before, mccs_before, aucs_before = evaluate_model(X_before, y, params_before)
accs_after, mccs_after, aucs_after = evaluate_model(X_after, y, params_after)


with open(RESULT_SAVE_PATH, "w") as f:
    f.write("GB (Feature Filtering before and after, 5-fold CV + Paired t-test)\n")
    f.write("=" * 70 + "\n")
    f.write(paired_t_test(accs_before, accs_after, "Accuracy"))
    f.write(paired_t_test(mccs_before, mccs_after, "MCC"))
    f.write(paired_t_test(aucs_before, aucs_after, "ROC-AUC"))

print(f" {RESULT_SAVE_PATH}")