import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── Settings ──────────────────────────────────────────────
train_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
feature_files = [
    "C:/Users/LG_LAB/Desktop/SSELCPP/Filtermethod/Featureimportance/feature_importance_1.csv",
    "C:/Users/LG_LAB/Desktop/SSELCPP/Filtermethod/Featureimportance/feature_importance_2.csv",
    "C:/Users/LG_LAB/Desktop/SSELCPP/Filtermethod/Featureimportance/feature_importance_3.csv",
    "C:/Users/LG_LAB/Desktop/SSELCPP/Filtermethod/Featureimportance/feature_importance_4.csv"
]
seeds = list(range(50))

# Hyperparameter grid
n_estimators_list = [100, 300, 600, 800]
max_features_list = ['sqrt']
min_samples_leaf_list = [5, 6, 7, 8]

# ── Load dataset ──────────────────────────────────────────
df = pd.read_csv(train_path)
df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x.lower() else 0)
y = df["Label"].values

# Store best hyperparameters
best_params = {}

# ── Hyperparameter tuning ─────────────────────────────────
for i, feature_file in enumerate(feature_files):
    method = f"Method_{i+1}"
    print(f"\n Hyperparameter tuning for: {method}")
    feature_df = pd.read_csv(feature_file)
    features = feature_df["Feature"].tolist()
    X = df[features].values

    param_list = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        clf = RandomForestClassifier(random_state=seed)
        grid = GridSearchCV(
            clf,
            {
                'n_estimators': n_estimators_list,
                'max_features': max_features_list,
                'min_samples_leaf': min_samples_leaf_list
            },
            cv=cv,
            scoring='accuracy',
            n_jobs=8
        )
        grid.fit(X, y)
        best = grid.best_params_
        param_list.append((best['n_estimators'], best['max_features'], best['min_samples_leaf']))

    # Extract mode (most common) hyperparameters
    mode_param = Counter(param_list).most_common(1)[0][0]
    best_params[method] = {
        'n_estimators': mode_param[0],
        'max_features': mode_param[1],
        'min_samples_leaf': mode_param[2]
    }
    print(f" Most frequent hyperparameters: {best_params[method]}")

# ── Save optimal parameters ───────────────────────────────
pd.DataFrame(best_params).T.to_csv("best_rf_params_per_method.csv")
print("\n Extracted optimal Random Forest hyperparameters for all filter methods!")
