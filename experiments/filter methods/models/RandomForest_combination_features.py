import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from tqdm import tqdm
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent

# ── Settings (relative paths) ──────────────────────────────────────────
train_path = base_dir / "data" / "mordred2dfeature_knn.csv"
feature_files = [
    base_dir / "data" / "feature_importance" / "feature_importance_1.csv",
    base_dir / "data" / "feature_importance" / "feature_importance_2.csv",
    base_dir / "data" / "feature_importance" / "feature_importance_3.csv",
    base_dir / "data" / "feature_importance" / "feature_importance_4.csv"
]
best_param_file = base_dir / "data" / "best_rf_params_per_method.csv"
output_dir = base_dir / "results" / "combination_results"

# ── Ensure output directory exists ─────────────────────────────────────
output_dir.mkdir(parents=True, exist_ok=True)

eval_seeds = list(range(50))

# Number of top features to evaluate
feature_counts = [50, 100, 150, 200, 250, 300]

# ── Load best hyperparameters ─────────────────────────────
best_param_df = pd.read_csv(best_param_file, index_col=0)
best_params = best_param_df.to_dict(orient='index')

# ── Load dataset ──────────────────────────────────────────
df = pd.read_csv(train_path)
df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x.lower() else 0)
y = df["Label"].values

# ── Run experiments ───────────────────────────────────────
for num_features in feature_counts:
    results = []

    print(f"\n=== Using Top {num_features} Features ===")
    
    # 4x4 combination experiments
    for src_method, src_params in best_params.items():
        for i, feature_file in enumerate(feature_files):
            tgt_method = f"Method_{i+1}"
            print(f"\n▶ Combination: {src_method} → {tgt_method} (Top {num_features} Features)")
            feature_df = pd.read_csv(feature_file)

            # Select top-N features
            features = feature_df["Feature"].head(num_features).tolist()
            X = df[features].values

            for seed in tqdm(eval_seeds, desc=f"Evaluating {src_method} → {tgt_method} (Top {num_features})"):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                accs, aucs, mccs = [], [], []

                for train_idx, val_idx in skf.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    model = RandomForestClassifier(
                        n_estimators=int(src_params['n_estimators']),
                        max_features=src_params['max_features'],
                        min_samples_leaf=int(src_params['min_samples_leaf']),
                        random_state=seed, n_jobs=8
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    y_proba = model.predict_proba(X_val)[:, 1]

                    accs.append(accuracy_score(y_val, y_pred))
                    aucs.append(roc_auc_score(y_val, y_proba))
                    mccs.append(matthews_corrcoef(y_val, y_pred))

                # Store results
                results.append({
                    'source_method': src_method,
                    'target_method': tgt_method,
                    'seed': seed,
                    'num_features': num_features,
                    'CV Accuracy': np.mean(accs),
                    'CV AUC': np.mean(aucs),
                    'CV MCC': np.mean(mccs),
                    'ACC Std': np.std(accs),
                    'AUC Std': np.std(aucs),
                    'MCC Std': np.std(mccs)
                })

    # Save results
    output_csv = output_dir / f"4x4_rf_combination_results_top{num_features}.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"\n All combination experiments completed (Top {num_features} Features). Results saved to '{output_csv}'.")