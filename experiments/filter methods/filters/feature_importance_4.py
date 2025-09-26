import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent

# ── File paths (relative paths) ────────────────────────────────────────
input_file_path = base_dir / "data" / "mordred2d_normalized.csv"
importance_file_path = base_dir / "data" / "feature_importance" / "feature_importance_4.csv"

# ── Ensure output directory exists ─────────────────────────────────────
importance_file_path.parent.mkdir(parents=True, exist_ok=True)

# Pearson correlation threshold
CORRELATION_LIMIT = 0.9

# ── Load dataset ────────────────────────────────────────────
df = pd.read_csv(input_file_path)

#  Create binary label (positive → 1, negative → 0)
df['label'] = df['id'].apply(
    lambda x: 1 if 'positive' in str(x).lower() else (0 if 'negative' in str(x).lower() else np.nan)
)
df = df.dropna(subset=["label"])
df['label'] = df['label'].astype(int)

# Separate features (exclude id and label)
features = df.drop(columns=['id', 'label'], errors='ignore')

print(" Using CPU with 5 processes...")
pearson_corr_matrix = features.corr().values
pearson_corr_matrix[np.tril_indices_from(pearson_corr_matrix)] = np.nan

# Extract highly correlated feature pairs
feature_names = features.columns
high_corr_pairs = []
for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):
        if not np.isnan(pearson_corr_matrix[i, j]) and pearson_corr_matrix[i, j] >= CORRELATION_LIMIT:
            high_corr_pairs.append((feature_names[i], feature_names[j]))

print(f" Found {len(high_corr_pairs)} highly correlated feature pairs.")

#  Function: compare absolute Kendall's tau
def compare_kendall_abstau(pair):
    f1, f2 = pair
    x1, x2 = df[f1], df[f2]
    y = df['label']

    mask1 = x1.notna() & y.notna()
    mask2 = x2.notna() & y.notna()

    tau1, _ = kendalltau(x1[mask1], y[mask1])
    tau2, _ = kendalltau(x2[mask2], y[mask2])

    # Drop the feature with the lower |tau|
    return f1 if (pd.isna(tau1) or abs(tau1) < abs(tau2)) else f2

#  Parallel execution to collect features to remove
if __name__ == '__main__':
    with Pool(processes=min(5, cpu_count())) as pool:
        features_to_remove = pool.map(compare_kendall_abstau, high_corr_pairs)

    features_to_remove = list(set(features_to_remove))
    print(f" Removing {len(features_to_remove)} features with lower |Kendall's tau|.")

    # Drop redundant features
    df_filtered = df.drop(columns=features_to_remove)

    #  Compute Kendall’s tau + p-value for remaining features
    feature_importance = []
    for feature in df_filtered.columns:
        if feature not in ["id", "label"]:
            x = df_filtered[feature]
            y = df_filtered["label"]
            mask = x.notna() & y.notna()
            tau, p_value = kendalltau(x[mask], y[mask])
            feature_importance.append((feature, tau, p_value))

    #  Sort by Kendall’s tau (signed, descending)
    sorted_features = sorted(
        feature_importance,
        key=lambda x: (x[1] if x[1] is not None else -np.inf),
        reverse=True
    )

    #  Save results
    importance_df = pd.DataFrame(sorted_features, columns=["Feature", "Kendall_Tau", "p-value"])
    importance_df.to_csv(importance_file_path, float_format='%.6f', index=False)

    # Print top features
    print("\n Feature Importance (Top Features)")
    for i, (feature, tau, p_value) in enumerate(sorted_features[:20]):
        print(f"{i+1}. {feature}: τ = {tau:.4f}, p = {p_value:.6f}")

    print(f"\n Feature importance saved → {importance_file_path}")
